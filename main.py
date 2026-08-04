
# debugger.py  (place this file in your repo root)
import os, sys
from pathlib import Path
import pickle
import argparse
from pathlib import Path
import torch

GPU_INDEX = 3  # GPU 0,1,2 or 3.
DEVICE = torch.device( f"cuda:{GPU_INDEX}" if torch.cuda.is_available() else "cpu")
# Avoid GUI backends on a headless server
os.environ.setdefault("MPLBACKEND", "Agg")

# --- Project paths -----------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent # put this file in the repo root
os.chdir(PROJECT_DIR)                           # important: many functions use CWD
print("CWD:", Path.cwd())

# --- Import pipeline functions ----------------------------------------------
from func_utils import (func1, func2, func3, func4, func5, func6_1, func6_2, func7)

# --- Fixed parameters (EDIT THESE ONCE) -------------------------------------


### measurements microtubules from Feb 2026
# Optical + acquisition
M               = 100       # system magnification
NA              = 1.45      # Numerical aperture of the objective lens
n_immersion     = 1.518     # Refractive index of the immersion medium. Typically: 1.518 for oil, 1.0 for air
lamda           = 0.67      # Emission wavelength in um
n_sample        = 1.33      # Refractive index of measured sample. typically 1.33 for water.
f_TL            = 200000    # Tube lens focal length; Typically 200mm for Nikon microscope. can also appear as f_4f in this code.
ps_camera       = 11        # Camera physical pixel size in um
ps_BFP          = 80        # Back-focal-plane sampling rate. used for phase retrieval.
external_mask   = "None"    # or absolute path to a .mat mask

# beads from 16/02/2026
 #Beads in air (on coverslip) for 21/01/2026
 # directory path to data files
data_dir_path = '/bigdata/ori_cohen/Field-of-view-dependance-Phase-Retrieval/'
Data_Dir = Path(data_dir_path).resolve()
# bead calibration on-axis z_stack files
zstack_file     = str(Data_Dir / "Microtubules_dSTORM/Calibration_stack/" / "step0.2um_027_center_x477_y573_2um.tif")
centralBeadCoordinates_pixel = [573, 477]  # on-axis bead coordinates in pixels ([r,c]). serves as optical axis reference.
# off-axis stack location:
offaxis_zstack_files = [
    str(Data_Dir /"Microtubules_dSTORM/Calibration_stack//" / "step0.2um_027_bottomRight_x866_y765_2um.tif"),
    str(Data_Dir /"Microtubules_dSTORM/Calibration_stack/" / "step0.2um_027_topRight_x851_y173_2um.tif"),
    str(Data_Dir / "Microtubules_dSTORM/Calibration_stack/"  / "step0.2um_027_top_x364_y152_2um.tif"),
    str(Data_Dir / "Microtubules_dSTORM/Calibration_stack/"  / "step0.2um_027_Left_x146_y417_2um.tif"),

    str(Data_Dir / "Microtubules_dSTORM/Calibration_stack/" / "step0.2um_027_centerTop_x582_y358.tif"),
    str(Data_Dir / "Microtubules_dSTORM/Calibration_stack/" / "step0.2um_027_centerBottom_x368_y826.tif"),
    str(Data_Dir / "Microtubules_dSTORM/Calibration_stack/" / "step0.2um_025_bottom_x682_y982.tif")  # bottom - area of interest
]
## off-axis beads pixel coordinates. coresponding to the same order they appear in off-axis stack locations
# The Structure is for easy, small offsets: ([base_r, base_c], [offset_r, offset_c])
raw_data = [
    ([765, 866], [0, 0]),
    ([180, 851], [0, 0]),
    ([152, 364], [0, 0]),
    ([417, 146], [0, 0]),
    ([358, 582], [0, 0]),
    ([826, 368], [0, 0]),
    ([982, 682], [0, 0])
]
# Compute element-wise sums
offaxis_coords_pixel = [[base[0] + offset[0]*0, base[1] + offset[1]*0] for base, offset in raw_data]

nfp_text        = "-5.0, -1.0, 21"  # beads stack range in um. "start, end, count"
# measurement NFP. beads stack range offset should be considered here.
# for example, if nfp_text input is "-2.0, 2.0, 21", this means that the system NFP is 0, and measurement NFP should be as usual, e.g. NFP=1.6 (focus plane is 1.6um inside the sample)
# However, if nfp_text input is "-5.0, -1.0, 21", this means that the system NFP offset is -3.0, and NFP should account for it: e.g. NFP = -3 + 1.6
NFP =  -3.0 + 1.6
zrange = [0.2, 3.2]  # Experimental zrange, format is [start, finish], the reconstruction will be at this range.

snr_roi         = "550, 550, 650, 650"     # r0,c0,r1,c1 (pixels)
max_pv          = 110 #80     # maximum pixel value in graylevel on the camera. NOT taking noise into account. this can be considered as maximum pixel value above noise in the experiment.
# ROI used by the SNR/noise-estimation stage. In the current workflow we manually define the simulation noise via shot_noise_background_range and noise_offset_range,
# so snr_roi may not directly affect training data, but it does use this to estimate photon-count range (Nsig_range).
# However, this parameter is for compatibility and future automatic noise estimation: func3() will use this ROI to estimate experimental signal/noise automatically like in AutoDS3D

# Simulated camera/background noise parameters
shot_noise_background_range = (8.0, 16.0)  # shot noise background (noise std)
noise_offset_range = (10.0, 30.0)   # background dark noise (noise avg)
projection_01   = 0                 # keep 0
num_z_voxel     = 81                # number of z voxels. voxel axial dim is ( zrange[1]-zrange[0] ) / nu_z_voxel
training_im_size= 1200              # image size in pixel of trained image (image size will be training_im_size x training_im_size). currently rectangular.
us_factor       = 1                 # up-scaling factor. keep 1.
num_tiles = 8 // 1                  # Training FOV tiling. value 8 means dividing the fov into 8x8 tiles
max_num_particles = 15*(num_tiles**2)        # over the entire FOV trained image size.
num_training_images = 100          # number of full frame training images. number of tiles: (training_im_size^2 x num_tiles^2)
test_idx        = 30              # index of frame number used in test stage
threshold       = 20              # localization intensity threshold.
center_fraction = 0.75              # Fraction of each inference tile retained as the trusted central region. # Lower values increase overlap between neighboring tiles, used to prevent inter-tile artifacts

# Raw data (blinking images) path
raw_image_folder= str(Data_Dir / "Microtubules_dSTORM/" /  "Experiment_50_frames")  # mes3 of 16/02/2025 experiment

# Optional: reuse a previous param_dict pickle produced by training (func5)
previous_param_dict = "None"   # e.g. "param_dict_01-23_17-02.pickle". Use "None".

# Where we cache GUI-like state between steps
STATE_PICKLE = PROJECT_DIR / ".debug_state.pkl"
f_4f = f_TL
def load_state():
    if STATE_PICKLE.exists():
        with open(STATE_PICKLE, "rb") as f:
            return pickle.load(f)
    return {"param_dict": {}}

def save_state(state):
    with open(STATE_PICKLE, "wb") as f:
        pickle.dump(state, f)

def run_step(step):
    state = load_state()

    args = (
        M, NA, n_immersion, lamda, n_sample, f_4f, ps_camera, ps_BFP, external_mask,
        zstack_file, nfp_text, NFP, zrange, raw_image_folder, snr_roi, max_pv, projection_01,
        num_z_voxel, training_im_size, us_factor, max_num_particles, num_training_images,
        previous_param_dict, test_idx, threshold, state
    )

    state.setdefault("param_dict", {})
    state["param_dict"]["device"] = DEVICE
    state["param_dict"]["centralBeadCoordinates_pixel"] = centralBeadCoordinates_pixel
    state["param_dict"]["offaxis_zstack_files"] = offaxis_zstack_files
    state["param_dict"]["offaxis_coords_pixel"] = offaxis_coords_pixel
    state["param_dict"]["debug_max_emitters"] = len(offaxis_coords_pixel) + 1
    state["param_dict"]["NFP"] = NFP
    state["param_dict"]["zrange"] = zrange
    state["param_dict"]["shot_noise_background_range"] = (shot_noise_background_range)
    state["param_dict"]["noise_offset_range"] = noise_offset_range
    if not 0.0 < center_fraction <= 1.0:
        raise ValueError("center_fraction must be larger than 0 and ""no larger than 1.")
    state["param_dict"]["center_fraction"] = float(center_fraction)
    state["param_dict"]["num_tiles"] = int(num_tiles)

    if step == "psf":
        msg = func1(*args)
    elif step == "preproc":
        msg = func2(*args)
    elif step == "snr":
        msg = func3(*args)
    elif step == "td":  # training data
        msg = func4(*args)
    elif step == "train":
        #msg = func5(*args)
        msg = func5(*args)
    elif step == "test":
        msg = func6_1(*args)
    elif step == "localize":
        msg = func6_2(*args)
    elif step == "all":
        msg = func7(*args)
    else:
        raise ValueError("Unknown step")

    # Persist state (so the next step sees updated param_dict)
    save_state(state)
    print("\n=== STEP OUTPUT ===\n", msg)


# ~~~~~~~~~~~~~~~~
# =============================================================================
# RUNNER
# =============================================================================

VALID_STEPS = {"psf", "preproc", "snr", "td", "train", "test", "localize"}
def run_custom_steps(steps):
    for step in steps:
        if step not in VALID_STEPS:
            raise ValueError(
                f"Unknown step '{step}'. "
                f"Allowed steps are: {sorted(VALID_STEPS)}"
            )

        print(f"\n~~~~~~~~~~~~~~~~~~~~~~ running {step} stage ~~~~~~~~~~~~~~~~~~~~~~")
        run_step(step)


# =============================================================================
# RUN SETTINGS
# =============================================================================
# Choose the stages to run, in order.
# In PyCharm: edit this list and press Run.
#
# Available stages:
#   "psf"       - phase-mask / PSF calibration
#   "preproc"   - background removal
#   "snr"       - optional automatic SNR/noise estimation
#   "td"        - synthetic training-data generation
#   "train"     - network training
#   "test"      - test reconstruction on one frame
#   "localize"  - localization on the experimental movie
#
# Examples:
#   CUSTOM_STEPS = ["psf"]
#   CUSTOM_STEPS = ["td", "train"]
#   CUSTOM_STEPS = ["train", "test", "localize"]
#   CUSTOM_STEPS = ["psf", "preproc", "snr", "td", "train", "test", "localize"]

CUSTOM_STEPS = ["psf", "preproc", "snr", "td", "train", "test", "localize"]

if __name__ == "__main__":
    run_custom_steps(CUSTOM_STEPS)


