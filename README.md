[README.md](https://github.com/user-attachments/files/30707978/README.md)
# FOV-dependent 3D SMLM Reconstruction with Displaced-Mask PSF Engineering

This repository contains a field-of-view-aware 3D single-molecule localization microscopy (SMLM) reconstruction pipeline. It was developed for experiments in which a phase mask is placed in an arbitrary plane between the objective lens and the tube lens, displaced from the shift-invariant conjugate Fourier/pupil plane. In this configuration, the engineered point-spread function (PSF) can depend on the emitter position in the camera field of view.

The workflow combines bead-based calibration of a displaced phase-mask with optical forward model, synthetic training-data generation, neural-network training, and localization of experimental blinking data.

> **Status:** research code under active development. Paths, microscope parameters, GPU index, and acquisition-specific settings should be edited in `main.py` before running.

---

## Main Idea

In a conventional shift-invariant PSF model, one calibrated PSF is assumed to represent all emitters in the camera field because the Fourier/pupil plane; e.g. a space invariant system.

In the displaced-mask configuration used here, off-axis emitters illuminate shifted regions of the phase mask. Depending on the nominal focal plane and the mask displacement, the beam can also be converging or diverging at the mask plane. This creates field-dependent PSF distortions that may need to be included in the reconstruction model.

This pipeline therefore:

1. Retrieves a phase mask from beads z-stack calibration.
2. Uses a displaced-mask forward model to simulate PSFs at different field positions.
3. Generates synthetic training frames over the full camera field of view.
4. Splits full synthetic frames into smaller training tiles while preserving their global FOV coordinates.
5. Trains a localization network to predict a 3D emitter volume.
6. Applies the trained network to experimental blinking frames using overlapping FOV-aware inference.
7. Exports localization lists as CSV files.

---

## Repository Structure

```text
.
├── main.py              # Main point; run pipeline steps and edit experiment parameters here
├── func_utils.py        # High-level pipeline steps: PSF, preprocessing, training data, training, inference
├── app_utils.py         # Phase retrieval, preprocessing, training-data generation, training, inference
├── app.py               # GUI entry point, if used
└── DS3Dplus/
    ├── ds3d_utils.py    # Optical model, sampling, dataset, volume-to-XYZ conversion, losses
    └── training_utils.py# Training loop and checkpoint handling
```

The recommended workflow is through `main.py`.

---

## Installation / Environment

This is research code and is intended to run in a Python environment with GPU support.

The code uses standard Python scientific-computing and deep-learning libraries. No special custom installation is required beyond a working Python environment with PyTorch.

The main packages used by the code are:

- PyTorch
- NumPy
- SciPy
- scikit-image
- matplotlib
- scikit-learn
- tqdm

The code is intended to run with GPU support. CPU execution may work for small tests, but phase retrieval, training-data generation, network training, and localization are expected to be slow without CUDA.

The recommended workflow is to open the project in PyCharm or another Python IDE, select the appropriate Python environment, edit `main.py` for parameter and which steps of the pipeline to run, and press **Run**.

## Required Input Data

The pipeline expects three main types of data.

### 1. Calibration bead z-stack

A central/on-axis bead z-stack is used for phase retrieval or PSF characterization. The file should be a multi-plane TIFF z-stack.

Example first image from a bead z-stack:

<img width="225" height="225" alt="image" src="https://github.com/user-attachments/assets/da4667d8-77d2-4408-8aa8-db0ee23f576f" />

Configured in `main.py`:

```python
zstack_file = "path/to/central_bead_zstack.tif"
centralBeadCoordinates_pixel = [row, col]
```
Coordinates are in camera pixels and use `[row, col]` ordering.

Example demo file from the microtubules experiment, relative to the repository root:

```text
/Microtubules_dSTORM/Calibration_stack/step0.2um_027_center_x477_y573_2um.tif
```

### 2. Off-axis bead z-stacks

Off-axis z-stacks and their camera coordinates are used to constrain the FOV-dependent phase-retrieval model.

Example first images from off-axis bead z-stacks:

<p align="left">
  <img width="225" height="225" alt="image" src="https://github.com/user-attachments/assets/1fb1e356-ece1-42ec-8547-4171b391c2c1" />
  <img width="225" height="225" alt="image" src="https://github.com/user-attachments/assets/8c3439f4-db17-400b-8660-e610384c5705" />
  <img width="225" height="225" alt="image" src="https://github.com/user-attachments/assets/8aed3e0a-337c-4ceb-91af-d956369e9157" />
</p>

Configured in `main.py`:

```python
offaxis_zstack_files = [
    "path/to/offaxis_bead_1.tif",
    "path/to/offaxis_bead_2.tif",
    "path/to/offaxis_bead_3.tif",
]

offaxis_coords_pixel = [
    [row1, col1],
    [row2, col2],
    [row3, col3],
]
```

Coordinates are in camera pixels and use `[row, col]` ordering.

Example demo file from the microtubules experiment, relative to the repository root:

```text
/Microtubules_dSTORM/Calibration_stack/step0.2um_027_bottomRight_x866_y765_2um.tif
/Microtubules_dSTORM/Calibration_stack/step0.2um_027_topRight_x851_y173_2um.tif
/Microtubules_dSTORM/Calibration_stack/step0.2um_027_top_x364_y152_2um.tif
/Microtubules_dSTORM/Calibration_stack/step0.2um_027_Left_x146_y417_2um.tif
/Microtubules_dSTORM/Calibration_stack/step0.2um_027_centerTop_x582_y358.tif
/Microtubules_dSTORM/Calibration_stack/step0.2um_027_centerBottom_x368_y826.tif
/Microtubules_dSTORM/Calibration_stack/step0.2um_025_bottom_x682_y982.tif
```


### 3. Experimental blinking frames

A folder containing raw experimental SMLM frames as TIFF images.

```python
raw_image_folder = "path/to/blinking_frames"
```

The preprocessing stage creates a background-removed folder with the suffix `_br`.

The experimental frame size should be divisible by the number of tiles. For example, a 1200 px frame can be divided by 8, 6, 5, 4, 3, or 2.

It is recommended to inspect several background-removed frames and estimate the noise range manually.

Example noise/histogram inspection:

<img width="476" height="504" alt="image" src="https://github.com/user-attachments/assets/743870ff-142f-4e5c-a2d5-2090f664bb42" />

<img width="234.5" height="192" alt="image" src="https://github.com/user-attachments/assets/010f43f8-f19c-46ff-899b-0074cf739aee" />

In this example, the selected ROI has a mean gray level of approximately 29, interpreted as the background offset, and a standard deviation of approximately 14, interpreted as the noise level.

Example demo file from the microtubules experiment, relative to the repository root:

```text
/Microtubules_dSTORM/Experiment_50_frames/
```

---

## Main Configuration Parameters

Most experiment-specific parameters should be edited in `main.py`.

### Optical and acquisition parameters

```python
M = 100
NA = 1.45
n_immersion = 1.518
lamda = 0.67       # emission wavelength [um]
n_sample = 1.33
f_4f = 200000      # [um]
ps_camera = 11     # camera pixel size [um]
ps_BFP = 80        # BFP/mask-plane sampling [um]
```

### Axial calibration and reconstruction range

```python
nfp_text = "-5.0, -1.0, 21"  # bead-stack start, end, number of planes [um]
NFP = -3.0 + 1.6              # nominal focal-plane parameter [um]
zrange = [0.2, 3.2]           # reconstructed emitter z-range [um]
num_z_voxel = 81              # number of output z voxels
```

Important distinction:

- `nfp_text` describes the bead z-stack calibration range.
- `zrange` defines the simulated training-data and network-output z-range.
- `NFP` is used when generating simulated images and should account for any offset in the bead-stack range.

By default, the bead z-stack calibration range is usually expected to be symmetric around 0, for example `"-2.0, 2.0, 21"`. In the displaced-mask model, however, this range affects the phase-retrieval result and may need to be determined empirically.

Example:

```python
nfp_text = "-2.0, 2.0, 21"
NFP = 1.6
```

means that the system nominal focal plane is centered at 0 and the measured focal plane is 1.6 um inside the sample.

If instead:

```python
nfp_text = "-5.0, -1.0, 21"
NFP = -3.0 + 1.6
```

then the bead-stack calibration range is centered at -3.0 um, and this offset should be included in `NFP`. Otherwise, the generated training PSFs may be systematically out of focus.

Future versions should either optimize this offset or provide a calibration method to estimate it.

### Training image size and tiling

```python
training_im_size = 1200   # full synthetic camera/canvas size in pixels
num_tiles = 8             # 8 means 8x8 tiles; 4 means 4x4 tiles; 1 means no tiling
```

`training_im_size` is the size of the full synthetic field of view before tiling. The actual network input tile size is:

```text
tile_size_px = training_im_size / num_tiles
```

Example:

```python
training_im_size = 1200
num_tiles = 8
```

This produces 150 x 150 px training tiles.

Example of a single training tile:

<img width="225" height="225" alt="image" src="https://github.com/user-attachments/assets/4d485e02-802d-42e6-906a-c662f32e81e2" />


Tile filename format:

```text
00800_FOV_00012.tif
```

means frame number 800, tile number 12. Tile number 0 is the top-left tile, and tile number `num_tiles**2 - 1` is the bottom-right tile.

### Inference overlap

```python
center_fraction = 0.75
```

This controls how much of each inference tile is trusted during overlapping tiled inference. Smaller values increase overlap and can reduce edge artifacts, but increase runtime.

### Noise and signal simulation

Manual simulation noise settings can be defined in `main.py`:

```python
shot_noise_background_range = (8.0, 16.0)
noise_offset_range = (10.0, 30.0)
```

`snr_roi` is kept for compatibility with the optional automatic SNR/noise-estimation stage inherited from AutoDS3D. In `func3`, `snr_roi` and `max_pv` are used to estimate experimental signal/noise parameters, including the photon-count range for simulated emitters. In the current workflow, these values are usually set manually after inspecting background-removed images.

Future improvements should restore or improve automatic SNR estimation.

---

## Important Parameters in `func_utils.py`

### Gaussian blur parameter

```python
param_dict["g_sigma"] = 1.4
```

This is the image-plane blur parameter used during phase retrieval and training-data simulation. A typical useful range is approximately 0.8 to 1.8. Phase retrieval also optimizes this parameter, preferably with a low learning rate for fine tuning.

### Phase-retrieval dictionary

Important phase-retrieval parameters include:

```python
d_bounds_um = (15000.0, 35000.0)
mask_iris_diameter_mm = 5.3
```

For a Nikon 100x/1.45 objective, the expected mask displacement may be around 27 mm, so the above bounds bracket that value.

The forward model can introduce lateral shifts for off-axis emitters. To improve robustness, phase retrieval allows lateral and axial shifts per off-axis bead when comparing simulation to experiment:

```python
fine_defocus_range_um = 0.4
fine_defocus_step_um = 0.1
max_shift_px = 15
```

The current lateral shift is cyclic, so the emitter ROI should be large enough to avoid cropping. If the allowed shift is too large relative to the bead crop, the PSF may be cropped and phase retrieval may fail. Future versions should handle this automatically.

---

## Running the Pipeline

The recommended workflow is to open `main.py`, edit the configuration values, choose the stages in `CUSTOM_STEPS`, and press **Run** in PyCharm.

Use this short runner at the bottom of `main.py`:

```python
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

CUSTOM_STEPS = ["psf", "preproc", "snr", "td", "train", "test", "localize"]

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


if __name__ == "__main__":
    run_custom_steps(CUSTOM_STEPS)
```

Then edit only `CUSTOM_STEPS` depending on what you want to run:

```python
CUSTOM_STEPS = ["psf"]
CUSTOM_STEPS = ["td", "train"]
CUSTOM_STEPS = ["train", "test", "localize"]
CUSTOM_STEPS = ["localize"]
```

Available pipeline stages:

| Stage | Description |
|---|---|
| `psf` | Phase retrieval / PSF characterization |
| `preproc` | Background removal for experimental frames |
| `snr` | Optional automatic SNR/noise estimation |
| `td` | Synthetic training-data generation |
| `train` | Network training |
| `test` | Test inference on one simulated and one experimental frame |
| `localize` | Full experimental localization |

Before training from scratch, make sure that `resume_net_file` in `func_utils.py` is set to:

```python
resume_net_file = None
```

During training, checkpoints are saved in `training_results/`. The `last_net_*.pt` checkpoint can be used to resume training.

---

## Output Files and Folders

Typical outputs include:

```text
phase_retrieval_outputs/       # retrieved phase mask, fitted parameters, simulated/experimental comparisons
phase_retrieval_results.jpg    # phase-retrieval summary figure
PSFs.jpg                       # model PSF examples
training_data/                 # generated synthetic training data
training_data/x/               # saved training image tiles
training_data/y.pickle         # labels for training images
training_data/param.pickle     # parameter dictionary used for training data
training_results/              # trained network checkpoints and training curves
sim_exp.tif                    # side-by-side simulated/experimental comparison
loss_curves.jpg                # training/validation loss curves
sim_loc_gt_rec.jpg             # test localization comparison
sim_im_gt_rec.jpg              # simulated image/reconstruction overlay
exp_im_gt_rec.jpg              # experimental test image/reconstruction overlay
localizations_*_chunk_*.csv    # final localization chunks, divided by frame chunks
```

By default, localization results may be written in chunks, for example one CSV per 5000 frames.

---

## Notes for Editing or Extending the Code

### Keep global FOV information

For FOV-dependent reconstruction, each training tile must know where it came from in the full camera image. The code stores this using:

```python
param_dict["camera_size_px"]
param_dict["tile_grid"]
param_dict["tile_size_px"]
```

These values are used to build global x/y coordinate channels for training and inference.

### Use compatible training and inference settings

The following parameters should remain consistent between training and inference:

```text
training_im_size
num_tiles
center_fraction
zrange
num_z_voxel
us_factor
threshold
camera_size_px
tile_grid
tile_size_px
```

Changing these after training can lead to incorrect coordinate maps or mismatched network input sizes.

### Full canvas size must be divisible by `num_tiles`

For regular tiling, the full simulated image size should be divisible by the number of tiles.

Examples:

```text
1200 / 8 = 150  OK
1200 / 4 = 300  OK
1152 / 8 = 144  OK
1128 / 8 = 141  OK
```

If the image size does not divide evenly, crop or pad the image so that the full FOV is divisible by `num_tiles`.

---

## Troubleshooting

### CUDA/GPU issues

Set the GPU index in `main.py`:

```python
GPU_INDEX = 1
DEVICE = torch.device(f"cuda:{GPU_INDEX}" if torch.cuda.is_available() else "cpu")
```

If CUDA is unavailable, the code falls back to CPU, but runtime may be impractical for large datasets.

### SNR stage

The SNR stage is inherited from AutoDS3D. It uses an input ROI to estimate background noise and photon count. In the current workflow, manual noise settings are often preferred.

If the SNR stage is skipped, make sure the photon-count range is defined manually before generating training data.

### Empty or poor localizations

If the network output is empty, sparse, or poor, check:

- `threshold`
- `zrange`
- `NFP`
- `center_fraction`
- `training_im_size`
- `num_tiles`
- generated training images in `training_data/x/`
- whether `sim_exp.tif` looks similar to the experimental data
- whether the phase-retrieval output matches the bead z-stacks

### Tiling artifacts

If localization artifacts appear near tile borders, reduce:

```python
center_fraction
```

The value should be between 0 and 1. Reducing `center_fraction` increases overlap during inference.

---

## Citation

If this code is used in a publication, cite the associated manuscript/preprint and the original AutoDS3D work on which the neural-network localization framework is based.

Suggested placeholder:

```text
Author list. Plug-and-play 3D localization microscopy with field-dependent reconstruction. bioRxiv, YEAR.
```

---

## License

Add the repository license here. If no license is provided, the code should be considered unavailable for reuse without permission from the authors.

---

## Contact

For questions, bugs, or data-specific configuration issues, contact the repository maintainer.
