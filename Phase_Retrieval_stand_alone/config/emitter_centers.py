# This file contains paths to multiple experiment records
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

# PSF retrieval inputs

# beads from 16/02/2026
# Beads in air (on coverslip) for 21/01/2026
# inputs mes3 (main microtubuls) - behind_obj_exc_640nm_oil_x100_145_step0.2um_027
ZSTACK_FILES_PATH = Path("2026_02_16_microtubules/beads_shiftedFOV_afterExperiment/")
ZSTACK_FILE = "center_x477_y573_2um.tif"
CENTRAL_BEAD_COORDINATES_PIXEL = [573, 477]  # optical axis reference ([y,x])

OFFAXIS_ZSTACK_FILES = [
    "bottomRight_x866_y765_2um.tif",
    "topRight_x851_y173_2um.tif",
    "top_x364_y152_2um.tif",
    "Left_x146_y417_2um.tif",

    "centerTop_x582_y358.tif",
    "centerBottom_x368_y826.tif",
    "bottom_x682_y982.tif"  # bottom - area of interest so 2um_025
]

OFFAXIS_COORDS_PIXEL = [
    [765, 866],   # [r,c] in full camera coordinates
    [180, 851],
    [152, 364],
    [417, 146],

    [358, 582],
    [826, 368],
    [982, 682]
]

# Raw data (blinking images)
# mes3 of 16/02/2025 experiment
RAW_IMAGE_FOLDER = str(PROJECT_DIR / "2026_02_16_microtubules/mes3_shiftedFOV/" /  "behindObj_exc_640nm_oil_x100_145_031")