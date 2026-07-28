# main.py
# Run AutoDS3D steps without the GUI. Edit the PARAMS below and run any step.

# Steps to install:
# 1. create a virtual environment and activate it:
#     (python -m venv .venv; .\.venv\Scripts\activate on Windows, or source .venv/bin/activate on Linux/Mac)
# 2. pip install -r requirements.txt
# 3. python main.py

import argparse
import os
from pathlib import Path
from config.config import Config, UserConfig
from config.emitter_centers import (
    ZSTACK_FILE, CENTRAL_BEAD_COORDINATES_PIXEL, OFFAXIS_ZSTACK_FILES, OFFAXIS_COORDS_PIXEL, RAW_IMAGE_FOLDER
)
from func_utils import characterize_PSF
from gui import build_demo

# TODO (RK): Delete and remove import if unnecessary
# Avoid GUI backends on a headless server
os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_DIR = Path(__file__).resolve().parent

# =============================================================================
# EDIT THIS BLOCK to configure your experiment
# =============================================================================
cfg = Config(
    user=UserConfig(
        zstack_file=str(ZSTACK_FILE),
        central_bead_coordinates_pixel=CENTRAL_BEAD_COORDINATES_PIXEL,
        offaxis_zstack_files=OFFAXIS_ZSTACK_FILES,
        offaxis_coords_pixel=OFFAXIS_COORDS_PIXEL,
        external_mask=None,           # or absolute path to a .npy mask
    )
)
# =============================================================================


def run_characterize_PSF():
    characterize_PSF(cfg)
    cfg.save(str(PROJECT_DIR / "config" / "config.json"))


def main():
    parser = argparse.ArgumentParser(description="DeepSTORM3D PSF characterization")
    parser.add_argument("--no-gui", action="store_true", help="Run headlessly without opening the browser GUI")
    args = parser.parse_args()
    if args.no_gui:
        run_characterize_PSF()
    else:
        build_demo().launch()


if __name__ == "__main__":
    main()
