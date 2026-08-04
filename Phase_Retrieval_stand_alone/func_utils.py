from pathlib import Path
from config.config import Config
from app_utils import (
    phase_retrieval,
    show_z_psf,
    fit_mask_offset_from_offaxis_stacks
)
import numpy as np
import torch
import scipy.io as sio

PROJECT_DIR = Path(__file__).resolve().parent


def characterize_PSF(cfg: Config):
    param_dict = cfg.generate_param_dict()
    pr_dict = cfg.generate_pr_dict()

    device = torch.device(param_dict['device'] if torch.cuda.is_available() else 'cpu')
    param_dict['device'] = device
    print(f'device used (characterize_PSF): {device}')

    # TODO (RK): make external_mask a starting point rather than an override of phase retrieval
    if cfg.user.external_mask is None:
        phase_mask, g_sigma, ccs = phase_retrieval(param_dict, pr_dict)
        print(f'Phase mask is retrieved. blue sigma: {np.round(g_sigma, decimals=2)}.')
        print(f'PSF modeling accuracy: average cc of {np.round(np.mean(ccs), decimals=4)}.')
    else:
        mask_dict = sio.loadmat(cfg.user.external_mask)
        mask_name = list(mask_dict.keys())[3]
        phase_mask = mask_dict[mask_name]
        g_sigma = 0.6

    param_dict['g_sigma'] = (np.round(0.8*g_sigma, decimals=2), np.round(1.0*g_sigma, decimals=2))
    param_dict['phase_mask'] = phase_mask

    show_z_psf(param_dict)

    save_dir = param_dict.get('mask_fit_save_dir') or str(PROJECT_DIR / 'mask_fit_outputs')
    NFP_exp = param_dict['NFP']
    param_dict['NFP'] = 0.0
    fit_mask_offset_from_offaxis_stacks(param_dict, save_dir=save_dir)
    param_dict['NFP'] = NFP_exp

    return ("PSF characterization is done. Check "
            "\nphase_retrieval_results.jpg "
            "\nPSFs.jpg")
