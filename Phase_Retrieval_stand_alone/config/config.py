from __future__ import annotations
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List


@dataclass
class UserConfig:
    # --- Microscope optics ---
    M: float = 100
    NA: float = 1.45
    n_immersion: float = 1.518
    lamda: float = 0.67       # emission wavelength (um)
    n_sample: float = 1.33
    f_4f: float = 200_000     # 4f relay focal length (um)
    ps_camera: float = 11     # camera pixel size (um)
    ps_BFP: float = 80        # BFP pixel size (um)

    # --- Experiment geometry ---
    NFP: float = -3.9         # nominal focal plane (um)
    nfp_text: str = "-7.5, -3.5, 21"  # "start, end, count" (um)
    # TODO RK: take count (21) from bead stacks (num images in z)
    zrange: str = "0.0, 3.2"  # display z-range (um)

    # --- Data (no defaults — must be set explicitly per experiment) ---
    project_dir: str = ""     # root directory holding experiment data; "" = resolve relative to cwd
    zstack_folder: str = ""   # subfolder (relative to project_dir) holding the z-stack .tif files
    zstack_file: str = ""     # filename only, resolved against project_dir/zstack_folder
    central_bead_coordinates_pixel: List[int] = field(default_factory=list)  # [row, col]
    offaxis_zstack_files: List[str] = field(default_factory=list)  # filenames only
    offaxis_coords_pixel: List[List[int]] = field(default_factory=list)
    external_mask: Optional[str] = None  # path to .npy mask, or None to run phase retrieval
    # TODO (RK): make external_mask a starting point rather than an override of phase retrieval

    # --- Derived: computed from nfp_text in __post_init__, excluded from serialization ---
    nfps: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.nfps = self._parse_nfps()

    def _parse_nfps(self) -> np.ndarray:
        start, stop, n = [x.strip() for x in self.nfp_text.split(',')]
        return np.linspace(float(start), float(stop), int(n))

    def _resolve_data_path(self, filename: str) -> str:
        return str(Path(self.project_dir) / self.zstack_folder / filename)

    @property
    def zstack_file_path(self) -> str:
        return self._resolve_data_path(self.zstack_file)

    @property
    def offaxis_zstack_file_paths(self) -> List[str]:
        return [self._resolve_data_path(f) for f in self.offaxis_zstack_files]


@dataclass
class AdvancedConfig:
    # --- Phase retrieval optimisation ---
    epochs: int = 250
    learning_rate: float = 0.001
    loss_label: int = 1          # 1=Gaussian log-likelihood, 2=L2
    r_bead: float = 0.02         # bead radius (um)
    adam_betas: tuple = (0.9, 0.99)   # Adam (beta1, beta2)
    lr_phase_mult: float = 100000     # phase mask LR = lr_phase_mult * learning_rate
    lr_sigma_mult: float = 0          # g_sigma LR multiplier; 0 = frozen
    lr_d_mult: float = 500            # mask displacement LR = lr_d_mult * learning_rate

    # --- Per-bead fine alignment ---
    fine_defocus_range_um: float = 0.2
    fine_defocus_step_um: float = 0.1
    max_shift_px: int = 10

    # --- Forward model internals ---
    g_sigma: float = 1.2         # initial Gaussian blur sigma (um); tuned 17/12/2025
    g_size: int = 9              # blur kernel size (pixels)
    circ_scale: float = 5.3/5.8  # aperture scaling; tuned 26/01/2026
    d_min_um: float = 15000      # mask displacement lower bound (um)
    d_max_um: float = 30000      # mask displacement upper bound (um)

    # --- Camera / noise ---
    bitdepth: int = 16
    baseline: Optional[float] = None
    read_std: Optional[float] = None
    bg: Optional[float] = None
    non_uniform_noise_flag: bool = True

    # --- Runtime / debug ---
    device: str = "cuda:3"
    mask_fit_save_dir: Optional[str] = None  # None -> PROJECT_DIR/mask_fit_outputs
    debug_bfp: bool = True
    debug_every: int = 250
    debug_max_emitters: Optional[int] = None  # None -> len(offaxis_coords_pixel) + 1


@dataclass
class Config:
    user: UserConfig = field(default_factory=UserConfig)
    advanced: AdvancedConfig = field(default_factory=AdvancedConfig)

    def generate_param_dict(self) -> dict:
        """
        Optical model parameters consumed by ImModel_pr and ImModelTraining.
        Describes the physical microscope: optics, pixel grids, aperture, blur, device.
        """
        u, a = self.user, self.advanced
        return {
            # optics
            'M': u.M, 'NA': u.NA, 'lamda': u.lamda,
            'n_immersion': u.n_immersion, 'n_sample': u.n_sample,
            'f_4f': u.f_4f, 'ps_camera': u.ps_camera, 'ps_BFP': u.ps_BFP,
            'NFP': u.NFP, 'nfps': u.nfps,
            # bead geometry
            'centralBeadCoordinates_pixel': u.central_bead_coordinates_pixel,
            'offaxis_zstack_files': u.offaxis_zstack_file_paths,
            'offaxis_coords_pixel': u.offaxis_coords_pixel,
            # display
            'zrange': tuple(float(x) for x in u.zrange.split(',')),
            # model internals
            'g_sigma': a.g_sigma, 'g_size': a.g_size,
            'circ_scale': a.circ_scale,
            'd_min_um': a.d_min_um, 'd_max_um': a.d_max_um,
            # camera / noise
            'bitdepth': a.bitdepth,
            'baseline': a.baseline, 'read_std': a.read_std, 'bg': a.bg,
            'non_uniform_noise_flag': a.non_uniform_noise_flag,
            # runtime
            'device': a.device,
            'mask_fit_save_dir': a.mask_fit_save_dir,
            'debug_bfp': a.debug_bfp,
            'debug_every': a.debug_every,
            'debug_max_emitters': a.debug_max_emitters if a.debug_max_emitters is not None
                                  else len(u.offaxis_coords_pixel) + 1,
        }

    def generate_pr_dict(self) -> dict:
        """
        Phase retrieval training configuration consumed only by phase_retrieval().
        Describes the optimization run: data path, epochs, LR, per-bead alignment.
        """
        u, a = self.user, self.advanced
        return {
            'zstack_file_path': u.zstack_file_path,
            'nfps': u.nfps,
            'r_bead': a.r_bead,
            'epochs': a.epochs,
            'loss_label': a.loss_label,
            'learning_rate': a.learning_rate,
            'fine_defocus_range_um': a.fine_defocus_range_um,
            'fine_defocus_step_um': a.fine_defocus_step_um,
            'max_shift_px': a.max_shift_px,
            'adam_betas': a.adam_betas,
            'lr_phase_mult': a.lr_phase_mult,
            'lr_sigma_mult': a.lr_sigma_mult,
            'lr_d_mult': a.lr_d_mult,
        }

    # --- Serialization ---

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dict. nfps is excluded (derived from nfp_text)."""
        d = asdict(self)
        d['user'].pop('nfps', None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Config:
        return cls(
            user=UserConfig(**d['user']),
            advanced=AdvancedConfig(**d['advanced']),
        )

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> Config:
        with open(path) as f:
            return cls.from_dict(json.load(f))
