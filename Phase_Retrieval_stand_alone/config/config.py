from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict, fields as dataclass_fields
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
    nfp_range_um: float = 4.0  # CRITICAL: fixed length of the experiment's z-range (um), not fitted
    zrange: str = "0.0, 3.2"  # display z-range (um)

    # --- Data (no defaults — must be set explicitly per experiment) ---
    project_dir: str = ""     # root directory holding experiment data; "" = resolve relative to cwd
    zstack_folder: str = ""   # subfolder (relative to project_dir) holding the z-stack .tif files
    zstack_file: str = ""     # filename only, resolved against project_dir/zstack_folder
    central_bead_coordinates_pixel: List[int] = field(default_factory=list)  # [row, col]
    offaxis_zstack_files: List[str] = field(default_factory=list)  # filenames only
    offaxis_coords_pixel: List[List[int]] = field(default_factory=list)
    external_mask: Optional[str] = None  # starting-guess mask (.npy/.mat) for phase retrieval, or None for zero-init

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
    lr_sigma_mult: float = 1          # g_sigma LR multiplier;
    lr_d_mult: float = 5000           # mask displacement LR = lr_d_mult * learning_rate; matches root pipeline default
    lr_nfp_mult: float = 20           # NFP center-offset LR = lr_nfp_mult * learning_rate;
    mask_warmup_epochs: int = 100      # initial epochs fitting phase_mask only from the on-axis bead, d/NFP/g_sigma frozen;
    live_debug_every_epochs: int = 20  # how often (epochs) the live GUI panel (loss/param graphs + PSF grid) refreshes

    # --- Per-bead fine alignment ---
    fine_defocus_range_um: float = 0.2
    fine_defocus_step_um: float = 0.1
    max_shift_px: int = 10

    # --- Forward model internals ---
    g_sigma: float = 1.0         # initial Gaussian blur sigma (um); tuned 17/12/2025
    g_size: int = 9              # blur kernel size (pixels)
    circ_scale: float = 5.3/5.8  # aperture scaling; tuned 26/01/2026
    d_min_um: float = 15000      # mask displacement lower bound (um)
    d_max_um: float = 35000      # mask displacement upper bound (um)
    d_init_um: Optional[float] = None   # initial guess for d; None = midpoint of [d_min_um, d_max_um]
    nfp_offset_init_um: Optional[float] = None  # initial guess for the NFP offset; None = midpoint of bounds. Not critical
    nfp_offset_min_um: float = -10      # lower bound for the learned NFP offset (um), sanity limit
    nfp_offset_max_um: float = 10       # upper bound for the learned NFP offset (um), sanity limit

    # --- Camera / noise ---
    bitdepth: int = 16
    baseline: Optional[float] = None
    read_std: Optional[float] = None
    bg: Optional[float] = None
    non_uniform_noise_flag: bool = True

    # --- Runtime / debug ---
    mask_fit_save_dir: Optional[str] = None  # None -> PROJECT_DIR/mask_fit_outputs
    debug_bfp: bool = True
    debug_every_num_epoch: int = 10  # save on-disk debug PNGs every N epochs (not forward calls)
    debug_max_emitters: Optional[int] = None  # None -> len(offaxis_coords_pixel) + 1


@dataclass
class TrainingDataConfig:
    # --- Primary (adjust in the "Generate Training Data" tab, then Update Preview) ---
    signal_range: str = "500, 3000"        # Nsig_range (photons/emitter)
    background_range: str = "80, 150"      # shot-noise VARIANCE (counts²), not a level; sqrt'd into shot_noise_background_range
    density_range: str = "1, 35"           # num_particles_range (emitters/frame); 35 matches root's max_num_particles default
    zrange_um: str = ""                    # "" => inherit UserConfig.zrange at generation time

    # --- Secondary / advanced ---
    canvas_size_px: int = 121              # H = W of generated frames
    num_z_voxel: int = 81                  # D; matches root's num_z_voxel default
    us_factor: int = 1
    blob_r: int = 3
    blob_sigma: float = 0.65
    blob_maxv: int = 5000
    noise_offset_range: str = "0, 0"       # additive dark/readout offset


@dataclass
class Config:
    user: UserConfig = field(default_factory=UserConfig)
    advanced: AdvancedConfig = field(default_factory=AdvancedConfig)
    training: TrainingDataConfig = field(default_factory=TrainingDataConfig)

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
            'nfp_range_um': u.nfp_range_um,
            'nfp_offset_init_um': a.nfp_offset_init_um,
            'nfp_offset_min_um': a.nfp_offset_min_um, 'nfp_offset_max_um': a.nfp_offset_max_um,
            # bead geometry
            'centralBeadCoordinates_pixel': u.central_bead_coordinates_pixel,
            'offaxis_zstack_files': u.offaxis_zstack_file_paths,
            'offaxis_coords_pixel': u.offaxis_coords_pixel,
            # display
            'zrange': tuple(float(x) for x in u.zrange.split(',')),
            # model internals
            'g_sigma': a.g_sigma, 'g_size': a.g_size,
            'circ_scale': a.circ_scale,
            'd_min_um': a.d_min_um, 'd_max_um': a.d_max_um, 'd_init_um': a.d_init_um,
            # camera / noise
            'bitdepth': a.bitdepth,
            'baseline': a.baseline, 'read_std': a.read_std, 'bg': a.bg,
            'non_uniform_noise_flag': a.non_uniform_noise_flag,
            # runtime
            'mask_fit_save_dir': a.mask_fit_save_dir,
            'debug_bfp': a.debug_bfp,
            'debug_every_num_epoch': a.debug_every_num_epoch,
            'debug_max_emitters': a.debug_max_emitters if a.debug_max_emitters is not None
                                  else len(u.offaxis_coords_pixel) + 1,
        }

    def generate_training_param_dict(self, pr_results: dict) -> dict:
        """
        Parameters consumed by Sampling and ImModelTraining (DS3Dplus/ds3d_utils.py) to
        generate simulated training frames using the phase-retrieval-fitted PSF.
        `pr_results` is the dict returned by app_utils.load_phase_retrieval_results().
        """
        from DS3Dplus.ds3d_utils import select_device

        u, t = self.user, self.training
        H = W = int(t.canvas_size_px)
        D = int(t.num_z_voxel)
        us = int(t.us_factor)
        if us < 1:
            raise ValueError(f"Up-sampling factor must be >= 1 (got {us}).")
        psf_half_size_px = 20  # margin (px) kept clear of the canvas edge for pasted PSF patches
        if H <= 2 * psf_half_size_px:
            raise ValueError(
                f"Training-frame canvas size ({H}px) must be greater than "
                f"{2 * psf_half_size_px}px (2x the PSF-patch edge margin)."
            )
        zrange_source = t.zrange_um if t.zrange_um.strip() else u.zrange
        zmin, zmax = (float(x) for x in zrange_source.split(','))
        ps_xy = u.ps_camera / u.M
        bg_lo, bg_hi = (float(x) for x in t.background_range.split(','))
        g_sigma_fitted = pr_results['g_sigma']

        return {
            'device': select_device(),
            'M': u.M, 'NA': u.NA, 'n_immersion': u.n_immersion, 'lamda': u.lamda,
            'n_sample': u.n_sample, 'f_4f': u.f_4f, 'ps_camera': u.ps_camera, 'ps_BFP': u.ps_BFP,
            'NFP': pr_results['nfp_offset_um'],
            'H': H, 'W': W,
            'phase_mask': pr_results['phase_mask'],
            'g_sigma': (g_sigma_fitted, g_sigma_fitted),
            'mask_offset_in_um': pr_results['d_um'],
            'centralBeadCoordinates_pixel': [H / 2, W / 2],
            'bitdepth': self.advanced.bitdepth,
            'baseline': self.advanced.baseline, 'read_std': self.advanced.read_std,
            'non_uniform_noise_flag': self.advanced.non_uniform_noise_flag,
            'D': D, 'us_factor': us,
            'HH': int(H * us), 'WW': int(W * us),
            'buffer_HH': int(psf_half_size_px * us), 'buffer_WW': int(psf_half_size_px * us),
            'vs_xy': ps_xy / us, 'vs_z': (zmax - zmin) / D, 'zrange': (zmin, zmax),
            'Nsig_range': tuple(float(x) for x in t.signal_range.split(',')),
            'num_particles_range': [int(float(x)) for x in t.density_range.split(',')],
            'blob_r': t.blob_r, 'blob_sigma': t.blob_sigma, 'blob_maxv': t.blob_maxv,
            'shot_noise_background_range': (bg_lo ** 0.5, bg_hi ** 0.5),
            'noise_offset_range': tuple(float(x) for x in t.noise_offset_range.split(',')),
        }

    def generate_pr_dict(self) -> dict:
        """
        Phase retrieval training configuration consumed only by phase_retrieval().
        Describes the optimization run: data path, epochs, LR, per-bead alignment.
        """
        u, a = self.user, self.advanced
        return {
            'zstack_file_path': u.zstack_file_path,
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
            'lr_nfp_mult': a.lr_nfp_mult,
            'mask_warmup_epochs': a.mask_warmup_epochs,
            'live_debug_every_epochs': a.live_debug_every_epochs,
        }

    # --- Serialization ---

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Config:
        user_fields = {f.name for f in dataclass_fields(UserConfig) if f.init}
        adv_fields = {f.name for f in dataclass_fields(AdvancedConfig) if f.init}
        training_fields = {f.name for f in dataclass_fields(TrainingDataConfig) if f.init}
        user_kwargs = {k: v for k, v in d['user'].items() if k in user_fields}
        adv_kwargs = {k: v for k, v in d['advanced'].items() if k in adv_fields}
        training_kwargs = {k: v for k, v in d.get('training', {}).items() if k in training_fields}
        return cls(
            user=UserConfig(**user_kwargs),
            advanced=AdvancedConfig(**adv_kwargs),
            training=TrainingDataConfig(**training_kwargs),
        )

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> Config:
        with open(path) as f:
            return cls.from_dict(json.load(f))
