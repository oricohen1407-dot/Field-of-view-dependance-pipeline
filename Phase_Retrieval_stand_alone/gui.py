"""Gradio web interface for DeepSTORM3D PSF characterization."""
import io
import json
import os
import queue
import sys
import threading
from pathlib import Path

import gradio as gr
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from config.config import Config, UserConfig, AdvancedConfig
from config.emitter_centers import (
    PROJECT_DIR as DATA_ROOT_DIR, ZSTACK_FILES_PATH,
    ZSTACK_FILE, CENTRAL_BEAD_COORDINATES_PIXEL, OFFAXIS_ZSTACK_FILES, OFFAXIS_COORDS_PIXEL,
)
from func_utils import characterize_PSF

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_SAVE_PATH = str(PROJECT_DIR / "config" / "config.json")
MICROSCOPES_PATH = str(PROJECT_DIR / "config" / "microscopes.json")

MICROSCOPE_FIELDS = ["M", "NA", "n_immersion", "f_4f", "ps_camera", "ps_BFP", "n_sample", "bitdepth"]

CRITICAL_CSS = """
.critical-config {
    border: 2px solid #d9534f;
    border-radius: 10px;
    padding: 14px;
    background: rgba(217, 83, 79, 0.06);
}
"""


def _default_config() -> Config:
    """Same experiment defaults main.py uses, for when no saved config.json exists yet.

    project_dir points directly at the folder holding the z-stack files (folded together with
    ZSTACK_FILES_PATH) rather than relying on zstack_folder, since the GUI no longer exposes
    that field — it always constructs UserConfig with zstack_folder defaulted to "".
    """
    return Config(
        user=UserConfig(
            project_dir=str(DATA_ROOT_DIR / ZSTACK_FILES_PATH),
            zstack_file=ZSTACK_FILE,
            central_bead_coordinates_pixel=CENTRAL_BEAD_COORDINATES_PIXEL,
            offaxis_zstack_files=OFFAXIS_ZSTACK_FILES,
            offaxis_coords_pixel=OFFAXIS_COORDS_PIXEL,
            external_mask=None,
        )
    )


def _load_microscopes() -> dict:
    """Read the named-microscope-preset library, tolerating a missing/corrupt file."""
    try:
        with open(MICROSCOPES_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_microscopes(microscopes: dict):
    with open(MICROSCOPES_PATH, "w") as f:
        json.dump(microscopes, f, indent=2)


class _StreamToQueue(io.TextIOBase):
    """Redirect stdout writes into a thread-safe queue for GUI streaming."""

    def __init__(self, q: queue.SimpleQueue):
        self._q = q

    def write(self, s: str) -> int:
        if s:
            self._q.put(s)
        return len(s)

    def flush(self):
        pass


def _build_live_figure(live_box: dict):
    """Build the training-progress panel from the latest app_utils._update_live_panel snapshot:
    mask-plane phase (bead-shifted) + effective BFP phase, stacked on the left, next to a rotating
    bead's multi-slice PSF grid (calculated vs. experimental) on top, loss/parameter history
    graphs on the bottom.

    Uses the matplotlib object-oriented API + an explicit Agg canvas (no pyplot global state),
    since this runs on the GUI polling thread while the training worker thread makes its own
    bare plt.* calls (phase_retrieval_with_displacement_iteration/iteration_<epoch>.jpg) — sharing
    pyplot's global figure stack across threads would race.
    """
    phase = live_box.get('phase')
    mask_phase = live_box.get('mask_phase')
    mask_shift_px = live_box.get('mask_shift_px', (0, 0))
    pred_slices = live_box.get('pred_slices')
    target_slices = live_box.get('target_slices')
    if phase is None or mask_phase is None or pred_slices is None or target_slices is None:
        return None

    slice_zi = live_box.get('slice_zi', [])
    slice_nfp = live_box.get('slice_nfp_um', [])
    bead_name = live_box.get('bead_name', '?')
    meta = live_box.get('meta', {})
    loss_hist = live_box.get('loss_history', [])
    d_hist = live_box.get('d_history', [])
    nfp_hist = live_box.get('nfp_offset_history', [])
    g_hist = live_box.get('g_sigma_history', [])

    n_slices = pred_slices.shape[0]
    fig = Figure(figsize=(2.5 + 2.0 * n_slices, 8), constrained_layout=True)
    FigureCanvasAgg(fig)
    subfig_top, subfig_bottom = fig.subfigures(2, 1, height_ratios=[2.2, 1])

    # ---- top: mask-plane phase (row 0) + effective BFP phase (row 1) on the left,
    top_gs = subfig_top.add_gridspec(2, 2 + n_slices, width_ratios=[1.3, 0.08] + [1] * n_slices)

    ax_mask_phase = subfig_top.add_subplot(top_gs[0, 0])
    im_mask_phase = ax_mask_phase.imshow(mask_phase, cmap="twilight")
    dx_px, dy_px = mask_shift_px
    ax_mask_phase.set_title(f"mask-plane phase\n(bead-shifted, Δ=({dx_px},{dy_px})px)", fontsize=9)
    ax_mask_phase.axis("off")
    subfig_top.colorbar(im_mask_phase, ax=ax_mask_phase, fraction=0.046, pad=0.04)

    ax_phase = subfig_top.add_subplot(top_gs[1, 0])
    im_phase = ax_phase.imshow(phase, cmap="twilight")
    ax_phase.set_title("effective BFP phase", fontsize=9)
    ax_phase.axis("off")
    subfig_top.colorbar(im_phase, ax=ax_phase, fraction=0.046, pad=0.04)

    ax_sep = subfig_top.add_subplot(top_gs[:, 1])
    ax_sep.set_xlim(0, 1)
    ax_sep.axvline(0.5, color="black", linewidth=3, alpha=0.8)
    ax_sep.axis("off")

    for col in range(n_slices):
        zi = slice_zi[col] if col < len(slice_zi) else col
        nfp = slice_nfp[col] if col < len(slice_nfp) else float('nan')

        ax_pred = subfig_top.add_subplot(top_gs[0, col + 2])
        ax_pred.imshow(pred_slices[col], cmap="gray")
        ax_pred.set_title(f"z{zi}\nNFP={nfp:.2f}um", fontsize=8)
        ax_pred.set_xticks([]); ax_pred.set_yticks([])
        if col == 0:
            ax_pred.set_ylabel("calculated", fontsize=9)

        ax_tgt = subfig_top.add_subplot(top_gs[1, col + 2])
        ax_tgt.imshow(target_slices[col], cmap="gray")
        ax_tgt.set_xticks([]); ax_tgt.set_yticks([])
        if col == 0:
            ax_tgt.set_ylabel("experimental", fontsize=9)

    subfig_top.suptitle(f"bead: {bead_name}", fontsize=10)

    # ---- bottom: loss + learned parameters over epochs ----
    ax_loss, ax_d, ax_nfp, ax_g = subfig_bottom.subplots(1, 4)
    x = list(range(len(loss_hist)))

    ax_loss.plot(x, loss_hist)
    ax_loss.set_yscale("log")
    ax_loss.set_title(f"loss={meta.get('loss', float('nan')):.4g}", fontsize=9)
    ax_loss.set_xlabel("epoch")

    ax_d.plot(x, d_hist, color="tab:orange")
    ax_d.set_title(f"d={meta.get('d', float('nan')):.1f}um", fontsize=9)
    ax_d.set_xlabel("epoch")

    ax_nfp.plot(x, nfp_hist, color="tab:green")
    ax_nfp.set_title(f"nfp_offset={meta.get('nfp_offset', float('nan')):.2f}um", fontsize=9)
    ax_nfp.set_xlabel("epoch")

    ax_g.plot(x, g_hist, color="tab:red")
    ax_g.set_title(f"g_sigma={meta.get('g_sigma', float('nan')):.3f}", fontsize=9)
    ax_g.set_xlabel("epoch")

    fig.suptitle(f"epoch {meta.get('step', '?')}")
    return fig


# ── Config ↔ field helpers ────────────────────────────────────────────────────

def _opt_float(v):
    s = str(v).strip() if v is not None else ""
    return None if s == "" else float(s)

def _opt_int(v):
    s = str(v).strip() if v is not None else ""
    return None if s == "" else int(float(s))

def _opt_str(v):
    s = str(v).strip() if v is not None else ""
    return None if s == "" else s


def config_to_fields(cfg: Config) -> list:
    """Flatten a Config into the ordered list of Gradio field values (47 items)."""
    u, a = cfg.user, cfg.advanced
    return [
        # ── Microscope preset fields, part 1 (7 of 8 — bitdepth is with AdvancedConfig below) ──
        u.M, u.NA, u.n_immersion, u.lamda, u.n_sample,
        u.f_4f, u.ps_camera, u.ps_BFP,
        # ── UserConfig geometry / data (8) ─────────────────────────────────
        u.nfp_range_um, u.zrange,
        u.project_dir,
        u.zstack_file,
        json.dumps(u.central_bead_coordinates_pixel),
        "\n".join(u.offaxis_zstack_files),
        json.dumps(u.offaxis_coords_pixel),
        u.external_mask or "",
        # ── AdvancedConfig (30) ──────────────────────────────────────────────
        a.epochs, a.learning_rate, a.loss_label, a.r_bead,
        json.dumps(list(a.adam_betas)),
        a.lr_phase_mult, a.lr_sigma_mult, a.lr_d_mult,
        a.fine_defocus_range_um, a.fine_defocus_step_um, a.max_shift_px,
        a.g_sigma, a.g_size, a.circ_scale,
        a.d_min_um, a.d_max_um,
        "" if a.d_init_um is None else str(a.d_init_um),
        a.bitdepth,
        "" if a.baseline is None else str(a.baseline),
        "" if a.read_std is None else str(a.read_std),
        "" if a.bg is None else str(a.bg),
        a.non_uniform_noise_flag,
        a.mask_fit_save_dir or "",
        a.debug_bfp,
        a.debug_every_num_epoch,
        "" if a.debug_max_emitters is None else str(a.debug_max_emitters),
        # ── NFP center offset (learned) — appended, keeps every index above stable ───
        a.lr_nfp_mult,
        "" if a.nfp_offset_init_um is None else str(a.nfp_offset_init_um),
        a.nfp_offset_min_um,
        a.nfp_offset_max_um,
        a.mask_warmup_epochs,
    ]


def fields_to_config(
    # Microscope preset fields, part 1 (7 of 8)
    M, NA, n_immersion, lamda, n_sample,
    f_4f, ps_camera, ps_BFP,
    # UserConfig geometry / data (8)
    nfp_range_um, zrange,
    project_dir,
    zstack_file,
    central_bead_json, offaxis_files_text, offaxis_coords_json,
    external_mask,
    # AdvancedConfig (30)
    epochs, learning_rate, loss_label, r_bead,
    adam_betas_json,
    lr_phase_mult, lr_sigma_mult, lr_d_mult,
    fine_defocus_range_um, fine_defocus_step_um, max_shift_px,
    g_sigma, g_size, circ_scale,
    d_min_um, d_max_um, d_init_um,
    bitdepth,
    baseline, read_std, bg,
    non_uniform_noise_flag,
    mask_fit_save_dir,
    debug_bfp, debug_every_num_epoch, debug_max_emitters,
    lr_nfp_mult, nfp_offset_init_um, nfp_offset_min_um, nfp_offset_max_um,
    mask_warmup_epochs,
) -> Config:
    """Parse ordered Gradio field values back into a Config object."""
    offaxis_files = [
        ln.strip()
        for ln in str(offaxis_files_text).strip().split("\n")
        if ln.strip()
    ]
    return Config(
        user=UserConfig(
            M=float(M), NA=float(NA), n_immersion=float(n_immersion),
            lamda=float(lamda), n_sample=float(n_sample),
            f_4f=float(f_4f), ps_camera=float(ps_camera), ps_BFP=float(ps_BFP),
            nfp_range_um=float(nfp_range_um), zrange=str(zrange),
            project_dir=str(project_dir).strip(),
            zstack_file=str(zstack_file).strip(),
            central_bead_coordinates_pixel=json.loads(str(central_bead_json)),
            offaxis_zstack_files=offaxis_files,
            offaxis_coords_pixel=json.loads(str(offaxis_coords_json)),
            external_mask=_opt_str(external_mask),
        ),
        advanced=AdvancedConfig(
            epochs=int(float(epochs)),
            learning_rate=float(learning_rate),
            loss_label=int(float(loss_label)),
            r_bead=float(r_bead),
            adam_betas=tuple(json.loads(str(adam_betas_json))),
            lr_phase_mult=float(lr_phase_mult),
            lr_sigma_mult=float(lr_sigma_mult),
            lr_d_mult=float(lr_d_mult),
            fine_defocus_range_um=float(fine_defocus_range_um),
            fine_defocus_step_um=float(fine_defocus_step_um),
            max_shift_px=int(float(max_shift_px)),
            g_sigma=float(g_sigma),
            g_size=int(float(g_size)),
            circ_scale=float(circ_scale),
            d_min_um=float(d_min_um),
            d_max_um=float(d_max_um),
            d_init_um=_opt_float(d_init_um),
            bitdepth=int(float(bitdepth)),
            baseline=_opt_float(baseline),
            read_std=_opt_float(read_std),
            bg=_opt_float(bg),
            non_uniform_noise_flag=bool(non_uniform_noise_flag),
            mask_fit_save_dir=_opt_str(mask_fit_save_dir),
            debug_bfp=bool(debug_bfp),
            debug_every_num_epoch=int(float(debug_every_num_epoch)),
            debug_max_emitters=_opt_int(debug_max_emitters),
            lr_nfp_mult=float(lr_nfp_mult),
            nfp_offset_init_um=_opt_float(nfp_offset_init_um),
            nfp_offset_min_um=float(nfp_offset_min_um),
            nfp_offset_max_um=float(nfp_offset_max_um),
            mask_warmup_epochs=int(float(mask_warmup_epochs)),
        ),
    )


# ── UI ────────────────────────────────────────────────────────────────────────

def build_demo() -> gr.Blocks:
    defaults = config_to_fields(_default_config())
    if Path(DEFAULT_SAVE_PATH).exists():
        try:
            defaults = config_to_fields(Config.load(DEFAULT_SAVE_PATH))
        except Exception:
            pass

    with gr.Blocks(title="DeepSTORM3D — FOV-dependance") as demo:
        gr.HTML(f"<style>{CRITICAL_CSS}</style>")
        gr.Markdown("# DeepSTORM3D — FOV-dependance PSF Characterization")

        with gr.Tabs():
            with gr.Tab("Configure"):
                # ── Load / Save ──────────────────────────────────────────────────────
                with gr.Row():
                    load_file = gr.File(
                        label="Load Config from JSON",
                        file_types=[".json"],
                        type="filepath",
                    )
                    with gr.Column():
                        save_btn = gr.Button("Save Config to Disk")
                        save_status = gr.Textbox(
                            show_label=False, interactive=False,
                            placeholder="Save status appears here",
                        )

                # ── Critical config ──────────────────────────────────────────────────
                with gr.Group(elem_classes=["critical-config"]):
                    gr.Markdown("## ⚠️ Critical — configure before running")
                    gr.Markdown("These vary per experiment — double-check before every run.")

                    gr.Markdown(
                        "**Calibration folder** — browse to auto-fill the files below, "
                        "pick the on-axis file, then add coordinates."
                    )
                    with gr.Row():
                        folder_upload = gr.File(
                            label="Browse for calibration data folder",
                            file_count="directory",
                        )
                        with gr.Column():
                            onaxis_picker = gr.Dropdown(
                                label="Which file is the on-axis (central) bead?", choices=[],
                            )
                            move_onaxis_btn = gr.Button("Move to Central Bead field")
                    scan_status = gr.Textbox(
                        show_label=False, interactive=False,
                        placeholder="Folder scan status appears here",
                    )
                    with gr.Row():
                        u_nfp_range = gr.Number(label="NFP z-range (µm)", value=defaults[8])
                        u_lamda    = gr.Number(label="λ emission (µm)",   value=defaults[3])
                    with gr.Row():
                        a_d_min    = gr.Number(label="d_min (µm)", value=defaults[30])
                        a_d_max    = gr.Number(label="d_max (µm)", value=defaults[31])
                    u_zstack      = gr.Textbox(label="Central bead file (filename only)", value=defaults[11])
                    u_central     = gr.Textbox(label="Central bead coords [row, col] (JSON)", value=defaults[12])
                    u_offax_files = gr.Textbox(
                        label="Off-axis files (one per line)", value=defaults[13], lines=5,
                    )
                    u_offax_coord = gr.Textbox(
                        label="Off-axis coords [[row, col], ...] (JSON)", value=defaults[14], lines=3,
                    )

                # ── Microscope setup (named presets) ────────────────────────────────
                with gr.Group():
                    gr.Markdown("### Microscope Setup")
                    gr.Markdown("Fixed for a given physical setup — save/load as a named preset.")
                    microscopes = _load_microscopes()
                    with gr.Row():
                        m_dropdown = gr.Dropdown(
                            label="Microscope preset",
                            choices=list(microscopes.keys()),
                            value="Default" if "Default" in microscopes else None,
                        )
                    with gr.Row():
                        m_M        = gr.Number(label="Magnification (M)",      value=defaults[0])
                        m_NA       = gr.Number(label="NA",                      value=defaults[1])
                        m_n_imm    = gr.Number(label="n_immersion",             value=defaults[2])
                        m_n_sample = gr.Number(label="n_sample",                value=defaults[4])
                    with gr.Row():
                        m_f4f      = gr.Number(label="f_4f (µm)",               value=defaults[5])
                        m_ps_cam   = gr.Number(label="Camera pixel size (µm)",  value=defaults[6])
                        m_ps_BFP   = gr.Number(label="BFP pixel size (µm)",     value=defaults[7])
                        m_bitdepth = gr.Number(label="Bit depth",               value=defaults[33], precision=0)
                    with gr.Row():
                        m_name     = gr.Textbox(label="Save current values as new microscope named:")
                        m_save_btn = gr.Button("Save as Microscope")
                    m_status = gr.Textbox(show_label=False, interactive=False, placeholder="Microscope save status appears here")

                # ── Other settings ───────────────────────────────────────────────────
                with gr.Group():
                    gr.Markdown("### Other settings")
                    u_zrange   = gr.Textbox(label='zrange ("min, max" µm, display only)', value=defaults[9])
                    u_project_dir = gr.Textbox(
                        label="Project root dir (auto-filled by Browse above — points at a "
                              "temp upload copy; edit manually if needed)",
                        value=defaults[10],
                    )
                    u_ext_mask = gr.Textbox(
                        label="Starting-guess mask for phase retrieval (.npy/.mat path, optional)",
                        value=defaults[15],
                    )

                # ── Advanced Config ──────────────────────────────────────────────────
                with gr.Accordion("Advanced Config", open=False):
                    gr.Markdown("**Phase retrieval optimisation**")
                    with gr.Row():
                        a_epochs   = gr.Number(label="Epochs",               value=defaults[16], precision=0)
                        a_lr       = gr.Number(label="Learning rate",         value=defaults[17])
                        a_loss     = gr.Number(label="Loss (1=Gauss, 2=L2)", value=defaults[18], precision=0)
                        a_r_bead   = gr.Number(label="Bead radius (µm)",      value=defaults[19])
                        a_mask_warmup = gr.Number(label="Mask warmup epochs (on-axis only, d/NFP frozen)", value=defaults[46], precision=0)
                    with gr.Row():
                        a_betas    = gr.Textbox(label="Adam betas [β1, β2] (JSON)", value=defaults[20])
                        a_lr_phase = gr.Number(label="lr_phase_mult",         value=defaults[21])
                        a_lr_sigma = gr.Number(label="lr_sigma_mult",         value=defaults[22])
                        a_lr_d     = gr.Number(label="lr_d_mult",             value=defaults[23])
                        a_lr_nfp   = gr.Number(label="lr_nfp_mult",           value=defaults[42])

                    gr.Markdown("**Per-bead fine alignment**")
                    with gr.Row():
                        a_fd_range = gr.Number(label="Defocus range (µm)",   value=defaults[24])
                        a_fd_step  = gr.Number(label="Defocus step (µm)",     value=defaults[25])
                        a_max_sh   = gr.Number(label="Max shift (px)",         value=defaults[26], precision=0)

                    gr.Markdown("**Forward model**")
                    with gr.Row():
                        a_g_sigma  = gr.Number(label="g_sigma (µm)",          value=defaults[27])
                        a_g_size   = gr.Number(label="g_size (px)",            value=defaults[28], precision=0)
                        a_circ     = gr.Number(label="circ_scale",             value=defaults[29])
                    a_d_init   = gr.Textbox(label="d_init (µm, empty=midpoint of [d_min, d_max] above)", value=defaults[32])
                    gr.Markdown("*Only the NFP window's CENTER OFFSET is learned (the range length above "
                                "is fixed). These bounds/init are sanity limits, not critical:*")
                    with gr.Row():
                        a_nfp_offset_init = gr.Textbox(label="nfp_offset init (µm, empty=midpoint of bounds)", value=defaults[43])
                        a_nfp_offset_min  = gr.Number(label="nfp_offset min (µm)", value=defaults[44])
                        a_nfp_offset_max  = gr.Number(label="nfp_offset max (µm)", value=defaults[45])

                    gr.Markdown("**Camera / noise**")
                    with gr.Row():
                        a_baseline = gr.Textbox(label="Baseline (empty=None)", value=defaults[34])
                        a_read_std = gr.Textbox(label="Read std (empty=None)", value=defaults[35])
                        a_bg       = gr.Textbox(label="BG (empty=None)",       value=defaults[36])
                    a_noisy        = gr.Checkbox(label="Non-uniform noise",    value=defaults[37])

                    gr.Markdown("**Runtime / debug**")
                    a_save_dir = gr.Textbox(label="mask_fit_save_dir (empty=auto)",  value=defaults[38])
                    with gr.Row():
                        a_dbg_ev   = gr.Number(label="Debug every N epochs",            value=defaults[40], precision=0)
                        a_dbg_max  = gr.Textbox(label="debug_max_emitters (empty=auto)", value=defaults[41])

            with gr.Tab("Run"):
                a_dbg_bfp = gr.Checkbox(label="Save debug images to disk", value=defaults[39])
                with gr.Row():
                    run_btn = gr.Button("Run Characterize PSF", variant="primary")
                    stop_btn = gr.Button("Stop", variant="stop", interactive=False)
                gr.Markdown("**Latest debug snapshot (live)**")
                live_plot = gr.Plot(show_label=False)
                log_out = gr.Textbox(label="Output Log", lines=20, interactive=False)

        # component list — order MUST match config_to_fields / fields_to_config
        all_fields = [
            m_M, m_NA, m_n_imm, u_lamda, m_n_sample,
            m_f4f, m_ps_cam, m_ps_BFP,
            u_nfp_range, u_zrange,
            u_project_dir,
            u_zstack, u_central, u_offax_files, u_offax_coord, u_ext_mask,
            a_epochs, a_lr, a_loss, a_r_bead,
            a_betas, a_lr_phase, a_lr_sigma, a_lr_d,
            a_fd_range, a_fd_step, a_max_sh,
            a_g_sigma, a_g_size, a_circ,
            a_d_min, a_d_max, a_d_init,
            m_bitdepth, a_baseline, a_read_std, a_bg,
            a_noisy, a_save_dir,
            a_dbg_bfp, a_dbg_ev, a_dbg_max,
            a_lr_nfp, a_nfp_offset_init, a_nfp_offset_min, a_nfp_offset_max,
            a_mask_warmup,
        ]

        # microscope preset fields, in the fixed order used by microscopes.json entries
        microscope_fields = [m_M, m_NA, m_n_imm, m_f4f, m_ps_cam, m_ps_BFP, m_n_sample, m_bitdepth]

        # runtime-only state shared between run_handler and stop_handler — not part of Config,
        # never persisted. "busy" is an explicit one-run-at-a-time guard, kept even though
        # demo.queue()'s default concurrency_limit=1 already serializes Run clicks process-wide.
        _run_state = {"stop_event": None, "busy": False}

        # ── Handlers ─────────────────────────────────────────────────────────

        def load_handler(filepath):
            if not filepath:
                return [gr.update()] * len(all_fields)
            cfg = Config.load(filepath)
            return config_to_fields(cfg)

        def save_handler(*vals):
            try:
                cfg = fields_to_config(*vals)
                cfg.save(DEFAULT_SAVE_PATH)
                return f"Saved to {DEFAULT_SAVE_PATH}"
            except Exception as exc:
                return f"[ERROR] {exc}"

        def scan_folder_handler(file_paths):
            if not file_paths:
                return gr.skip(), gr.skip(), gr.skip(), "No folder selected."
            tif_paths = [f for f in file_paths if str(f).lower().endswith((".tif", ".tiff"))]
            if not tif_paths:
                return gr.skip(), gr.skip(), gr.skip(), "No .tif files found in the selected folder."
            folder = os.path.commonpath(tif_paths)
            names = sorted(os.path.basename(p) for p in tif_paths)
            return (
                folder, "\n".join(names),
                gr.update(choices=names, value=None),
                f"Found {len(names)} .tif file(s) in {folder}",
            )

        def move_onaxis_handler(selected, offaxis_text):
            if not selected:
                return gr.skip(), gr.skip(), gr.skip(), "Pick a file from the dropdown first."
            lines = [ln.strip() for ln in str(offaxis_text).strip().split("\n") if ln.strip()]
            if selected not in lines:
                return gr.skip(), gr.skip(), gr.skip(), f"'{selected}' is not in the off-axis list."
            lines.remove(selected)
            return (
                selected, "\n".join(lines), gr.update(choices=lines, value=None),
                f"Moved '{selected}' to the central-bead field.",
            )

        def microscope_load_handler(name):
            data = _load_microscopes()
            preset = data.get(name)
            if preset is None:
                return [gr.update()] * len(MICROSCOPE_FIELDS)
            return [preset.get(k, gr.update()) for k in MICROSCOPE_FIELDS]

        def microscope_save_handler(name, M, NA, n_immersion, f_4f, ps_camera, ps_BFP, n_sample, bitdepth):
            name = str(name).strip()
            if not name:
                return gr.update(), "[ERROR] Enter a microscope name first."
            data = _load_microscopes()
            data[name] = {
                "M": float(M), "NA": float(NA), "n_immersion": float(n_immersion),
                "f_4f": float(f_4f), "ps_camera": float(ps_camera), "ps_BFP": float(ps_BFP),
                "n_sample": float(n_sample), "bitdepth": int(float(bitdepth)),
            }
            _save_microscopes(data)
            return gr.update(choices=list(data.keys()), value=name), f"Saved microscope '{name}'."

        def stop_handler():
            if _run_state["stop_event"] is not None:
                _run_state["stop_event"].set()
                gr.Info(
                    "Stop requested — finishing the current epoch and saving results. "
                    "This can take a little while.",
                    duration=6,
                )
            return gr.update(value="⏳ Stopping…", interactive=False)

        def run_handler(*vals):
            if _run_state["busy"]:
                yield "[ERROR] A run is already in progress.", gr.skip(), gr.update(interactive=False), gr.update(interactive=True)
                return

            try:
                cfg = fields_to_config(*vals)
            except Exception as exc:
                yield f"[CONFIG ERROR] {exc}", gr.skip(), gr.update(interactive=True), gr.update(interactive=False)
                return

            q: queue.SimpleQueue = queue.SimpleQueue()
            old_stdout = sys.stdout
            sys.stdout = _StreamToQueue(q)
            done_evt = threading.Event()
            run_error: list = [None]
            live_box: dict = {}
            stop_event = threading.Event()
            _run_state["stop_event"] = stop_event
            _run_state["busy"] = True

            def _worker():
                try:
                    characterize_PSF(cfg, live_box=live_box, stop_event=stop_event)
                except Exception as exc:
                    q.put(f"\n[EXCEPTION] {exc}\n")
                    run_error[0] = exc
                finally:
                    sys.stdout = old_stdout
                    _run_state["busy"] = False
                    done_evt.set()

            threading.Thread(target=_worker, daemon=True).start()

            log = ""
            last_seen_version = 0
            while True:
                try:
                    chunk = q.get(timeout=0.2)
                    log += chunk
                except queue.Empty:
                    if done_evt.is_set():
                        break
                    # heartbeat keeps the WebSocket alive — fall through to yield below

                version = live_box.get("version", 0)
                if version != last_seen_version:
                    last_seen_version = version
                    plot_update = _build_live_figure(live_box)
                else:
                    plot_update = gr.skip()
                # once Stop has been clicked, stop_handler already set the "Stopping…" label —
                # keep the button disabled (don't touch its value) instead of re-enabling it
                stop_btn_update = gr.skip() if stop_event.is_set() else gr.update(interactive=True)
                yield log, plot_update, gr.update(interactive=False), stop_btn_update

            while not q.empty():
                log += q.get_nowait()

            version = live_box.get("version", 0)
            if version != last_seen_version:
                plot_update = _build_live_figure(live_box)
            else:
                plot_update = gr.skip()

            _run_state["stop_event"] = None
            log += "\n\n--- DONE ---" if run_error[0] is None else f"\n\n--- FAILED: {run_error[0]} ---"
            yield log, plot_update, gr.update(interactive=True), gr.update(value="Stop", interactive=False)

        load_file.change(fn=load_handler, inputs=load_file, outputs=all_fields)
        save_btn.click(fn=save_handler, inputs=all_fields, outputs=save_status)
        folder_upload.upload(
            fn=scan_folder_handler, inputs=[folder_upload],
            outputs=[u_project_dir, u_offax_files, onaxis_picker, scan_status],
        )
        move_onaxis_btn.click(
            fn=move_onaxis_handler, inputs=[onaxis_picker, u_offax_files],
            outputs=[u_zstack, u_offax_files, onaxis_picker, scan_status],
        )
        m_dropdown.change(fn=microscope_load_handler, inputs=m_dropdown, outputs=microscope_fields)
        m_save_btn.click(fn=microscope_save_handler, inputs=[m_name] + microscope_fields, outputs=[m_dropdown, m_status])
        run_btn.click(fn=run_handler, inputs=all_fields, outputs=[log_out, live_plot, run_btn, stop_btn])
        stop_btn.click(fn=stop_handler, outputs=stop_btn)

    demo.queue()
    return demo
