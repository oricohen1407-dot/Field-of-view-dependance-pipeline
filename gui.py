"""Gradio web interface for DeepSTORM3D PSF characterization."""
import io
import json
import queue
import sys
import threading
from pathlib import Path

import gradio as gr

from config.config import Config, UserConfig, AdvancedConfig
from config.emitter_centers import (
    PROJECT_DIR as DATA_ROOT_DIR, ZSTACK_FILES_PATH,
    ZSTACK_FILE, CENTRAL_BEAD_COORDINATES_PIXEL, OFFAXIS_ZSTACK_FILES, OFFAXIS_COORDS_PIXEL,
)
from func_utils import characterize_PSF

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_SAVE_PATH = str(PROJECT_DIR / "config" / "config.json")


def _default_config() -> Config:
    """Same experiment defaults main.py uses, for when no saved config.json exists yet."""
    return Config(
        user=UserConfig(
            project_dir=str(DATA_ROOT_DIR),
            zstack_folder=str(ZSTACK_FILES_PATH),
            zstack_file=ZSTACK_FILE,
            central_bead_coordinates_pixel=CENTRAL_BEAD_COORDINATES_PIXEL,
            offaxis_zstack_files=OFFAXIS_ZSTACK_FILES,
            offaxis_coords_pixel=OFFAXIS_COORDS_PIXEL,
            external_mask=None,
        )
    )


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
    """Flatten a Config into the ordered list of Gradio field values (44 items)."""
    u, a = cfg.user, cfg.advanced
    return [
        # ── UserConfig (18) ──────────────────────────────────────────────────
        u.M, u.NA, u.n_immersion, u.lamda, u.n_sample,
        u.f_4f, u.ps_camera, u.ps_BFP,
        u.NFP, u.nfp_text, u.zrange,
        u.project_dir, u.zstack_folder,
        u.zstack_file,
        json.dumps(u.central_bead_coordinates_pixel),
        "\n".join(u.offaxis_zstack_files),
        json.dumps(u.offaxis_coords_pixel),
        u.external_mask or "",
        # ── AdvancedConfig (26) ──────────────────────────────────────────────
        a.epochs, a.learning_rate, a.loss_label, a.r_bead,
        json.dumps(list(a.adam_betas)),
        a.lr_phase_mult, a.lr_sigma_mult, a.lr_d_mult,
        a.fine_defocus_range_um, a.fine_defocus_step_um, a.max_shift_px,
        a.g_sigma, a.g_size, a.circ_scale,
        a.d_min_um, a.d_max_um,
        a.bitdepth,
        "" if a.baseline is None else str(a.baseline),
        "" if a.read_std is None else str(a.read_std),
        "" if a.bg is None else str(a.bg),
        a.non_uniform_noise_flag,
        a.device,
        a.mask_fit_save_dir or "",
        a.debug_bfp,
        a.debug_every,
        "" if a.debug_max_emitters is None else str(a.debug_max_emitters),
    ]


def fields_to_config(
    # UserConfig (18)
    M, NA, n_immersion, lamda, n_sample,
    f_4f, ps_camera, ps_BFP,
    NFP, nfp_text, zrange,
    project_dir, zstack_folder,
    zstack_file,
    central_bead_json, offaxis_files_text, offaxis_coords_json,
    external_mask,
    # AdvancedConfig (26)
    epochs, learning_rate, loss_label, r_bead,
    adam_betas_json,
    lr_phase_mult, lr_sigma_mult, lr_d_mult,
    fine_defocus_range_um, fine_defocus_step_um, max_shift_px,
    g_sigma, g_size, circ_scale,
    d_min_um, d_max_um,
    bitdepth,
    baseline, read_std, bg,
    non_uniform_noise_flag,
    device, mask_fit_save_dir,
    debug_bfp, debug_every, debug_max_emitters,
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
            NFP=float(NFP), nfp_text=str(nfp_text), zrange=str(zrange),
            project_dir=str(project_dir).strip(),
            zstack_folder=str(zstack_folder).strip(),
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
            bitdepth=int(float(bitdepth)),
            baseline=_opt_float(baseline),
            read_std=_opt_float(read_std),
            bg=_opt_float(bg),
            non_uniform_noise_flag=bool(non_uniform_noise_flag),
            device=str(device).strip(),
            mask_fit_save_dir=_opt_str(mask_fit_save_dir),
            debug_bfp=bool(debug_bfp),
            debug_every=int(float(debug_every)),
            debug_max_emitters=_opt_int(debug_max_emitters),
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

    with gr.Blocks(title="DeepSTORM3D") as demo:
        gr.Markdown("# DeepSTORM3D — PSF Characterization")

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

        # ── User Config ──────────────────────────────────────────────────────
        with gr.Group():
            gr.Markdown("### User Config")

            gr.Markdown("**Microscope optics**")
            with gr.Row():
                u_M        = gr.Number(label="Magnification (M)",      value=defaults[0])
                u_NA       = gr.Number(label="NA",                      value=defaults[1])
                u_n_imm    = gr.Number(label="n_immersion",             value=defaults[2])
                u_lamda    = gr.Number(label="λ emission (µm)",         value=defaults[3])
            with gr.Row():
                u_n_sample = gr.Number(label="n_sample",                value=defaults[4])
                u_f4f      = gr.Number(label="f_4f (µm)",               value=defaults[5])
                u_ps_cam   = gr.Number(label="Camera pixel size (µm)",  value=defaults[6])
                u_ps_BFP   = gr.Number(label="BFP pixel size (µm)",    value=defaults[7])

            gr.Markdown("**Experiment geometry**")
            with gr.Row():
                u_NFP      = gr.Number(label="NFP (µm)",                value=defaults[8])
                u_nfp_text = gr.Textbox(label='nfp_text ("start, end, count")', value=defaults[9])
                u_zrange   = gr.Textbox(label='zrange ("min, max" µm)', value=defaults[10])

            gr.Markdown("**Data paths**")
            with gr.Row():
                u_project_dir   = gr.Textbox(label="Project root dir", value=defaults[11])
                u_zstack_folder = gr.Textbox(label="Z-stack folder (relative to project root)", value=defaults[12])
            u_zstack      = gr.Textbox(label="Z-stack file (central bead, filename only)", value=defaults[13])
            u_central     = gr.Textbox(
                label="Central bead coords [row, col] (JSON)", value=defaults[14],
            )
            u_offax_files = gr.Textbox(
                label="Off-axis Z-stack files (filenames only, one per line)",
                value=defaults[15], lines=5,
            )
            u_offax_coord = gr.Textbox(
                label="Off-axis pixel coords [[row, col], ...] (JSON)",
                value=defaults[16], lines=3,
            )
            u_ext_mask    = gr.Textbox(
                label="External mask (.npy path) — leave empty to run phase retrieval",
                value=defaults[17],
            )

        # ── Advanced Config ──────────────────────────────────────────────────
        with gr.Accordion("Advanced Config", open=False):
            gr.Markdown("**Phase retrieval optimisation**")
            with gr.Row():
                a_epochs   = gr.Number(label="Epochs",               value=defaults[18], precision=0)
                a_lr       = gr.Number(label="Learning rate",         value=defaults[19])
                a_loss     = gr.Number(label="Loss (1=Gauss, 2=L2)", value=defaults[20], precision=0)
                a_r_bead   = gr.Number(label="Bead radius (µm)",      value=defaults[21])
            with gr.Row():
                a_betas    = gr.Textbox(label="Adam betas [β1, β2] (JSON)", value=defaults[22])
                a_lr_phase = gr.Number(label="lr_phase_mult",         value=defaults[23])
                a_lr_sigma = gr.Number(label="lr_sigma_mult",         value=defaults[24])
                a_lr_d     = gr.Number(label="lr_d_mult",             value=defaults[25])

            gr.Markdown("**Per-bead fine alignment**")
            with gr.Row():
                a_fd_range = gr.Number(label="Defocus range (µm)",   value=defaults[26])
                a_fd_step  = gr.Number(label="Defocus step (µm)",     value=defaults[27])
                a_max_sh   = gr.Number(label="Max shift (px)",         value=defaults[28], precision=0)

            gr.Markdown("**Forward model**")
            with gr.Row():
                a_g_sigma  = gr.Number(label="g_sigma (µm)",          value=defaults[29])
                a_g_size   = gr.Number(label="g_size (px)",            value=defaults[30], precision=0)
                a_circ     = gr.Number(label="circ_scale",             value=defaults[31])
            with gr.Row():
                a_d_min    = gr.Number(label="d_min (µm)",             value=defaults[32])
                a_d_max    = gr.Number(label="d_max (µm)",             value=defaults[33])

            gr.Markdown("**Camera / noise**")
            with gr.Row():
                a_bitdepth = gr.Number(label="Bit depth",              value=defaults[34], precision=0)
                a_baseline = gr.Textbox(label="Baseline (empty=None)", value=defaults[35])
                a_read_std = gr.Textbox(label="Read std (empty=None)", value=defaults[36])
                a_bg       = gr.Textbox(label="BG (empty=None)",       value=defaults[37])
            a_noisy        = gr.Checkbox(label="Non-uniform noise",    value=defaults[38])

            gr.Markdown("**Runtime / debug**")
            with gr.Row():
                a_device   = gr.Textbox(label="Device",                          value=defaults[39])
                a_save_dir = gr.Textbox(label="mask_fit_save_dir (empty=auto)",  value=defaults[40])
            with gr.Row():
                a_dbg_bfp  = gr.Checkbox(label="Debug BFP",                     value=defaults[41])
                a_dbg_ev   = gr.Number(label="Debug every N epochs",             value=defaults[42], precision=0)
                a_dbg_max  = gr.Textbox(label="debug_max_emitters (empty=auto)", value=defaults[43])

        # ── Run ──────────────────────────────────────────────────────────────
        run_btn = gr.Button("Run Characterize PSF", variant="primary")
        log_out = gr.Textbox(label="Output Log", lines=20, interactive=False)

        # component list — order MUST match config_to_fields / fields_to_config
        all_fields = [
            u_M, u_NA, u_n_imm, u_lamda, u_n_sample,
            u_f4f, u_ps_cam, u_ps_BFP,
            u_NFP, u_nfp_text, u_zrange,
            u_project_dir, u_zstack_folder,
            u_zstack, u_central, u_offax_files, u_offax_coord, u_ext_mask,
            a_epochs, a_lr, a_loss, a_r_bead,
            a_betas, a_lr_phase, a_lr_sigma, a_lr_d,
            a_fd_range, a_fd_step, a_max_sh,
            a_g_sigma, a_g_size, a_circ,
            a_d_min, a_d_max,
            a_bitdepth, a_baseline, a_read_std, a_bg,
            a_noisy, a_device, a_save_dir,
            a_dbg_bfp, a_dbg_ev, a_dbg_max,
        ]

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

        def run_handler(*vals):
            try:
                cfg = fields_to_config(*vals)
            except Exception as exc:
                yield f"[CONFIG ERROR] {exc}"
                return

            q: queue.SimpleQueue = queue.SimpleQueue()
            old_stdout = sys.stdout
            sys.stdout = _StreamToQueue(q)
            done_evt = threading.Event()
            run_error: list = [None]

            def _worker():
                try:
                    characterize_PSF(cfg)
                except Exception as exc:
                    q.put(f"\n[EXCEPTION] {exc}\n")
                    run_error[0] = exc
                finally:
                    sys.stdout = old_stdout
                    done_evt.set()

            threading.Thread(target=_worker, daemon=True).start()

            log = ""
            while True:
                try:
                    chunk = q.get(timeout=0.2)
                    log += chunk
                    yield log
                except queue.Empty:
                    if done_evt.is_set():
                        break
                    yield log  # heartbeat keeps the WebSocket alive

            while not q.empty():
                log += q.get_nowait()

            log += "\n\n--- DONE ---" if run_error[0] is None else f"\n\n--- FAILED: {run_error[0]} ---"
            yield log

        load_file.change(fn=load_handler, inputs=load_file, outputs=all_fields)
        save_btn.click(fn=save_handler, inputs=all_fields, outputs=save_status)
        run_btn.click(fn=run_handler, inputs=all_fields, outputs=log_out)

    demo.queue()
    return demo
