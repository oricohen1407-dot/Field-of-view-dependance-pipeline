import os
import csv
import math
import time
import torch
import numpy as np
import tifffile
from skimage import io
from scipy import ndimage
from datetime import datetime
import matplotlib.pyplot as plt
import torch.nn.functional as F
from image_model import ImModel_pr
from DS3Dplus.ds3d_utils import ImModel

def _load_zstack_with_count(path: str):
    """Z (slice count) is authoritative from the file itself — these TIFFs carry no
    z-spacing/calibration metadata. Cross-checks ImageJ's 'slices' tag if present (warn only)."""
    zst = io.imread(path).astype(np.float32)
    if zst.ndim != 3:
        raise ValueError(f"{path}: expected a 3D (Z,H,W) z-stack TIFF, got shape {zst.shape}")
    Z = int(zst.shape[0])
    try:
        with tifffile.TiffFile(path) as tf:
            slices_tag = (tf.imagej_metadata or {}).get('slices')
        if slices_tag is not None and int(slices_tag) != Z:
            print(f"[PR] WARNING: {path}: ImageJ 'slices' tag={slices_tag} but frame count={Z}; using {Z}.")
    except Exception as exc:
        print(f"[PR] NOTE: could not read ImageJ metadata from {path} ({exc}); using frame count {Z}.")
    return zst, Z

def _norm01_sum(im):
    im = im.astype(np.float32, copy=False)
    im = im - im.min()
    s = float(im.sum())
    if s > 0:
        im /= s
    return im

def _norm_zm_unit(x, eps=1e-6):
    x = x - x.mean()
    return x / (x.std() + eps)

@torch.no_grad()
def cc_score(a, b, eps=1e-6):
    a = _norm_zm_unit(a.float(), eps).flatten()
    b = _norm_zm_unit(b.float(), eps).flatten()
    return (a @ b) / (a.norm() * b.norm() + eps)

@torch.no_grad()
def phasecorr_shift_int(a, b, max_shift_px=None, eps=1e-6):
    """
    Returns (dy, dx) integer shift that best aligns b to a (translation only),
    using phase correlation. a,b: [H,W] torch tensors.
    """
    H, W = a.shape
    a0 = _norm_zm_unit(a.float(), eps)
    b0 = _norm_zm_unit(b.float(), eps)

    A = torch.fft.fftn(a0)
    B = torch.fft.fftn(b0)
    R = A * torch.conj(B)
    R = R / (torch.abs(R) + eps)
    cc = torch.fft.ifftn(R).real  # [H,W]

    # peak index
    k = torch.argmax(cc)
    py = (k // W).item()
    px = (k %  W).item()

    # wrap to signed
    if py > H // 2: py -= H
    if px > W // 2: px -= W

    # optional: clamp shifts (prevents crazy jumps)
    if max_shift_px is not None:
        py = int(max(-max_shift_px, min(max_shift_px, py)))
        px = int(max(-max_shift_px, min(max_shift_px, px)))

    return py, px

def calculate_cc(output, target):
    # output: rank 3, target: rank 3
    output_mean = np.mean(output, axis=(1, 2), keepdims=True)
    target_mean = np.mean(target, axis=(1, 2), keepdims=True)
    ccs = (np.sum((output - output_mean) * (target - target_mean), axis=(1, 2)) /
           (np.sqrt(np.sum((output - output_mean) ** 2, axis=(1, 2)) * np.sum((target - target_mean) ** 2,
                                                                              axis=(1, 2))) + 1e-9))
    return ccs

def phase_retrieval(param_dict, pr_dict, fig_flag=True):
    device = param_dict['device']

    # ----------------------------
    # Collect stacks: on-axis + off-axis
    # ----------------------------
    stacks = []

    # RK: on-axis stack (x=y=0)
    zstack_on, Z = _load_zstack_with_count(pr_dict['zstack_file_path'])
    stacks.append(("onaxis", zstack_on, 0.0, 0.0))

    # off-axis stacks (if provided)
    if 'offaxis_zstack_files' in param_dict and len(param_dict['offaxis_zstack_files']) > 0:
        r0, c0 = param_dict['centralBeadCoordinates_pixel']
        ps_cam = float(param_dict['ps_camera'])
        M = float(param_dict['M'])

        for f, (rr, cc) in zip(param_dict['offaxis_zstack_files'], param_dict['offaxis_coords_pixel']):
            zst = io.imread(f).astype(np.float32)
            dx_pix = float(cc) - float(c0)
            dy_pix = float(rr) - float(r0)

            # ✅ correct physical conversion (sample-plane um)
            x_um = dx_pix * (ps_cam / M)
            y_um = dy_pix * (ps_cam / M)

            stacks.append((os.path.splitext(os.path.basename(f))[0], zst, x_um, y_um))

    # ----------------------------
    # Normalize and pack into one training batch
    # ----------------------------
    y_list = []
    xyz_list = []
    zi_list = []
    bead_id_list = []
    stack_index = 0

    is_onaxis_list = []  # <-- add

    for name, zst, x_um, y_um in stacks:
        stack_index += 1
        bead_id = stack_index - 1  # added on 15/03/2026
        if zst.shape[0] != Z:
            raise ValueError(
                f"{name}: Z mismatch. stack has {zst.shape[0]} but on-axis calibration stack has {Z}"
            )

        # ---------- ORIGINAL PR BACKGROUND CLEANUP ----------
        corner_size = max(7, int(0.1 * zst.shape[1]))

        patches = np.concatenate(
            (
                np.concatenate(
                    (zst[:, :corner_size, :corner_size],
                     zst[:, :corner_size, -corner_size:]),
                    axis=2
                ),
                np.concatenate(
                    (zst[:, -corner_size:, :corner_size],
                     zst[:, -corner_size:, -corner_size:]),
                    axis=2
                ),
            ),
            axis=1
        )

        means = np.mean(patches, axis=(1, 2), keepdims=True)
        stds = np.std(patches, axis=(1, 2), keepdims=True)

        zst = zst - means
        mask = (zst > stds*1)

        struct = ndimage.generate_binary_structure(2, 1)
        mask = np.array([
            ndimage.binary_dilation(
                ndimage.binary_erosion(mask[i], struct),
                struct
            )
            for i in range(mask.shape[0])
        ], dtype=np.float32)

        zst = zst * mask
        # ---------- END CLEANUP ----------
        # Autocorrelation
        '''# after:
        # zst = zst * mask

        if pr_dict.get("recenter_offaxis", True) and (name != "onaxis"):
            zst, shifts = recenter_stack_per_slice(zst, ref_mode="midz", upsample=10)
        #    if name == "onaxis":
        #        print(f"[recenter] {name}: dy={dy:.2f}px dx={dx:.2f}px")'''
        # end autocorrelation


        # normalize AFTER cleanup (as in original PR)
        #zst = zst / (np.sum(zst, axis=(1, 2), keepdims=True) + 1e-12)
        zst = np.clip(zst, 0.0, None).astype(np.float32)
        z_photons = np.sum(zst, axis=(1, 2)).astype(np.float32)
        for zi in range(Z):
            #y_list.append(zst[zi])
            #y_list.append(zst[zi]/z_photons[Z//2])  # normalize according to center

            #if stack_index == 1:
            #norm_factor = z_photons[Z//2]
            norm_factor = z_photons[zi]
            y_list.append(zst[zi] / norm_factor)  # normalize according to center
            #xyz_list.append([x_um, y_um, 0.0, float(z_photons[zi])])  # <-- photons restored
            xyz_list.append([x_um, y_um, 0.0, 1.0])
            zi_list.append(zi)
            bead_id_list.append(bead_id)
            is_onaxis_list.append(name == "onaxis")  # <-- add

        # normalize AFTER masking
        #zst = zst / (np.sum(zst, axis=(1, 2), keepdims=True) + 1e-12)

    y_true = torch.from_numpy(np.stack(y_list, 0)).to(device)  # [B,H,W]
    xyzps = torch.from_numpy(np.asarray(xyz_list, np.float32)).to(device)  # [B,4]

    # per-sample slice index; NFPs itself is recomputed from im_model.nfps(zi_tensor) each epoch
    zi_tensor = torch.tensor(np.asarray(zi_list, np.int64), device=device)
    bead_ids = torch.tensor(np.asarray(bead_id_list, np.int64), device=device)  # added on 15/03/2026
    is_onaxis = torch.tensor(is_onaxis_list, device=device)  # [B] bool

    # ----------------------------
    # Build PR model (now includes d via ASM)
    # ----------------------------

    # zstack is numpy (Z,Hroi,Wroi)
    Hroi, Wroi = y_true.shape[-2], y_true.shape[-1]  # <-- ALWAYS matches the training target
    param_dict['H'] = int(Hroi)
    param_dict['W'] = int(Wroi)

    params_pr = dict(param_dict)
    params_pr['H'] = int(Hroi)
    params_pr['W'] = int(Wroi)

    # initial d: explicit Config override > warm start from a prior run > bounds midpoint
    d_init_um = param_dict['d_init_um']
    if d_init_um is None:
        d_init_um = param_dict.get('mask_offset_in_um')  # warm start, e.g. resuming a prior fit
    if d_init_um is None:
        d_init_um = 0.5 * (param_dict['d_min_um'] + param_dict['d_max_um'])
    d_init_um = float(d_init_um)

    if not (param_dict['d_min_um'] <= d_init_um <= param_dict['d_max_um']):
        raise ValueError(
            f"d_init_um={d_init_um} must be inside "
            f"[{param_dict['d_min_um']}, {param_dict['d_max_um']}]."
        )

    params_pr['mask_offset_in_um'] = d_init_um
    # end ori's edit from 26/01/2026 for improved pr with displacement

    # initial NFP offset: explicit Config override > warm start from a prior run > bounds midpoint
    nfp_range_um = param_dict['nfp_range_um']
    nfp_offset_min_um = param_dict['nfp_offset_min_um']
    nfp_offset_max_um = param_dict['nfp_offset_max_um']

    nfp_offset_init = param_dict['nfp_offset_init_um']
    if nfp_offset_init is None:
        nfp_offset_init = param_dict.get('nfp_offset_um')  # warm start, e.g. resuming a prior fit
    if nfp_offset_init is None:
        nfp_offset_init = 0.5 * (nfp_offset_min_um + nfp_offset_max_um)
    nfp_offset_init = float(nfp_offset_init)

    if not (nfp_offset_min_um <= nfp_offset_init <= nfp_offset_max_um):
        raise ValueError(
            f"nfp_offset_init_um={nfp_offset_init} must be inside "
            f"[{nfp_offset_min_um}, {nfp_offset_max_um}]."
        )

    params_pr['nfp_range_um'] = nfp_range_um
    params_pr['nfp_offset_min_um'] = nfp_offset_min_um
    params_pr['nfp_offset_max_um'] = nfp_offset_max_um
    params_pr['nfp_offset_init_um'] = nfp_offset_init
    params_pr['Z'] = Z

    im_model = ImModel_pr(params_pr).to(device)

    im_model.train()

    opt = torch.optim.Adam(
        [
            {'params': [im_model.phase_mask], 'lr': pr_dict['lr_phase_mult'] * pr_dict['learning_rate']},
            {'params': [im_model.g_sigma],    'lr': pr_dict['lr_sigma_mult'] * pr_dict['learning_rate']},
            {'params': [im_model.d_raw],      'lr': pr_dict['lr_d_mult']     * pr_dict['learning_rate']},
            {'params': [im_model.nfp_offset_raw], 'lr': pr_dict['lr_nfp_mult'] * pr_dict['learning_rate']},
        ],
        betas=tuple(pr_dict['adam_betas'])
    )

    # ----------------------------
    # Live GUI panel: dense per-epoch history + periodic PSF-grid snapshot
    # ----------------------------
    live_box = param_dict.get('live_box')
    live_debug_every_epochs = int(pr_dict['live_debug_every_epochs'])
    bead_names = [name for name, _, _, _ in stacks]

    loss_history = []
    d_history = []
    nfp_offset_history = []
    g_sigma_history = []
    bead_cursor = 0
    live_panel_time_s = 0.0  # cumulative wall-clock time spent inside _refresh_live_panel

    def _refresh_live_panel(step, loss_val, d_now, nfp_offset_now, g_sigma_now,
                             pred_disp, target_disp, local_bead_ids, local_zi):
        """Unconditionally (re)builds the live_box payload from the current state.
        Callers gate on live_box/refresh-cadence; this never appends to history."""
        nonlocal bead_cursor, live_panel_time_s
        _panel_t0 = time.perf_counter()
        available_beads = torch.unique(local_bead_ids).tolist()
        bead = available_beads[bead_cursor % len(available_beads)]
        bead_cursor += 1

        idx = torch.where(local_bead_ids == bead)[0]
        order = torch.argsort(local_zi[idx])
        idx = idx[order]

        pred_bead = pred_disp[idx].detach().cpu().numpy()      # [Zb,H,W]
        target_bead = target_disp[idx].detach().cpu().numpy()  # [Zb,H,W]

        Zb = pred_bead.shape[0]
        n_slices = min(7, Zb)
        slice_idx = np.round(np.linspace(0, Zb - 1, n_slices)).astype(int)

        pred_slices = np.stack([pred_bead[i] / (pred_bead[i].max() + 1e-12) for i in slice_idx], axis=0)
        target_slices = np.stack([target_bead[i] / (target_bead[i].max() + 1e-12) for i in slice_idx], axis=0)

        with torch.no_grad():
            nfp_vals = im_model.nfps(torch.tensor(slice_idx, device=device)).detach().cpu().numpy()

        mid = idx[len(idx) // 2]
        live_box['phase'] = im_model.last_ef_bfp_phase[mid].cpu().numpy()
        live_box['mask_phase'] = im_model.last_mask_plane_phase[mid].cpu().numpy()
        shift_px = im_model.last_mask_shift_px[mid].cpu().tolist()
        live_box['mask_shift_px'] = (int(shift_px[0]), int(shift_px[1]))
        live_box['pred_slices'] = pred_slices
        live_box['target_slices'] = target_slices
        live_box['slice_zi'] = slice_idx.tolist()
        live_box['slice_nfp_um'] = nfp_vals.tolist()
        live_box['bead_name'] = bead_names[bead]
        live_box['loss_history'] = list(loss_history)
        live_box['d_history'] = list(d_history)
        live_box['nfp_offset_history'] = list(nfp_offset_history)
        live_box['g_sigma_history'] = list(g_sigma_history)
        live_box['meta'] = {
            'step': step, 'loss': loss_val, 'd': d_now,
            'nfp_offset': nfp_offset_now, 'g_sigma': g_sigma_now,
            'bead_name': bead_names[bead],
        }
        live_box['version'] = live_box.get('version', 0) + 1
        live_panel_time_s += time.perf_counter() - _panel_t0

    def _update_live_panel(step, loss_val, d_now, nfp_offset_now, g_sigma_now,
                            pred_disp, target_disp, local_bead_ids, local_zi):
        loss_history.append(loss_val)
        d_history.append(d_now)
        nfp_offset_history.append(nfp_offset_now)
        g_sigma_history.append(g_sigma_now)

        if live_box is None or (step % live_debug_every_epochs) != 0:
            return
        _refresh_live_panel(step, loss_val, d_now, nfp_offset_now, g_sigma_now,
                             pred_disp, target_disp, local_bead_ids, local_zi)

    ccs = []
    fine_defocus_range_um = float(pr_dict.get("fine_defocus_range_um", 0.6))
    fine_defocus_step_um = float(pr_dict.get("fine_defocus_step_um", 0.1))
    max_shift_px = int(pr_dict.get("max_shift_px", 10))

    delta_candidates = np.arange(
        -fine_defocus_range_um,
        fine_defocus_range_um + 0.5 * fine_defocus_step_um,
        fine_defocus_step_um,
        dtype=np.float32
    )
    # end
    stop_event = param_dict.get('stop_event')

    _timing_t0 = time.perf_counter()  # covers warmup + main loop only, for the calc-vs-display breakdown below

    # --- Phase A: mask-only warmup from the on-axis bead alone, d/NFP/g_sigma held fixed ---
    # d's gradient depends on the mask having real structure (see phase_retrieval physics
    # notes); this gives the mask (fast LR) a head start before off-axis beads and NFP join in.
    # g_sigma is held fixed too so the optimizer can't lower loss via blur instead of real mask structure.
    mask_warmup_epochs = int(pr_dict['mask_warmup_epochs'])
    if mask_warmup_epochs > 0:
        onaxis_idx = torch.where(is_onaxis)[0]
        xyzps_onaxis = xyzps[onaxis_idx]
        y_onaxis = y_true[onaxis_idx]
        zi_onaxis = zi_tensor[onaxis_idx]

        held_fixed = [im_model.d_raw, im_model.nfp_offset_raw, im_model.g_sigma]

        for warmup_epoch in range(mask_warmup_epochs):
            if stop_event is not None and stop_event.is_set():
                print(f"[PR][warmup] stop requested — halting at epoch {warmup_epoch}")
                break
            im_model.current_epoch = warmup_epoch
            opt.zero_grad()
            pred = im_model(xyzps_onaxis, im_model.nfps(zi_onaxis), targets=y_onaxis)
            loss = F.mse_loss(pred, y_onaxis)
            loss.backward()

            snapshots = [p.detach().clone() for p in held_fixed]
            opt.step()
            with torch.no_grad():
                for p, snap in zip(held_fixed, snapshots):
                    p.copy_(snap)
                im_model.g_sigma.clamp_(min=1e-3, max=20.0)

            is_last_warmup_epoch = warmup_epoch == mask_warmup_epochs - 1
            if (warmup_epoch % 10) == 0 or is_last_warmup_epoch:
                print(f"[PR][warmup] epoch {warmup_epoch:4d} loss={float(loss.item()):.6g}")

            if live_box is not None and ((warmup_epoch % live_debug_every_epochs) == 0 or is_last_warmup_epoch):
                _refresh_live_panel(
                    warmup_epoch, float(loss.item()),
                    float(im_model.d_um().detach().cpu().item()),
                    float(im_model.nfp_offset_um().detach().cpu().item()),
                    float(im_model.g_sigma.item()),
                    pred, y_onaxis,
                    bead_ids[onaxis_idx], zi_onaxis,
                )

        print(f"[PR] mask warmup done ({mask_warmup_epochs} epochs, on-axis only) "
              f"— d/NFP/g_sigma now free to move, Adam momentum already warmed up")

    # Track the best (lowest-loss) main-loop state so the run's final output reflects the
    # best point found, not wherever training happened to end up — loss can tick back up after
    # its minimum (e.g. g_sigma drifting late, noisy per-bead alignment search), so the last
    # epoch isn't necessarily the best one. Only main-loop epochs are compared (all beads, same
    # loss formulation) — warmup loss (on-axis only) isn't comparable, per existing design.
    best_loss = float('inf')
    best_state = None

    pred_display = target_display = d_now = nfp_offset_now = None
    for epoch in range(pr_dict['epochs']):
        if stop_event is not None and stop_event.is_set():
            print(f"[PR] stop requested — halting at epoch {epoch}")
            break
        # continues on from the warmup phase's own 0..mask_warmup_epochs-1 numbering, so
        # on-disk debug filenames never collide between the two phases
        im_model.current_epoch = mask_warmup_epochs + epoch
        opt.zero_grad()
        NFPs = im_model.nfps(zi_tensor)  # depends on the live nfp_offset_raw
        apply_off_axis_space_invariance = (max_shift_px > 0)

        if not apply_off_axis_space_invariance:
            pred = im_model(xyzps, NFPs, targets=y_true)  #original
            loss = F.mse_loss(pred, y_true)  #original
            # added on 15/03/2026 for small defocus robustness in pr
        else:
            # --------------------------------------------------
            # bead-wise robust alignment:
            # 1) one fixed shift per bead across z
            # 2) one fixed fine-defocus offset per bead across z
            # --------------------------------------------------
            with torch.no_grad():
                y_aligned = y_true.clone()
                nfp_offsets = torch.zeros_like(NFPs)

                unique_beads = torch.unique(bead_ids)

                for bid in unique_beads.tolist():
                    idx = torch.where(bead_ids == bid)[0]

                    # keep on-axis fixed
                    if bool(is_onaxis[idx[0]].item()):
                        continue

                    target_bead = y_true[idx]  # [Z,H,W]

                    best_loss = None
                    best_dd = 0.0
                    best_target_shifted = target_bead.clone()

                    for dd in delta_candidates:
                        nfp_cand = NFPs[idx] + float(dd)
                        pred_cand = im_model(xyzps[idx], nfp_cand, targets=target_bead)  # [Z,H,W] removed on 15/03/2026



                        target_shifted = target_bead.clone()

                        # allow a different shift for every z slice
                        # TODO (RK): goal here is to find the best cc per bead and not per z slice, this can potentially break the bead in half
                        for zi in range(pred_cand.shape[0]):
                            a = pred_cand[zi]
                            b = target_bead[zi]

                            dy, dx = phasecorr_shift_int(a, b, max_shift_px=max_shift_px)

                            # TODO (RK): check if roll is needed. probably not
                            b1 = torch.roll(b, shifts=(dy, dx), dims=(0, 1))
                            b2 = torch.roll(b, shifts=(-dy, -dx), dims=(0, 1))
                            if cc_score(a, b2) > cc_score(a, b1):
                                b1 = b2

                            target_shifted[zi] = b1

                        eps = 1e-12
                        pred_cand_n = pred_cand / (pred_cand.sum(dim=(1, 2), keepdim=True) + eps)
                        cand_loss = F.mse_loss(pred_cand_n, target_shifted).item()

                        if (best_loss is None) or (cand_loss < best_loss):
                            best_loss = cand_loss
                            best_dd = float(dd)
                            best_target_shifted = target_shifted.clone()

                    nfp_offsets[idx] = best_dd
                    y_aligned[idx] = best_target_shifted
            ''' replaced to make shift invariant per slice rather than per bead
            with torch.no_grad():
                y_aligned = y_true.clone()
                nfp_offsets = torch.zeros_like(NFPs)

                unique_beads = torch.unique(bead_ids)

                for bid in unique_beads.tolist():
                    idx = torch.where(bead_ids == bid)[0]

                    # on-axis bead: keep nominal NFP, no shift search
                    if bool(is_onaxis[idx[0]].item()):
                        continue

                    best_loss = None
                    best_dd = 0.0
                    best_shift = (0, 0)

                    target_bead = y_true[idx]  # [Z,H,W]

                    for dd in delta_candidates:
                        nfp_cand = NFPs[idx] + float(dd)
                        pred_cand = im_model(xyzps[idx], nfp_cand)  # [Z,H,W]

                        # one shift for the whole bead stack:
                        # use sum over z to estimate a single robust shift
                        a_ref = pred_cand.sum(dim=0)  # [H,W]
                        b_ref = target_bead.sum(dim=0)  # [H,W]

                        dy, dx = phasecorr_shift_int(a_ref, b_ref, max_shift_px=max_shift_px)

                        # sign ambiguity: test both directions on the summed image
                        b_ref_1 = torch.roll(b_ref, shifts=(dy, dx), dims=(0, 1))
                        b_ref_2 = torch.roll(b_ref, shifts=(-dy, -dx), dims=(0, 1))
                        if cc_score(a_ref, b_ref_2) > cc_score(a_ref, b_ref_1):
                            dy, dx = -dy, -dx

                        target_shifted = torch.roll(target_bead, shifts=(dy, dx), dims=(1, 2))

                        eps = 1e-12
                        pred_cand_n = pred_cand / (pred_cand.sum(dim=(1, 2), keepdim=True) + eps)
                        cand_loss = F.mse_loss(pred_cand_n, target_shifted).item()

                        if (best_loss is None) or (cand_loss < best_loss):
                            best_loss = cand_loss
                            best_dd = float(dd)
                            best_shift = (int(dy), int(dx))

                    # save best bead-wise alignment
                    nfp_offsets[idx] = best_dd
                    y_aligned[idx] = torch.roll(
                        target_bead,
                        shifts=best_shift,
                        dims=(1, 2)
                    )
                    ''' # replaced

            # forward again WITH grad, using the chosen per-bead fine defocus
            pred = im_model(xyzps, NFPs + nfp_offsets, targets=y_aligned)

            eps = 1e-12
            pred_n = pred / (pred.sum(dim=(1, 2), keepdim=True) + eps)
            loss = F.mse_loss(pred_n, y_aligned)
           
        # keep some MSE to prevent "degenerate" solutions
        #loss = 0.2 * loss_mse + 0.8 * loss_ac
        #loss = 0.0 * loss_mse + 1.0 * loss_ac

        # whichever pred/target the loss actually used this epoch, for the live panel
        pred_display = pred if not apply_off_axis_space_invariance else pred_n
        target_display = y_true if not apply_off_axis_space_invariance else y_aligned

        loss.backward()

        # snapshot the state that produced this epoch's loss (pre-step — opt.step() below
        # would otherwise move phase_mask/g_sigma/d_raw/nfp_offset_raw past it)
        loss_val = float(loss.item())
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {
                'phase_mask': im_model.phase_mask.detach().clone(),
                'g_sigma': im_model.g_sigma.detach().clone(),
                'd_raw': im_model.d_raw.detach().clone(),
                'nfp_offset_raw': im_model.nfp_offset_raw.detach().clone(),
            }

        if epoch == 0:
            print("d_um:", im_model.d_um().detach().item())
            print("grad(d_raw):", None if im_model.d_raw.grad is None else im_model.d_raw.grad.detach().item())
            print("nfp_offset_um:", im_model.nfp_offset_um().detach().item())
            print("grad(nfp_offset_raw):", None if im_model.nfp_offset_raw.grad is None else im_model.nfp_offset_raw.grad.detach().item())

        opt.step()
        Visualize_mask = True
        # visualization
        if Visualize_mask:
            if epoch % 10 == 0:
                with torch.no_grad():
                    mask = im_model.phase_mask.detach().cpu().numpy()

                    # wrap phase to [-pi, pi] for visualization
                    mask_wrapped = np.angle(np.exp(1j * mask))

                    plt.figure(figsize=(4, 4))
                    plt.imshow(mask, cmap="twilight")
                    plt.colorbar()
                    plt.title(f"Phase mask, epoch {epoch}")
                    plt.tight_layout()

                    path2save = 'phase_retrieval_with_displacement_iteration'
                    if not (os.path.isdir(path2save)):
                        os.mkdir(path2save)
                    plt.savefig(os.path.join(path2save, 'iteration_' + str(epoch) +  '.jpg'), bbox_inches='tight', dpi=300)
                    plt.close()
                    # End visualization

        # keep sigma sane (optional but helps)
        with torch.no_grad():
            im_model.g_sigma.clamp_(min=1e-3, max=20.0)

        # monitor
        with torch.no_grad():
            pred2 = im_model(xyzps, NFPs, targets=y_true)
            cc = calculate_cc(pred2.detach().cpu().numpy(), y_true.detach().cpu().numpy())
            ccs.append(cc)

            d_now = float(im_model.d_um().detach().cpu().item())
            nfp_offset_now = float(im_model.nfp_offset_um().detach().cpu().item())
            if (epoch % 10) == 0:
                print(
                    f"[PR] epoch {epoch:4d} loss={float(loss.item()):.6g}  d={d_now:.2f} um  "
                    f"g_sigma={float(im_model.g_sigma.item()):.4f}  "
                    f"nfp_offset={nfp_offset_now:.3f} um")

        _update_live_panel(
            epoch, float(loss.item()), d_now, nfp_offset_now,
            float(im_model.g_sigma.item()), pred_display, target_display,
            bead_ids, zi_tensor,
        )

    if live_box is not None and pred_display is not None and (epoch % live_debug_every_epochs) != 0:
        _refresh_live_panel(
            epoch, float(loss.item()), d_now, nfp_offset_now,
            float(im_model.g_sigma.item()), pred_display, target_display,
            bead_ids, zi_tensor,
        )

    _total_time_s = time.perf_counter() - _timing_t0
    _calc_time_s = max(0.0, _total_time_s - im_model.debug_save_time_s - live_panel_time_s)
    if _total_time_s > 0:
        print(
            f"[PR] timing breakdown (warmup+main loop): total={_total_time_s:.1f}s  "
            f"calculation={_calc_time_s:.1f}s ({100*_calc_time_s/_total_time_s:.1f}%)  "
            f"debug_png_dump={im_model.debug_save_time_s:.1f}s ({100*im_model.debug_save_time_s/_total_time_s:.1f}%)  "
            f"live_panel={live_panel_time_s:.1f}s ({100*live_panel_time_s/_total_time_s:.1f}%)"
        )

    # Restore the best-loss main-loop state (if any main-loop epoch ran) so everything saved
    # below — d, g_sigma, phase_mask, the fitted NFP offset, and the exported sim stacks — comes
    # from the lowest-loss point found, not just wherever the last epoch happened to land.
    if best_state is not None:
        with torch.no_grad():
            im_model.phase_mask.copy_(best_state['phase_mask'])
            im_model.g_sigma.copy_(best_state['g_sigma'])
            im_model.d_raw.copy_(best_state['d_raw'])
            im_model.nfp_offset_raw.copy_(best_state['nfp_offset_raw'])
        print(f"[PR] restoring best-loss state (loss={best_loss:.6g}) for final outputs")

    # save final values back
    param_dict['mask_offset_in_um'] = float(im_model.d_um().detach().cpu().item())
    print(f"[PR] done. best d = {param_dict['mask_offset_in_um']:.2f} um")
    phase_mask = im_model.phase_mask.detach().cpu().numpy()
    g_sigma = float(im_model.g_sigma.detach().cpu().numpy())

    with torch.no_grad():
        NFPs = im_model.nfps(zi_tensor).detach()

    param_dict['nfp_offset_um'] = float(im_model.nfp_offset_um().detach().cpu().item())
    param_dict['nfp_range_um'] = float(im_model.nfp_range_um)
    param_dict['nfp_fitted_Z'] = int(Z)
    print(f"[PR] done. fitted NFP offset={param_dict['nfp_offset_um']:.3f} um "
          f"(range={param_dict['nfp_range_um']} um, Z={Z})")

    #print(f"[PR] done. best d = {d:.1f} um")

    # ----------------------------
    # SAVE OUTPUTS (after PR is done)
    # ----------------------------
    save_dir = pr_dict.get("save_dir", os.path.join(os.getcwd(), "phase_retrieval_outputs"))
    os.makedirs(save_dir, exist_ok=True)

    # save mask + scalar params
    np.save(os.path.join(save_dir, "phase_mask.npy"), phase_mask)
    with open(os.path.join(save_dir, "g_sigma_and_d.txt"), "w") as f:
        f.write(f"g_sigma = {g_sigma}\n")
        f.write(f"mask_offset_in_um (d) = {float(param_dict['mask_offset_in_um'])}\n")
        f.write(f"nfp_offset_um = {param_dict['nfp_offset_um']}\n")
        f.write(f"nfp_range_um = {param_dict['nfp_range_um']}\n")

    # helper: float stack -> uint16 for viewing
    def _to_u16(st):
        st = st.astype(np.float32)
        out = np.zeros_like(st, dtype=np.uint16)
        for i in range(st.shape[0]):
            mx = float(st[i].max())
            if mx > 0:
                out[i] = (np.clip(st[i] / mx, 0, 1) * 65535.0).astype(np.uint16)
        return out

    im_model.eval()
    cnt = -1
    with torch.no_grad():
        for name, zst, x_um, y_um in stacks:
            cnt+=1
            # z is always 0 — axial variation flows entirely through NFPs
            xyz_bead = np.stack([[x_um, y_um, 0.0, 1.0] for _ in range(Z)], axis=0).astype(np.float32)
            xyz_bead_t = torch.from_numpy(xyz_bead).to(device)

            pred = im_model(xyz_bead_t, im_model.nfps()).detach().cpu().numpy()  # [Z,H,W]; NFPs is num_beads*Z long, wrong shape here
            exp = zst / (np.sum(zst, axis=(1, 2), keepdims=True) + 1e-12)   # [Z,H,W] (same norm as training)

            exp_u16 = _to_u16(exp)
            sim_u16 = _to_u16(pred)

            io.imsave(os.path.join(save_dir, f"exp_stack_{cnt}_{name}.tif"), exp_u16, check_contrast=False)
            io.imsave(os.path.join(save_dir, f"sim_stack_{cnt}_{name}.tif"), sim_u16, check_contrast=False)

            # montage per z: left=exp, right=sim
            Z0, H0, W0 = exp_u16.shape
            montage = np.zeros((Z0, H0, 2 * W0), dtype=np.uint16)
            montage[:, :, :W0] = exp_u16
            montage[:, :, W0:] = sim_u16
            io.imsave(os.path.join(save_dir, f"montage_{cnt}_{name}.tif"), montage, check_contrast=False)

    # --- Final per-bead debug dump (central z only) ---
    # Put debug outputs inside the same phase_retrieval_outputs folder
    im_model.debug_bfp = True
    im_model.debug_every_num_epoch = 1
    im_model.current_epoch = 0
    im_model._last_debug_epoch = None
    im_model.debug_dir = os.path.join(save_dir, "per_bead_phase")

    # number of beads you used in PR (onaxis + offaxis)
    num_beads = len(stacks)
    im_model.debug_max_emitters = num_beads

    # Optional: give names to beads (so folders aren't emitter_000, emitter_001...)
    im_model.debug_names = [name for (name, _, _, _) in stacks]

    # Take ONLY the central z/NFP slice from each bead:
    zi = Z // 2
    idxs = [b * Z + zi for b in range(num_beads)]  # assumes your packing is bead-major then z

    xyz_mid = xyzps[idxs]

    #xyz_mid[:,0]  =  xyz_mid[:,0] * 100
    #xyz_mid[:,1]  =  xyz_mid[:,1] * 100

    NFPs_mid = NFPs[idxs]

    with torch.no_grad():
        _ = im_model(xyz_mid, NFPs_mid, targets=y_true[idxs])  # triggers _maybe_save_debug once

    return phase_mask, g_sigma, ccs


def show_z_psf(param_dict):
    model = ImModel(param_dict)
    model.model_demo(np.linspace(param_dict['zrange'][0], param_dict['zrange'][1], 5))  # check PSFs

def _center_crop(im, out_hw):
    out_h, out_w = out_hw
    h, w = im.shape
    y0 = max(0, (h - out_h) // 2)
    x0 = max(0, (w - out_w) // 2)
    return im[y0:y0 + out_h, x0:x0 + out_w]

def _cc(a, b):
    a = a.astype(np.float32, copy=False).ravel()
    b = b.astype(np.float32, copy=False).ravel()
    a = a - a.mean()
    b = b - b.mean()
    den = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b) / den)


def _to_uint16_stack(stack, mode="per_slice_max"):
    st = stack.astype(np.float32, copy=False)

    if mode == "global_max":
        mx = float(st.max())
        if mx <= 0:
            return np.zeros_like(st, dtype=np.uint16)
        return (np.clip(st / mx, 0, 1) * 65535.0).astype(np.uint16)

    # per_slice_max
    out = np.zeros_like(st, dtype=np.uint16)
    mx = st.reshape(st.shape[0], -1).max(axis=1)
    for i in range(st.shape[0]):
        if mx[i] > 0:
            out[i] = (np.clip(st[i] / float(mx[i]), 0, 1) * 65535.0).astype(np.uint16)
    return out


def fit_mask_offset_from_offaxis_stacks(
    param_dict,
    #d_search_um=(00000.0, 70000.0),#d_search_um=(00000.0, 80000.0),
    d_search_um=(00000.0, 70000.0),#d_search_um=(00000.0, 80000.0),
    d_coarse_step_um=5000.0,
    d_fine_step_um=250.0,
    photons_for_sim=1e4,
    save_dir=None,
    save_uint16_mode="per_slice_max",
    make_montage=True,
):
    if not param_dict.get("offaxis_zstack_files"):
        print("[fit d] No off-axis stacks provided. Skipping.")
        return None

    # --- output directory (ONE place only) ---
    if save_dir is None:
        time_now = datetime.today().strftime("%Y%m%d_%H%M%S")
        #save_dir = os.path.join(os.getcwd(), f"mask_offset_fit_{time_now}")
        save_dir = os.path.join(os.getcwd(), f"mask_fit_outputs")
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # NFP sweep from the offset phase_retrieval() already fitted, not a raw/unfit guess
    nfp_offset_um = param_dict["nfp_offset_um"]
    nfp_range_um = param_dict["nfp_range_um"]
    Z_expected = int(param_dict["nfp_fitted_Z"])
    nfp_start_um = nfp_offset_um - nfp_range_um / 2
    nfp_end_um = nfp_offset_um + nfp_range_um / 2
    nfps = np.linspace(nfp_start_um, nfp_end_um, Z_expected, dtype=np.float32)

    # --- model ---
    from DS3Dplus.ds3d_utils import ImModelTraining
    model = ImModelTraining(param_dict)
    model.eval()

    r0, c0 = map(float, param_dict["centralBeadCoordinates_pixel"])
    ps_cam = float(param_dict["ps_camera"])
    M = (param_dict["M"])

    # --- load stacks once ---
    stacks = []
    for f, (rr, cc) in zip(param_dict["offaxis_zstack_files"], param_dict["offaxis_coords_pixel"]):
        zstack = io.imread(f).astype(np.float32)  # (Z,H,W)

        if zstack.shape[0] != Z_expected:
            raise ValueError(f"[fit d] Z mismatch: {f} has Z={zstack.shape[0]} but nfps has {Z_expected}.")

        dx_pix = float(cc)# - c0
        dy_pix = float(rr)# - r0
        x_um = dx_pix * (ps_cam/M)
        y_um = dy_pix * (ps_cam/M)

        stacks.append({
            "file": f,
            "name": os.path.splitext(os.path.basename(f))[0],
            "exp": zstack,
            "H": zstack.shape[1],
            "W": zstack.shape[2],
            "x_um": float(x_um),
            "y_um": float(y_um),
        })


    def simulate_stack_for_d(d_um, st):
        model.mask_offset_in_um = float(d_um)

        Z, Hroi, Wroi = st["exp"].shape[0], st["H"], st["W"]
        sim_stack = np.zeros((Z, Hroi, Wroi), dtype=np.float32)
        cc_per_z = np.zeros((Z,), dtype=np.float32)

        x_um, y_um = st["x_um"], st["y_um"]

        oldNFP = float(model.NFP)

        for zi, nfp_um in enumerate(nfps):
            exp_im = _norm01_sum(st["exp"][zi])
            model.NFP = float(nfp_um)  # scan -> NFP
            xyzp = np.array([x_um, y_um, 0.0, float(photons_for_sim)], dtype=np.float32)

            sim = model.psf_patch_clean(xyzp)
            # IMPORTANT FIX: float32 (prevents Float vs Double mismatch in torch)
            #xyzp = np.array([x_um, y_um, float(z_um), float(photons_for_sim)], dtype=np.float32)

            sim = _center_crop(sim, (Hroi, Wroi))
            sim = _norm01_sum(sim)


            Blur = True
            if Blur:
                g_sigma = param_dict["g_sigma"]
                g_sigma = torch.tensor(g_sigma)
                g_size = 9 #hard coded! to fix
                g_r = int(g_size / 2)
                #g_xs = torch.linspace(-g_r, g_r, g_size, device=device).type(torch.float64)
                g_xs = torch.linspace(-g_r, g_r, g_size).type(torch.float64)
                g_xx, g_yy = torch.meshgrid(g_xs, g_xs, indexing='xy')

                # blur
                # blur (batched)
                blur_kernel = 1 / (2 * math.pi * g_sigma[0] ** 2) * (
                    torch.exp(-0.5 * (g_xx ** 2 + g_yy ** 2) / g_sigma[0] ** 2)   )
                sim_tensor = torch.tensor(sim)
                sim_tensor = F.conv2d(sim_tensor.unsqueeze(0).unsqueeze(0), blur_kernel.type_as(sim_tensor).unsqueeze(0).unsqueeze(0), padding='same' ).squeeze(1)
                '''sim = F.conv2d(
                    sim.unsqueeze(1),
                    blur_kernel.unsqueeze(0).unsqueeze(0).type_as(sim),
                    padding='same'
                ).squeeze(1)'''
                # photon normalization
                #sim = sim / torch.sum(psfs, dim=(1, 2), keepdims=True) * xyzps[:, 3:4].unsqueeze(         1)  # photon normalization
                # sim = sim[:, self.idx05 - self.h05:self.idx05 + self.h05 + 1, self.idx05 - self.w05:self.idx05 + self.w05 + 1]
                #sim = sim[:, self.r0:self.r0 + self.H, self.c0:self.c0 + self.W]
            sim_tensor = sim_tensor.squeeze(0)
            sim_stack[zi] = sim_tensor
            cc_per_z[zi] = _cc(exp_im, sim_tensor.numpy())



        model.NFP = oldNFP  # turning it back to experimental nfp
        return sim_stack, cc_per_z

    def score_for_d(d_um):
        ccs = []
        for st in stacks:
            _, cc_per_z = simulate_stack_for_d(d_um, st)
            ccs.append(cc_per_z)
        ccs = np.concatenate(ccs) if ccs else np.array([-1e9], dtype=np.float32)
        return float(ccs.mean())

    # ---- coarse search ----
    d_search_um = param_dict["mask_offset_in_um"], param_dict["mask_offset_in_um"]+1e-6
    d0, d1 = map(float, d_search_um)

    d_vals = np.arange(d0, d1 + 1e-6, float(d_coarse_step_um), dtype=np.float32)
    scores = [score_for_d(d) for d in d_vals]
    best_d = float(d_vals[int(np.argmax(scores))])

    # ---- fine search ----
    lo = max(d0, best_d - 2 * float(d_coarse_step_um))
    hi = min(d1, best_d + 2 * float(d_coarse_step_um))
    d_vals2 = np.arange(lo, hi + 1e-6, float(d_fine_step_um), dtype=np.float32)
    scores2 = [score_for_d(d) for d in d_vals2]
    best_d2 = float(d_vals2[int(np.argmax(scores2))])
    best_s2 = float(max(scores2))

    # save to param_dict
    param_dict["mask_offset_in_um"] = best_d2
    param_dict["mask_offset_fit_info"] = {
        "best_d_um": best_d2,
        "best_cc": best_s2,
        "coarse": {"d": d_vals.tolist(), "cc": [float(x) for x in scores]},
        "fine": {"d": d_vals2.tolist(), "cc": [float(x) for x in scores2]},
        "save_dir": save_dir,
        "nfps_used": nfps.tolist(),
    }

    print(f"[fit d] best mask_offset_in_um = {best_d2:.1f} um, mean CC={best_s2:.4f}")
    print(f"[fit d] saving outputs to: {save_dir}")

    # ---- save curves ----
    with open(os.path.join(save_dir, "cc_curve_coarse.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["d_um", "mean_cc"])
        w.writerows([[float(d), float(s)] for d, s in zip(d_vals, scores)])

    with open(os.path.join(save_dir, "cc_curve_fine.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["d_um", "mean_cc"])
        w.writerows([[float(d), float(s)] for d, s in zip(d_vals2, scores2)])

    plt.figure(figsize=(6, 4))
    plt.plot(d_vals, scores, marker="o", linewidth=1)
    plt.plot(d_vals2, scores2, marker="o", linewidth=1)
    plt.axvline(best_d2, linestyle="--")
    plt.xlabel("mask_offset_in_um (d) [um]")
    plt.ylabel("mean CC")
    plt.title(f"Best d = {best_d2:.1f} um, mean CC={best_s2:.4f}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cc_curve.png"), dpi=200)
    plt.close()

    # ---- save exp/sim/montage for best d (ALL in same dir) ----
    for st in stacks:
        sim_stack, cc_per_z = simulate_stack_for_d(best_d2, st)  # <-- removed bogus save_results

        with open(os.path.join(save_dir, f"cc_per_z_{st['name']}.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["z_um", "cc"])
            w.writerows([[float(z), float(cc)] for z, cc in zip(nfps, cc_per_z)])

        exp_u16 = _to_uint16_stack(st["exp"], mode=save_uint16_mode)
        sim_u16 = _to_uint16_stack(sim_stack, mode=save_uint16_mode)

        io.imsave(os.path.join(save_dir, f"exp_stack_{st['name']}.tif"), exp_u16, check_contrast=False)
        io.imsave(os.path.join(save_dir, f"sim_stack_bestd_{st['name']}.tif"), sim_u16, check_contrast=False)

        if make_montage:
            Z, H, W = exp_u16.shape
            montage = np.zeros((Z, H, 2 * W), dtype=np.uint16)
            montage[:, :, :W] = exp_u16
            montage[:, :, W:] = sim_u16
            io.imsave(os.path.join(save_dir, f"comparison_montage_{st['name']}.tif"), montage, check_contrast=False)

    return best_d2
