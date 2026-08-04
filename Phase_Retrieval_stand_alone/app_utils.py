import os
import csv
import math
import torch
import numpy as np
from skimage import io
from scipy import ndimage
from datetime import datetime
import matplotlib.pyplot as plt
import torch.nn.functional as F
from image_model import ImModel_pr
from DS3Dplus.ds3d_utils import ImModel

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

    nfps = np.asarray(param_dict['nfps'], dtype=np.float32)
    Z = len(nfps)

    # ----------------------------
    # Collect stacks: on-axis + off-axis
    # ----------------------------
    stacks = []

    # RK: on-axis stack (x=y=0)
    zstack_on = io.imread(pr_dict['zstack_file_path']).astype(np.float32)
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
    nfp_list = []
    bead_id_list = []
    stack_index = 0

    is_onaxis_list = []  # <-- add

    for name, zst, x_um, y_um in stacks:
        stack_index += 1
        bead_id = stack_index - 1  # added on 15/03/2026
        if zst.shape[0] != Z:
            raise ValueError(
                f"{name}: Z mismatch. stack has {zst.shape[0]} but nfps has {Z}"
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
            nfp_list.append(float(nfps[zi]))
            bead_id_list.append(bead_id)
            is_onaxis_list.append(name == "onaxis")  # <-- add

        # normalize AFTER masking
        #zst = zst / (np.sum(zst, axis=(1, 2), keepdims=True) + 1e-12)

    y_true = torch.from_numpy(np.stack(y_list, 0)).to(device)  # [B,H,W]
    xyzps = torch.from_numpy(np.asarray(xyz_list, np.float32)).to(device)  # [B,4]

    # NFP per sample (constant)
    #NFPs = torch.full((xyzps.shape[0],), float(param_dict['NFP']), device=device)  # removed on 26/01/2026
    NFPs = torch.tensor(np.asarray(nfp_list, np.float32), device=device)
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

    # initial d:
    params_pr['mask_offset_in_um'] = float(param_dict.get("mask_offset_in_um", 0.0))
    # end ori's edit from 26/01/2026 for improved pr with displacement

    im_model = ImModel_pr(params_pr).to(device)

    im_model.train()

    opt = torch.optim.Adam(
        [
            {'params': [im_model.phase_mask], 'lr': pr_dict['lr_phase_mult'] * pr_dict['learning_rate']},
            {'params': [im_model.g_sigma],    'lr': pr_dict['lr_sigma_mult'] * pr_dict['learning_rate']},
            {'params': [im_model.d_raw],      'lr': pr_dict['lr_d_mult']     * pr_dict['learning_rate']},
        ],
        betas=tuple(pr_dict['adam_betas'])
    )

    ccs = []
    # TODO (RK): 
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
    for epoch in range(pr_dict['epochs']):
        opt.zero_grad()
        apply_off_axis_space_invariance = (max_shift_px > 0)

        if not apply_off_axis_space_invariance:
            pred = im_model(xyzps, NFPs)  #original
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
                        pred_cand = im_model(xyzps[idx], nfp_cand)  # [Z,H,W] removed on 15/03/2026



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
            pred = im_model(xyzps, NFPs + nfp_offsets)

            eps = 1e-12
            pred_n = pred / (pred.sum(dim=(1, 2), keepdim=True) + eps)
            loss = F.mse_loss(pred_n, y_aligned)
           
        # keep some MSE to prevent "degenerate" solutions
        #loss = 0.2 * loss_mse + 0.8 * loss_ac
        #loss = 0.0 * loss_mse + 1.0 * loss_ac

        loss.backward()

        if epoch == 0:
            print("d_um:", im_model.d_um().detach().item())
            print("grad(d_raw):", None if im_model.d_raw.grad is None else im_model.d_raw.grad.detach().item())

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
                    plt.clf()
                    # End visualization

        # keep sigma sane (optional but helps)
        with torch.no_grad():
            im_model.g_sigma.clamp_(min=1e-3, max=20.0)

        # monitor
        with torch.no_grad():
            pred2 = im_model(xyzps, NFPs)
            cc = calculate_cc(pred2.detach().cpu().numpy(), y_true.detach().cpu().numpy())
            ccs.append(cc)

            if (epoch % 10) == 0:
                d_now = float(im_model.d_um().detach().cpu().item())
                print(
                    f"[PR] epoch {epoch:4d} loss={float(loss.item()):.6g}  d={d_now:.2f} um  g_sigma={float(im_model.g_sigma.item()):.4f}")

    # save final values back
    param_dict['mask_offset_in_um'] = float(im_model.d_um().detach().cpu().item())
    print(f"[PR] done. best d = {param_dict['mask_offset_in_um']:.2f} um")
    phase_mask = im_model.phase_mask.detach().cpu().numpy()
    g_sigma = float(im_model.g_sigma.detach().cpu().numpy())

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
            # build xyz for this bead (Z samples)
            #xyz_bead = np.stack([[x_um, y_um, float(nfps[zi]), 1.0] for zi in range(Z)], axis=0).astype(np.float32)
            xyz_bead = np.stack([[x_um, y_um, float(nfps[zi])*0, 1.0] for zi in range(Z)], axis=0).astype(np.float32)
            xyz_bead_t = torch.from_numpy(xyz_bead).to(device)
            NFPs_bead = torch.full((Z,), float(param_dict['NFP']), device=device)
            #NFPs_bead = torch.full((Z,), float(param_dict['NFP'])*0, device=device)

            #pred = im_model(xyz_bead_t, NFPs_bead).detach().cpu().numpy()  # [Z,H,W]
            pred = im_model(xyz_bead_t, torch.tensor(nfps).to(device)).detach().cpu().numpy()  # [Z,H,W]
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
    im_model.debug_every = 1
    im_model.debug_dir = os.path.join(save_dir, "per_bead_phase")
    im_model._debug_call_idx = 0

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
        _ = im_model(xyz_mid, NFPs_mid)  # triggers _maybe_save_debug once

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

    # --- load nfps & enforce monotonic (fixes “flip”) ---
    #nfps_raw = np.asarray(param_dict["nfps"], dtype=np.float32)
    #Z_expected = len(nfps_raw)

    nfps = np.asarray(param_dict["nfps"], dtype=np.float32)




    # If z values are strictly decreasing, reverse them AND reverse every exp z-stack
    if np.all(np.diff(nfps) < 0):
        nfps = nfps[::-1].copy()
        reverse_exp_stacks = True
    else:
        reverse_exp_stacks = False

    Z_expected = len(nfps)

    '''
    # If nfps is not strictly increasing, replace with a monotonic grid
    if not np.all(np.diff(nfps_raw) > 0):
        z0, z1 = float(nfps_raw.min()), float(nfps_raw.max())
        nfps = np.linspace(z0, z1, Z_expected, dtype=np.float32)
        print(f"[fit d] WARNING: nfps not monotonic -> using linspace({z0:.3f}, {z1:.3f}, {Z_expected})")
    else:
        nfps = nfps_raw
    '''
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
        #if reverse_exp_stacks:
            #zstack = zstack[::-1].copy()
            #nfps = nfps[::-1].copy()


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
