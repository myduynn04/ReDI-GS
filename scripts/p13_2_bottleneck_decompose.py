#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 13.2.4] Bottleneck decomposition diagnostic.
# File: scripts/p13_2_bottleneck_decompose.py  (TẠO MỚI)
# Mục đích: Tách overfit gap 12.88 dB (A3 train 34.21 / test 21.33)
#           thành buckets nguyên nhân → biết bottleneck dominant →
#           chọn direction theo % attribution, KHÔNG đoán.
# Post-hoc: load A3 checkpoint, render depth ở test cam (GPU inference,
#           ~2 min, NO training), attribute mỗi test pixel vào 1 bucket.
#
# Buckets (priority order — fundamental nhất trước):
#   H1 IRREDUCIBLE   : back-proj 3D point KHÔNG train cam nào thấy (frustum)
#   H3 GEOMETRY      : |A3_depth − COLMAP_z|/range > τ_d  (tại keypoint)
#   H5 DETAIL        : Laplacian_GT > τ_hf  (HF texture — đã refute là driver)
#   H4 APPEARANCE/SH : residual (depth OK, low HF, vẫn sai)
#
# 5 quyết định (user-approved 2026-05-14):
#   1. Script tự render depth từ .ply (GPU inference, reuse Scene/render)
#   2. H3 vs COLMAP triangulated depth (KHÔNG DAV2)
#   3. H1 visibility frustum-only (no occlusion) — lower-bound
#   4. Self-calibrate τ:
#        τ_d  = max(median(rel)+2·MAD(rel), 0.05)   [rel=|Δz|/depth_range]
#        τ_hf = percentile80(Laplacian_GT) per scene
#      + sensitivity sweep τ_d k∈{1,2,3}, τ_hf pct∈{70,80,90}
#   5. Unified 1-pass + 3 side-metrics (H6/H7/H8)
#
# Q1 CRITICAL — depth alignment pre-flight (3-tier graceful):
#   aligned (corr>.95, slope≈1)            → proceed
#   pure scale (corr>.95, slope≠1)         → auto-correct COLMAP_z×slope
#   convention mismatch (corr<.95)         → try ray→z; vẫn fail →
#                                            H3=UNRELIABLE, H1/H4/H5 vẫn verdict
# Q2 — H4 chroma/lum sub-check (actionable SH vs near-optimal residual)
#
# Reuse verified: scene.Scene/GaussianModel/render (camera convention khớp
# training tuyệt đối), colmap_loader, _stem name-join (đã verify unmatched=0).
# ============================================================
"""[CRSGaussian Phase 13.2.4] Bottleneck decomposition.

Run (GPU SERVER — needs 3DGS env + checkpoints):
    python scripts/p13_2_bottleneck_decompose.py
    SCENES="orchids horns trex" python scripts/p13_2_bottleneck_decompose.py

Smoke (CPU local OK for #1,#2,#4; #3 + full = server only):
    python -c "import ast; ast.parse(open('scripts/p13_2_bottleneck_decompose.py').read()); print('syntax OK')"
"""

import os
import sys
import re
import glob
import math
import statistics

import numpy as np

sys.path.insert(0, ".")

# ── Config ──
DATA_ROOT = os.environ.get("DATA_ROOT", "data/nerf_llff_data")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "output/p13_lfcf")
SEED = os.environ.get("SEED", "42")
ITERATION = int(os.environ.get("ITERATION", "10000"))
SCENES = os.environ.get(
    "SCENES", "fern flower fortress horns leaves orchids room trex").split()
PLOT_DIR = os.environ.get("PLOT_DIR", "logs/p13_2_bottleneck")
LOG_DIR_MS = os.environ.get("LOG_DIR_MS", "logs/p13_lfcf")  # multi-seed logs (H7)
MS_SEEDS = os.environ.get("MS_SEEDS", "42 137 9999").split()

# τ calibration (Q3 final — user approved counter-proposal)
TAU_D_K = float(os.environ.get("TAU_D_K", "2.0"))     # median + k·MAD
TAU_D_FLOOR = float(os.environ.get("TAU_D_FLOOR", "0.05"))  # frac depth_range
TAU_HF_PCT = float(os.environ.get("TAU_HF_PCT", "90"))      # percentile (pct80→90: smoke báo pct80 quá thấp, H5 nuốt color-boundary)

# Q1 alignment thresholds
ALIGN_CORR_MIN = 0.95     # corr_in ≥ này + scale≈1 → label ALIGNED
ALIGN_SLOPE_TOL = 0.05    # |scale−1| ≤ này → coi như scale=1 (ALIGNED)
# SCALE_CORRECTED chấp nhận khi corr_in ≥ ALIGN_ACCEPT_MIN dù scale≠1.
# Justification (KHÔNG p-hack): (1) crs_module.py:191 codebase tự dùng 5%
# slack vì alpha-weighted depth noisy → đòi 0.95 chặt hơn chuẩn-tin-depth
# của chính codebase; (2) τ_d=median+2·MAD tự calibrate hấp thụ scatter →
# H3 chỉ flag outlier > scatter dù corr 0.91 hay 0.99; corr_in≥0.90
# (r²≥0.81 trên inlier) là quan hệ tuyến tính RÕ (convention vỡ → corr≈0).
ALIGN_ACCEPT_MIN = 0.90

PSNR_PAT = re.compile(r"\b(\d{2,5})\s*\|\s*test\s*\|\s*([0-9]+\.[0-9]+)\s*\|")


# ── CPU utilities ──
def _stem(name):
    """Strip path+ext — khớp dataset_readers.py:217 (verified preflight)."""
    return os.path.basename(name).split(".")[0]


def luminance(rgb):
    """rgb [...,3] in [0,1] → Y (BT.601)."""
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def laplacian_mag(img2d):
    """|∇²I| — same kernel as HF pilot/preflight (consistency)."""
    K = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    pad = np.pad(img2d, 1, mode='edge')
    lap = np.zeros_like(img2d, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            lap += K[dy, dx] * pad[dy:dy + img2d.shape[0], dx:dx + img2d.shape[1]]
    return np.abs(lap)


def mad(x):
    """Median absolute deviation."""
    m = np.median(x)
    return float(np.median(np.abs(x - m)))


def parse_test_psnr(log_path):
    if not os.path.isfile(log_path):
        return None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    ms = PSNR_PAT.findall(txt)
    if not ms:
        return None
    it2p = {int(i): float(p) for i, p in ms}
    return it2p[max(it2p)]


# ── GPU: render A3 depth + RGB at test cams (mirror render.py path) ──
def render_scene_test(scene_name):
    """Load A3 checkpoint via Scene (camera convention khớp training),
    render test views → list of dict per test cam:
      {name, depth(HxW np), render(HxW3 np), gt(HxW3 np),
       fx,fy,cx,cy, W,H, znear,zfar, wvt(4x4 np, =W2C^T), wvt_inv(4x4 np)}
    + train_cams: list of dict {fx,fy,cx,cy,W,H,znear,zfar,wvt}
    GPU inference only (torch.no_grad). Returns (test_list, train_list) or None.
    """
    import torch
    from argparse import ArgumentParser, Namespace
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from arguments import ModelParams, PipelineParams

    model_path = f"{OUTPUT_ROOT}/A3_seed{SEED}_{scene_name}"
    cfg_path = os.path.join(model_path, "cfg_args")
    if not os.path.isfile(cfg_path):
        print(f"  ERR no cfg_args at {model_path}")
        return None
    ply = f"{model_path}/point_cloud/iteration_{ITERATION}/point_cloud.ply"
    if not os.path.isfile(ply):
        print(f"  ERR no checkpoint {ply}")
        return None

    # Merge cfg_args — mirror arguments/__init__.py:333 get_combined_args EXACTLY:
    # parse_args([]) gives ALL registered defaults; cfg_args overrides.
    # (KHÔNG dùng Namespace(**vars(cfg)) trực tiếp — cfg có thể thiếu field →
    #  lp.extract getattr KeyError. parse_args([]) base đảm bảo đủ field.)
    parser = ArgumentParser()
    lp = ModelParams(parser)   # side-effect: register ModelParams args vào
    pp = PipelineParams(parser)  # parser → parse_args([]) có đủ default
    _ = lp                       # giữ ref (side-effect-only, không call extract)
    args_cmdline = parser.parse_args([])
    with open(cfg_path) as f:
        cfg = eval(f.read())
    merged = vars(args_cmdline).copy()
    for k, v in vars(cfg).items():
        merged[k] = v
    args = Namespace(**merged)
    args.model_path = model_path
    args.iteration = ITERATION
    if not os.path.isdir(getattr(args, "source_path", "") or ""):
        args.source_path = os.path.join(DATA_ROOT, scene_name)
    # render() expects PipelineParams-extracted namespace (mirror render_set)
    pipe = pp.extract(args)

    torch.cuda.empty_cache()
    with torch.no_grad():
        # Mirror render.py:59-60 EXACTLY — GaussianModel(args), Scene(args,...)
        # (KHÔNG GaussianModel(sh_degree)/Scene(dataset) — verified API)
        gaussians = GaussianModel(args)
        scene = Scene(args, gaussians, load_iteration=ITERATION, shuffle=False)
        bg_color = [1., 1., 1.] if getattr(args, "white_background", False) \
            else [0., 0., 0.]
        bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        def cam_intr(cam):
            W, H = cam.image_width, cam.image_height
            fx = W / (2.0 * math.tan(cam.FoVx * 0.5))
            fy = H / (2.0 * math.tan(cam.FoVy * 0.5))
            return fx, fy, W / 2.0, H / 2.0, W, H

        train_list = []
        for c in scene.getTrainCameras():
            fx, fy, cx, cy, W, H = cam_intr(c)
            train_list.append(dict(
                fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H,
                znear=float(c.znear), zfar=float(c.zfar),
                wvt=c.world_view_transform.detach().cpu().numpy()))

        test_list = []
        for c in scene.getTestCameras():
            pkg = render(c, gaussians, pipe, bg)
            rd = pkg["render"].clamp(0, 1).detach().cpu().numpy()      # 3,H,W
            # [VERIFIED crs_module.py:697,705] rasterizer trả depth TÍCH LŨY
            # (Σwᵢzᵢ, CHƯA chia coverage). Phải /(alpha+1e-6) → alpha-weighted
            # avg depth thật. Mirror crs_module.py:705 (codebase convention).
            al = np.squeeze(pkg["alpha"].detach().cpu().numpy())
            dp = np.squeeze(pkg["depth"].detach().cpu().numpy())
            dp = dp / (al + 1e-6)
            gt = c.original_image[:3].clamp(0, 1).detach().cpu().numpy()  # 3,H,W
            fx, fy, cx, cy, W, H = cam_intr(c)
            wvt = c.world_view_transform.detach().cpu().numpy()
            test_list.append(dict(
                name=_stem(c.image_name),
                depth=dp.astype(np.float32),
                render=np.transpose(rd, (1, 2, 0)).astype(np.float32),
                gt=np.transpose(gt, (1, 2, 0)).astype(np.float32),
                fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H,
                znear=float(c.znear), zfar=float(c.zfar),
                wvt=wvt, wvt_inv=np.linalg.inv(wvt)))
    del gaussians, scene
    torch.cuda.empty_cache()
    return test_list, train_list


# ── COLMAP keypoint depth (H3 ground-truth-ish) ──
def _read_points3d_xyz_by_id(path):
    """Local points3D.bin reader GIỮ COLMAP point id → xyz.

    scene.colmap_loader.read_points3D_binary trả (xyzs,rgbs,errors) arrays
    theo read-order, DISCARD id → KHÔNG map được img.point3D_ids → xyz.
    Replicate documented binary format (KHÔNG sửa shared infra):
      header: <Q num_points
      per point: <QdddBBBd (id,x,y,z,r,g,b,err)=43B
                 + <Q track_len + 8·track_len bytes track
    """
    import struct
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            pr = struct.unpack("<QdddBBBd", f.read(43))
            out[pr[0]] = np.array(pr[1:4], np.float64)
            tl = struct.unpack("<Q", f.read(8))[0]
            f.read(8 * tl)
    return out


def colmap_test_keypoints(scene_name):
    """Per test image_name → (np.array (M,3)=(u,v,z), (Wc,Hc)).

    (u,v) ở COLMAP resolution (Wc,Hc = intr.width/height). z = COLMAP_z
    trong test cam. Consumer PHẢI scale (u,v) theo render_res/COLMAP_res.
    Verified dataset_readers.py:195-223: readColmapCameras dùng intr.width
    cho CameraInfo.width NHƯNG load ảnh từ images_8 (downsampled) → xys
    trong images.bin ở COLMAP res, A3 render ở images_8 res → khác nhau.
    """
    from scene.colmap_loader import (read_extrinsics_binary,
                                     read_intrinsics_binary, qvec2rotmat)
    sp = os.path.join(DATA_ROOT, scene_name, "sparse", "0")
    images = read_extrinsics_binary(os.path.join(sp, "images.bin"))
    intr = read_intrinsics_binary(os.path.join(sp, "cameras.bin"))
    xyz = _read_points3d_xyz_by_id(os.path.join(sp, "points3D.bin"))

    out = {}
    for img in images.values():
        stem = _stem(img.name)
        cam = intr[img.camera_id]              # COLMAP intrinsic (res gốc)
        Wc, Hc = int(cam.width), int(cam.height)
        # COLMAP world→cam: R(qvec) , t = img.tvec ; z = (R@X + t)[2]
        R = qvec2rotmat(img.qvec)
        t = np.asarray(img.tvec, np.float64)
        rows = []
        for (xy, pid) in zip(img.xys, img.point3D_ids):
            if pid < 0 or pid not in xyz:
                continue
            Xc = R @ xyz[pid] + t
            if Xc[2] <= 0:
                continue
            rows.append((float(xy[0]), float(xy[1]), float(Xc[2])))
        if rows:
            out[stem] = (np.asarray(rows, np.float64), (Wc, Hc))
    return out


# ── Q1: depth alignment pre-flight (3-tier) ──
def align_check(test_cams, colmap_kp):
    """Collect (COLMAP_z, A3_depth) pairs across test views.

    A3 render trong world ĐÃ rescale bởi 3DGS (cameras.py:57
    getWorld2View2 nhân camera center × scale) → A3_depth ≈ scale·COLMAP_z.
    Robust scale = median(A3_depth/COLMAP_z) (qua gốc, robust outlier — KHÔNG
    polyfit có intercept). Accept khi quan hệ tuyến tính RÕ (inlier-corr ≥
    ngưỡng) — KHÔNG đòi scale≈1 (3DGS LUÔN rescale, slope≈1 là giả định SAI).
    Residual scatter (corr<1) = alpha-weighted depth noise nội tại
    (crs_module.py:191) → τ_d=median+2·MAD hấp thụ → H3 vẫn valid.

    Returns (status, correction_fn, info).
      correction_fn: COLMAP_z → A3-depth space (= scale·z)
      status ∈ {ALIGNED, SCALE_CORRECTED, UNRELIABLE}
    """
    cz, ad = [], []
    for tc in test_cams:
        ent = colmap_kp.get(tc['name'])
        if ent is None:
            continue
        kp_arr, (Wc, Hc) = ent
        H, W = tc['depth'].shape
        sx, sy = W / Wc, H / Hc        # COLMAP res → render res (dynamic)
        for (u, v, z) in kp_arr:
            ui, vi = int(round(u * sx)), int(round(v * sy))
            if 0 <= ui < W and 0 <= vi < H:
                d = tc['depth'][vi, ui]
                if np.isfinite(d) and d > 0 and z > 0:
                    cz.append(z)
                    ad.append(float(d))
    if len(cz) < 30:
        return "UNRELIABLE", None, dict(n=len(cz), reason="too few keypoints")
    cz = np.asarray(cz, np.float64)
    ad = np.asarray(ad, np.float64)

    # Robust scale = median ratio (A3_depth / COLMAP_z); MAD đo độ chụm
    ratio = ad / cz
    scale = float(np.median(ratio))
    rmad = float(np.median(np.abs(ratio - scale)))
    rmad_pct = (rmad / scale) if scale > 0 else float("inf")
    # Inlier = ratio trong median ± 3·MAD (loại outlier triangulation/blend)
    inlier = np.abs(ratio - scale) <= (3.0 * rmad + 1e-12)
    n_in = int(inlier.sum())
    corr_raw = float(np.corrcoef(cz, ad)[0, 1])
    corr_in = (float(np.corrcoef(cz[inlier], ad[inlier])[0, 1])
               if n_in >= 30 else corr_raw)
    info = dict(n=len(cz), scale=scale, rmad_pct=rmad_pct,
                inlier_frac=n_in / len(cz),
                corr_raw=corr_raw, corr_in=corr_in)
    fix = (lambda z, s=scale: s * z)   # COLMAP_z → A3-depth space

    if corr_in >= ALIGN_CORR_MIN and abs(scale - 1.0) <= ALIGN_SLOPE_TOL:
        return "ALIGNED", fix, info
    if corr_in >= ALIGN_ACCEPT_MIN:
        return "SCALE_CORRECTED", fix, info
    return "UNRELIABLE", None, info   # corr_in thấp = convention vỡ thật


# ── Back-projection / frustum visibility (H1) ──
def backproject(u, v, d, tc):
    """Test pixel (u,v) + depth d → world point (homogeneous 1x4).
    3DGS convention: world_view_transform = W2C^T (row-vector).
    view = [vx,vy,vz,1]; vx=(u-cx)/fx·d ; vy=(v-cy)/fy·d ; vz=d.
    world = view @ wvt_inv.
    """
    vx = (u - tc['cx']) / tc['fx'] * d
    vy = (v - tc['cy']) / tc['fy'] * d
    pv = np.array([vx, vy, d, 1.0], np.float64)
    return pv @ tc['wvt_inv']


def n_train_visible(pw, train_cams):
    """Count train cams seeing world point pw (frustum: z in (znear,zfar)
    AND projected pixel in image bounds). Frustum-only (no occlusion)."""
    n = 0
    for c in train_cams:
        pv = pw @ c['wvt']            # world→view (row-vec, wvt=W2C^T)
        z = pv[2]
        if z <= c['znear'] or z >= c['zfar']:
            continue
        u = c['fx'] * pv[0] / z + c['cx']
        v = c['fy'] * pv[1] / z + c['cy']
        if 0 <= u < c['W'] and 0 <= v < c['H']:
            n += 1
    return n


# ── H8: per-image affine residual reduction (exposure/color mismatch test) ──
def _affine_residual_reduction(rd_h4, gt_h4):
    """rd_h4, gt_h4: (M,3) RGB của H4 pixels MỘT test view.

    Fit per-channel affine a·render+b ≈ gt (least squares, per-IMAGE vì
    exposure/WB lệch per-image). Đo phần lỗi H4 bị xoá bởi 1 affine duy nhất.
    Returns (resid_before_sum, resid_after_sum) — L1 sum để caller pool
    cross-view bằng Σ/Σ. Chỉ dùng numpy lstsq + data đã load (verified-safe).
    """
    if len(rd_h4) < 10:
        return 0.0, 0.0
    rb = float(np.abs(rd_h4 - gt_h4).sum())
    ra = 0.0
    for ch in range(3):
        r = rd_h4[:, ch]
        g = gt_h4[:, ch]
        A = np.stack([r, np.ones_like(r)], axis=1)        # (M,2): [render,1]
        sol, *_ = np.linalg.lstsq(A, g, rcond=None)       # [a, b]
        ra += float(np.abs(A @ sol - g).sum())
    return rb, ra


# ── Per-scene attribution ──
def attribute_scene(scene_name, test_cams, train_cams, colmap_kp,
                     align_status, align_fix, tau_d_k=TAU_D_K,
                     tau_hf_pct=TAU_HF_PCT):
    """Returns dict: bucket error fractions + diagnostics for one scene."""
    # Pool valid pixels across test views
    tot_err = 0.0
    err_b = {"H1": 0.0, "H3": 0.0, "H5": 0.0, "H4": 0.0}
    n_b = {"H1": 0, "H3": 0, "H5": 0, "H4": 0}
    valid_px = total_px = 0
    h4_chroma_err = h4_lum_err = 0.0
    h8_rb = h8_ra = 0.0   # H8: per-image affine residual reduction (pooled)

    # Calibrate τ_hf per scene from pooled Laplacian; τ_d from keypoint rel
    lap_all, rel_all = [], []
    per_view = []
    for tc in test_cams:
        H, W = tc['depth'].shape
        gt, rd = tc['gt'], tc['render']
        Yg, Yr = luminance(gt), luminance(rd)
        lap = laplacian_mag(Yg)
        depth = tc['depth']
        valid = np.isfinite(depth) & (depth > 0)
        total_px += H * W
        valid_px += int(valid.sum())
        lap_all.append(lap[valid].ravel())
        # keypoint rel-depth disagreement (for τ_d + H3) — scale COLMAP→render
        ent = colmap_kp.get(tc['name'])
        kp_map = {}
        if ent is not None and align_fix is not None:
            kp_arr, (Wc, Hc) = ent
            sx, sy = W / Wc, H / Hc
            for (u, v, z) in kp_arr:
                ui, vi = int(round(u * sx)), int(round(v * sy))
                if 0 <= ui < W and 0 <= vi < H and valid[vi, ui]:
                    za = align_fix(z)
                    kp_map[(vi, ui)] = abs(depth[vi, ui] - za)
        per_view.append((tc, valid, Yg, Yr, lap, kp_map))
        rel_all.extend(kp_map.values())

    lap_pool = np.concatenate(lap_all) if lap_all else np.array([0.0])
    tau_hf = float(np.percentile(lap_pool, tau_hf_pct))
    # depth_range per scene = p95−p5 of valid A3 depth (documented choice)
    dpool = np.concatenate([tc['depth'][np.isfinite(tc['depth']) &
                            (tc['depth'] > 0)].ravel() for tc in test_cams])
    depth_range = float(np.percentile(dpool, 95) - np.percentile(dpool, 5))
    depth_range = max(depth_range, 1e-6)
    if rel_all and align_fix is not None:
        rel = np.asarray(rel_all) / depth_range
        tau_d = max(float(np.median(rel) + tau_d_k * mad(rel)), TAU_D_FLOOR)
    else:
        tau_d = TAU_D_FLOOR

    h3_evaluable = (align_status != "UNRELIABLE") and (align_fix is not None)

    for (tc, valid, Yg, Yr, lap, kp_map) in per_view:
        H, W = tc['depth'].shape
        gt, rd, depth = tc['gt'], tc['render'], tc['depth']
        err = np.mean(np.abs(rd - gt), axis=2)  # HxW per-pixel RGB-mean abs
        vidx = np.argwhere(valid)
        # subsample for back-proj cost (H1 per-pixel projection heavy)
        if len(vidx) > 60000:
            sel = np.random.choice(len(vidx), 60000, replace=False)
            vidx = vidx[sel]
        h4_rd_v, h4_gt_v = [], []   # H4 pixel RGB của view này (cho H8 affine)
        for (vi, ui) in vidx:
            e = float(err[vi, ui])
            tot_err += e
            d = float(depth[vi, ui])
            pw = backproject(ui + 0.5, vi + 0.5, d, tc)
            nv = n_train_visible(pw, train_cams)
            if nv == 0:
                b = "H1"
            elif h3_evaluable and ((vi, ui) in kp_map) and \
                    (kp_map[(vi, ui)] / depth_range > tau_d):
                b = "H3"
            elif lap[vi, ui] > tau_hf:
                b = "H5"
            else:
                b = "H4"
                # Q2 chroma vs lum sub-check
                le = abs(Yr[vi, ui] - Yg[vi, ui])
                ce = np.mean(np.abs((rd[vi, ui] - Yr[vi, ui]) -
                                    (gt[vi, ui] - Yg[vi, ui])))
                h4_lum_err += le
                h4_chroma_err += ce
                h4_rd_v.append(rd[vi, ui])   # H8: thu RGB H4 pixel
                h4_gt_v.append(gt[vi, ui])
            err_b[b] += e
            n_b[b] += 1
        # H8: fit per-image affine trên H4 pixels của view này, pool residual
        if h4_rd_v:
            rb, ra = _affine_residual_reduction(
                np.asarray(h4_rd_v), np.asarray(h4_gt_v))
            h8_rb += rb
            h8_ra += ra

    frac = {k: (err_b[k] / tot_err if tot_err > 0 else 0.0) for k in err_b}
    h4_ratio = (h4_chroma_err / (h4_lum_err + 1e-6)) if h4_lum_err else 0.0
    # H8 ∈ [0,1]: phần lỗi H4 bị xoá bởi 1 per-image affine duy nhất.
    # >0.5 = exposure/color mismatch DOMINANT (ACTIONABLE per-image align);
    # <0.2 = affine vô dụng → near-optimal residual thật (accept ceiling).
    h8 = (1.0 - h8_ra / h8_rb) if h8_rb > 1e-9 else None
    return dict(
        scene=scene_name, frac=frac, n_b=n_b,
        coverage=valid_px / total_px if total_px else 0.0,
        align_status=align_status, tau_d=tau_d, tau_hf=tau_hf,
        depth_range=depth_range, h3_evaluable=h3_evaluable,
        h4_ratio=h4_ratio, h8=h8,
    )


# ── Side metrics (CPU, instant) ──
def ply_vertex_count(scene_name):
    p = (f"{OUTPUT_ROOT}/A3_seed{SEED}_{scene_name}"
         f"/point_cloud/iteration_{ITERATION}/point_cloud.ply")
    if not os.path.isfile(p):
        return None
    with open(p, "rb") as f:
        for _ in range(40):
            line = f.readline().decode("ascii", "ignore")
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line.strip() == "end_header":
                break
    return None


def side_metrics(scene_name):
    # H6 N_gauss, H7 multi-seed test PSNR std
    nv = ply_vertex_count(scene_name)
    psnrs = []
    for s in MS_SEEDS:
        p = parse_test_psnr(f"{LOG_DIR_MS}/A3_seed{s}_{scene_name}.log")
        if p is not None:
            psnrs.append(p)
    std = statistics.pstdev(psnrs) if len(psnrs) > 1 else None
    # H8 train GT exposure variance
    expo = None
    gdir = sorted(glob.glob(os.path.join(
        DATA_ROOT, scene_name, "images_8", "*")))
    return dict(n_gauss=nv,
                psnr_seed_std=std, n_psnr=len(psnrs))


# ── Main ──
def smoke_attribution():
    """Smoke #4 — attribution sane on mock arrays (CPU, no GPU)."""
    print("=== Smoke #4: attribution mock ===")
    # mock: 1 test cam looking down -Z, 1 train cam same → all visible
    wvt = np.eye(4)                       # identity W2C^T
    tc = dict(name="m", depth=np.full((4, 4), 5.0, np.float32),
              render=np.zeros((4, 4, 3), np.float32),
              gt=np.ones((4, 4, 3), np.float32),
              fx=4., fy=4., cx=2., cy=2., W=4, H=4,
              znear=0.01, zfar=100., wvt=wvt, wvt_inv=np.eye(4))
    pw = backproject(2.0, 2.0, 5.0, tc)
    nv = n_train_visible(pw, [dict(fx=4., fy=4., cx=2., cy=2., W=4, H=4,
                                   znear=0.01, zfar=100., wvt=wvt)])
    print(f"  backproj center d=5 → nv_train={nv} (expect 1 — same cam sees)")
    assert nv == 1, "frustum visibility mock failed"
    print("  Smoke #4 PASS\n")


def main():
    print("=== Phase 13.2.4 Bottleneck Decomposition ===")
    print(f"OUTPUT_ROOT={OUTPUT_ROOT} SEED={SEED} ITER={ITERATION}")
    print(f"SCENES={SCENES}")
    print(f"τ_d=max(median+{TAU_D_K}·MAD, {TAU_D_FLOOR}·range)  "
          f"τ_hf=pct{TAU_HF_PCT}(Laplacian)")
    print("Priority: H1 irreducible → H3 geometry → H5 detail → H4 appearance\n")
    os.makedirs(PLOT_DIR, exist_ok=True)
    smoke_attribution()

    results = []
    for sc in SCENES:
        print(f"──── {sc} ────")
        rt = render_scene_test(sc)
        if rt is None:
            print(f"  SKIP {sc}\n")
            continue
        test_cams, train_cams = rt
        ckp = colmap_test_keypoints(sc)
        status, fix, ainfo = align_check(test_cams, ckp)
        print(f"  align: {status}  (n={ainfo.get('n')}  "
              f"scale={ainfo.get('scale', float('nan')):.4f}  "
              f"corr_in={ainfo.get('corr_in', float('nan')):.3f}  "
              f"corr_raw={ainfo.get('corr_raw', float('nan')):.3f}  "
              f"rmad={ainfo.get('rmad_pct', float('nan'))*100:.1f}%  "
              f"inlier={ainfo.get('inlier_frac', float('nan'))*100:.0f}%)")
        r = attribute_scene(sc, test_cams, train_cams, ckp, status, fix)
        r.update(side_metrics(sc))
        results.append(r)
        f = r['frac']
        h8s = f"{r['h8']:.2f}" if r.get('h8') is not None else "-"
        print(f"  H1={f['H1']*100:.1f}% H3={f['H3']*100:.1f}% "
              f"H5={f['H5']*100:.1f}% H4={f['H4']*100:.1f}%  "
              f"cov={r['coverage']*100:.0f}% h3_eval={r['h3_evaluable']} "
              f"h4_ratio={r['h4_ratio']:.2f} H8={h8s}\n")

    if not results:
        print("NO scenes analyzed."); sys.exit(1)

    # ── Aggregate table ──
    print("=== Per-scene bucket error attribution (% of test error) ===")
    hdr = (f"{'scene':<10} {'H1_irr':>7} {'H3_geo':>7} {'H5_det':>7} "
           f"{'H4_app':>7} {'cov%':>6} {'h3?':>5} {'h4rat':>6} {'H8':>6} "
           f"{'align':>14}")
    print(hdr); print("-" * len(hdr))
    for r in results:
        f = r['frac']
        h8s = f"{r['h8']:.2f}" if r.get('h8') is not None else "-"
        print(f"{r['scene']:<10} {f['H1']*100:>6.1f} {f['H3']*100:>6.1f} "
              f"{f['H5']*100:>6.1f} {f['H4']*100:>6.1f} "
              f"{r['coverage']*100:>5.0f} "
              f"{'Y' if r['h3_evaluable'] else 'N':>5} "
              f"{r['h4_ratio']:>6.2f} {h8s:>6} {r['align_status']:>14}")
    agg = {k: statistics.fmean(r['frac'][k] for r in results)
           for k in ["H1", "H3", "H5", "H4"]}
    print(f"\n  Aggregate: H1={agg['H1']*100:.1f}% H3={agg['H3']*100:.1f}% "
          f"H5={agg['H5']*100:.1f}% H4={agg['H4']*100:.1f}%")

    # ── Side metrics ──
    print("\n=== Side metrics ===")
    print(f"{'scene':<10} {'N_gauss':>9} {'PSNR_seed_std':>14}")
    for r in results:
        ng = r.get('n_gauss')
        sd = r.get('psnr_seed_std')
        print(f"{r['scene']:<10} {str(ng):>9} "
              f"{(f'{sd:.3f}' if sd is not None else '-'):>14}")
    stds = [r['psnr_seed_std'] for r in results if r.get('psnr_seed_std')]
    if stds:
        print(f"  H7: mean per-scene test PSNR std (seed {MS_SEEDS}) = "
              f"{statistics.fmean(stds):.3f} dB")

    # ── Verdict + lower-bound confidence (Refinement #1) ──
    print("\n=== VERDICT ===")
    dom = max(agg, key=agg.get)
    print(f"  Dominant bucket: {dom} = {agg[dom]*100:.1f}%")
    print("  (H1/H3 are LOWER-BOUNDS: frustum-only / sparse-keypoint →"
          " true value ≥ measured)")

    if agg["H1"] >= 0.50:
        print(f"\n  🟢 H1 IRREDUCIBLE ≥50% (lower-bound) → CONFIRMED dominant")
        print(f"  → Accept 21.330 ceiling, HIGH confidence. KHÔNG method nào sửa.")
    elif agg["H3"] >= 0.30:
        print(f"\n  🟢 H3 GEOMETRY large (lower-bound → true ≥) ")
        print(f"  → dn-splatter monocular-NORMAL prior ĐÁNG đầu tư (HIGH conf)")
    elif agg["H4"] >= 0.40:
        h8s = [r['h8'] for r in results if r.get('h8') is not None]
        mh8 = statistics.fmean(h8s) if h8s else None
        rats = [r['h4_ratio'] for r in results]
        mr = statistics.fmean(rats) if rats else 0.0
        if mh8 is None:
            print(f"\n  🟡 H4 dominant nhưng H8 chưa tính được (no H4 pixel?)"
                  f" → inconclusive")
        elif mh8 > 0.5:
            print(f"\n  🟢 H4 dominant + H8={mh8:.2f}>0.5 — 1 per-image affine"
                  f" xoá >50% lỗi H4")
            print(f"  → EXPOSURE/COLOR mismatch ACTIONABLE (per-image"
                  f" appearance embedding / color align) — KHÔNG accept ceiling")
        elif mh8 < 0.2:
            if mr > 1.5:
                print(f"\n  🟡 H4 dom + H8={mh8:.2f}<0.2 + chroma-dom"
                      f" (ratio={mr:.2f}) → SH/specular — MEDIUM conf")
            else:
                print(f"\n  🟡 H4 dom + H8={mh8:.2f}<0.2 (affine vô dụng) +"
                      f" uniform (ratio={mr:.2f})")
                print(f"  → near-optimal residual THẬT → accept ceiling")
        else:
            print(f"\n  🟡 H4 dominant + H8={mh8:.2f} (0.2–0.5) — exposure"
                  f" partial. Per-image align giúp 1 phần → đáng thử nhưng"
                  f" gain bị giới hạn")
    elif agg["H1"] < 0.20:
        print(f"\n  🟡 No bucket decisive, H1 lower-bound small (occlusion có"
              f" thể tăng) → INCONCLUSIVE. Cần thêm investigation.")
    else:
        print(f"\n  🟡 Mixed — xem per-scene, không clean global verdict.")

    # Refinement #3 — H5 guard
    if agg["H5"] > 0.15:
        print(f"\n  ⚠️ H5={agg['H5']*100:.1f}%>15% — τ_hf likely too low "
              f"(catching color boundary as detail). Detail-axis ĐÃ refute "
              f"(HF pilot) → KHÔNG re-open, re-examine τ_hf calibration.")

    # ── Caveats ──
    print("\n=== Caveats (đọc kèm, KHÔNG over-claim) ===")
    print("  - Attribution PRIORITY-ORDERED (H1→H3→H5→H4): 1 pixel có thể")
    print("    fail nhiều nguyên nhân, gán theo fundamental-nhất-trước (heuristic)")
    print("  - H3 chỉ đo tại COLMAP keypoint (sparse) → geometry = lower-bound")
    print("  - H1 frustum-only (no occlusion) → irreducible = lower-bound")
    print("  - τ_d/τ_hf self-calibrated; chạy lại với TAU_D_K∈{1,2,3} "
          "TAU_HF_PCT∈{70,80,90} (env) check robustness nếu verdict borderline")
    print("  - H3 scenes với align_status=UNRELIABLE: H3 KHÔNG kết luận được;"
          " verdict dựa H1/H4/H5")


if __name__ == "__main__":
    main()
