#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 14] Dense-init substrate Gate (matcher-AGNOSTIC).
# File: scripts/p14_denseinit_gate.py  (TẠO MỚI)
# Mục đích: TRƯỚC khi tốn công clone/integrate matcher (PDCNet/RoMa/
#   RoMav2) + 8-scene preprocess, trả lời 1 câu DUY NHẤT (no-train):
#     "Lỗi render của A3 có nằm trong vùng matcher CÓ THỂ giúp không?"
#   Matcher giúp được ⟺ điểm 3D đó (a) ≥2 train-view cùng thấy
#   (covisible → triangulate được; covis<2 → KHÔNG matcher nào tạo
#   điểm được, kể cả RoMa — giới hạn cấu trúc) VÀ (b) COLMAP-sparse
#   CHƯA cover (nếu đã có keypoint gần đó → sparse-init đã seed →
#   dense-init thừa = saturate).
#
#   → Attribute ERROR-MASS test của A3 vào 3 bucket:
#     NO_HELP        : err ∧ covis<2  (cấu trúc — matcher bất lực)
#     ALREADY_COVERED: err ∧ covis≥2 ∧ COLMAP-keypoint gần (đã seed)
#     DENSE_SUBSTRATE: err ∧ covis≥2 ∧ KHÔNG COLMAP gần  ← chỗ DUY
#                      NHẤT dense-init có thể thêm value
#
#   Verdict: DENSE_SUBSTRATE nhỏ → dense-init đóng (matcher-agnostic,
#   khỏi đụng RoMa). Non-trivial → matcher-choice + option-B mới đáng.
#
# MATCHER-AGNOSTIC: KHÔNG chạy PDCNet/RoMa — chỉ đo VỊ TRÍ lỗi A3 so
# covisibility + COLMAP-coverage. Trả lời "có việc không" trước "dùng
# tool nào". covis = frustum-only (no occlusion) → NO_HELP là LOWER-
# BOUND (true ≥) → nếu lower-bound đã lớn, kết luận robust.
#
# Reuse VERIFIED scripts/p13_2_bottleneck_decompose.py:128-370
#   (render_scene_test, colmap_test_keypoints, backproject,
#    n_train_visible, _stem, _read_points3d_xyz_by_id). Standalone,
#   no-train, KHÔNG đụng production (Gate discipline).
# ============================================================
"""[CRSGaussian Phase 14] Dense-init substrate Gate (no-train, matcher-agnostic).

Run (GPU SERVER):
    python scripts/p14_denseinit_gate.py
    COLMAP_RADII="8 16 24" python scripts/p14_denseinit_gate.py
Smoke (CPU local — syntax):
    python -c "import ast; ast.parse(open('scripts/p14_denseinit_gate.py').read()); print('OK')"
"""

import os
import sys
import math
import struct
import statistics

import numpy as np

sys.path.insert(0, ".")

DATA_ROOT = os.environ.get("DATA_ROOT", "data/nerf_llff_data")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "output/p13_lfcf")
SEED = os.environ.get("SEED", "42")
ITERATION = int(os.environ.get("ITERATION", "10000"))
SCENES = os.environ.get(
    "SCENES", "fern flower fortress horns leaves orchids room trex").split()
# "COLMAP-covered" = pixel trong R px của 1 COLMAP-keypoint-projection
# (proxy: sparse point seed Gaussian phủ 1 lân cận). Sweep sensitivity.
COLMAP_RADII = [float(x) for x in os.environ.get("COLMAP_RADII",
                                                 "8 16 24").split()]
# err "cao" = err > median + K·MAD (self-calibrate, KHÔNG bịa hằng số —
# đúng kỷ luật bottleneck τ self-calibration).
ERR_K = float(os.environ.get("ERR_K", "1.0"))


def _stem(name):
    return os.path.basename(name).split(".")[0]


def _read_points3d_xyz_by_id(path):
    """points3D.bin id→xyz (verified bottleneck:226-245)."""
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            pr = struct.unpack("<QdddBBBd", f.read(43))
            out[pr[0]] = np.array(pr[1:4], np.float64)
            tl = struct.unpack("<Q", f.read(8))[0]
            f.read(8 * tl)
    return out


def render_scene_test(scene_name):
    """Load A3 ckpt + render test views. Mirror VERIFIED
    bottleneck_decompose.py:128-222 (depth /(alpha+1e-6), camera
    convention wvt=W2C^T). Returns (test_list, train_list) or None."""
    import torch
    from argparse import ArgumentParser, Namespace
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from arguments import ModelParams, PipelineParams

    model_path = f"{OUTPUT_ROOT}/A3_seed{SEED}_{scene_name}"
    cfg_path = os.path.join(model_path, "cfg_args")
    ply = f"{model_path}/point_cloud/iteration_{ITERATION}/point_cloud.ply"
    if not (os.path.isfile(cfg_path) and os.path.isfile(ply)):
        print(f"  ERR missing ckpt/cfg {model_path}")
        return None
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    _ = lp
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
    pipe = pp.extract(args)
    torch.cuda.empty_cache()
    with torch.no_grad():
        gaussians = GaussianModel(args)
        scene = Scene(args, gaussians, load_iteration=ITERATION,
                      shuffle=False)
        bg = torch.tensor([0., 0., 0.], dtype=torch.float32, device="cuda")

        def cam_intr(c):
            W, H = c.image_width, c.image_height
            fx = W / (2.0 * math.tan(c.FoVx * 0.5))
            fy = H / (2.0 * math.tan(c.FoVy * 0.5))
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
            rd = pkg["render"].clamp(0, 1).detach().cpu().numpy()       # 3HW
            al = np.squeeze(pkg["alpha"].detach().cpu().numpy())
            dp = np.squeeze(pkg["depth"].detach().cpu().numpy())
            dp = dp / (al + 1e-6)                                       # #2
            gt = c.original_image[:3].clamp(0, 1).detach().cpu().numpy()
            fx, fy, cx, cy, W, H = cam_intr(c)
            wvt = c.world_view_transform.detach().cpu().numpy()
            test_list.append(dict(
                name=_stem(c.image_name),
                depth=dp.astype(np.float32),
                render=np.transpose(rd, (1, 2, 0)).astype(np.float32),
                gt=np.transpose(gt, (1, 2, 0)).astype(np.float32),
                fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H,
                wvt=wvt, wvt_inv=np.linalg.inv(wvt)))
    del gaussians, scene
    torch.cuda.empty_cache()
    return test_list, train_list


def colmap_test_keypoints(scene_name):
    """test image_name → (np (M,2)=(u,v) @COLMAP-res, (Wc,Hc)).
    Verified bottleneck:248-282 (xys @intr.width; consumer scale)."""
    from scene.colmap_loader import (read_extrinsics_binary,
                                     read_intrinsics_binary, qvec2rotmat)
    sp = os.path.join(DATA_ROOT, scene_name, "sparse", "0")
    images = read_extrinsics_binary(os.path.join(sp, "images.bin"))
    intr = read_intrinsics_binary(os.path.join(sp, "cameras.bin"))
    xyz = _read_points3d_xyz_by_id(os.path.join(sp, "points3D.bin"))
    out = {}
    for img in images.values():
        stem = _stem(img.name)
        cam = intr[img.camera_id]
        Wc, Hc = int(cam.width), int(cam.height)
        R = qvec2rotmat(img.qvec)
        t = np.asarray(img.tvec, np.float64)
        rows = []
        for (xy, pid) in zip(img.xys, img.point3D_ids):
            if pid < 0 or pid not in xyz:
                continue
            Xc = R @ xyz[pid] + t
            if Xc[2] <= 0:
                continue
            rows.append((float(xy[0]), float(xy[1])))
        if rows:
            out[stem] = (np.asarray(rows, np.float64), (Wc, Hc))
    return out


def backproject(u, v, d, tc):
    """test pixel(u,v)+depth d → world (1x4). bottleneck:345-354."""
    vx = (u - tc['cx']) / tc['fx'] * d
    vy = (v - tc['cy']) / tc['fy'] * d
    return np.array([vx, vy, d, 1.0], np.float64) @ tc['wvt_inv']


def n_train_visible(pw, train_cams):
    """#train cams thấy world pt (frustum-only, no occlusion → covis
    lower-bound). bottleneck:357-370."""
    n = 0
    for c in train_cams:
        pv = pw @ c['wvt']
        z = pv[2]
        if z <= c['znear'] or z >= c['zfar']:
            continue
        u = c['fx'] * pv[0] / z + c['cx']
        v = c['fy'] * pv[1] / z + c['cy']
        if 0 <= u < c['W'] and 0 <= v < c['H']:
            n += 1
    return n


def _colmap_mask(tc, ckp, radius):
    """bool (H,W): pixel trong `radius` px của 1 COLMAP-kp-projection
    (scale COLMAP-res→render-res). Proxy 'sparse-init đã seed gần'."""
    H, W = tc['depth'].shape
    m = np.zeros((H, W), bool)
    ent = ckp.get(tc['name'])
    if ent is None:
        return m
    kp, (Wc, Hc) = ent
    sx, sy = W / Wc, H / Hc
    r = int(math.ceil(radius))
    for (u, v) in kp:
        ui, vi = int(round(u * sx)), int(round(v * sy))
        if 0 <= ui < W and 0 <= vi < H:
            y0, y1 = max(0, vi - r), min(H, vi + r + 1)
            x0, x1 = max(0, ui - r), min(W, ui + r + 1)
            m[y0:y1, x0:x1] = True
    return m


def gate_scene(scene_name, test_cams, train_cams, ckp, radius):
    """→ dict: error-mass fraction per bucket cho 1 scene, 1 radius."""
    tot = 0.0
    b = {"NO_HELP": 0.0, "ALREADY": 0.0, "SUBSTRATE": 0.0}
    for tc in test_cams:
        H, W = tc['depth'].shape
        err = np.mean(np.abs(tc['render'] - tc['gt']), axis=2)   # (H,W)
        depth = tc['depth']
        valid = np.isfinite(depth) & (depth > 0)
        ev = err[valid]
        if ev.size == 0:
            continue
        med = float(np.median(ev))
        mad = float(np.median(np.abs(ev - med)))
        thr = med + ERR_K * mad                       # self-calibrate
        cmask = _colmap_mask(tc, ckp, radius)
        vidx = np.argwhere(valid & (err > thr))
        if len(vidx) > 60000:
            sel = np.random.choice(len(vidx), 60000, replace=False)
            vidx = vidx[sel]
        for (vi, ui) in vidx:
            e = float(err[vi, ui])
            tot += e
            pw = backproject(ui + 0.5, vi + 0.5, float(depth[vi, ui]), tc)
            nv = n_train_visible(pw, train_cams)
            if nv < 2:
                b["NO_HELP"] += e
            elif cmask[vi, ui]:
                b["ALREADY"] += e
            else:
                b["SUBSTRATE"] += e
    if tot <= 0:
        return None
    return {k: v / tot for k, v in b.items()}


def main():
    print("=== Phase 14 Dense-init Gate (no-train, matcher-AGNOSTIC) ===")
    print(f"OUTPUT_ROOT={OUTPUT_ROOT} SEED={SEED} ITER={ITERATION}")
    print(f"SCENES={SCENES}  COLMAP_RADII={COLMAP_RADII}px  "
          f"err_thr=median+{ERR_K}·MAD")
    print("Bucket A3 test-error-mass: NO_HELP(covis<2) | "
          "ALREADY(covis≥2 ∧ COLMAP-near) | SUBSTRATE(covis≥2 ∧ no-COLMAP)\n")

    loaded = {}
    for sc in SCENES:
        print(f"──── {sc} ────")
        rt = render_scene_test(sc)
        if rt is None:
            print(f"  SKIP {sc}\n"); continue
        tc, trc = rt
        ckp = colmap_test_keypoints(sc)
        loaded[sc] = (tc, trc, ckp)
        for R in COLMAP_RADII:
            r = gate_scene(sc, tc, trc, ckp, R)
            if r is None:
                print(f"  R={R:.0f}px: no error mass"); continue
            print(f"  R={R:>4.0f}px | NO_HELP={r['NO_HELP']*100:5.1f}%  "
                  f"ALREADY={r['ALREADY']*100:5.1f}%  "
                  f"SUBSTRATE={r['SUBSTRATE']*100:5.1f}%")
        print()

    if not loaded:
        print("NO scenes."); sys.exit(1)

    print("=== AGGREGATE (mean over scenes, per radius) ===")
    any_substrate = False
    for R in COLMAP_RADII:
        rows = []
        for sc, (tc, trc, ckp) in loaded.items():
            r = gate_scene(sc, tc, trc, ckp, R)
            if r:
                rows.append(r)
        if not rows:
            continue
        nh = statistics.fmean(x["NO_HELP"] for x in rows)
        al = statistics.fmean(x["ALREADY"] for x in rows)
        su = statistics.fmean(x["SUBSTRATE"] for x in rows)
        # SUBSTRATE = LOWER-BOUND-favorable: covis frustum-only (true
        # NO_HELP ≥ đo) → true SUBSTRATE ≤ đo. Verdict dùng đo (ceiling).
        if su >= 0.20:
            tag = "🎯 SUBSTRATE ≥20% → matcher-choice + option-B đáng cân"
            any_substrate = True
        elif su >= 0.10:
            tag = "🟡 10–20% — biên; ceiling thật ≤ này (covis lower-bound)"
        else:
            tag = "❌ <10% → dense-init ĐÓNG (matcher-agnostic, khỏi RoMa)"
        print(f"  R={R:>4.0f}px | NO_HELP={nh*100:5.1f}%  "
              f"ALREADY={al*100:5.1f}%  SUBSTRATE={su*100:5.1f}%  {tag}")

    print("\n=== Kết luận ===")
    if any_substrate:
        print("  🎯 ≥1 radius cho SUBSTRATE ≥20% → CÓ đất dense-init.")
        print("  → BƯỚC TIẾP: option-B (dense=CRS-prior, KHÔNG replace")
        print("    init → né Phase-10A recipe-risk) + so PDCNet vs RoMa")
        print("    (RoMa wide-baseline-robust mới relevant ở bước này).")
    else:
        print("  ❌ Mọi radius SUBSTRATE <20% (và là CEILING — covis")
        print("    frustum lower-bound → thật còn ≤). Lỗi A3 = NO_HELP")
        print("    (covis<2, cấu trúc) + ALREADY (sparse đã seed) →")
        print("    dense-init KHÔNG chạm bottleneck, MỌI matcher vô")
        print("    nghĩa (PDCNet=RoMa=RoMav2). ĐÓNG dense-init sạch.")

    print("\n=== Caveat (đọc kèm) ===")
    print("  - covis frustum-only (no occlusion) → NO_HELP lower-bound,")
    print("    SUBSTRATE là CEILING (thật ≤). <20% ceiling = kết luận")
    print("    đóng RẤT robust; ≥20% = necessary KHÔNG sufficient (vẫn")
    print("    còn recipe-risk Phase-10A + matcher-quality + horns-frag).")
    print("  - COLMAP-covered = proxy bán-kính dilate (sweep R sensitivity).")
    print("  - err_thr self-calibrate median+K·MAD (KHÔNG hằng số bịa).")
    print("  - Gate = 'có việc cho matcher không', KHÔNG 'dense-init sẽ")
    print("    work'. GO = đáng điều tra tiếp; NO-GO = đóng dứt điểm.")


if __name__ == "__main__":
    main()
