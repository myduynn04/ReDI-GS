#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 14] L_consist saturation pre-check.
# File: scripts/p14_lconsist_precheck.py  (TẠO MỚI)
# Mục đích: TRƯỚC khi tốn 24 pilot-run, trả lời "A3 còn 'đất' cho
#           L_consist sửa không?" — đo CHÍNH residual L_consist thật
#           trên A3 checkpoint (no-train, ~20 phút, standalone).
#
# Cơ chế (verified Binocular train.py:123-136):
#   từ 1 train view → camera dịch ngang baseline B → render 2 cam →
#   disp = focal·(−B)/depth → inverse-warp shifted→orig → L1 vs GT.
#
# substrate = res(B │ valid_mask ∧ ~Lambertian)
#           − res(B→0 │ CÙNG region)
#   • res(B→0): warp→identity → = train-fit floor (common term, triệt
#     tiêu trong hiệu) — KHÔNG phải interp-noise (verified: B→0 ⇒
#     shifted_cam→orig_cam, disp→0, warp→identity).
#   • ~Lambertian mask: tiny-rotate-about-CENTER probe (zero parallax →
#     cô lập view-dependent SH/specular). Loại specular-false-accept.
#   • Bilinear-interp = known small +bias (mọc theo B, KHÔNG khử được)
#     → kết luận "có đất" CHỈ khi substrate vượt floor ĐỦ BIÊN.
#   • UNMASKED substrate = UPPER-BOUND (Lamb-mask chỉ giảm) → nếu
#     upper-bound ≈ 0 thì KẾT LUẬN reject chắc chắn (true ≤ upper).
#
# Verified-from-code (KHÔNG đoán — kỷ luật GDAGS):
#   - model-load: mirror scripts/p13_2_bottleneck_decompose.py:137-183
#   - depth ACCUMULATED → /(alpha+1e-6): bottleneck:203-208,
#     crs_module.py:697-708 (#2 fix, silent-bug)
#   - shifted-cam: PseudoCamera (scene/cameras.py:66-87) + W2C=wvt.T
#     (d_cycle.py:31), trans = world-disp của [B,0,0] camera-local (#1)
#   - rotate-about-center: getWorld2View2 (graphics_utils.py:38-49):
#       center = −R·t ; c2w_rot = R  ⇒  giữ center: t' = R'ᵀ·R·t
#   - fov2focal (graphics_utils.py:100): px/(2·tan(fov/2))
#
# Standalone — KHÔNG đụng production code/train.py (Gate discipline).
# ============================================================
"""[CRSGaussian Phase 14] L_consist saturation pre-check (no-train).

Run (GPU SERVER, sau khi Bước-0 PASS):
    # MẶC ĐỊNH = FULL 8-scene (quyết-định-reject phải full-8)
    python scripts/p14_lconsist_precheck.py
    # Smoke nhanh 3-scene (CHỈ để debug script, KHÔNG dùng để reject):
    SCENES="orchids horns trex" python scripts/p14_lconsist_precheck.py

Smoke (CPU local — syntax only):
    python -c "import ast; ast.parse(open('scripts/p14_lconsist_precheck.py').read()); print('OK')"
"""

import os
import sys
import math

import numpy as np

sys.path.insert(0, ".")

# ── Config ──
DATA_ROOT = os.environ.get("DATA_ROOT", "data/nerf_llff_data")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "output/p13_lfcf")
SEED = os.environ.get("SEED", "42")
ITERATION = int(os.environ.get("ITERATION", "10000"))
# FULL-8 mặc định: pre-check RA QUYẾT ĐỊNH REJECT (no-substrate → bỏ cả
# hướng) → per feedback_full_8scene_ablation + per-scene heterogeneity cao
# (GDAGS horns −0.605, bottleneck leaves H1 92.6% vs agg 21%) → KHÔNG
# subset-reject. No-train nên 8 scene cost thấp. Override SCENES= để smoke.
SCENES = os.environ.get(
    "SCENES", "fern flower fortress horns leaves orchids room trex").split()
# B = baseline dịch ngang, tính theo FRACTION của scene cameras_extent
# (#3: Bino 0.4 là world-scale RIÊNG của Bino → KHÔNG copy; ta param theo
#  extent + sweep để pre-check tự lộ parallax/disocclusion tradeoff).
B_FRACS = [float(x) for x in os.environ.get("B_FRACS", "0.02 0.05 0.10").split()]
B_EPS_FRAC = float(os.environ.get("B_EPS_FRAC", "0.0005"))  # ≈0 baseline
ROT_DELTA_DEG = float(os.environ.get("ROT_DELTA_DEG", "0.5"))  # Lamb probe yaw
# ~Lambertian = pixel mà tiny-rotate đổi màu < pct LAMB_PCT của phân bố
LAMB_PCT = float(os.environ.get("LAMB_PCT", "60"))
# substrate "có đất" cần vượt floor ≥ MARGIN_FRAC × floor (hấp thụ
# known interp +bias) VÀ ≥ MARGIN_ABS tuyệt đối (L1 image-space).
MARGIN_FRAC = float(os.environ.get("MARGIN_FRAC", "0.50"))
MARGIN_ABS = float(os.environ.get("MARGIN_ABS", "0.010"))


# ── Camera builders (verified-from-code) ──
def _make_shifted_cam(cam, B):
    """PseudoCamera dịch camera-center +B theo trục X-local (R giữ nguyên
    → rectified stereo). W2C = wvt.T (d_cycle.py:31). pt_world = C2W@[B,0,0,1];
    trans = pt_world − camera_center (đúng Binocular getShiftedCamera logic,
    nhưng dùng wvt thay get_camera_matrix mà CRS Camera KHÔNG có — #1 fix)."""
    import torch
    from scene.cameras import PseudoCamera
    W2C = cam.world_view_transform.T                       # (4,4) actual W2C
    pt_cam = torch.tensor([float(B), 0.0, 0.0, 1.0],
                          device=W2C.device, dtype=W2C.dtype)
    pt_world = torch.inverse(W2C) @ pt_cam                 # C2W @ pt_cam
    trans = (pt_world[:3] - cam.camera_center).detach().cpu().numpy()
    return PseudoCamera(R=cam.R, T=cam.T, FoVx=cam.FoVx, FoVy=cam.FoVy,
                        width=cam.image_width, height=cam.image_height,
                        trans=trans, scale=float(getattr(cam, "scale", 1.0)))


def _make_rotated_cam(cam, deg):
    """PseudoCamera xoay quanh CHÍNH camera-center góc nhỏ `deg` (yaw,
    trục Y-local) → ZERO parallax, chỉ đổi view-direction → cô lập
    view-dependent SH/specular (Lambertian probe).

    Derived từ getWorld2View2 (graphics_utils.py:38-49):
      center = −R·t ;  c2w_rot = R.
    Giữ center khi đổi R→R': t' = R'ᵀ · R · t  (⇒ −R'·t' = −R·t = center).
    """
    R = np.asarray(cam.R, np.float64)
    T = np.asarray(cam.T, np.float64)
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    Ry = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], np.float64)
    Rp = R @ Ry                                    # rotate trong camera-frame
    Tp = Rp.T @ R @ T                              # giữ center bất biến
    from scene.cameras import PseudoCamera
    return PseudoCamera(R=Rp, T=Tp, FoVx=cam.FoVx, FoVy=cam.FoVy,
                        width=cam.image_width, height=cam.image_height,
                        trans=np.array([0.0, 0.0, 0.0]),
                        scale=float(getattr(cam, "scale", 1.0)))


def _inverse_warp_cols(img, disparity):
    """warped[:,r,c] = bilinear(img[:,r, c+disparity[r,c]]) — column-only
    inverse warp (rectified stereo), faithful Binocular inverse_warp_images
    (graphics_utils.py:80-118 semantics) qua grid_sample bilinear.

    Args: img (3,H,W) tensor; disparity (H,W) tensor (pixels).
    Returns: warped (3,H,W), valid (H,W) bool (sample in-bounds)."""
    import torch
    C, H, W = img.shape
    ys, xs = torch.meshgrid(
        torch.arange(H, device=img.device, dtype=img.dtype),
        torch.arange(W, device=img.device, dtype=img.dtype),
        indexing="ij")
    xsrc = xs + disparity                              # source column
    valid = (xsrc >= 0) & (xsrc <= (W - 1))
    gx = 2.0 * xsrc / max(W - 1, 1) - 1.0
    gy = 2.0 * ys / max(H - 1, 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)  # (1,H,W,2)
    out = torch.nn.functional.grid_sample(
        img.unsqueeze(0), grid, mode="bilinear",
        padding_mode="zeros", align_corners=True).squeeze(0)
    return out, valid


# ── Model load (mirror bottleneck render path, verified) ──
def _load_scene(scene_name):
    import torch
    from argparse import ArgumentParser, Namespace
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from arguments import ModelParams, PipelineParams

    model_path = f"{OUTPUT_ROOT}/A3_seed{SEED}_{scene_name}"
    cfg_path = os.path.join(model_path, "cfg_args")
    ply = f"{model_path}/point_cloud/iteration_{ITERATION}/point_cloud.ply"
    if not (os.path.isfile(cfg_path) and os.path.isfile(ply)):
        print(f"  ERR missing ckpt/cfg at {model_path}")
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
    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians, load_iteration=ITERATION, shuffle=False)
    bg = torch.tensor([0., 0., 0.], dtype=torch.float32, device="cuda")
    return gaussians, scene, pipe, bg, render


def _focal_x(cam):
    from utils.graphics_utils import fov2focal
    return fov2focal(cam.FoVx, cam.image_width)


# ── Per-scene substrate ──
def _scene_substrate(scene_name):
    import torch
    loaded = _load_scene(scene_name)
    if loaded is None:
        return None
    gaussians, scene, pipe, bg, render = loaded
    extent = float(getattr(scene, "cameras_extent", 0.0)) or 1.0

    rows = []
    with torch.no_grad():
        train_cams = scene.getTrainCameras()
        for cam in train_cams:
            gt = cam.original_image[:3].clamp(0, 1)              # (3,H,W)
            pkg_o = render(cam, gaussians, pipe, bg)
            depth = pkg_o["depth"].squeeze(0)
            alpha = pkg_o["alpha"].squeeze(0)
            depth_n = depth / (alpha + 1e-6)                     # #2 fix
            fx = _focal_x(cam)

            # ~Lambertian mask: tiny-rotate (zero parallax) → đổi ít = Lamb
            rot_cam = _make_rotated_cam(cam, ROT_DELTA_DEG)
            img_o = pkg_o["render"].clamp(0, 1)
            img_rot = render(rot_cam, gaussians, pipe, bg)["render"].clamp(0, 1)
            view_dep = (img_o - img_rot).abs().mean(0)           # (H,W)
            lamb_thr = torch.quantile(view_dep.flatten(), LAMB_PCT / 100.0)
            lamb_mask = view_dep <= lamb_thr

            def _resid(Bfrac):
                B = Bfrac * extent
                sc = _make_shifted_cam(cam, B)
                img_s = render(sc, gaussians, pipe, bg)["render"].clamp(0, 1)
                disp = fx * (-B) / (depth_n + 1e-5)              # Binocular sign
                warped, vmask = _inverse_warp_cols(img_s, disp)
                l1 = (warped - gt).abs().mean(0)                 # (H,W)
                return l1, vmask

            # region = valid(B_target) ∧ ~Lambertian — CỐ ĐỊNH cho cả 2 hạng
            l1_eps, v_eps = _resid(B_EPS_FRAC)
            for bf in B_FRACS:
                l1_B, v_B = _resid(bf)
                region = v_B & lamb_mask
                if int(region.sum()) < 200:
                    continue
                res_B = float(l1_B[region].mean())
                res_0 = float(l1_eps[region].mean())
                # unmasked upper-bound (chỉ valid, KHÔNG Lamb) — strict UB
                ub_reg = v_B
                res_B_ub = float(l1_B[ub_reg].mean())
                res_0_ub = float(l1_eps[ub_reg].mean())
                rows.append(dict(
                    scene=scene_name, cam=cam.image_name, B_frac=bf,
                    res_B=res_B, res_0=res_0, sub=res_B - res_0,
                    sub_ub=res_B_ub - res_0_ub,
                    cov=float(region.float().mean())))
    del gaussians, scene
    torch.cuda.empty_cache()
    return rows


def _agg(rows, key):
    v = [r[key] for r in rows]
    return float(np.mean(v)) if v else float("nan")


def main():
    print("=== Phase 14 — L_consist saturation pre-check (no-train) ===")
    print(f"OUTPUT_ROOT={OUTPUT_ROOT} SEED={SEED} ITER={ITERATION}")
    print(f"SCENES={SCENES}  B_FRACS={B_FRACS} (×cameras_extent)  "
          f"B_eps={B_EPS_FRAC}  rot={ROT_DELTA_DEG}°  Lamb=pct{LAMB_PCT}")
    print(f"GO threshold: substrate ≥ max({MARGIN_FRAC}×floor, {MARGIN_ABS})\n")

    all_rows = []
    for sc in SCENES:
        print(f"──── {sc} ────")
        r = _scene_substrate(sc)
        if not r:
            print(f"  SKIP {sc}\n")
            continue
        all_rows += r
        for bf in B_FRACS:
            sub = [x for x in r if x["B_frac"] == bf]
            if not sub:
                continue
            print(f"  B={bf:.3f}·ext | res_B={_agg(sub,'res_B'):.4f} "
                  f"res_0(floor)={_agg(sub,'res_0'):.4f} "
                  f"substrate={_agg(sub,'sub'):+.4f} "
                  f"(UB unmasked={_agg(sub,'sub_ub'):+.4f}) "
                  f"cov={_agg(sub,'cov')*100:.0f}%")
        print()

    if not all_rows:
        print("NO data — checkpoints missing?"); sys.exit(1)

    print("=== VERDICT (per B, aggregate over scenes) ===")
    any_go = False
    for bf in B_FRACS:
        sub = [x for x in all_rows if x["B_frac"] == bf]
        if not sub:
            continue
        s = _agg(sub, "sub")
        s_ub = _agg(sub, "sub_ub")
        floor = _agg(sub, "res_0")
        gate = max(MARGIN_FRAC * floor, MARGIN_ABS)
        if s_ub <= MARGIN_ABS:
            tag = "❌ NO SUBSTRATE (upper-bound≈0 → reject CHẮC CHẮN)"
        elif s >= gate:
            tag = "🎯 SUBSTRATE (vượt floor đủ biên → đáng pilot)"
            any_go = True
        elif s > MARGIN_ABS:
            tag = "🟡 MARGINAL (không vượt biên → coi như no-go)"
        else:
            tag = "⚪ ~0 (Lamb-masked ≈ floor → no substrate)"
        print(f"  B={bf:.3f}·ext: substrate={s:+.4f} "
              f"(UB={s_ub:+.4f}, gate={gate:.4f}, floor={floor:.4f}) {tag}")

    print("\n=== Kết luận ===")
    if any_go:
        print("  🎯 CÓ ĐẤT ở ≥1 baseline → đi tiếp Bước 2 (Phase-0 verify) "
              "+ Bước 3 implement 2×2.")
    else:
        print("  ❌/⚪ KHÔNG đủ substrate vượt biên ở mọi B → A3 đã "
              "binocular-consistent → REJECT SỚM, không tốn 24 pilot-run.")

    print("\n=== Caveats (đọc kèm, KHÔNG over-claim) ===")
    print("  - Pre-check = go/no-go THÔ, KHÔNG dự đoán Δ_PSNR chính xác.")
    print("  - UNMASKED substrate = UPPER-BOUND (Lamb-mask chỉ giảm) →")
    print("    UB≈0 ⇒ reject chắc; UB lớn + masked nhỏ ⇒ specular-inflated.")
    print("  - Known +bias còn lại = bilinear-interp (mọc theo B) → đã")
    print("    chặn bằng MARGIN; marginal = no-go (không phải gain).")
    print("  - ~Lambertian = tiny-rotate proxy (zero-parallax), KHÔNG đụng")
    print("    SH coeff. floor=res(B→0)=train-fit (warp→identity).")
    print("  - Single-seed checkpoint; substrate là proxy directional —")
    print("    GO = đáng pilot, KHÔNG = đảm bảo gain (verdict ở Bước 5).")


if __name__ == "__main__":
    main()
