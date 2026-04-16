# ============================================================
# [CRSGaussian DIAG E2] Geometry Verification
# File: scripts/diagnose_geometry.py  (TẠO MỚI)
# Mục đích:
#   Verify Gaussians có đứng đúng vị trí 3D không + phân tích
#   gap 18dB có do geometry sai hay không.
#
#   E2a — Visual depth inspection (side-by-side PNG)
#   E2b — Quantitative depth error (train vs test depth consistency)
#   E2c — Point cloud colored by CRS (floater vs surface)
#
# Usage:
#   python scripts/diagnose_geometry.py \
#       --model_path output/diag_e1_sh/E1_baseline_fern \
#       --source_path data/nerf_llff_data/fern \
#       --eval -r 8 --n_views 3 \
#       --use_depth_prior --dav2_path ../Depth-Anything-V2 \
#       --iteration 10000
# ============================================================

import os
import sys
import torch
import numpy as np
from argparse import ArgumentParser
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene, GaussianModel
from utils.depth import precompute_depth_priors, align_depth_to_colmap
from utils.depth.depth_warping import find_nearest_training_cam, forward_warp_depth
from gaussian_renderer import render


# ════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════
def to_numpy(t):
    return t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)


def depth_to_vis(depth_np, vmin=None, vmax=None):
    """Normalize depth để visualize. Zero pixels → NaN (hiện trắng)."""
    d = depth_np.copy().astype(np.float32)
    mask = d > 1e-6
    if vmin is None and mask.any():
        vmin = float(np.percentile(d[mask], 5))
    if vmax is None and mask.any():
        vmax = float(np.percentile(d[mask], 95))
    d = np.clip(d, vmin, vmax)
    d = (d - vmin) / max(vmax - vmin, 1e-6)
    d[~mask] = np.nan
    return d, vmin, vmax


def save_train_panel(path, gt_img, rend_rgb, rend_depth, aligned_depth, diff):
    """E2a — Side-by-side PNG cho training cam: 5 subplots."""
    fig, axes = plt.subplots(1, 5, figsize=(25, 4))
    axes[0].imshow(gt_img.transpose(1, 2, 0))
    axes[0].set_title("GT image")
    axes[1].imshow(rend_rgb.transpose(1, 2, 0).clip(0, 1))
    axes[1].set_title("Rendered RGB")
    rd_vis, vmin, vmax = depth_to_vis(rend_depth)
    axes[2].imshow(rd_vis, cmap="turbo")
    axes[2].set_title(f"Rendered depth\n[{vmin:.2f}, {vmax:.2f}]")
    ad_vis, _, _ = depth_to_vis(aligned_depth, vmin=vmin, vmax=vmax)
    axes[3].imshow(ad_vis, cmap="turbo")
    axes[3].set_title("Aligned DAV2 depth")
    axes[4].imshow(diff, cmap="coolwarm", vmin=-0.3, vmax=0.3)
    axes[4].set_title("Diff (rend - aligned)/range")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_test_panel(path, gt_img, rend_rgb, rend_depth, warp_depth):
    """E2a — Test cam: GT | rendered RGB | rendered depth | warped depth."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    axes[0].imshow(gt_img.transpose(1, 2, 0))
    axes[0].set_title("GT image")
    axes[1].imshow(rend_rgb.transpose(1, 2, 0).clip(0, 1))
    axes[1].set_title("Rendered RGB")
    rd_vis, vmin, vmax = depth_to_vis(rend_depth)
    axes[2].imshow(rd_vis, cmap="turbo")
    axes[2].set_title(f"Rendered depth\n[{vmin:.2f}, {vmax:.2f}]")
    wd_vis, _, _ = depth_to_vis(warp_depth, vmin=vmin, vmax=vmax)
    axes[3].imshow(wd_vis, cmap="turbo")
    axes[3].set_title("Warped depth (from nearest train)")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def write_ply_colored(xyz, rgb, path):
    """Write binary-less ASCII PLY. rgb in [0,1]."""
    n = xyz.shape[0]
    rgb_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            f.write(f"{xyz[i,0]:.6f} {xyz[i,1]:.6f} {xyz[i,2]:.6f} "
                    f"{rgb_u8[i,0]} {rgb_u8[i,1]} {rgb_u8[i,2]}\n")


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = ArgumentParser(description="[DIAG E2] Geometry verification")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--iteration", type=int, default=10000)
    # Top-level args cần cho Scene/GaussianModel constructor
    parser.add_argument("--train_bg", action="store_true")
    parser.add_argument("--gaussiansN", type=int, default=1)
    parser.add_argument("--coreg", action="store_true")
    parser.add_argument("--coprune", action="store_true")
    parser.add_argument("--coprune_threshold", type=int, default=5)
    parser.add_argument("--reg_sample_rate", type=float, default=0.5)
    parser.add_argument("--mask_training", action="store_true")
    parser.add_argument("--save_log_images", action="store_true")
    parser.add_argument("--onlyrgb", action="store_true")
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--max_train_panels", type=int, default=3,
                        help="Số training panels tối đa (tiết kiệm disk)")
    parser.add_argument("--max_test_panels", type=int, default=5,
                        help="Số test panels tối đa")
    args = parser.parse_args(sys.argv[1:])

    out_dir = os.path.join(args.model_path, "geometry_diag")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print(" [CRSGaussian DIAG E2] Geometry Verification")
    print("=" * 70)
    print(f"  Model path : {args.model_path}")
    print(f"  Source path: {args.source_path}")
    print(f"  Iteration  : {args.iteration}")
    print(f"  Output dir : {out_dir}")

    # ── Load scene ──
    print("\nLoading scene + trained model ...")
    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False)
    train_cams = scene.getTrainCameras()
    test_cams = scene.getTestCameras()
    print(f"  N gaussians  : {gaussians.get_xyz.shape[0]}")
    print(f"  N train cams : {len(train_cams)}")
    print(f"  N test cams  : {len(test_cams)}")

    # ── Recompute aligned depth (train cams only) ──
    print("\nRecomputing aligned depth dict ...")
    depth_prior_dict = precompute_depth_priors(
        train_cams, args.dav2_path, encoder=args.dav2_encoder)
    aligned_depth_dict, depth_range = align_depth_to_colmap(
        depth_prior_dict, train_cams, args.source_path, args.n_views)
    del depth_prior_dict
    print(f"  depth_range = {depth_range:.3f}")

    # ── Pipeline params + bg color cho render ──
    pipe = pp.extract(args)
    bg_color = torch.tensor([0.0, 0.0, 0.0], device="cuda", dtype=torch.float32)

    # ════════════════════════════════════════════════════════
    # E2a + E2b — Training views: rendered depth vs aligned depth
    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(" E2a/E2b — TRAINING VIEWS")
    print("=" * 70)

    train_l1_list, train_pearson_list, train_rgb_l1_list = [], [], []
    train_psnr_list = []

    with torch.no_grad():
        for idx, cam in enumerate(train_cams):
            if cam.uid not in aligned_depth_dict:
                continue
            res = render(cam, gaussians, pipe, bg_color)
            rend_rgb = res["render"]                         # (3,H,W)
            rend_depth = res["depth"].squeeze(0)             # (H,W)
            aligned = aligned_depth_dict[cam.uid]            # (H,W)

            # Metrics
            valid = (aligned > 1e-6) & (rend_depth > 1e-6)
            if valid.any():
                diff = (rend_depth - aligned) / depth_range
                l1 = float(diff[valid].abs().mean().item())
                a = rend_depth[valid].cpu().numpy().astype(np.float64)
                b = aligned[valid].cpu().numpy().astype(np.float64)
                if a.std() > 1e-9 and b.std() > 1e-9:
                    pearson = float(np.corrcoef(a, b)[0, 1])
                else:
                    pearson = float("nan")
                train_l1_list.append(l1)
                train_pearson_list.append(pearson)

            # RGB metrics
            gt = cam.original_image.to("cuda").clamp(0, 1)
            rgb_l1 = float((rend_rgb - gt).abs().mean().item())
            mse = float(((rend_rgb - gt) ** 2).mean().item())
            psnr = -10.0 * np.log10(max(mse, 1e-12))
            train_rgb_l1_list.append(rgb_l1)
            train_psnr_list.append(psnr)

            # Save panel for first K cams
            if idx < args.max_train_panels:
                diff_np = to_numpy((rend_depth - aligned) / depth_range)
                save_train_panel(
                    os.path.join(out_dir, f"train_cam{idx:02d}.png"),
                    to_numpy(gt), to_numpy(rend_rgb),
                    to_numpy(rend_depth), to_numpy(aligned), diff_np,
                )

    if train_l1_list:
        t_l1_m, t_l1_s = float(np.mean(train_l1_list)), float(np.std(train_l1_list))
        t_p_m, t_p_s = float(np.nanmean(train_pearson_list)), float(np.nanstd(train_pearson_list))
    else:
        t_l1_m = t_l1_s = t_p_m = t_p_s = float("nan")
    t_rgb_m = float(np.mean(train_rgb_l1_list)) if train_rgb_l1_list else float("nan")
    t_psnr_m = float(np.mean(train_psnr_list)) if train_psnr_list else float("nan")

    print(f"\n  Train depth L1 (norm by range): {t_l1_m:.4f} ± {t_l1_s:.4f}")
    print(f"  Train depth Pearson           : {t_p_m:.4f} ± {t_p_s:.4f}")
    print(f"  Train RGB L1 vs GT            : {t_rgb_m:.4f}")
    print(f"  Train RGB PSNR vs GT          : {t_psnr_m:.2f} dB")

    # ════════════════════════════════════════════════════════
    # E2a + E2b — Test views: rendered depth vs warped depth
    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(" E2a/E2b — TEST VIEWS (depth consistency via warp)")
    print("=" * 70)

    test_l1_list, test_rgb_l1_list, test_psnr_list = [], [], []

    with torch.no_grad():
        for idx, cam in enumerate(test_cams):
            res = render(cam, gaussians, pipe, bg_color)
            rend_rgb = res["render"]
            rend_depth = res["depth"].squeeze(0)

            # RGB
            gt = cam.original_image.to("cuda").clamp(0, 1)
            rgb_l1 = float((rend_rgb - gt).abs().mean().item())
            mse = float(((rend_rgb - gt) ** 2).mean().item())
            psnr = -10.0 * np.log10(max(mse, 1e-12))
            test_rgb_l1_list.append(rgb_l1)
            test_psnr_list.append(psnr)

            # Depth consistency: warp aligned depth từ nearest train cam
            nearest = find_nearest_training_cam(cam, train_cams)
            warp_depth = None
            if nearest is not None and nearest.uid in aligned_depth_dict:
                warp_depth, valid_mask = forward_warp_depth(
                    aligned_depth_dict[nearest.uid], nearest, cam)
                valid = valid_mask & (rend_depth > 1e-6)
                if valid.any():
                    diff = (rend_depth - warp_depth) / depth_range
                    l1 = float(diff[valid].abs().mean().item())
                    test_l1_list.append(l1)

            if idx < args.max_test_panels:
                save_test_panel(
                    os.path.join(out_dir, f"test_cam{idx:02d}.png"),
                    to_numpy(gt), to_numpy(rend_rgb),
                    to_numpy(rend_depth),
                    to_numpy(warp_depth) if warp_depth is not None else np.zeros_like(to_numpy(rend_depth)),
                )

    te_l1_m = float(np.mean(test_l1_list)) if test_l1_list else float("nan")
    te_rgb_m = float(np.mean(test_rgb_l1_list)) if test_rgb_l1_list else float("nan")
    te_psnr_m = float(np.mean(test_psnr_list)) if test_psnr_list else float("nan")

    print(f"\n  Test depth consistency L1 (vs warped): {te_l1_m:.4f}")
    print(f"  Test RGB L1 vs GT                    : {te_rgb_m:.4f}")
    print(f"  Test RGB PSNR vs GT                  : {te_psnr_m:.2f} dB")

    # ════════════════════════════════════════════════════════
    # E2c — Export point cloud colored by CRS
    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(" E2c — POINT CLOUD COLORED BY CRS")
    print("=" * 70)

    xyz_np = to_numpy(gaussians.get_xyz)              # (N,3)
    if hasattr(gaussians, "_crs_score") and gaussians._crs_score.numel() > 0:
        crs_np = to_numpy(gaussians.get_crs).squeeze()  # (N,)
    else:
        # .ply loader không có stored CRS → recompute quick logit→sigmoid=0.5
        crs_np = np.full(xyz_np.shape[0], 0.5, dtype=np.float32)
        print("  [INFO] _crs_score missing (loaded from .ply) → using 0.5 fallback")

    # Color: red = low CRS (floater candidate), green = high CRS (surface)
    colors = np.zeros((xyz_np.shape[0], 3), dtype=np.float32)
    colors[:, 0] = 1.0 - crs_np
    colors[:, 1] = crs_np

    ply_path = os.path.join(out_dir, "gaussians_crs.ply")
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_np.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        o3d.io.write_point_cloud(ply_path, pcd)
        print(f"  [o3d] Saved: {ply_path}")
    except Exception as e:
        print(f"  [WARN] open3d failed ({e}); writing ASCII PLY manually.")
        write_ply_colored(xyz_np, colors, ply_path)
        print(f"  [ascii] Saved: {ply_path}")

    n = xyz_np.shape[0]
    n_low = int((crs_np < 0.35).sum())
    n_high = int((crs_np > 0.65).sum())
    print(f"  Total Gaussians     : {n}")
    print(f"  Floater (CRS<0.35)  : {n_low} ({100*n_low/n:.1f}%)")
    print(f"  Surface (CRS>0.65)  : {n_high} ({100*n_high/n:.1f}%)")

    # ════════════════════════════════════════════════════════
    # Summary + interpretation
    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(" === GEOMETRY DIAGNOSIS ===")
    print("=" * 70)
    print(f"Training views:")
    print(f"  Depth L1 (normalized): {t_l1_m:.4f} ± {t_l1_s:.4f}")
    print(f"  Depth Pearson        : {t_p_m:.4f} ± {t_p_s:.4f}")
    print(f"  RGB L1 vs GT         : {t_rgb_m:.4f}")
    print(f"  RGB PSNR             : {t_psnr_m:.2f} dB")
    print(f"Test views:")
    print(f"  Depth L1 vs warped   : {te_l1_m:.4f}")
    print(f"  RGB L1 vs GT         : {te_rgb_m:.4f}")
    print(f"  RGB PSNR             : {te_psnr_m:.2f} dB")
    print(f"Train-Test RGB gap     : {t_psnr_m - te_psnr_m:.2f} dB")

    print("\nInterpretation:")
    if not np.isnan(t_p_m) and t_p_m > 0.9:
        print(f"  ✓ Depth fits prior well (pearson={t_p_m:.3f})")
    else:
        print(f"  ⚠ Depth does not fit prior well (pearson={t_p_m:.3f})")

    if not np.isnan(t_l1_m) and t_l1_m < 0.05:
        print(f"  ✓ Train geometry consistent with DAV2 prior (L1={t_l1_m:.3f})")
    else:
        print(f"  ⚠ Large train depth error (L1={t_l1_m:.3f}) — geometry may be wrong")

    if not np.isnan(te_l1_m):
        if te_l1_m < 2 * t_l1_m:
            print(f"  ✓ Test depth consistent with train (L1_test={te_l1_m:.3f} ≈ L1_train)")
            print(f"    → Geometry OK across views. Gap likely from APPEARANCE (SH).")
        else:
            print(f"  ⚠ Test depth inconsistent with train (L1_test={te_l1_m:.3f} >> L1_train={t_l1_m:.3f})")
            print(f"    → Geometry OVERFIT to train views. Gap from GEOMETRY, not just SH.")

    print("\n" + "=" * 70)
    print(f" Visualizations saved in: {out_dir}")
    print("=" * 70)
