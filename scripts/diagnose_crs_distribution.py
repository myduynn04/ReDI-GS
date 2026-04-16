# ============================================================
# [CRSGaussian DIAG E2B] CRS Distribution Analysis on Trained Model
# File: scripts/diagnose_crs_distribution.py  (TẠO MỚI)
# Mục đích:
#   Load 1 model đã trained → compute D_i, R_i, CRS cho tất cả Gaussians
#   → in distribution stats + bins → biết geometry "đúng" theo CRS không.
#
# Diễn giải:
#   - CRS mean > 0.7, std > 0.10  → geometry stable, R_i đồng thuận
#   - CRS mean < 0.6              → nhiều Gaussians có disagreement
#   - % CRS < 0.35 cao            → nhiều "floater candidates" còn sót
#   - D_i mean cao, R_i mean thấp → depth OK nhưng color không nhất quán
#                                     → SH overfit
#   - D_i mean thấp, R_i mean cao → geometry sai, color không phải vấn đề
#
# Usage:
#   python scripts/diagnose_crs_distribution.py \
#       --source_path data/nerf_llff_data/fern \
#       -m output/<trained_model> \
#       --eval -r 8 --n_views 3 \
#       --use_depth_prior --dav2_path ../Depth-Anything-V2 \
#       --iteration 10000
# ============================================================

import os
import sys
import torch
import numpy as np
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene, GaussianModel
from utils.depth import precompute_depth_priors, align_depth_to_colmap
from utils.crs.crs_module import compute_depth_consistency, compute_reprojection_consistency


def histogram_text(values, bins=10, width=40):
    """Vẽ histogram text-based đơn giản."""
    if len(values) == 0:
        return "(empty)"
    hist, edges = np.histogram(values, bins=bins)
    max_count = hist.max() if hist.max() > 0 else 1
    lines = []
    for i, c in enumerate(hist):
        bar_len = int(width * c / max_count)
        bar = "█" * bar_len
        lines.append(f"  [{edges[i]:6.3f}, {edges[i+1]:6.3f}): {c:>6} {bar}")
    return "\n".join(lines)


def stats_block(name, values):
    if len(values) == 0:
        print(f"  {name}: empty")
        return
    arr = np.asarray(values)
    print(f"  {name}:")
    print(f"    n      = {len(arr)}")
    print(f"    mean   = {arr.mean():.4f}")
    print(f"    std    = {arr.std():.4f}")
    print(f"    min    = {arr.min():.4f}")
    print(f"    max    = {arr.max():.4f}")
    print(f"    median = {np.median(arr):.4f}")
    print(f"    p10    = {np.percentile(arr, 10):.4f}")
    print(f"    p25    = {np.percentile(arr, 25):.4f}")
    print(f"    p75    = {np.percentile(arr, 75):.4f}")
    print(f"    p90    = {np.percentile(arr, 90):.4f}")
    print(f"    < 0.35 = {(arr < 0.35).sum()} ({100*(arr<0.35).mean():.2f}%)")
    print(f"    > 0.65 = {(arr > 0.65).sum()} ({100*(arr>0.65).mean():.2f}%)")
    print(f"  histogram (10 bins):")
    print(histogram_text(arr, bins=10))


# ════════════════════════════════════════
# Main
# ════════════════════════════════════════
if __name__ == "__main__":
    parser = ArgumentParser(description="Diagnose CRS distribution on trained model")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--iteration", type=int, default=10000,
                        help="Iteration to load (must exist in point_cloud/iteration_*)")
    # Top-level args (cần để Scene/GaussianModel không crash)
    parser.add_argument("--train_bg", action="store_true")
    parser.add_argument("--gaussiansN", type=int, default=1)
    parser.add_argument("--coreg", action="store_true")
    parser.add_argument("--coprune", action="store_true")
    parser.add_argument("--coprune_threshold", type=int, default=5)
    parser.add_argument("--reg_sample_rate", type=float, default=0.5)
    parser.add_argument("--mask_training", action="store_true")
    parser.add_argument("--save_log_images", action="store_true")
    parser.add_argument("--onlyrgb", action="store_true")
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])

    print("=" * 70)
    print(" [CRSGaussian DIAG E2B] CRS Distribution Analysis")
    print("=" * 70)
    print(f"  Model path : {args.model_path}")
    print(f"  Source path: {args.source_path}")
    print(f"  Iteration  : {args.iteration}")

    # ── Load scene + trained model ──
    print("\nLoading scene + trained model ...")
    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False)
    train_cams = scene.getTrainCameras()
    test_cams = scene.getTestCameras()

    print(f"  N gaussians    : {gaussians.get_xyz.shape[0]}")
    print(f"  N train cams   : {len(train_cams)}")
    print(f"  N test cams    : {len(test_cams)}")

    # ── Recompute aligned depth (cần làm lại vì không lưu khi train) ──
    print("\nRecomputing aligned depth dict ...")
    depth_prior_dict = precompute_depth_priors(
        train_cams, args.dav2_path, encoder=args.dav2_encoder)
    aligned_depth_dict, depth_range = align_depth_to_colmap(
        depth_prior_dict, train_cams, args.source_path, args.n_views)
    del depth_prior_dict
    print(f"  depth_range = {depth_range:.3f}")

    # ── Compute D_i và R_i (no_grad) ──
    # Note: signature nhận xyz tensor, không phải gaussians object
    xyz = gaussians.get_xyz.detach()

    print("\nComputing D_i (depth consistency) ...")
    with torch.no_grad():
        D = compute_depth_consistency(
            xyz, train_cams, aligned_depth_dict, depth_range)

    print("Computing R_i (reprojection / color consistency) ...")
    with torch.no_grad():
        R = compute_reprojection_consistency(xyz, train_cams)

    D_np = D.detach().cpu().numpy().squeeze()
    R_np = R.detach().cpu().numpy().squeeze()

    # ── Compute CRS như công thức trong update_crs ──
    # crs_logit = 5.0 * (0.5 * D + 0.5 * R - 0.5)
    # CRS = sigmoid(crs_logit)
    crs_logit = 5.0 * (0.5 * D_np + 0.5 * R_np - 0.5)
    CRS_np = 1.0 / (1.0 + np.exp(-crs_logit))

    # ── Stored CRS từ training (chỉ có khi load từ chkpnt .pth, không phải .ply) ──
    # `.ply` chỉ lưu xyz/scaling/rotation/opacity/SH, không lưu _crs_score.
    # Nếu attribute không tồn tại → skip stored, chỉ dùng recomputed.
    stored_crs = None
    if hasattr(gaussians, '_crs_score') and gaussians._crs_score is not None:
        try:
            stored_crs = gaussians.get_crs.detach().cpu().numpy().squeeze()
        except Exception as e:
            print(f"  [WARN] Could not extract stored CRS: {e}")
            stored_crs = None
    else:
        print("  [INFO] _crs_score not present (loaded from .ply, not chkpnt). Skipping stored CRS.")

    print("\n" + "=" * 70)
    print(" RESULTS")
    print("=" * 70)

    print("\n--- D_i (depth consistency, [0,1]) ---")
    stats_block("D_i", D_np)

    print("\n--- R_i (reprojection / color consistency, [0,1]) ---")
    stats_block("R_i", R_np)

    print("\n--- CRS (recomputed at this snapshot) ---")
    stats_block("CRS_recomputed", CRS_np)

    if stored_crs is not None:
        print("\n--- CRS (stored — final EMA state at end of training) ---")
        stats_block("CRS_stored", stored_crs)
    else:
        print("\n--- CRS (stored) ---")
        print("  [SKIP] No stored CRS available (model loaded from .ply)")

    # ── Cross-stat: D_i vs R_i ──
    print("\n--- D_i vs R_i breakdown ---")
    high_D = D_np > 0.7
    low_D = D_np < 0.3
    high_R = R_np > 0.7
    low_R = R_np < 0.3
    print(f"  High D AND High R (geometry+color OK)   : {(high_D & high_R).sum()} "
          f"({100*(high_D & high_R).mean():.1f}%)")
    print(f"  High D AND Low R  (geometry OK, SH bad) : {(high_D & low_R).sum()} "
          f"({100*(high_D & low_R).mean():.1f}%)")
    print(f"  Low D  AND High R (geometry bad, SH OK) : {(low_D & high_R).sum()} "
          f"({100*(low_D & high_R).mean():.1f}%)")
    print(f"  Low D  AND Low R  (both bad)            : {(low_D & low_R).sum()} "
          f"({100*(low_D & low_R).mean():.1f}%)")

    # Pearson correlation D_i vs R_i
    if len(D_np) > 1 and D_np.std() > 1e-9 and R_np.std() > 1e-9:
        corr = float(np.corrcoef(D_np, R_np)[0, 1])
        print(f"\n  Pearson(D_i, R_i) = {corr:.4f}")
        if corr > 0.5:
            print("    → Strong positive: D và R đồng thuận. 1 metric đủ.")
        elif corr < 0.1:
            print("    → Weak/no correlation: D và R orthogonal — capture aspects khác nhau.")
        else:
            print("    → Mild positive: complementary signals.")

    # ── Diagnosis hints ──
    print("\n" + "=" * 70)
    print(" DIAGNOSIS HINTS")
    print("=" * 70)

    d_mean = D_np.mean()
    r_mean = R_np.mean()
    crs_mean = CRS_np.mean()

    print(f"  D_i mean = {d_mean:.3f}, R_i mean = {r_mean:.3f}, CRS mean = {crs_mean:.3f}")

    if d_mean > 0.75 and r_mean > 0.75:
        print("  → Both D and R high → geometry + color both fit prior well.")
        print("    Train-test gap likely from BOTH overdensification AND")
        print("    SH overfit memorizing training views (need E1 to confirm).")
    elif d_mean > 0.75 and r_mean < 0.6:
        print("  → D high, R low → geometry OK, COLOR INCONSISTENT cross-views.")
        print("    Strong evidence for SH OVERFIT. Run E1B (freeze SH).")
    elif d_mean < 0.6 and r_mean > 0.75:
        print("  → D low, R high → GEOMETRY WRONG, color consistent (Gaussians")
        print("    in wrong 3D positions but happen to render similar colors).")
        print("    Need geometry regularization, not SH freeze.")
    else:
        print("  → Mixed: both moderate. Need E1 to isolate.")

    if (CRS_np < 0.35).mean() > 0.10:
        print(f"  ⚠ {100*(CRS_np<0.35).mean():.1f}% Gaussians có CRS < 0.35")
        print("    → CRS pruning chưa loại được hết floater candidates")
    else:
        print(f"  ✓ Chỉ {100*(CRS_np<0.35).mean():.1f}% Gaussians CRS < 0.35 (OK)")

    print("\n" + "=" * 70)
