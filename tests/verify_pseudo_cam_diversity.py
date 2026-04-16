# ============================================================
# [CRSGaussian] Verify Pseudo-Cam Angular Diversity
# File: tests/verify_pseudo_cam_diversity.py  (TẠO MỚI)
# Mục đích: Đo angular distance distribution của 10000 pseudo cams
#           với training cams gần nhất. Kiểm tra hypothesis H4:
#           pseudo cams có thực sự "novel" hay chỉ perturbation
#           nhỏ của training cams?
#
# Output:
#   - min/max/mean/median/p10/p25/p75/p90/p99 angular distance
#   - histogram bins (0-1°, 1-2°, 2-5°, 5-10°, 10-20°, >20°)
#   - % cams pass thresholds {3°, 5°, 8°, 10°, 15°}
#
# Quyết định fix:
#   max < 5°    → F2b (perturb forward direction trong generation)
#   max 5-10°   → F1 (filter trong training loop, threshold 5°)
#   max > 10°   → F1 (filter, threshold 5° hoặc 8°)
#
# Chạy: python tests/verify_pseudo_cam_diversity.py \
#         --source_path data/nerf_llff_data/fern \
#         --eval -r 8 --n_views 3
# ============================================================

import os
import sys
import numpy as np
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene, GaussianModel


def angular_distance_deg(R1, R2):
    """Angular distance giữa 2 forward directions (cột 3 của C2W rotation).

    cam.R = C2W rotation → cam.R[:, 2] = forward direction trong world.
    """
    f1 = np.asarray(R1)[:, 2]
    f2 = np.asarray(R2)[:, 2]
    f1 = f1 / (np.linalg.norm(f1) + 1e-12)
    f2 = f2 / (np.linalg.norm(f2) + 1e-12)
    cos = np.clip(np.dot(f1, f2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def nearest_train_angle(pseudo_cam, train_cams):
    """Tìm góc nhỏ nhất giữa pseudo cam và bất kỳ training cam nào."""
    return min(angular_distance_deg(pseudo_cam.R, tc.R) for tc in train_cams)


# ════════════════════════════════════════
# Main
# ════════════════════════════════════════
if __name__ == "__main__":
    parser = ArgumentParser(description="Measure pseudo-cam angular diversity")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    op = OptimizationParams(parser)
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

    print("Loading scene ...")
    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians, shuffle=False)

    train_cams = scene.getTrainCameras()
    pseudo_cams = scene.getPseudoCameras()

    print(f"\n  Training cameras: {len(train_cams)}")
    print(f"  Pseudo cameras  : {len(pseudo_cams)}")

    # ── Training cam pairwise angular distances (reference) ──
    print("\n" + "=" * 60)
    print(" Training cam pairwise angular distances (reference)")
    print("=" * 60)
    n_train = len(train_cams)
    pair_angles = []
    for i in range(n_train):
        for j in range(i + 1, n_train):
            ang = angular_distance_deg(train_cams[i].R, train_cams[j].R)
            pair_angles.append(ang)
            print(f"  cam{i}-cam{j}: {ang:.2f}°")

    if pair_angles:
        print(f"  >> Train pair: min={min(pair_angles):.2f}° "
              f"max={max(pair_angles):.2f}° "
              f"mean={np.mean(pair_angles):.2f}°")

    # ── Compute angle pseudo→nearest training cam cho TẤT CẢ pseudo cams ──
    print("\n" + "=" * 60)
    print(" Pseudo cam → nearest training cam (10000 pseudo)")
    print("=" * 60)
    print("  Computing angles for all pseudo cams ...")

    angles = []
    for i, pc in enumerate(pseudo_cams):
        ang = nearest_train_angle(pc, train_cams)
        angles.append(ang)
        if (i + 1) % 2000 == 0:
            print(f"  ... {i + 1}/{len(pseudo_cams)} done")

    angles = np.asarray(angles, dtype=np.float64)

    # ── Statistics ──
    print("\n  Distribution statistics:")
    print(f"    n         = {len(angles)}")
    print(f"    min       = {angles.min():.3f}°")
    print(f"    max       = {angles.max():.3f}°")
    print(f"    mean      = {angles.mean():.3f}°")
    print(f"    median    = {np.median(angles):.3f}°")
    print(f"    std       = {angles.std():.3f}°")
    print(f"    p10       = {np.percentile(angles, 10):.3f}°")
    print(f"    p25       = {np.percentile(angles, 25):.3f}°")
    print(f"    p75       = {np.percentile(angles, 75):.3f}°")
    print(f"    p90       = {np.percentile(angles, 90):.3f}°")
    print(f"    p99       = {np.percentile(angles, 99):.3f}°")

    # ── Histogram bins ──
    print("\n  Histogram bins:")
    bins = [(0, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, 1e9)]
    for lo, hi in bins:
        n = int(((angles >= lo) & (angles < hi)).sum())
        pct = 100 * n / len(angles)
        bar = '█' * int(pct / 2)
        hi_str = f"{hi}°" if hi < 1e8 else "∞"
        print(f"    [{lo:>4}°, {hi_str:>4}): {n:>5} ({pct:>5.1f}%) {bar}")

    # ── Threshold pass rates ──
    print("\n  Threshold pass rates (% pseudo cams với angle > threshold):")
    for thr in [3, 5, 8, 10, 15]:
        n_pass = int((angles > thr).sum())
        pct = 100 * n_pass / len(angles)
        verdict = "✓" if pct > 30 else ("⚠" if pct > 5 else "✗")
        print(f"    > {thr:>2}° : {n_pass:>5} / {len(angles)} ({pct:>5.1f}%) {verdict}")

    # ── Quyết định fix ──
    print("\n" + "=" * 60)
    print(" Fix recommendation")
    print("=" * 60)
    max_ang = float(angles.max())
    n_above_5 = int((angles > 5).sum())
    pct_above_5 = 100 * n_above_5 / len(angles)

    if max_ang < 5:
        print(f"  ✗ MAX = {max_ang:.2f}° < 5°")
        print(f"     → F1 (filter) KHÔNG khả thi — không có pseudo cam đủ xa")
        print(f"     → Cần F2: sửa generate_random_poses_llff() để tạo true novel views")
        print(f"     → Đề xuất F2b: perturb forward direction sau khi sample position")
    elif pct_above_5 < 5:
        print(f"  ⚠ MAX = {max_ang:.2f}°, chỉ {pct_above_5:.1f}% pass threshold 5°")
        print(f"     → F1 khả thi nhưng tốn nhiều retry")
        print(f"     → Cân nhắc F2 để có distribution rộng hơn")
    elif pct_above_5 < 30:
        print(f"  ⚠ MAX = {max_ang:.2f}°, {pct_above_5:.1f}% pass threshold 5°")
        print(f"     → F1 OK với threshold 5°")
        print(f"     → ~{int(100/pct_above_5)} retries trung bình để có 1 cam pass")
    else:
        print(f"  ✓ MAX = {max_ang:.2f}°, {pct_above_5:.1f}% pass threshold 5°")
        print(f"     → F1 dễ — chỉ cần ~{int(100/pct_above_5)} retries trung bình")
        print(f"     → Threshold 5° an toàn, có thể thử 8° để stricter")
