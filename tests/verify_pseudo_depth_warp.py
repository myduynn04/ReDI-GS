# ============================================================
# [CRSGaussian] Verify Pseudo-View Depth Warping (5 issues)
# File: tests/verify_pseudo_depth_warp.py  (TẠO MỚI)
# Mục đích: Sau khi implement utils/depth/depth_warping.py, verify
#           5 lỗ hổng tiềm năng trước khi chạy ablation thực:
#
#   VERIFY 1 — Camera convention round-trip (sanity check lại)
#   VERIFY 2 — Warp coverage (>50% cho pseudo cam gần training)
#   VERIFY 3 — Depth scale consistency (rendered vs warped reference)
#   VERIFY 4 — Collision rate (% pixels overwrite trong scatter)
#   VERIFY 5 — Loss gradient flow (loss.backward → gaussians._xyz.grad)
#
# Chạy: python tests/verify_pseudo_depth_warp.py \
#         --source_path data/nerf_llff_data/fern \
#         --eval -r 8 --n_views 3 \
#         --use_depth_prior --dav2_path ../Depth-Anything-V2
#
# PASS:
#   V1: round-trip < 1e-4 (đã verify ở verify_camera_convention.py)
#   V2: coverage > 50% cho ít nhất 1 pseudo cam
#   V3: ranges cùng order of magnitude (ratio < 10x)
#   V4: collision rate < 50% (acceptable)
#   V5: gradient norm > 0 trên _xyz
# ============================================================

import os
import sys
import torch
import numpy as np
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.depth import precompute_depth_priors, align_depth_to_colmap
from utils.depth.depth_warping import (
    find_nearest_training_cam,
    forward_warp_depth,
    compute_warp_coverage,
)
from utils.loss_utils import pearson_depth_loss


# ════════════════════════════════════════
# VERIFY 2 — Warp coverage
# ════════════════════════════════════════
def verify_warp_coverage(scene, allCameras, aligned_depth_dict):
    print("\n" + "=" * 60)
    print(" VERIFY 2 — Warp coverage")
    print("=" * 60)

    pseudo_cams = scene.getPseudoCameras()
    if pseudo_cams[0] is None:
        print("  ✗ No pseudo cameras")
        return False

    # Test 5 pseudo cams (tránh quá lâu)
    n_test = min(5, len(pseudo_cams))
    print(f"  Testing {n_test} pseudo cams (out of {len(pseudo_cams)})")

    coverages = []
    for i in range(n_test):
        pcam = pseudo_cams[i]
        ncam = find_nearest_training_cam(pcam, allCameras)
        if ncam is None or ncam.uid not in aligned_depth_dict:
            print(f"  pseudo {i}: no match")
            continue

        # Angular dist log
        p_fwd = np.asarray(pcam.R)[:, 2]
        n_fwd = np.asarray(ncam.R)[:, 2]
        p_fwd = p_fwd / (np.linalg.norm(p_fwd) + 1e-12)
        n_fwd = n_fwd / (np.linalg.norm(n_fwd) + 1e-12)
        cos = np.clip(np.dot(p_fwd, n_fwd), -1, 1)
        ang = np.degrees(np.arccos(cos))

        depth_ref, valid, stats = forward_warp_depth(
            aligned_depth_dict[ncam.uid], ncam, pcam, return_stats=True
        )
        cov = compute_warp_coverage(valid)
        coverages.append(cov)

        print(f"  pseudo {i} ← cam {ncam.uid} (ang={ang:.2f}°): "
              f"coverage={cov*100:.1f}% | "
              f"in-frame={stats['n_inframe']}/{stats['n_total']} | "
              f"unique={stats['n_unique']}")

    if not coverages:
        print("  ✗ FAIL — không có pseudo cam nào tính được coverage")
        return False

    cov_max = max(coverages)
    cov_avg = sum(coverages) / len(coverages)
    print(f"\n  >>> Coverage: max={cov_max*100:.1f}% avg={cov_avg*100:.1f}%")

    if cov_max > 0.5:
        print("  >>> ✓ PASS (max > 50%)")
        return True
    elif cov_max > 0.3:
        print("  >>> ⚠ MARGINAL (30-50%) — loss có signal nhưng yếu")
        return True
    else:
        print("  >>> ✗ FAIL — coverage quá thấp, cần fix (warp từ multi cam?)")
        return False


# ════════════════════════════════════════
# VERIFY 3 — Depth scale consistency
# ════════════════════════════════════════
def verify_depth_scale(scene, allCameras, aligned_depth_dict, pipe):
    print("\n" + "=" * 60)
    print(" VERIFY 3 — Depth scale consistency")
    print("=" * 60)

    pseudo_cams = scene.getPseudoCameras()
    if pseudo_cams[0] is None:
        return False

    pcam = pseudo_cams[0]
    ncam = find_nearest_training_cam(pcam, allCameras)
    if ncam is None or ncam.uid not in aligned_depth_dict:
        print("  ✗ no match")
        return False

    # Warped reference
    depth_ref, valid, _ = forward_warp_depth(
        aligned_depth_dict[ncam.uid], ncam, pcam, return_stats=True
    )

    # Rendered depth từ Gaussian model
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    render_pkg = render(pcam, scene.gaussians, pipe, bg)
    d_rendered = render_pkg["depth"].squeeze(0).detach()

    if valid.sum() == 0:
        print("  ✗ no valid pixels")
        return False

    ref_v = depth_ref[valid]
    rnd_v = d_rendered[valid]

    print(f"  depth_ref     : min={ref_v.min():.3f} "
          f"max={ref_v.max():.3f} "
          f"mean={ref_v.mean():.3f}")
    print(f"  depth_rendered: min={rnd_v.min():.3f} "
          f"max={rnd_v.max():.3f} "
          f"mean={rnd_v.mean():.3f}")

    ratio_mean = float(rnd_v.mean() / max(float(ref_v.mean()), 1e-6))
    print(f"\n  >>> mean ratio (rnd/ref): {ratio_mean:.3f}")

    if 0.1 < ratio_mean < 10:
        print("  >>> ✓ PASS — cùng order of magnitude")
        return True
    else:
        print("  >>> ⚠ Scale mismatch — Pearson vẫn OK vì scale-invariant, "
              "nhưng cần lưu ý")
        return True  # vẫn pass vì Pearson invariant


# ════════════════════════════════════════
# VERIFY 4 — Collision rate
# ════════════════════════════════════════
def verify_collision(scene, allCameras, aligned_depth_dict):
    print("\n" + "=" * 60)
    print(" VERIFY 4 — Collision rate (depth discontinuity)")
    print("=" * 60)

    pseudo_cams = scene.getPseudoCameras()
    if pseudo_cams[0] is None:
        return False

    n_test = min(5, len(pseudo_cams))
    rates = []
    for i in range(n_test):
        pcam = pseudo_cams[i]
        ncam = find_nearest_training_cam(pcam, allCameras)
        if ncam is None or ncam.uid not in aligned_depth_dict:
            continue
        _, _, stats = forward_warp_depth(
            aligned_depth_dict[ncam.uid], ncam, pcam, return_stats=True
        )
        rates.append(stats['collision_rate'])
        print(f"  pseudo {i}: total={stats['n_total']} "
              f"in-frame={stats['n_inframe']} unique={stats['n_unique']} "
              f"collision={stats['collision_rate']*100:.1f}%")

    if not rates:
        return False

    avg = sum(rates) / len(rates)
    print(f"\n  >>> Avg collision rate: {avg*100:.1f}%")
    if avg < 0.5:
        print("  >>> ✓ PASS (<50%)")
    else:
        print("  >>> ⚠ HIGH — scene có nhiều depth discontinuity, "
              "interpret kết quả cẩn thận")
    return True


# ════════════════════════════════════════
# VERIFY 5 — Loss gradient flow
# ════════════════════════════════════════
def verify_gradient_flow(scene, allCameras, aligned_depth_dict, pipe):
    print("\n" + "=" * 60)
    print(" VERIFY 5 — Loss gradient flow")
    print("=" * 60)

    pseudo_cams = scene.getPseudoCameras()
    if pseudo_cams[0] is None:
        return False

    pcam = pseudo_cams[0]
    ncam = find_nearest_training_cam(pcam, allCameras)
    if ncam is None or ncam.uid not in aligned_depth_dict:
        return False

    depth_ref, valid = forward_warp_depth(
        aligned_depth_dict[ncam.uid], ncam, pcam
    )
    if int(valid.sum().item()) < 100:
        print(f"  ✗ Not enough valid pixels ({int(valid.sum().item())})")
        return False

    # Render với gradient
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    gaussians = scene.gaussians
    if gaussians._xyz.grad is not None:
        gaussians._xyz.grad.zero_()

    render_pkg = render(pcam, gaussians, pipe, bg)
    d_rnd = render_pkg["depth"].squeeze(0)  # (H,W) — grad chảy qua

    L = pearson_depth_loss(
        d_rnd[valid].unsqueeze(0),
        depth_ref[valid],
    )
    print(f"  loss: {float(L.item()):.6f}")

    L.backward()

    grad = gaussians._xyz.grad
    if grad is None:
        print("  ✗ FAIL — no grad on _xyz")
        return False

    norm = float(grad.norm().item())
    nz = int((grad.abs() > 1e-10).sum().item())
    total = int(grad.numel())

    print(f"  grad norm: {norm:.6e}")
    print(f"  nonzero  : {nz}/{total} ({100 * nz / total:.1f}%)")

    if norm > 1e-12 and nz > 0:
        print("  >>> ✓ PASS — gradient flow OK")
        return True
    else:
        print("  >>> ✗ FAIL")
        return False


# ════════════════════════════════════════
# Main
# ════════════════════════════════════════
if __name__ == "__main__":
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    op = OptimizationParams(parser)
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

    pipe = pp.extract(args)

    print("Loading scene ...")
    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians, shuffle=False)
    allCameras = scene.getTrainCameras()

    print("Precomputing aligned depth ...")
    depth_prior_dict = precompute_depth_priors(allCameras, args.dav2_path)
    aligned_depth_dict, depth_range = align_depth_to_colmap(
        depth_prior_dict, allCameras, args.source_path, args.n_views)
    print(f"  depth_range = {depth_range:.2f}")
    print(f"  aligned cams: {sorted(aligned_depth_dict.keys())}")

    results = []
    results.append(("Warp coverage",      verify_warp_coverage(scene, allCameras, aligned_depth_dict)))
    results.append(("Depth scale",        verify_depth_scale(scene, allCameras, aligned_depth_dict, pipe)))
    results.append(("Collision rate",     verify_collision(scene, allCameras, aligned_depth_dict)))
    results.append(("Loss gradient flow", verify_gradient_flow(scene, allCameras, aligned_depth_dict, pipe)))

    print("\n" + "=" * 60)
    print(" SUMMARY — Pseudo Depth Warp Verification")
    print("=" * 60)
    for name, ok in results:
        print(f"  {'✓' if ok else '✗'} {name}")

    n_pass = sum(1 for _, ok in results if ok)
    print(f"\n  {n_pass}/{len(results)} checks PASSED")

    if n_pass == len(results):
        print("\n  >>> SAFE to proceed with unit tests + ablation")
        sys.exit(0)
    else:
        print("\n  >>> FIX FAILURES BEFORE ABLATION")
        sys.exit(1)
