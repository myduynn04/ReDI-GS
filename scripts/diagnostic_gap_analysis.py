#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Diagnostic] Gap analysis — 4 tests
# File: scripts/diagnostic_gap_analysis.py (TẠO MỚI)
# Standalone script — KHÔNG sửa production code.
#
# Load checkpoint → chạy 4 diagnostic tests → print summary.
# Mục đích: phân tách residual gap 9.64 dB thành nguồn cụ thể.
#
# Usage:
#   python scripts/diagnostic_gap_analysis.py \
#       --model_path output/ablation_track_b/B1b_fern \
#       --source_path data/nerf_llff_data/fern \
#       --n_views 3 --iteration 10000
# ============================================================

import torch
import sys
import os
import json
import numpy as np
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.sh_utils import eval_sh
from utils.image_utils import psnr as compute_psnr
from utils.graphics_utils import fov2focal


# ──────────────────────────────────────────────────────────────
# Test 4: DC vs Rest contribution at test views
# ──────────────────────────────────────────────────────────────
def test4_dc_vs_rest(gaussians, test_cams, pipeline, bg):
    """[CRSGaussian Diagnostic] Render test views: full SH vs DC-only."""
    results = {}

    # Full SH render
    full_psnrs = []
    for cam in test_cams:
        pkg = render(cam, gaussians, pipeline, bg, disable_dropout=True)
        gt = cam.original_image[:3].cuda()
        full_psnrs.append(compute_psnr(pkg["render"], gt).mean().item())
    full_avg = np.mean(full_psnrs)

    # DC-only render: zero out _features_rest temporarily
    saved_rest = gaussians._features_rest.data.clone()
    gaussians._features_rest.data.zero_()

    dc_psnrs = []
    for cam in test_cams:
        pkg = render(cam, gaussians, pipeline, bg, disable_dropout=True)
        gt = cam.original_image[:3].cuda()
        dc_psnrs.append(compute_psnr(pkg["render"], gt).mean().item())
    dc_avg = np.mean(dc_psnrs)

    # Restore
    gaussians._features_rest.data.copy_(saved_rest)

    delta = dc_avg - full_avg
    if delta > 0.1:
        verdict = "REST_HURTING"
    elif delta > -0.2:
        verdict = "REST_NEUTRAL"
    else:
        verdict = "REST_HELPING"

    results = {
        "full_psnr": round(full_avg, 3),
        "dc_only_psnr": round(dc_avg, 3),
        "delta": round(delta, 3),
        "verdict": verdict,
    }

    print(f"\n=== Test 4: DC vs Rest contribution ===")
    print(f"  Full SH test PSNR:  {full_avg:.3f}")
    print(f"  DC-only test PSNR:  {dc_avg:.3f}")
    print(f"  Δ (DC_only - Full): {delta:+.3f}")
    print(f"  Verdict: {verdict}")
    return results


# ──────────────────────────────────────────────────────────────
# Test 5: Angular distance train→test
# ──────────────────────────────────────────────────────────────
def test5_angular_distance(train_cams, test_cams):
    """[CRSGaussian Diagnostic] Angular novelty of test views."""
    # Scene center = mean of train camera positions
    train_pos = np.array([c.camera_center.cpu().numpy() for c in train_cams])
    test_pos = np.array([c.camera_center.cpu().numpy() for c in test_cams])
    scene_center = train_pos.mean(axis=0)

    def angle_deg(pos_a, pos_b, center):
        va = pos_a - center
        vb = pos_b - center
        cos_a = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8)
        return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

    # Train-train pairwise
    train_pairs = []
    for i in range(len(train_pos)):
        for j in range(i + 1, len(train_pos)):
            train_pairs.append(angle_deg(train_pos[i], train_pos[j], scene_center))

    # Test→nearest train
    test_min_angles = []
    for tp in test_pos:
        dists = [angle_deg(tp, trp, scene_center) for trp in train_pos]
        test_min_angles.append(min(dists))

    tt_mean = np.mean(train_pairs) if train_pairs else 0
    tt_min = np.min(train_pairs) if train_pairs else 0
    tt_max = np.max(train_pairs) if train_pairs else 0

    te_mean = np.mean(test_min_angles)
    te_min = np.min(test_min_angles)
    te_max = np.max(test_min_angles)
    te_med = np.median(test_min_angles)

    if te_mean < 10:
        verdict = "CLOSE"
    elif te_mean < 20:
        verdict = "MODERATE"
    else:
        verdict = "FAR"

    results = {
        "n_train": len(train_cams),
        "n_test": len(test_cams),
        "train_pair_mean": round(tt_mean, 2),
        "train_pair_min": round(tt_min, 2),
        "train_pair_max": round(tt_max, 2),
        "test_nearest_mean": round(te_mean, 2),
        "test_nearest_min": round(te_min, 2),
        "test_nearest_max": round(te_max, 2),
        "test_nearest_median": round(te_med, 2),
        "verdict": verdict,
    }

    print(f"\n=== Test 5: Angular distance train→test ===")
    print(f"  N_train: {len(train_cams)}, N_test: {len(test_cams)}")
    print(f"  Train-train pairwise: mean={tt_mean:.1f}°, min={tt_min:.1f}°, max={tt_max:.1f}°")
    print(f"  Test→nearest_train:   mean={te_mean:.1f}°, min={te_min:.1f}°, max={te_max:.1f}°, median={te_med:.1f}°")
    print(f"  Verdict: {verdict}")
    return results


# ──────────────────────────────────────────────────────────────
# Test 2: SH angular divergence
# ──────────────────────────────────────────────────────────────
def test2_sh_divergence(gaussians, train_cams, test_cams):
    """[CRSGaussian Diagnostic] SH output variance at train/test/random angles."""
    xyz = gaussians.get_xyz  # (N, 3)
    N = xyz.shape[0]
    # get_features: (N, K, 3) → need (N, 3, K) for eval_sh
    shs = gaussians.get_features.transpose(1, 2)  # (N, 3, K)
    deg = gaussians.active_sh_degree

    def sh_colors_at_cams(cams, max_cams=5):
        """Eval SH color for all Gaussians from each camera direction."""
        colors = []
        for cam in cams[:max_cams]:
            dirs = xyz - cam.camera_center.unsqueeze(0)  # (N, 3)
            dirs = dirs / (dirs.norm(dim=1, keepdim=True) + 1e-8)
            c = eval_sh(deg, shs, dirs)  # (N, 3)
            c = torch.clamp(c + 0.5, 0.0, 1.0)
            colors.append(c)
        return torch.stack(colors, dim=0)  # (K_cams, N, 3)

    # Train directions
    train_stack = sh_colors_at_cams(train_cams, max_cams=len(train_cams))
    var_train = train_stack.var(dim=0).mean(dim=1)  # (N,)

    # Test directions
    test_stack = sh_colors_at_cams(test_cams, max_cams=5)
    var_test = test_stack.var(dim=0).mean(dim=1)  # (N,)

    # Random directions
    rand_colors = []
    for _ in range(5):
        rd = torch.randn(N, 3, device=xyz.device)
        rd = rd / (rd.norm(dim=1, keepdim=True) + 1e-8)
        c = eval_sh(deg, shs, rd)
        c = torch.clamp(c + 0.5, 0.0, 1.0)
        rand_colors.append(c)
    rand_stack = torch.stack(rand_colors, dim=0)
    var_rand = rand_stack.var(dim=0).mean(dim=1)  # (N,)

    vt_mean = var_train.mean().item()
    vt_p95 = var_train.quantile(0.95).item()
    ve_mean = var_test.mean().item()
    ve_p95 = var_test.quantile(0.95).item()
    vr_mean = var_rand.mean().item()
    vr_p95 = var_rand.quantile(0.95).item()

    ratio_te = ve_mean / (vt_mean + 1e-8)
    ratio_rand = vr_mean / (vt_mean + 1e-8)

    if ratio_te < 2:
        verdict = "LOW_DIVERGE"
    elif ratio_te < 5:
        verdict = "MODERATE"
    else:
        verdict = "HIGH"

    # Bonus: correlation with CRS for top-10% divergent Gaussians
    # CRS có thể không có trong PLY checkpoint → skip nếu lỗi
    crs_corr_str = ""
    try:
        crs = gaussians.get_crs.squeeze(-1)  # (N,) — may fail if no _crs_score
        sh_rest_norm = gaussians._features_rest.flatten(1).norm(dim=1)
        # Top 10% by test variance
        _, top_idx = var_test.topk(max(1, N // 10))
        crs_top = crs[top_idx].mean().item()
        shn_top = sh_rest_norm[top_idx].mean().item()
        crs_corr_str = (f"  Top-10% divergent Gaussians:\n"
                        f"    CRS mean:     {crs_top:.3f}\n"
                        f"    sh_norm mean: {shn_top:.4f}")
    except Exception:
        pass

    results = {
        "var_train_mean": round(vt_mean, 6),
        "var_train_p95": round(vt_p95, 6),
        "var_test_mean": round(ve_mean, 6),
        "var_test_p95": round(ve_p95, 6),
        "var_rand_mean": round(vr_mean, 6),
        "var_rand_p95": round(vr_p95, 6),
        "ratio_test_train": round(ratio_te, 2),
        "ratio_rand_train": round(ratio_rand, 2),
        "verdict": verdict,
    }

    print(f"\n=== Test 2: SH angular divergence ===")
    print(f"  SH variance at train dirs:  mean={vt_mean:.6f}, p95={vt_p95:.6f}")
    print(f"  SH variance at test dirs:   mean={ve_mean:.6f}, p95={ve_p95:.6f}")
    print(f"  SH variance at random dirs: mean={vr_mean:.6f}, p95={vr_p95:.6f}")
    print(f"  Ratio test/train:  {ratio_te:.2f}x")
    print(f"  Ratio rand/train:  {ratio_rand:.2f}x")
    print(f"  Verdict: {verdict}")
    if crs_corr_str:
        print(crs_corr_str)
    return results


# ──────────────────────────────────────────────────────────────
# Test 6: Error concentration (Pareto analysis)
# ──────────────────────────────────────────────────────────────
def test6_error_concentration(gaussians, test_cams, pipeline, bg):
    """[CRSGaussian Diagnostic] Per-Gaussian error Pareto analysis."""
    # Dùng tối đa 3 test cams, aggregate
    all_pareto = []
    N = gaussians.get_xyz.shape[0]
    # CRS có thể không có trong PLY checkpoint → fallback zeros
    try:
        crs_vals = gaussians.get_crs.squeeze(-1)  # (N,)
    except (AttributeError, RuntimeError):
        crs_vals = torch.full((N,), 0.5, device=gaussians.get_xyz.device)
    sh_norms = gaussians._features_rest.flatten(1).norm(dim=1)  # (N,)

    for cam in test_cams[:3]:
        pkg = render(cam, gaussians, pipeline, bg, disable_dropout=True)
        rendered = pkg["render"]  # (3, H, W)
        gt = cam.original_image[:3].cuda()
        error_map = (rendered - gt).abs().mean(dim=0)  # (H, W)
        vis = pkg["visibility_filter"]  # (N,) bool

        vis_idx = torch.nonzero(vis, as_tuple=True)[0]
        xyz_vis = gaussians.get_xyz[vis_idx]

        # Project to pixel
        W2C = cam.world_view_transform.T  # (4, 4)
        ones = torch.ones(xyz_vis.shape[0], 1, device=xyz_vis.device)
        xyz_hom = torch.cat([xyz_vis, ones], dim=1)  # (M, 4)
        pts_cam = (W2C @ xyz_hom.T).T  # (M, 4)
        depth = pts_cam[:, 2]
        H, W = int(cam.image_height), int(cam.image_width)
        fx = fov2focal(cam.FoVx, W)
        fy = fov2focal(cam.FoVy, H)
        px = (pts_cam[:, 0] / (depth + 1e-8) * fx + W / 2.0).long()
        py = (pts_cam[:, 1] / (depth + 1e-8) * fy + H / 2.0).long()

        valid = (depth > 0.01) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
        if valid.sum() < 100:
            continue
        px = px[valid].clamp(0, W - 1)
        py = py[valid].clamp(0, H - 1)
        valid_global_idx = vis_idx[valid]

        per_gauss_error = error_map[py, px]  # (M_valid,)
        sorted_err, sorted_local = per_gauss_error.sort(descending=True)
        cumsum = sorted_err.cumsum(0) / (sorted_err.sum() + 1e-8)

        pareto = {}
        n_total = len(sorted_err)
        for pct in [0.05, 0.10, 0.20, 0.50]:
            idx_n = min(int(pct * n_total), n_total - 1)
            pareto[f"top_{int(pct*100)}pct"] = round(cumsum[idx_n].item() * 100, 1)

        # Profile top-5%
        top5_n = max(1, int(0.05 * n_total))
        top5_global = valid_global_idx[sorted_local[:top5_n]]
        pareto["top5_crs_mean"] = round(crs_vals[top5_global].mean().item(), 3)
        pareto["top5_sh_norm_mean"] = round(sh_norms[top5_global].mean().item(), 4)
        pareto["top5_opacity_mean"] = round(
            gaussians.get_opacity[top5_global].mean().item(), 3)

        all_pareto.append(pareto)

    # Average across test views
    if not all_pareto:
        print("\n=== Test 6: Error concentration === SKIPPED (no valid projections)")
        return {}

    avg_pareto = {}
    for key in all_pareto[0]:
        vals = [p[key] for p in all_pareto if key in p]
        avg_pareto[key] = round(np.mean(vals), 2)

    top10_err = avg_pareto.get("top_10pct", 0)
    if top10_err > 60:
        verdict = "CONCENTRATED"
    elif top10_err > 40:
        verdict = "MODERATE"
    else:
        verdict = "DISTRIBUTED"

    avg_pareto["verdict"] = verdict

    print(f"\n=== Test 6: Error concentration (Pareto, avg {len(all_pareto)} test views) ===")
    for pct in [5, 10, 20, 50]:
        k = f"top_{pct}pct"
        if k in avg_pareto:
            print(f"  Top {pct:>2}% Gaussians cause {avg_pareto[k]:.1f}% error")
    print(f"  Verdict: {verdict}")
    print(f"  Top-5% error Gaussians profile:")
    print(f"    CRS mean:     {avg_pareto.get('top5_crs_mean', '?')}")
    print(f"    sh_norm mean: {avg_pareto.get('top5_sh_norm_mean', '?')}")
    print(f"    opacity mean: {avg_pareto.get('top5_opacity_mean', '?')}")
    return avg_pareto


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = ArgumentParser(description="[CRSGaussian Diagnostic] Gap analysis")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=10000, type=int)
    parser.add_argument("--save_json", action="store_true",
                        help="Save results to logs/diagnostic_gap/<scene>.json")
    args = get_combined_args(parser)

    with torch.no_grad():
        gaussians = GaussianModel(args)
        scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False)
        gaussians.active_sh_degree = gaussians.max_sh_degree

        bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
        bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        pipe = pipeline.extract(args)
        train_cams = scene.getTrainCameras()
        test_cams = scene.getTestCameras()

        scene_name = os.path.basename(args.source_path)
        n_gaussians = gaussians.get_xyz.shape[0]

        print(f"\n{'='*64}")
        print(f"  DIAGNOSTIC GAP ANALYSIS — Scene: {scene_name}")
        print(f"  Checkpoint: {args.model_path} @ iter {args.iteration}")
        print(f"  Gaussians: {n_gaussians}")
        print(f"  Train cams: {len(train_cams)}, Test cams: {len(test_cams)}")
        print(f"{'='*64}")

        # Run 4 tests
        r4 = test4_dc_vs_rest(gaussians, test_cams, pipe, bg)
        r5 = test5_angular_distance(train_cams, test_cams)
        r2 = test2_sh_divergence(gaussians, train_cams, test_cams)
        r6 = test6_error_concentration(gaussians, test_cams, pipe, bg)

        # Grand summary + recommendation
        print(f"\n{'='*64}")
        print(f"  RECOMMENDATION — {scene_name}")
        print(f"{'='*64}")

        if r4.get("verdict") == "REST_HURTING":
            print("  → Rest NET NEGATIVE: earlier freeze hoặc stronger regularization")
        elif r4.get("verdict") == "REST_NEUTRAL":
            if r2.get("verdict") == "HIGH":
                print("  → Rest is noise + SH diverge mạnh. Cross-view SH consistency loss may help")
            else:
                print("  → Rest neutral, SH stable. Gap chủ yếu do geometry / coverage limitation")
        else:
            print("  → Rest HELPING. SH smoothness loss recommended, không chặn thêm")

        if r6.get("verdict") == "CONCENTRATED":
            print("  → Error tập trung: targeted fix khả thi (CRS-guided SH lr, per-Gaussian regularization)")
        elif r6.get("verdict") == "DISTRIBUTED":
            print("  → Error phân tán: uniform approach (dropout/freeze) tiếp tục tốt hơn targeted")

        if r5.get("verdict") == "FAR":
            print("  → Test views rất xa train → gap phần lớn do viewpoint novelty, khó giảm hơn")
        elif r5.get("verdict") == "CLOSE":
            print("  → Test views gần train → gap do appearance overfit, còn room for improvement")

        # Save JSON
        if args.save_json:
            out_dir = "logs/diagnostic_gap"
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{scene_name}.json")
            results = {
                "scene": scene_name,
                "model_path": args.model_path,
                "iteration": args.iteration,
                "n_gaussians": n_gaussians,
                "test4": r4,
                "test5": r5,
                "test2": r2,
                "test6": r6,
            }
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n  [Saved] {out_path}")


if __name__ == "__main__":
    main()
