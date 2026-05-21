#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 17e — Step A] Per-pixel C1a improvement map
# File: scripts/p17e_step_a_pixel_diagnostic.py  (TẠO MỚI — keep local)
#
# Diagnostic-only (no train). Đo CHÍNH XÁC WHERE C1a-uniform giúp vs A3
# theo per-pixel L1 distance vs GT trên test views. Output heatmap per
# scene: positive (red) = C1a closer GT, negative (blue) = C1a farther.
#
# Workflow per scene:
#   for each seed in {42, 137, 9999}:
#     - Load A3 ckpt iter 10000 → render all test views → L1_A3(x,y) per view
#     - Load C1L10 ckpt iter 10000 → render same test views → L1_C1(x,y)
#   Aggregate L1 across (seeds × test views) → mean L1 per pixel per model
#   improvement(x,y) = L1_A3(x,y) - L1_C1(x,y)
#     > 0 ⟹ C1a render closer to GT ⟹ C1a thắng ở pixel này
#     < 0 ⟹ C1a hại
#
# KHÔNG đụng production. Đọc A3 (logs/p13_lfcf) + C1L10 (logs/p17_c1)
# ckpts có sẵn. Render qua API standard (gaussian_renderer.render).
#
# Sanity check: scene-level scalar improvement = mean(improvement_map)
# should correlate-sign với per-scene Δ_C1a paired N=24 đã có (CLAUDE.md
# Phase 17 table). Vd: fortress positive (Δ+0.473), foliage negative.
#
# Step B (next, sau khi user inspect heatmap): characterize win-regions
# bằng feature engineering (R̄ DSINE, DAV2 plane residual, depth grad,
# RGB texture) → derive data-driven gate criterion. KHÔNG pre-register
# gate ở Step A — observational mapping only.
# ============================================================
"""[CRSGaussian Phase 17e — Step A] per-pixel C1a-vs-A3 improvement map.

Server (no train, ~20-30 phút, 1 GPU; matplotlib optional cho PNG):
    mkdir -p logs/p17e
    CUDA_VISIBLE_DEVICES=0 python scripts/p17e_step_a_pixel_diagnostic.py \\
        2>&1 | tee logs/p17e/step_a.log
  → logs/p17e/improvement_<scene>.npy + improvement_<scene>.png
"""

import os
import sys
import math
import statistics
import numpy as np

sys.path.insert(0, ".")

DATA_ROOT       = os.environ.get("DATA_ROOT",       "data/nerf_llff_data")
OUTPUT_ROOT_A3  = os.environ.get("OUTPUT_ROOT_A3",  "output/p13_lfcf")
OUTPUT_ROOT_C1  = os.environ.get("OUTPUT_ROOT_C1",  "output/p17_c1")
ITERATION       = int(os.environ.get("ITERATION", "10000"))
SCENES = os.environ.get(
    "SCENES", "fern flower fortress horns leaves orchids room trex").split()
SEEDS  = os.environ.get("SEEDS", "42 137 9999").split()
OUT_DIR = os.environ.get("OUT_DIR", "logs/p17e")


def load_model_and_scene(output_root, name_prefix, seed, scene):
    """Mirror p17c_tier2 verified pattern: load Scene with cfg_args + load_iteration."""
    import torch
    from argparse import ArgumentParser, Namespace
    from scene import Scene, GaussianModel
    from arguments import ModelParams, PipelineParams

    model_path = f"{output_root}/{name_prefix}_seed{seed}_{scene}"
    cfg_path = os.path.join(model_path, "cfg_args")
    ply = f"{model_path}/point_cloud/iteration_{ITERATION}/point_cloud.ply"
    if not os.path.isfile(cfg_path) or not os.path.isfile(ply):
        return None, None, None

    parser = ArgumentParser()
    lp = ModelParams(parser); pp = PipelineParams(parser); _ = lp
    args_cmdline = parser.parse_args([])
    with open(cfg_path) as f:
        cfg = eval(f.read())
    merged = vars(args_cmdline).copy()
    for k, v in vars(cfg).items():
        merged[k] = v
    args = Namespace(**merged)
    args.model_path = model_path
    if not os.path.isdir(getattr(args, "source_path", "") or ""):
        args.source_path = os.path.join(DATA_ROOT, scene)
    pipe = pp.extract(args)

    gaussians = GaussianModel(args)
    scene_obj = Scene(args, gaussians, load_iteration=ITERATION, shuffle=False)
    return gaussians, scene_obj, pipe


def accumulate_L1_test_views(gaussians, scene_obj, pipe):
    """Render all test views, return (sum_L1_per_pixel, n_views) on GPU.
    sum_L1 shape (H, W). L1 = mean over RGB của |render - gt|."""
    import torch
    from gaussian_renderer import render
    bg = torch.tensor([0., 0., 0.], device="cuda")
    sum_L1 = None
    n = 0
    test_cams = scene_obj.getTestCameras()
    with torch.no_grad():
        for c in test_cams:
            rpkg = render(c, gaussians, pipe, bg)
            img = rpkg["render"].clamp(0.0, 1.0)              # (3,H,W) [0,1]
            gt = c.original_image[:3].to(img.device)          # (3,H,W) [0,1]
            L1 = (img - gt).abs().mean(0)                     # (H, W)
            if sum_L1 is None:
                sum_L1 = L1.detach().clone()
            else:
                sum_L1 += L1.detach()
            n += 1
    return sum_L1, n


def process_scene(scene):
    import torch
    print(f"\n────── {scene} ──────")
    sum_L1_A3 = None
    sum_L1_C1 = None
    n_total = 0
    H_W = None
    for sd in SEEDS:
        # A3 pass
        g_a3, sc_a3, pipe_a3 = load_model_and_scene(OUTPUT_ROOT_A3, "A3", sd, scene)
        if g_a3 is None:
            print(f"  seed{sd} A3: MISSING — skip")
            continue
        n_test_a3 = len(sc_a3.getTestCameras())
        sumL1_a3_seed, n_a3 = accumulate_L1_test_views(g_a3, sc_a3, pipe_a3)
        del g_a3, sc_a3
        torch.cuda.empty_cache()

        # C1L10 pass
        g_c1, sc_c1, pipe_c1 = load_model_and_scene(OUTPUT_ROOT_C1, "C1L10", sd, scene)
        if g_c1 is None:
            print(f"  seed{sd} C1L10: MISSING — skip (rollback A3 contribution)")
            del sumL1_a3_seed
            torch.cuda.empty_cache()
            continue
        n_test_c1 = len(sc_c1.getTestCameras())
        if n_test_a3 != n_test_c1:
            print(f"  seed{sd}: WARN n_test mismatch A3={n_test_a3} C1={n_test_c1}")
        sumL1_c1_seed, n_c1 = accumulate_L1_test_views(g_c1, sc_c1, pipe_c1)
        del g_c1, sc_c1
        torch.cuda.empty_cache()

        # Accumulate seed
        if sum_L1_A3 is None:
            sum_L1_A3 = sumL1_a3_seed.clone()
            sum_L1_C1 = sumL1_c1_seed.clone()
            H_W = sumL1_a3_seed.shape
        else:
            sum_L1_A3 += sumL1_a3_seed
            sum_L1_C1 += sumL1_c1_seed
        n_total += n_a3
        print(f"  seed{sd}: A3 mean_L1={sumL1_a3_seed.mean().item()/n_a3:.5f} "
              f"C1 mean_L1={sumL1_c1_seed.mean().item()/n_c1:.5f} "
              f"n_views={n_a3}")
        del sumL1_a3_seed, sumL1_c1_seed
        torch.cuda.empty_cache()

    if sum_L1_A3 is None or n_total == 0:
        print(f"  {scene}: NO DATA")
        return None, None

    mean_L1_A3 = (sum_L1_A3 / n_total).cpu().numpy()
    mean_L1_C1 = (sum_L1_C1 / n_total).cpu().numpy()
    improvement = mean_L1_A3 - mean_L1_C1           # (H,W) positive ⟹ C1a thắng

    # Scene-level summary
    imp_mean = float(improvement.mean())
    imp_pos_pct = float((improvement > 0).mean()) * 100
    imp_p95 = float(np.percentile(improvement, 95))
    imp_p5  = float(np.percentile(improvement, 5))
    print(f"  → scene {scene}: improvement.mean={imp_mean:+.5f} "
          f"(L1 scale, sign should match Δ_C1a)")
    print(f"    pixels C1a-thắng: {imp_pos_pct:.1f}%  "
          f"p95={imp_p95:+.5f}  p5={imp_p5:+.5f}")

    return improvement, dict(mean=imp_mean, pos_pct=imp_pos_pct,
                             p5=imp_p5, p95=imp_p95, H=H_W[0], W=H_W[1],
                             n_total=n_total)


def save_outputs(scene, improvement, summary):
    os.makedirs(OUT_DIR, exist_ok=True)
    npy_path = os.path.join(OUT_DIR, f"improvement_{scene}.npy")
    np.save(npy_path, improvement)
    print(f"  saved npy:  {npy_path}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Diverging colormap, symmetric range centered 0
        vmax = max(abs(summary["p95"]), abs(summary["p5"]))
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(improvement, cmap="RdBu", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{scene}: C1a improvement vs A3 (red=C1a wins)\n"
                     f"mean={summary['mean']:+.5f} pos_pct={summary['pos_pct']:.1f}%")
        plt.colorbar(im, ax=ax, fraction=0.04)
        ax.axis("off")
        plt.tight_layout()
        png = os.path.join(OUT_DIR, f"improvement_{scene}.png")
        plt.savefig(png, dpi=120); plt.close()
        print(f"  saved png:  {png}")
    except Exception as e:
        print(f"  PNG skipped (matplotlib err): {e}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 70)
    print("Phase 17e — Step A: per-pixel C1a-vs-A3 improvement map")
    print("=" * 70)
    print(f"SCENES = {SCENES}")
    print(f"SEEDS  = {SEEDS}  (atomicAdd √{len(SEEDS)}× noise reduction)")
    print(f"OUT_DIR = {OUT_DIR}")

    summaries = {}
    for sc in SCENES:
        imp, sm = process_scene(sc)
        if imp is not None:
            save_outputs(sc, imp, sm)
            summaries[sc] = sm

    # Final cross-scene summary
    print("\n" + "=" * 70)
    print(f"{'scene':<10} {'imp.mean':>10} {'pos_pct%':>10} "
          f"{'p5':>10} {'p95':>10} {'n_views':>8}")
    for sc in SCENES:
        if sc not in summaries:
            print(f"{sc:<10} (no data)"); continue
        s = summaries[sc]
        print(f"{sc:<10} {s['mean']:>+10.5f} {s['pos_pct']:>10.1f} "
              f"{s['p5']:>+10.5f} {s['p95']:>+10.5f} {s['n_total']:>8}")
    print()
    print("Sanity: imp.mean SIGN should match Δ_C1a per-scene from N=24 analyzer")
    print("  Expected POSITIVE (C1a thắng): fortress, flower, orchids, room")
    print("  Expected NEGATIVE (C1a thua): fern, leaves, horns, trex")
    print()
    print("Next: inspect heatmaps (logs/p17e/improvement_*.png) — visually")
    print("characterize WHERE C1a wins (planar? coherent? textureless?) →")
    print("Step B = quantify feature predictors → data-driven gate.")


if __name__ == "__main__":
    main()
