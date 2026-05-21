#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 17e — Step B] Uncertainty-fixation hypothesis test
# File: scripts/p17e_step_b_uncertainty_analysis.py  (TẠO MỚI — keep local)
#
# HYPOTHESIS (pre-registered, predict-before-test):
#   "C1a wins concentrate on pixels where A3 is uncertain (high A3 L1).
#    On fortress, those uncertain pixels = textureless rock face where
#    DSINE orientation prior is most informative → big PSNR gain.
#    On orchids, wins scatter randomly → no concentrated mechanism."
#
# Test: per pixel per scene compute:
#   - r = Pearson(L1_A3, improvement)
#   - top-10% mask by L1_A3 → what fraction of total positive improvement
#     comes from this 10% (uncertainty-fixation capture rate)
#
# Detection feasibility score per detector = top10_capture / oracle_gain.
# Decides whether uncertainty-gate is operationally viable.
#
# Diagnostic-only. KHÔNG đụng production. Reuse improvement.npy from Step A
# + re-render A3 (no C1a needed, ~half Step A cost ≈ 10 phút server).
#
# ────────────────────── PRE-REGISTRATION LOCKED ──────────────────────
# Metric: per-pixel correlation + top-decile capture analysis
#
# Per-scene classification:
#   substrate ⟺ r ≥ 0.30 AND top10_capture ≥ 30%
#   (Both conditions: corr direction + concentration effect)
#
# Aggregate verdict (LOCKED, single decision):
#   ≥5/8 scenes SUBSTRATE → commit gate pilot with L1_A3-threshold gate
#   ≥3/8 SUBSTRATE → AMBIGUOUS, user decides
#   <3/8 SUBSTRATE → push C1b (gate-by-uncertainty not viable)
#
# Oracle upper bound (informational, NOT verdict):
#   oracle_L1 = mean(max(improvement, 0)) per scene (pixel-level perfect gate)
#   Aggregate → rough PSNR-Δ estimate via L1→MSE→PSNR
#   If oracle <<< +0.10 PSNR even with perfect gate → mathematically blocked.
#
# Caveats (baked, không slip):
#   (a) N=8 scenes magnitude triage, not stat significance.
#   (b) L1_A3 measured on TEST views; gate during training would use TRAIN
#       view L1 → transferability assumed, not proven by this diagnostic.
#   (c) Per-pixel improvement has atomicAdd noise (mitigated by 3-seed avg).
#   (d) Per-pixel r has huge N (HxW) — high statistical power, but causal
#       interpretation depends on whether L1_A3 is measurable at train time
#       (it IS — A3 photometric loss computes this directly).
#   FORBID: 'p<0.05', 'noise=driver', '% variance' claims.
# ─────────────────────────────────────────────────────────────────────
# ============================================================
"""[CRSGaussian Phase 17e — Step B] Uncertainty-fixation hypothesis.

Server (no train, ~10 phút, 1 GPU; reuse Step A's improvement.npy):
    mkdir -p logs/p17e
    CUDA_VISIBLE_DEVICES=0 python scripts/p17e_step_b_uncertainty_analysis.py \\
        2>&1 | tee logs/p17e/step_b.log
  → logs/p17e/step_b_summary.txt + step_b_scatter_<scene>.png +
    step_b_l1_a3_<scene>.npy
"""

import os
import sys
import math
import statistics
import numpy as np

sys.path.insert(0, ".")

DATA_ROOT      = os.environ.get("DATA_ROOT",      "data/nerf_llff_data")
OUTPUT_ROOT_A3 = os.environ.get("OUTPUT_ROOT_A3", "output/p13_lfcf")
STEP_A_DIR     = os.environ.get("STEP_A_DIR",     "logs/p17e")
ITERATION      = int(os.environ.get("ITERATION", "10000"))
SCENES = os.environ.get(
    "SCENES", "fern flower fortress horns leaves orchids room trex").split()
SEEDS = os.environ.get("SEEDS", "42 137 9999").split()
OUT_DIR = os.environ.get("OUT_DIR", "logs/p17e")

# Pre-registered (Phase 17 N=24) per-scene Δ_C1a for cross-check
DELTA_C1A = {
    "fortress": +0.473, "orchids": +0.097, "flower": +0.085, "room": +0.032,
    "leaves":   -0.098, "fern":    -0.123, "horns":  -0.203, "trex":  -0.217,
}

R_SUBSTRATE = 0.30           # pre-registered locked
TOP10_CAPTURE_SUBSTRATE = 0.30  # 30% of total positive improvement from top-10% L1_A3 pixels


def load_a3_model(seed, scene):
    """Mirror p17c_tier2 verified load pattern."""
    from argparse import ArgumentParser, Namespace
    from scene import Scene, GaussianModel
    from arguments import ModelParams, PipelineParams

    model_path = f"{OUTPUT_ROOT_A3}/A3_seed{seed}_{scene}"
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


def compute_L1_A3(scene):
    """Re-render A3 test views across 3 seeds; return mean L1_A3 (H,W) numpy."""
    import torch
    from gaussian_renderer import render
    bg = torch.tensor([0., 0., 0.], device="cuda")
    sum_L1 = None
    n = 0
    with torch.no_grad():
        for sd in SEEDS:
            g, sc, pipe = load_a3_model(sd, scene)
            if g is None:
                print(f"    seed{sd}: MISSING — skip")
                continue
            for c in sc.getTestCameras():
                rpkg = render(c, g, pipe, bg)
                img = rpkg["render"].clamp(0.0, 1.0)
                gt = c.original_image[:3].to(img.device)
                L1 = (img - gt).abs().mean(0)
                if sum_L1 is None:
                    sum_L1 = L1.detach().clone()
                else:
                    sum_L1 += L1.detach()
                n += 1
            del g, sc
            torch.cuda.empty_cache()
    if sum_L1 is None or n == 0:
        return None
    return (sum_L1 / n).cpu().numpy()


def analyze_scene(scene, L1_A3, improvement):
    """Test uncertainty-fixation hypothesis. Return summary dict."""
    L1_A3_flat = L1_A3.ravel()
    imp_flat = improvement.ravel()

    # Pearson correlation (per pixel)
    mx, my = L1_A3_flat.mean(), imp_flat.mean()
    dx = L1_A3_flat - mx
    dy = imp_flat - my
    denom = math.sqrt((dx * dx).sum() * (dy * dy).sum())
    r = float((dx * dy).sum() / denom) if denom > 0 else 0.0

    # Oracle upper bound: pixel-level perfect gate (only positive)
    positive_mask = imp_flat > 0
    oracle_L1 = float(imp_flat[positive_mask].mean()) if positive_mask.any() else 0.0
    oracle_L1_per_pixel = float(np.where(positive_mask, imp_flat, 0).mean())
    total_positive_gain = float(imp_flat[positive_mask].sum())

    # Top-10% by L1_A3 = "highest A3 uncertainty pixels"
    thr_top10 = float(np.percentile(L1_A3_flat, 90))
    top10_mask = L1_A3_flat >= thr_top10
    top10_imp = imp_flat[top10_mask]
    top10_mean_imp = float(top10_imp.mean()) if top10_imp.size > 0 else 0.0
    top10_win_rate = float((top10_imp > 0).mean()) * 100 if top10_imp.size > 0 else 0.0
    # capture: % of total positive gain coming from top-10% pixels
    top10_positive_gain = float(top10_imp[top10_imp > 0].sum()) if top10_imp.size > 0 else 0.0
    top10_capture = (top10_positive_gain / total_positive_gain) if total_positive_gain > 0 else 0.0

    # Bottom-10% by L1_A3 = "easy A3 pixels" (counterpart check)
    thr_bot10 = float(np.percentile(L1_A3_flat, 10))
    bot10_mask = L1_A3_flat <= thr_bot10
    bot10_imp = imp_flat[bot10_mask]
    bot10_mean_imp = float(bot10_imp.mean()) if bot10_imp.size > 0 else 0.0

    # Decile-bin scatter (for plot)
    deciles = np.percentile(L1_A3_flat, [10, 20, 30, 40, 50, 60, 70, 80, 90])
    bin_indices = np.digitize(L1_A3_flat, deciles)
    bin_means = []
    for b in range(10):
        m = bin_indices == b
        bin_means.append(float(imp_flat[m].mean()) if m.any() else 0.0)

    # Substrate classification (LOCKED rule)
    substrate = (r >= R_SUBSTRATE) and (top10_capture >= TOP10_CAPTURE_SUBSTRATE)

    return {
        "scene": scene,
        "r": r,
        "oracle_L1": oracle_L1,
        "oracle_L1_per_pixel": oracle_L1_per_pixel,
        "top10_mean_imp": top10_mean_imp,
        "top10_win_rate": top10_win_rate,
        "top10_capture": top10_capture,
        "bot10_mean_imp": bot10_mean_imp,
        "bin_means": bin_means,
        "thr_top10": thr_top10,
        "substrate": substrate,
    }


def save_scatter(scene, summary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        deciles = list(range(1, 11))
        plt.figure(figsize=(7, 4))
        plt.bar(deciles, summary["bin_means"], color="steelblue")
        plt.axhline(0, color="black", lw=0.5)
        plt.xlabel("L1_A3 decile (1=lowest error, 10=highest error)")
        plt.ylabel("Mean C1a improvement (L1 scale)")
        plt.title(f"{scene}: improvement by A3-uncertainty decile\n"
                  f"r={summary['r']:+.3f}, top10_capture={summary['top10_capture']*100:.1f}%, "
                  f"substrate={'YES' if summary['substrate'] else 'NO'}")
        plt.xticks(deciles)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        path = os.path.join(OUT_DIR, f"step_b_scatter_{scene}.png")
        plt.savefig(path, dpi=120); plt.close()
        print(f"  scatter saved: {path}")
    except Exception as e:
        print(f"  scatter SKIPPED (matplotlib err: {e})")


def L1_to_psnr_delta_estimate(L1_gain, L1_baseline=0.06):
    """Rough heuristic: MSE ≈ 1.5×L1² (assuming Gaussian errors).
    ΔPSNR = -10·log10(MSE_new/MSE_old).
    L1_gain = L1_old - L1_new (positive = gain).
    Return rough ΔPSNR estimate (informational only)."""
    if L1_baseline <= L1_gain:
        return float("nan")
    L1_new = L1_baseline - L1_gain
    mse_old = 1.5 * L1_baseline ** 2
    mse_new = 1.5 * L1_new ** 2
    return -10 * math.log10(mse_new / mse_old)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 72)
    print("Phase 17e — Step B: uncertainty-fixation hypothesis (LOCKED)")
    print("=" * 72)
    print(f"SCENES = {SCENES}")
    print(f"SEEDS  = {SEEDS}")
    print(f"R_substrate ≥ {R_SUBSTRATE}, top10_capture ≥ {TOP10_CAPTURE_SUBSTRATE*100:.0f}%")
    print()

    summaries = {}
    for sc in SCENES:
        print(f"────── {sc} ──────")
        # Load Step A improvement
        imp_path = os.path.join(STEP_A_DIR, f"improvement_{sc}.npy")
        if not os.path.isfile(imp_path):
            print(f"  MISSING {imp_path} — skip")
            continue
        improvement = np.load(imp_path)
        print(f"  loaded improvement: shape={improvement.shape}")

        # Re-render A3 → L1_A3
        print(f"  re-rendering A3 (3 seeds × test views)...")
        L1_A3 = compute_L1_A3(sc)
        if L1_A3 is None:
            print(f"  A3 render FAILED — skip")
            continue
        # Save L1_A3 for later use
        np.save(os.path.join(OUT_DIR, f"step_b_l1_a3_{sc}.npy"), L1_A3)

        # Shape consistency
        if L1_A3.shape != improvement.shape:
            print(f"  shape mismatch: L1_A3={L1_A3.shape} vs improvement={improvement.shape} — skip")
            continue

        # Analyze
        s = analyze_scene(sc, L1_A3, improvement)
        summaries[sc] = s

        # Print
        delta = DELTA_C1A.get(sc, None)
        d_str = f"{delta:+.3f}" if delta is not None else "?"
        print(f"  N=24 Δ_C1a (PSNR) = {d_str}")
        print(f"  r(L1_A3, improvement) = {s['r']:+.4f}")
        print(f"  oracle pixel-mean gain = {s['oracle_L1_per_pixel']:+.5f} (L1 scale)")
        print(f"  top-10% L1_A3: mean_imp={s['top10_mean_imp']:+.5f} "
              f"win_rate={s['top10_win_rate']:.1f}% capture={s['top10_capture']*100:.1f}%")
        print(f"  bot-10% L1_A3: mean_imp={s['bot10_mean_imp']:+.5f}")
        print(f"  substrate (r≥{R_SUBSTRATE} ∧ top10_capture≥{TOP10_CAPTURE_SUBSTRATE*100:.0f}%): "
              f"{'✅ YES' if s['substrate'] else '❌ NO'}")

        save_scatter(sc, s)

    # Aggregate
    print()
    print("=" * 72)
    print(f"{'scene':<10} {'Δ_C1a':>8} {'r_unc':>7} {'top10_cap%':>11} "
          f"{'top10_win%':>11} {'oracle_L1':>10} {'subst':>6}")
    n_substrate = 0
    n_total = 0
    oracle_per_pixel_all = []
    for sc in SCENES:
        if sc not in summaries:
            print(f"{sc:<10} (no data)"); continue
        s = summaries[sc]
        n_total += 1
        if s["substrate"]:
            n_substrate += 1
        oracle_per_pixel_all.append(s["oracle_L1_per_pixel"])
        delta = DELTA_C1A.get(sc, None)
        d_str = f"{delta:+.3f}" if delta is not None else "?"
        print(f"{sc:<10} {d_str:>8} {s['r']:>+7.3f} {s['top10_capture']*100:>10.1f} "
              f"{s['top10_win_rate']:>10.1f} {s['oracle_L1_per_pixel']:>+10.5f} "
              f"{'YES' if s['substrate'] else 'NO':>6}")

    # Oracle aggregate (informational PSNR estimate)
    if oracle_per_pixel_all:
        oracle_mean = statistics.fmean(oracle_per_pixel_all)
        psnr_est = L1_to_psnr_delta_estimate(oracle_mean)
        print()
        print(f"Oracle aggregate (pixel-level perfect gate, INFORMATIONAL):")
        print(f"  mean L1 gain across scenes = {oracle_mean:+.5f}")
        print(f"  rough ΔPSNR estimate ≈ {psnr_est:+.3f} dB (Gaussian-error heuristic)")
        print(f"  Note: PSNR-Δ scales non-linearly with L1 + per-scene MSE structure.")
        print(f"  Compare to pre-registered bar +0.10 dB. If oracle >> 0.10 → headroom exists.")

    # Final verdict
    print()
    print("─" * 72)
    print(f"[Verdict] {n_substrate}/{n_total} scenes show SUBSTRATE for uncertainty-fixation")
    if n_substrate >= 5:
        verdict = "SUBSTRATE_EXISTS"
        action = "Commit gate pilot N=24 with L1_A3-threshold gate"
    elif n_substrate >= 3:
        verdict = "AMBIGUOUS"
        action = "User decides (substrate marginal)"
    else:
        verdict = "SUBSTRATE_ABSENT"
        action = "Uncertainty-gate KHÔNG viable → push C1b (CUDA per-Gaussian normal)"
    print(f"  → {verdict}")
    print(f"  Action: {action}")
    print("─" * 72)
    print("CAVEATS:")
    print("  (a) N=8 scenes — magnitude triage, NOT stat significance.")
    print("  (b) L1_A3 measured on TEST views; in-train gate would use TRAIN")
    print("      view L1 → transferability assumed.")
    print("  (c) atomicAdd noise mitigated by 3-seed × N-view averaging.")
    print("  FORBID downstream: 'p<0.05', '% variance', 'X = driver confirmed'.")
    print("─" * 72)
    print("\nDone.")


if __name__ == "__main__":
    main()
