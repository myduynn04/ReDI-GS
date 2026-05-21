#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 17e — Step C] Gate-by-L1_A3 simulation projection
# File: scripts/p17e_step_c_gate_simulation.py  (TẠO MỚI — keep local)
#
# Simulation OFFLINE: dùng SAME A3 + C1L10 renders đã tồn tại + per-view
# L1_A3 → tính composite render nếu gate-to-top-K% by L1_A3. Tính MSE
# composite per view → PSNR composite → Δ vs A3 baseline. Aggregate.
#
# Trả lời CHÍNH XÁC (trên dữ liệu đã có, ZERO new training):
#   "Nếu áp gate-by-L1_A3-top-K% lên 2 model đã train, PSNR shift bao nhiêu?"
#
# Caveat semantics: simulation = ENSEMBLE upper bound. Lấy 2 model trained
# uniform-A3 vs uniform-C1a, switch per-pixel = optimistic estimate. Real
# training-time gate train MỘT model with masked loss → kết quả thực tế
# có shrinkage so với simulation. Honest: nếu simulation < +0.10 → real
# pilot càng dưới → KHÔNG đáng commit pilot.
#
# Diagnostic-only. Re-render BOTH A3 + C1L10 (3 seeds × 8 scenes) ~15 min.
# KHÔNG đụng production.
#
# ─────────────────── PRE-REGISTRATION LOCKED ───────────────────
# K_PCT = 10%  (top-10% pixels by per-view L1_A3 = gate mask)
#   Lock NOW từ Step B observation top10_win >55% mọi scene.
#   KHÔNG sweep K (= bẫy λ-grid). Single config.
#
# Per-view mask: each view computes its own threshold (realistic vs
# in-train gate). Aggregate per scene → 3-seed × N-view paired mean.
#
# Verdict (simulation projection, LOCKED):
#   projected Δ_mean ≥ +0.10 ∧ ≥7/8 scene Δ_gated ≥ 0 ∧ horns Δ_gated ≥ −0.05
#     → SIMULATION_PASS → commit (B) pilot N=24 với high confidence
#   projected < +0.10 → SIMULATION_FAIL → push C1b (A); KHÔNG cần pilot
#   in-between → ambiguous, user quyết
#
# Note: real pilot expect shrinkage 20-50% vs simulation (training-time
# gate KHÔNG ensemble cleanly). Simulation = optimistic ceiling.
#
# Caveats:
#   (a) N=8 magnitude triage, không stat.
#   (b) Per-view L1_A3 measured on TEST views. Train-time gate would
#       use TRAIN view L1 → projection assumes transferability.
#   (c) Ensemble simulation = upper bound; real gate-trained model lower.
#   (d) atomicAdd noise floor ±0.05/scene paired (mitigated 3-seed avg).
#   FORBID: 'p<0.05', '% variance', 'commit guaranteed'.
# ───────────────────────────────────────────────────────────────
# ============================================================
"""[CRSGaussian Phase 17e — Step C] Gate-by-L1_A3 simulation.

Server (no train, ~15 min, 1 GPU):
    mkdir -p logs/p17e
    CUDA_VISIBLE_DEVICES=0 python scripts/p17e_step_c_gate_simulation.py \\
        2>&1 | tee logs/p17e/step_c.log
  → logs/p17e/step_c_summary.txt
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
SEEDS = os.environ.get("SEEDS", "42 137 9999").split()
OUT_DIR = os.environ.get("OUT_DIR", "logs/p17e")

K_PCT = float(os.environ.get("K_PCT", "0.10"))  # pre-registered locked

# Pre-registered Phase 17 N=24 per-scene Δ_C1a (PSNR) for cross-check
DELTA_C1A_N24 = {
    "fortress": +0.473, "orchids": +0.097, "flower": +0.085, "room": +0.032,
    "leaves":   -0.098, "fern":    -0.123, "horns":  -0.203, "trex":  -0.217,
}


def load_model_and_scene(output_root, name_prefix, seed, scene):
    from argparse import ArgumentParser, Namespace
    from scene import Scene, GaussianModel
    from arguments import ModelParams, PipelineParams
    import torch  # noqa

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


def simulate_per_view_gated(scene):
    """Per scene, per seed: render A3 + C1L10 trên test views, compute
    composite render qua gate-by-L1_A3-top-K%, return list of (psnr_A3,
    psnr_C1_uniform, psnr_C1_gated) per (seed, view)."""
    import torch
    from gaussian_renderer import render

    bg = torch.tensor([0., 0., 0.], device="cuda")
    per_view = []  # list of (psnr_A3, psnr_C1u, psnr_gated)

    for sd in SEEDS:
        # --- A3 render pass ---
        g_a3, sc_a3, pipe = load_model_and_scene(OUTPUT_ROOT_A3, "A3", sd, scene)
        if g_a3 is None:
            print(f"    seed{sd} A3 MISSING — skip")
            continue
        a3_data = []  # per view: (sq_a3_3HW, L1_a3_HW)
        test_cams = sc_a3.getTestCameras()
        with torch.no_grad():
            for c in test_cams:
                r = render(c, g_a3, pipe, bg)["render"].clamp(0.0, 1.0)
                gt = c.original_image[:3].to(r.device)
                err = r - gt
                sq = err * err                                      # (3,H,W)
                L1 = err.abs().mean(0)                              # (H,W)
                a3_data.append((sq.detach().cpu(), L1.detach().cpu()))
        del g_a3, sc_a3
        torch.cuda.empty_cache()

        # --- C1L10 render pass on same test cams ---
        g_c1, sc_c1, _ = load_model_and_scene(OUTPUT_ROOT_C1, "C1L10", sd, scene)
        if g_c1 is None:
            print(f"    seed{sd} C1L10 MISSING — skip")
            continue
        with torch.no_grad():
            for i, c in enumerate(sc_c1.getTestCameras()):
                r = render(c, g_c1, pipe, bg)["render"].clamp(0.0, 1.0)
                gt = c.original_image[:3].to(r.device)
                err = r - gt
                sq_c1 = err * err                                  # (3,H,W) GPU
                sq_a3 = a3_data[i][0].to(r.device)
                L1_a3 = a3_data[i][1].to(r.device)
                # Per-view mask: top-K% pixels by L1_A3
                thr = torch.quantile(L1_a3.flatten(), 1.0 - K_PCT)
                mask = (L1_a3 >= thr).float().unsqueeze(0)         # (1,H,W)
                composite_sq = mask * sq_c1 + (1.0 - mask) * sq_a3 # (3,H,W)
                mse_a3 = sq_a3.mean().item()
                mse_c1 = sq_c1.mean().item()
                mse_gated = composite_sq.mean().item()
                psnr_a3 = -10.0 * math.log10(max(mse_a3, 1e-12))
                psnr_c1 = -10.0 * math.log10(max(mse_c1, 1e-12))
                psnr_gated = -10.0 * math.log10(max(mse_gated, 1e-12))
                per_view.append((psnr_a3, psnr_c1, psnr_gated))
        del g_c1, sc_c1
        del a3_data
        torch.cuda.empty_cache()

    return per_view


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 72)
    print("Phase 17e — Step C: Gate-by-L1_A3 simulation projection (LOCKED)")
    print("=" * 72)
    print(f"K_PCT = {K_PCT*100:.0f}% (top-K by per-view L1_A3 mask)")
    print(f"SCENES = {SCENES}")
    print(f"SEEDS  = {SEEDS}")
    print()
    print(f"{'scene':<10} {'N':>4} {'Δ_unif':>9} {'Δ_unif_24':>10} "
          f"{'Δ_gated':>9} {'gate_gain':>10}")
    print("-" * 60)

    scene_summary = {}
    for sc in SCENES:
        print(f"  computing {sc}...", end=" ", flush=True)
        per_view = simulate_per_view_gated(sc)
        if not per_view:
            print("NO DATA")
            continue
        # Per-view Δ
        deltas_uniform = [psnr_c1 - psnr_a3 for (psnr_a3, psnr_c1, _) in per_view]
        deltas_gated   = [psnr_g - psnr_a3 for (psnr_a3, _, psnr_g) in per_view]
        mean_uniform = statistics.fmean(deltas_uniform)
        mean_gated   = statistics.fmean(deltas_gated)
        gate_gain    = mean_gated - mean_uniform
        n24 = DELTA_C1A_N24.get(sc, None)
        n24_str = f"{n24:+.3f}" if n24 is not None else "?"
        scene_summary[sc] = dict(
            n=len(per_view),
            mean_uniform=mean_uniform,
            mean_gated=mean_gated,
            gate_gain=gate_gain,
            n24=n24,
        )
        print(f"DONE")
        print(f"  {sc:<10} {len(per_view):>4} {mean_uniform:>+9.4f} "
              f"{n24_str:>10} {mean_gated:>+9.4f} {gate_gain:>+10.4f}")

    # Aggregate (paired N=24 ≈ 8 scenes × 3 seeds × ~varied views, but per-view averaged)
    print()
    print("=" * 72)
    print("[Aggregate per-scene means, equivalent paired N=24 across scenes]")
    print(f"{'scene':<10} {'Δ_uniform':>11} {'Δ_unif_N24':>12} "
          f"{'Δ_gated':>10} {'gain_gate':>10}")
    print("-" * 60)
    uniform_means = []
    gated_means = []
    for sc in SCENES:
        if sc not in scene_summary:
            print(f"{sc:<10} (no data)"); continue
        s = scene_summary[sc]
        uniform_means.append(s["mean_uniform"])
        gated_means.append(s["mean_gated"])
        n24_str = f"{s['n24']:+.3f}" if s['n24'] is not None else "?"
        print(f"{sc:<10} {s['mean_uniform']:>+11.4f} {n24_str:>12} "
              f"{s['mean_gated']:>+10.4f} {s['gate_gain']:>+10.4f}")

    if uniform_means:
        mean_unif_agg = statistics.fmean(uniform_means)
        mean_gated_agg = statistics.fmean(gated_means)
        gain_agg = mean_gated_agg - mean_unif_agg
        n_gated_pos = sum(1 for v in gated_means if v >= 0)
        horns_gated = scene_summary.get("horns", {}).get("mean_gated", None)
        # Bar check
        c1 = mean_gated_agg >= 0.10
        c3a = n_gated_pos >= 7
        c3b = (horns_gated is not None and horns_gated >= -0.05)
        print()
        print(f"  mean Δ_uniform (sanity vs N=24 +0.006) = {mean_unif_agg:+.4f}")
        print(f"  mean Δ_gated (projection)              = {mean_gated_agg:+.4f}")
        print(f"  gate vs uniform gain                   = {gain_agg:+.4f}")
        print(f"  # scenes Δ_gated ≥ 0                   = {n_gated_pos}/{len(gated_means)}")
        print(f"  horns Δ_gated                          = "
              f"{horns_gated if horns_gated is None else f'{horns_gated:+.4f}'}")

        print()
        print("─" * 60)
        print("[Simulation verdict per pre-registered rule]")
        print(f"  C1: Δ_mean ≥ +0.10            : {mean_gated_agg:+.4f}  "
              f"{'PASS' if c1 else 'FAIL'}")
        print(f"  C3a: ≥7/8 scenes Δ_gated ≥ 0  : {n_gated_pos}/8  "
              f"{'PASS' if c3a else 'FAIL'}")
        h_str = f"{horns_gated:+.4f}" if horns_gated is not None else "?"
        print(f"  C3b: horns Δ_gated ≥ −0.05    : {h_str}  "
              f"{'PASS' if c3b else 'FAIL'}")
        SIM_PASS = c1 and c3a and c3b
        print()
        if SIM_PASS:
            print("  ✅ SIMULATION_PASS — projection clears bar.")
            print("     Action: COMMIT pilot N=24 với gate-by-L1_A3-top-10% (option B).")
            print("     Caveat: real pilot có shrinkage 20-50% so với simulation")
            print("     (ensemble upper bound vs training-time single model).")
        elif mean_gated_agg < 0.05:
            print("  ❌ SIMULATION_FAIL (rõ ràng) — projection xa bar.")
            print("     Action: SKIP gate pilot → push C1b (option A).")
        else:
            print("  ⚠️  SIMULATION_MARGINAL — projection near bar.")
            print(f"     Δ_gated={mean_gated_agg:+.4f} dB, bar +0.10.")
            print("     Real pilot likely below; user quyết.")
        print("─" * 60)
        print("CAVEATS (locked):")
        print("  (a) N=8 magnitude triage, not stat.")
        print("  (b) Per-view L1_A3 measured on TEST; in-train gate uses TRAIN view L1.")
        print("  (c) Simulation = ENSEMBLE upper bound (2 trained models switched).")
        print("      Real gate-trained model: 1 model with masked loss → shrinkage.")
        print("  (d) atomicAdd noise floor ±0.05/scene paired, mitigated 3-seed.")
        print("  FORBID: 'p<0.05', 'commit guaranteed', '% explained'.")
        print("─" * 60)

    print("\nDone.")


if __name__ == "__main__":
    main()
