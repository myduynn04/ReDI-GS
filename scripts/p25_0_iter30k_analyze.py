#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 25_0] 30k iter scaling FULL N=24 ANALYZE
# File: scripts/p25_0_iter30k_analyze.py
#
# Mục đích: Compare 30k recipe results với 10k anchors:
#   (a) Phase 22 anchor pilot N=24 = 21.918 (10k)
#   (b) Phase 24 c0_base N=24 = 21.882 (10k re-run)
#   (c) Phase 24 post-cleanup N=24 = 21.869 (10k post-cleanup)
#   (d) 10k mean of (a,b,c) = 21.890
#
# Hypothesis: PSNR(30k) ≈ 21.89 ± 0.10 → saturation confirmed
# Alt H1: PSNR(30k) > 21.99 → 30k significantly better (switch to 30k)
# Alt H2: PSNR(30k) < 21.79 → 30k overfit (10k optimal)
#
# Usage:
#   python scripts/p25_0_iter30k_analyze.py
# ============================================================

import re
import os
import sys
import numpy as np

LOG_ROOT = "logs/p25_0_iter30k_full"
SCENES = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]
SEEDS = [42, 137, 9999]

# 10k anchors (3 N=24 runs cùng recipe)
P22_PILOT = 21.918       # Phase 22 pilot (2026-05-25)
P24_C0BASE = 21.882      # Phase 24 c0_base (2026-05-29 pre-cleanup)
P24_POSTCLEAN = 21.869   # Phase 24 post-cleanup (2026-05-29)
ANCHOR_10K_MEAN = np.mean([P22_PILOT, P24_C0BASE, P24_POSTCLEAN])
ANCHOR_10K_SEM = np.std([P22_PILOT, P24_C0BASE, P24_POSTCLEAN], ddof=1) / np.sqrt(3)

# External SOTA references
BINOCULAR_30K = 21.44    # Binocular3DGS LLFF 3-view (verified 30k)
DOC_GS = 21.38           # iter unknown
ICO_GS = 22.20           # iter unknown

# Per-scene Phase 22 anchor (for per-scene comparison)
P22_PER_SCENE = {
    "fern": 23.84, "flower": 21.41, "fortress": 25.57, "horns": 21.08,
    "leaves": 19.38, "orchids": 17.58, "room": 22.97, "trex": 23.51,
}

PSNR_RE = re.compile(r"Best test PSNR.*?(\d+\.\d+)")


def load_psnr(seed, scene):
    log_path = f"{LOG_ROOT}/A3_seed{seed}_{scene}.log"
    if not os.path.isfile(log_path):
        return None
    with open(log_path) as f:
        content = f.read()
    matches = PSNR_RE.findall(content)
    if not matches:
        return None
    return float(matches[-1])


def bootstrap_ci(values, n_resample=10000, alpha=0.05, seed=42):
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_resample, len(values)), replace=True)
    means = samples.mean(axis=1)
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return lo, hi


def main():
    # Load
    psnr_table = {}
    for seed in SEEDS:
        for scene in SCENES:
            psnr_table[(seed, scene)] = load_psnr(seed, scene)

    expected = len(SEEDS) * len(SCENES)
    done = sum(1 for v in psnr_table.values() if v is not None)
    print("=" * 75)
    print("COVERAGE")
    print("=" * 75)
    print(f"  Done: {done}/{expected}")
    if done < expected:
        print("  Missing:")
        for (s, sc), v in psnr_table.items():
            if v is None:
                print(f"    seed{s} {sc}")
    print()

    # Per-scene 3-seed mean
    print("=" * 75)
    print("PER-SCENE 30k PSNR vs Phase 22 10k reference")
    print("=" * 75)
    print(f"{'scene':<10} {'30k mean':<11} {'10k ref':<10} {'Δ vs 10k':<11} {'note'}")
    print("-" * 75)
    all_30k = []
    for scene in SCENES:
        vals = [psnr_table.get((s, scene)) for s in SEEDS]
        vals = [v for v in vals if v is not None]
        if not vals:
            print(f"{scene:<10} {'--':<11} {P22_PER_SCENE[scene]:<10.2f} {'no data':<11}")
            continue
        m = np.mean(vals)
        all_30k.extend(vals)
        ref = P22_PER_SCENE[scene]
        d = m - ref
        floor = 1.3 / np.sqrt(len(vals))
        if abs(d) <= floor:
            note = "≈"
        elif d > 0:
            note = "↑ 30k higher"
        else:
            note = "↓ 30k lower"
        print(f"{scene:<10} {m:<11.3f} {ref:<10.2f} {d:+.3f}     {note}")
    print()

    if not all_30k:
        print("❌ No 30k data — script not run?")
        sys.exit(1)

    # Overall
    mean_30k = np.mean(all_30k)
    N = len(all_30k)
    ci_lo, ci_hi = bootstrap_ci(all_30k)

    print("=" * 75)
    print("OVERALL — 30k vs 10k anchors + SOTA")
    print("=" * 75)
    print(f"  30k mean (N={N}):    {mean_30k:.3f}  95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
    print()
    print(f"  10k anchors:")
    print(f"    Phase 22 pilot:        {P22_PILOT:.3f}")
    print(f"    Phase 24 c0_base:      {P24_C0BASE:.3f}")
    print(f"    Phase 24 post-cleanup: {P24_POSTCLEAN:.3f}")
    print(f"    10k mean of 3 runs:    {ANCHOR_10K_MEAN:.3f} ± {ANCHOR_10K_SEM:.3f}")
    print()
    print(f"  External SOTA (LLFF 3-view):")
    print(f"    Binocular3DGS (30k):   {BINOCULAR_30K:.2f}")
    print(f"    DOC-GS (iter unknown): {DOC_GS:.2f}")
    print(f"    ICO-GS (iter unknown): {ICO_GS:.2f}")
    print()

    # Δ analyses
    d_30k_vs_10k = mean_30k - ANCHOR_10K_MEAN
    d_30k_vs_binocular = mean_30k - BINOCULAR_30K
    print("=" * 75)
    print("KEY DELTAS")
    print("=" * 75)
    print(f"  Δ(30k vs 10k mean):           {d_30k_vs_10k:+.3f}  (noise floor ±0.10)")
    print(f"  Δ(30k vs Binocular3DGS 30k):  {d_30k_vs_binocular:+.3f}  (fair, same budget)")
    print()

    # Verdict
    print("=" * 75)
    print("VERDICT — 30k vs 10k hypothesis")
    print("=" * 75)
    if abs(d_30k_vs_10k) <= 0.10:
        print(f"  ✅ SATURATION CONFIRMED — 30k ≈ 10k (|Δ|={abs(d_30k_vs_10k):.3f} ≤ 0.10)")
        print(f"  → Sparse-view 3-view saturates around iter 5000-10000")
        print(f"  → STRONGEST narrative: 'we match SOTA at 3× less compute'")
        print(f"  → Paper claim: Δ(ours 10k 21.89 vs Binocular 30k 21.44) = +0.45 @ 3× less compute")
    elif d_30k_vs_10k > 0.10:
        print(f"  ⚠ 30k SIGNIFICANTLY BETTER (Δ={d_30k_vs_10k:+.3f} > +0.10)")
        print(f"  → Consider switching to 30k recipe for paper")
        print(f"  → New best: {mean_30k:.3f}")
    else:
        print(f"  ⚠ 30k WORSE (Δ={d_30k_vs_10k:+.3f} < -0.10)")
        print(f"  → 10k is optimal — 30k overfits")
        print(f"  → Defense argument: '10k is empirically optimal, 30k overfit'")
    print()

    # vs Binocular fair compare
    print("=" * 75)
    print("VS BINOCULAR3DGS (same 30k budget — fair apples-to-apples)")
    print("=" * 75)
    if d_30k_vs_binocular > 0:
        print(f"  ✅ ours 30k {mean_30k:.3f} > Binocular 30k {BINOCULAR_30K:.2f}  (Δ +{d_30k_vs_binocular:.3f})")
    else:
        print(f"  ⚠ ours 30k {mean_30k:.3f} < Binocular 30k {BINOCULAR_30K:.2f}  (Δ {d_30k_vs_binocular:+.3f})")
    print()

    print("DONE.")


if __name__ == "__main__":
    main()
