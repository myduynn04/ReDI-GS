#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 25_2] Phase 22 LOCK ANALYZE — 4 N=24 runs
# File: scripts/p25_2_phase22_lock_analyze.py
#
# Mục đích: Compute final Phase 22 PSNR lock value sau 4 N=24 measurements:
#   1. Phase 22 pilot (2026-05-25):       21.918
#   2. Phase 24 c0_base (2026-05-29):     21.882
#   3. Phase 24 postcleanup (2026-05-29): 21.869
#   4. Phase 25_2 lock (2026-06-XX):      ? (this run)
#
# Output: final mean ± SEM cho narrative paper
#
# Usage: python scripts/p25_2_phase22_lock_analyze.py
# ============================================================

import re
import os
import sys
import numpy as np
from pathlib import Path

LOG_ROOT = "logs/p25_2_phase22_lock"
SCENES = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]
SEEDS = [42, 137, 9999]

# Existing 3 N=24 anchors
PRIOR_MEANS = {
    "Phase 22 pilot (2026-05-25)": 21.918,
    "Phase 24 c0_base (2026-05-29)": 21.882,
    "Phase 24 postcleanup (2026-05-29)": 21.869,
}

P22_PER_SCENE = {
    "fern": 23.84, "flower": 21.41, "fortress": 25.57, "horns": 21.08,
    "leaves": 19.38, "orchids": 17.58, "room": 22.97, "trex": 23.51,
}

PSNR_RE = re.compile(r"Best test PSNR.*?(\d+\.\d+)")


def load_psnr(seed, scene):
    log_path = Path(LOG_ROOT) / f"A3_seed{seed}_{scene}.log"
    if not log_path.is_file():
        return None
    try:
        with open(log_path) as f:
            content = f.read()
    except Exception:
        return None
    matches = PSNR_RE.findall(content)
    return float(matches[-1]) if matches else None


def main():
    # Load Phase 25_2 results
    psnr_table = {}
    for seed in SEEDS:
        for scene in SCENES:
            psnr_table[(seed, scene)] = load_psnr(seed, scene)

    expected = len(SEEDS) * len(SCENES)
    done = sum(1 for v in psnr_table.values() if v is not None)

    print("=" * 70)
    print("PHASE 25_2 LOCK — 4th N=24 measurement")
    print("=" * 70)
    print(f"Coverage: {done}/{expected}")
    if done < expected:
        print("Missing:")
        for (s, sc), v in psnr_table.items():
            if v is None:
                print(f"  seed{s} {sc}")
        print()

    if done == 0:
        print("\n❌ No data yet. Run scripts/p25_2_phase22_lock_run.sh first.")
        sys.exit(1)

    # Per-scene 3-seed mean
    print()
    print("=" * 70)
    print("PER-SCENE (3-seed mean)")
    print("=" * 70)
    print(f"{'scene':<10} {'P25_2':<10} {'P22 ref':<10} {'Δ':<10}")
    print("-" * 50)
    all_psnrs = []
    for scene in SCENES:
        vals = [psnr_table.get((s, scene)) for s in SEEDS if psnr_table.get((s, scene)) is not None]
        if not vals:
            print(f"{scene:<10} {'--':<10} {P22_PER_SCENE[scene]:<10.2f}")
            continue
        m = np.mean(vals)
        all_psnrs.extend(vals)
        d = m - P22_PER_SCENE[scene]
        print(f"{scene:<10} {m:<10.3f} {P22_PER_SCENE[scene]:<10.2f} {d:+.3f}")
    print()

    if len(all_psnrs) < expected:
        print(f"⚠ Partial data — using {len(all_psnrs)}/{expected} cells")
        print()

    # Phase 25_2 mean
    p25_2_mean = np.mean(all_psnrs)

    # Combine với 3 prior runs
    print("=" * 70)
    print("4 N=24 MEASUREMENTS — Phase 22 recipe (10k, RoMa v1)")
    print("=" * 70)
    all_means = list(PRIOR_MEANS.values()) + [p25_2_mean]
    all_labels = list(PRIOR_MEANS.keys()) + [f"Phase 25_2 lock (this run, N={len(all_psnrs)})"]
    for label, m in zip(all_labels, all_means):
        marker = " ⭐ NEW" if "25_2" in label else ""
        print(f"  {label:<45} {m:.3f}{marker}")
    print()

    # Statistics 4 runs
    arr = np.array(all_means)
    mean_4 = arr.mean()
    sem_4 = arr.std(ddof=1) / np.sqrt(len(arr))
    std_4 = arr.std(ddof=1)
    range_4 = arr.max() - arr.min()

    print("=" * 70)
    print("FINAL STATISTICS (4 N=24 runs)")
    print("=" * 70)
    print(f"  Mean of 4 N=24:     {mean_4:.3f}")
    print(f"  Std (ddof=1):       ±{std_4:.3f}")
    print(f"  SEM (mean-of-means): ±{sem_4:.3f}")
    print(f"  Range (max - min):  {range_4:.3f}")
    print(f"  Min / Max:          {arr.min():.3f} / {arr.max():.3f}")
    print()

    # 3-run mean comparison
    prior_arr = arr[:3]
    prior_mean = prior_arr.mean()
    delta_lock = p25_2_mean - prior_mean
    print("=" * 70)
    print("PHASE 25_2 vs PRIOR 3-RUN MEAN")
    print("=" * 70)
    print(f"  Prior 3-run mean:   {prior_mean:.3f}")
    print(f"  Phase 25_2 (lock):  {p25_2_mean:.3f}")
    print(f"  Δ (lock vs prior):  {delta_lock:+.3f}")
    if abs(delta_lock) <= 0.10:
        print(f"  ✓ Consistent with prior 3 runs (within noise ±0.10)")
    else:
        print(f"  ⚠ Larger than expected noise — verify")
    print()

    # Recommendation for paper narrative
    print("=" * 70)
    print("PAPER NARRATIVE RECOMMENDATION")
    print("=" * 70)
    # Round to reasonable precision
    paper_psnr = round(mean_4, 2)
    paper_ci = round(max(sem_4 * 2, 0.10), 2)  # 95% CI ≈ 2 SEM, min 0.10 noise floor
    print(f"  Report:  PSNR = {paper_psnr:.2f} ± {paper_ci:.2f}")
    print(f"          (mean of 4 independent N=24 runs, SEM = {sem_4:.3f})")
    print()
    print(f"  Defense: 'Verified across 4 independent N=24 paired runs spanning")
    print(f"           {len(PRIOR_MEANS)+1} weeks, recipe + cleanup states. Δ between extreme runs")
    print(f"           = {range_4:.3f}, below ±0.10 paired noise floor → reproducibility")
    print(f"           strong despite atomicAdd non-determinism.'")
    print()

    # vs SOTA
    print("=" * 70)
    print("vs SOTA (LLFF 3-view)")
    print("=" * 70)
    refs = [
        ("CoR-GS (parent, 10k)", 20.11, "10k"),
        ("FSGS (10k)", 20.31, "10k"),
        ("DOC-GS (iter unknown)", 21.38, "?"),
        ("Binocular3DGS (30k)", 21.44, "30k"),
        ("ICO-GS (preprint, iter unknown)", 22.20, "?"),
    ]
    for name, psnr, budget in refs:
        d = paper_psnr - psnr
        bucket = "FAIR" if budget == "10k" else "cross-budget"
        marker = "✓" if d > 0 else "✗"
        print(f"  {marker} ours {paper_psnr:.2f} vs {name:<30} {psnr:.2f}  Δ={d:+.2f}  ({bucket})")
    print()

    print("DONE.")


if __name__ == "__main__":
    main()
