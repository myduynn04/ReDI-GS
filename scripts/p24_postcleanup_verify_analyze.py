#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 24 POST-CLEANUP VERIFY] Analyze
# File: scripts/p24_postcleanup_verify_analyze.py
#
# Mục đích: So sánh Phase 22 A3-TRIM 8-module post-cleanup result với:
#   (a) Phase 22 anchor N=24 = 21.918 (pilot 2026-05-25)
#   (b) Phase 24 c0_base N=24 = 21.882 (pre-cleanup re-run 2026-05-29)
#
# PASS criteria:
#   - Drift |mean − 21.918| ≤ 0.10 (within noise floor)
#   - 8/8 scenes present
#   - Per-scene match Phase 22 references ± noise floor
#
# Usage: python scripts/p24_postcleanup_verify_analyze.py
# ============================================================

import re
import os
import sys
import numpy as np

LOG_ROOT = "logs/p24_postcleanup_verify"
SCENES = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]

# Phase 22 anchor PSNR (N=24 pilot 2026-05-25)
PHASE22_ANCHOR_MEAN = 21.918
PHASE22_PER_SCENE = {
    "fern": 23.84, "flower": 21.41, "fortress": 25.57, "horns": 21.08,
    "leaves": 19.38, "orchids": 17.58, "room": 22.97, "trex": 23.51,
}
# Phase 24 c0_base (N=24 pre-cleanup re-run 2026-05-29)
PHASE24_C0_BASE_MEAN = 21.882

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


def find_seeds():
    """Auto-detect seeds from existing log filenames."""
    if not os.path.isdir(LOG_ROOT):
        return []
    seeds = set()
    pattern = re.compile(r"A3_seed(\d+)_")
    for f in os.listdir(LOG_ROOT):
        m = pattern.match(f)
        if m:
            seeds.add(int(m.group(1)))
    return sorted(seeds)


def main():
    seeds = find_seeds()
    if not seeds:
        print(f"❌ No logs found in {LOG_ROOT}/")
        sys.exit(1)
    print(f"Detected seeds: {seeds}")
    print(f"Mode: {'MULTI N=24' if len(seeds) >= 3 else f'SINGLE N={len(seeds)*8}'}")
    print()

    # Load all
    psnr_table = {}
    for seed in seeds:
        for scene in SCENES:
            psnr_table[(seed, scene)] = load_psnr(seed, scene)

    # Coverage
    expected = len(seeds) * len(SCENES)
    done = sum(1 for v in psnr_table.values() if v is not None)
    print("=" * 70)
    print("COVERAGE")
    print("=" * 70)
    print(f"  Done: {done}/{expected}")
    if done < expected:
        print(f"  ⚠ Missing {expected - done} cells:")
        for (s, sc), v in psnr_table.items():
            if v is None:
                print(f"    seed{s} {sc}")
    print()

    # Per-scene 3-seed mean (or 1-seed value)
    all_psnrs = []
    print("=" * 70)
    print("PER-SCENE PSNR — compare to Phase 22 anchor")
    print("=" * 70)
    print(f"{'scene':<10} {'PSNR (post)':<14} {'Phase 22 ref':<14} {'Δ vs ref':<12} {'verdict'}")
    print("-" * 70)
    for scene in SCENES:
        vals = [psnr_table.get((s, scene)) for s in seeds]
        vals = [v for v in vals if v is not None]
        if not vals:
            print(f"{scene:<10} {'--':<14} {PHASE22_PER_SCENE[scene]:<14.2f} {'no data':<12} —")
            continue
        m = np.mean(vals)
        all_psnrs.extend(vals)
        ref = PHASE22_PER_SCENE[scene]
        d = m - ref
        # Single-scene noise floor ~±1.3 dB single-seed, ±0.5 multi-seed
        floor = 1.3 / np.sqrt(len(vals))
        v = "✓" if abs(d) <= floor else ("⚠" if abs(d) <= floor * 2 else "❌")
        print(f"{scene:<10} {m:<14.3f} {ref:<14.2f} {d:+.3f}      {v}")
    print()

    # Overall mean
    if all_psnrs:
        overall_mean = np.mean(all_psnrs)
        drift_anchor = overall_mean - PHASE22_ANCHOR_MEAN
        drift_c0 = overall_mean - PHASE24_C0_BASE_MEAN
        N = len(all_psnrs)
        # 8-scene paired noise floor: ±0.10 (N=24), ±0.18 (N=8)
        floor = 0.10 if N >= 24 else (0.18 if N >= 8 else 0.5)

        print("=" * 70)
        print("OVERALL DRIFT vs ANCHORS")
        print("=" * 70)
        print(f"  Post-cleanup mean: {overall_mean:.3f} (N={N})")
        print(f"  Phase 22 anchor:   {PHASE22_ANCHOR_MEAN:.3f}  | drift = {drift_anchor:+.3f}")
        print(f"  Phase 24 c0_base:  {PHASE24_C0_BASE_MEAN:.3f}  | drift = {drift_c0:+.3f}")
        print(f"  Noise floor at N={N}: ±{floor:.3f}")
        print()

        # Verdict
        print("=" * 70)
        print("VERDICT")
        print("=" * 70)
        if abs(drift_anchor) <= floor:
            print(f"  ✅ PASS — drift {drift_anchor:+.3f} trong noise floor ±{floor:.2f}")
            print(f"  Phase 22 recipe intact sau cleanup. Safe to proceed.")
        elif abs(drift_anchor) <= floor * 2:
            print(f"  ⚠ MARGINAL — drift {drift_anchor:+.3f} > floor ±{floor:.2f}")
            print(f"  Recommend chạy thêm seeds confirm trước khi tin.")
            if N < 24:
                print(f"  Suggestion: rerun với SEEDS_MODE=multi (3 seeds × 8 = N=24).")
        else:
            print(f"  ❌ FAIL — drift {drift_anchor:+.3f} > 2× floor ({floor*2:.2f})")
            print(f"  Cleanup có thể đã break Phase 22 recipe!")
            print(f"  Recommend: revert files từ .backup/ + debug.")

    print("\nDONE.")


if __name__ == "__main__":
    main()
