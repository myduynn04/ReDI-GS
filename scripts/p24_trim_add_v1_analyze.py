#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 24] Analyze trim-add re-test on RoMa v1 backbone
# File: scripts/p24_trim_add_v1_analyze.py
#
# Mục đích: tính paired Δ vs c0_base anchor cho 4 test configs
# (c1_informed, c2_crsprune, c3_rvisible, c4_stack3) + bootstrap
# 95% CI để verdict HELP/HURT/WASH.
#
# Verdict criteria (project standard):
#   ✅ HELP:  Δ_mean ≥ +0.10  AND  CI_lo > 0
#   ❌ HURT:  Δ_mean ≤ −0.10  AND  CI_hi < 0
#   ~ WASH:  else
#
# Drift sanity check:
#   c0_base mean (this run) phải ≈ 21.918 (Phase 22 anchor) ±0.10
#   Nếu lệch > 0.10 → code drift, kiểm tra trước khi tin Δ
#
# Usage:
#   python scripts/p24_trim_add_v1_analyze.py
# ============================================================

import re
import os
import sys
import numpy as np

# ── Config ──
LOG_ROOT = "logs/p24_trim_add_v1"
CONFIGS = ["c0_base", "c1_informed", "c2_crsprune", "c3_rvisible", "c4_stack3"]
SCENES = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]
SEEDS = [42, 137, 9999]
PHASE22_ANCHOR = 21.918  # Phase 22 pilot N=24 PROJECT BEST

PSNR_RE = re.compile(r"Best test PSNR.*?(\d+\.\d+)")


def load_psnr(cfg, seed, scene):
    """Extract last 'Best test PSNR' value from log."""
    log_path = f"{LOG_ROOT}/{cfg}/A3_seed{seed}_{scene}.log"
    if not os.path.isfile(log_path):
        return None
    with open(log_path) as f:
        content = f.read()
    matches = PSNR_RE.findall(content)
    if not matches:
        return None
    return float(matches[-1])


def bootstrap_ci(deltas, n_resample=10000, alpha=0.05, seed=42):
    """Bootstrap 95% CI on mean of paired deltas."""
    deltas = np.asarray(deltas)
    rng = np.random.default_rng(seed)
    samples = rng.choice(deltas, size=(n_resample, len(deltas)), replace=True)
    means = samples.mean(axis=1)
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return lo, hi


def verdict(delta_mean, ci_lo, ci_hi):
    if delta_mean >= 0.10 and ci_lo > 0:
        return "✅ HELP"
    if delta_mean <= -0.10 and ci_hi < 0:
        return "❌ HURT"
    return "~ WASH"


def main():
    # ── Load all logs ──
    psnr_table = {}
    for cfg in CONFIGS:
        psnr_table[cfg] = {}
        for seed in SEEDS:
            for scene in SCENES:
                psnr_table[cfg][(seed, scene)] = load_psnr(cfg, seed, scene)

    # ── Coverage check ──
    print("=" * 80)
    print("COVERAGE (target: 24 per config)")
    print("=" * 80)
    total = 0
    for cfg in CONFIGS:
        done = sum(1 for v in psnr_table[cfg].values() if v is not None)
        total += done
        status = "✓" if done == 24 else f"⚠ missing {24 - done}"
        print(f"  {cfg:<14} {done}/24  {status}")
    print(f"  TOTAL        {total}/120")
    print()

    # ── Drift sanity check ──
    base_vals_all = [v for v in psnr_table["c0_base"].values() if v is not None]
    if not base_vals_all:
        print("❌ c0_base no data — cannot proceed")
        sys.exit(1)
    c0_mean = np.mean(base_vals_all)
    drift = c0_mean - PHASE22_ANCHOR
    print("=" * 80)
    print("DRIFT SANITY CHECK")
    print("=" * 80)
    print(f"  Phase 22 anchor (pilot N=24): {PHASE22_ANCHOR:.3f}")
    print(f"  c0_base re-run mean:          {c0_mean:.3f}")
    print(f"  Drift:                        {drift:+.3f}")
    if abs(drift) <= 0.10:
        print(f"  ✓ Drift within noise floor (±0.10), anchor consistent")
    else:
        print(f"  ⚠ Drift > 0.10 — code/GPU state may differ from Phase 22 pilot")
        print(f"    Δ comparisons still valid (paired) but reproduction questionable")
    print()

    # ── Per-config summary ──
    print("=" * 80)
    print("PER-CONFIG SUMMARY (paired vs c0_base)")
    print("=" * 80)
    print(f"{'config':<14} {'N':<4} {'PSNR':<9} {'Δ':<9} {'95% CI':<22} {'verdict'}")
    print("-" * 80)
    base_psnrs = psnr_table["c0_base"]
    for cfg in CONFIGS:
        cfg_psnrs = psnr_table[cfg]
        # Paired: (seed, scene) where BOTH cfg and c0_base have values
        keys = [k for k in cfg_psnrs if cfg_psnrs[k] is not None and base_psnrs[k] is not None]
        if not keys:
            print(f"  {cfg}: no paired data")
            continue
        cfg_vals = [cfg_psnrs[k] for k in keys]
        cfg_mean = np.mean(cfg_vals)
        if cfg == "c0_base":
            print(f"{cfg:<14} {len(keys):<4} {cfg_mean:<9.3f} {'anchor':<9} {'—':<22} —")
            continue
        deltas = [cfg_psnrs[k] - base_psnrs[k] for k in keys]
        d_mean = np.mean(deltas)
        ci_lo, ci_hi = bootstrap_ci(deltas)
        v = verdict(d_mean, ci_lo, ci_hi)
        ci_str = f"[{ci_lo:+.3f}, {ci_hi:+.3f}]"
        print(f"{cfg:<14} {len(keys):<4} {cfg_mean:<9.3f} {d_mean:+.3f}    {ci_str:<22} {v}")
    print()

    # ── Per-scene PSNR table ──
    print("=" * 80)
    print("PER-SCENE PSNR (3-seed mean)")
    print("=" * 80)
    header = f"{'scene':<10}" + "".join(f"{c:<13}" for c in CONFIGS)
    print(header)
    print("-" * len(header))
    for scene in SCENES:
        row = f"{scene:<10}"
        for cfg in CONFIGS:
            vals = [psnr_table[cfg].get((s, scene)) for s in SEEDS]
            vals = [v for v in vals if v is not None]
            if vals:
                row += f"{np.mean(vals):<13.3f}"
            else:
                row += f"{'--':<13}"
        print(row)
    print()

    # ── Per-scene Δ vs c0_base ──
    print("=" * 80)
    print("PER-SCENE Δ vs c0_base (3-seed mean)")
    print("=" * 80)
    cfgs_no_base = [c for c in CONFIGS if c != "c0_base"]
    header = f"{'scene':<10}" + "".join(f"{c:<13}" for c in cfgs_no_base)
    print(header)
    print("-" * len(header))
    for scene in SCENES:
        row = f"{scene:<10}"
        base_vals = [psnr_table["c0_base"].get((s, scene)) for s in SEEDS]
        base_vals = [v for v in base_vals if v is not None]
        if not base_vals:
            row += "  (no base)"
        else:
            base_mean = np.mean(base_vals)
            for cfg in cfgs_no_base:
                cfg_vals = [psnr_table[cfg].get((s, scene)) for s in SEEDS]
                cfg_vals = [v for v in cfg_vals if v is not None]
                if cfg_vals:
                    d = np.mean(cfg_vals) - base_mean
                    row += f"{d:+.3f}{'':<7}"
                else:
                    row += f"{'--':<13}"
        print(row)
    print()

    # ── Per-seed paired Δ ──
    print("=" * 80)
    print("PER-SEED PAIRED Δ vs c0_base (8-scene mean per seed)")
    print("=" * 80)
    header = f"{'seed':<8}" + "".join(f"{c:<13}" for c in cfgs_no_base)
    print(header)
    print("-" * len(header))
    for seed in SEEDS:
        row = f"{seed:<8}"
        for cfg in cfgs_no_base:
            ds = []
            for scene in SCENES:
                b = psnr_table["c0_base"].get((seed, scene))
                c = psnr_table[cfg].get((seed, scene))
                if b is not None and c is not None:
                    ds.append(c - b)
            if ds:
                row += f"{np.mean(ds):+.3f}{'':<7}"
            else:
                row += f"{'--':<13}"
        print(row)
    print()

    # ── Decision summary ──
    print("=" * 80)
    print("DECISION SUMMARY")
    print("=" * 80)
    print("Phase 20 TRIM verify on RoMa v1 backbone:")
    print()
    for cfg in cfgs_no_base:
        cfg_psnrs = psnr_table[cfg]
        keys = [k for k in cfg_psnrs if cfg_psnrs[k] is not None and base_psnrs[k] is not None]
        if not keys:
            continue
        deltas = [cfg_psnrs[k] - base_psnrs[k] for k in keys]
        d_mean = np.mean(deltas)
        ci_lo, ci_hi = bootstrap_ci(deltas)
        v = verdict(d_mean, ci_lo, ci_hi)
        action = {
            "✅ HELP":  "→ KHÔNG remove. Update Phase 22 recipe (9-module).",
            "❌ HURT":  "→ Confirm Phase 20 TRIM remove. Insight: v1 cross-confirm âm.",
            "~ WASH":  "→ Confirm Phase 20 TRIM safe to remove. v1 cross-confirm.",
        }[v]
        print(f"  {cfg:<14}  Δ={d_mean:+.3f}  {v}  {action}")
    print()
    print("DONE.")


if __name__ == "__main__":
    main()
