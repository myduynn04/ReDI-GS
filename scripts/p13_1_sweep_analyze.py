#!/usr/bin/env python3
"""[CRSGaussian Phase 13.1] LFCF intensity sweep analyzer.

Compare 3 sweep variants (M/H/X) vs A3 baseline (existing Round 1+2 data).

Variants:
  Base (A3) = scaler=1.5, interval=2 (Phase 13 confirmed winner, reuse logs/p13_lfcf/A3_seed42_*)
  M (medium) = scaler=1.8, interval=2
  H (high)   = scaler=2.0, interval=1
  X (extra)  = scaler=2.5, interval=1

Metrics:
  - Test PSNR per variant
  - Train PSNR (overfit check)
  - Δ_test vs A3 baseline (paired per scene, seed 42)
  - Δ_train vs A3 (capacity check)
  - Δ_gap vs A3 (overfit reduction check)

Decision logic (single-seed N=8 — per memory project_3dgs_variance_floor.md):
  Best variant Δ_test ≥ +0.10 → REQUIRED multi-seed verify before adopt
  Best variant +0.05..+0.10  → marginal, multi-seed verify recommended
  Best variant < +0.05       → A3 base optimal, lock recipe

Run:
    python scripts/p13_1_sweep_analyze.py
    SCENES="fern flower ..." python scripts/p13_1_sweep_analyze.py
"""

import os
import re
import statistics

SCENES = os.environ.get("SCENES", "fern flower fortress horns leaves orchids room trex").split()
SEED = os.environ.get("SEED", "42")  # sweep chỉ seed 42
SWEEP_LOG_DIR = os.environ.get("SWEEP_LOG_DIR", "logs/p13_1_sweep")
A3_LOG_DIR = os.environ.get("A3_LOG_DIR", "logs/p13_lfcf")  # baseline data

VARIANTS = ['Base', 'M', 'H', 'X']
VARIANT_DESCRIPTIONS = {
    'Base': 'scaler=1.5, interval=2 (A3 reference)',
    'M':    'scaler=1.8, interval=2 (medium)',
    'H':    'scaler=2.0, interval=1 (high)',
    'X':    'scaler=2.5, interval=1 (extra)',
}

PSNR_PAT = re.compile(r"Best test PSNR:\s*([\d\.eE\-\+]+)")
TABLE_PAT = re.compile(
    r"\b(\d{2,5})\s*\|\s*(test|train)\s*\|\s*([0-9]+\.[0-9]+)\s*\|"
)

PHASE_8_FAIR_BASELINE = 21.160
A3_BASE_NUMBER = 21.330   # multi-seed confirmed


def get_log_path(variant, scene):
    """Map variant + scene to log path."""
    if variant == 'Base':
        return f"{A3_LOG_DIR}/A3_seed{SEED}_{scene}.log"
    return f"{SWEEP_LOG_DIR}/{variant}_seed{SEED}_{scene}.log"


def parse_test_train(log_path):
    """Returns (test_psnr, train_psnr) at largest iter found."""
    if not os.path.isfile(log_path):
        return None, None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    matches = TABLE_PAT.findall(text)
    if not matches:
        return None, None
    iter_to_split = {}
    for it, split, psnr in matches:
        iter_to_split.setdefault(int(it), {})[split] = float(psnr)
    if not iter_to_split:
        return None, None
    max_iter = max(iter_to_split.keys())
    return iter_to_split[max_iter].get("test"), iter_to_split[max_iter].get("train")


def fmt(v, prec=3):
    return f"{v:.{prec}f}" if v is not None else "    -"


def main():
    print("=== Phase 13.1 LFCF Intensity Sweep Analyzer ===")
    print(f"SEED={SEED}  SCENES={SCENES}")
    print(f"Base = A3 reference từ {A3_LOG_DIR}/A3_seed{SEED}_*.log")
    print(f"M/H/X = sweep variants từ {SWEEP_LOG_DIR}/{{M,H,X}}_seed{SEED}_*.log")
    print()
    for v in VARIANTS:
        print(f"  {v:<5} = {VARIANT_DESCRIPTIONS[v]}")
    print()

    # Collect: data[variant][scene] = (test, train)
    data = {v: {} for v in VARIANTS}
    parsed = 0
    total = len(VARIANTS) * len(SCENES)
    for v in VARIANTS:
        for sc in SCENES:
            path = get_log_path(v, sc)
            test, train = parse_test_train(path)
            data[v][sc] = (test, train)
            if test is not None and train is not None:
                parsed += 1
    print(f"Parsed {parsed}/{total} (variant, scene) cells with full test+train data")

    # ── Per-scene PSNR table ──
    print("\n=== Per-scene PSNR by variant (seed 42) ===")
    print(f"{'scene':<10}  {'Base test':>10}  {'M test':>9}  {'H test':>9}  {'X test':>9}  "
          f"|  {'Base train':>11}  {'M train':>9}  {'H train':>9}  {'X train':>9}")
    print("-" * 110)
    for sc in SCENES:
        row = f"{sc:<10}  "
        # Test PSNRs
        for v in VARIANTS:
            test, _ = data[v].get(sc, (None, None))
            row += f"  {fmt(test):>9}"
        row += "  |  "
        # Train PSNRs
        for v in VARIANTS:
            _, train = data[v].get(sc, (None, None))
            row += f"  {fmt(train):>9}"
        print(row)

    # ── Per-variant 8-scene means ──
    print("\n=== Per-variant 8-scene means ===")
    print(f"{'variant':<8}  {'N':>3}  {'test mean':>10}  {'train mean':>11}  {'gap':>8}  {'vs A3 paper':>11}")
    print("-" * 70)
    var_stats = {}
    for v in VARIANTS:
        tests = [data[v][sc][0] for sc in SCENES if data[v].get(sc) and data[v][sc][0] is not None]
        trains = [data[v][sc][1] for sc in SCENES if data[v].get(sc) and data[v][sc][1] is not None]
        if not tests or not trains:
            print(f"{v:<8}  NO DATA")
            continue
        test_mean = statistics.fmean(tests)
        train_mean = statistics.fmean(trains)
        gap = train_mean - test_mean
        vs_a3 = test_mean - A3_BASE_NUMBER
        var_stats[v] = {'test': test_mean, 'train': train_mean, 'gap': gap}
        print(f"{v:<8}  {len(tests):>3}  {test_mean:>10.3f}  {train_mean:>11.3f}  "
              f"{gap:>8.3f}  {vs_a3:>+11.3f}")

    # ── Paired Δ vs Base (A3 reference) ──
    print("\n=== Paired Δ vs Base (A3 baseline) — per scene ===")
    print(f"{'scene':<10}  {'M_Δtest':>9}  {'M_Δtrain':>10}  {'M_Δgap':>9}  |  "
          f"{'H_Δtest':>9}  {'H_Δtrain':>10}  {'H_Δgap':>9}  |  "
          f"{'X_Δtest':>9}  {'X_Δtrain':>10}  {'X_Δgap':>9}")
    print("-" * 130)
    deltas = {v: {'test': [], 'train': [], 'gap': []} for v in ['M', 'H', 'X']}
    for sc in SCENES:
        base = data['Base'].get(sc)
        if base is None or None in base:
            continue
        b_test, b_train = base
        b_gap = b_train - b_test
        row = f"{sc:<10}"
        for v in ['M', 'H', 'X']:
            entry = data[v].get(sc)
            if entry is None or None in entry:
                row += f"  {'-':>9}  {'-':>10}  {'-':>9}  |"
                continue
            t, tr = entry
            g = tr - t
            dt = t - b_test
            dtr = tr - b_train
            dg = g - b_gap
            deltas[v]['test'].append(dt)
            deltas[v]['train'].append(dtr)
            deltas[v]['gap'].append(dg)
            row += f"  {dt:>+9.3f}  {dtr:>+10.3f}  {dg:>+9.3f}  |"
        print(row)

    # ── Stats summary ──
    print("\n=== Paired Δ summary vs Base (N=scenes_with_data) ===")
    print(f"{'variant':<8}  {'N':>3}  {'Δtest mean':>11}  {'SEM':>7}  {'95% CI':>22}  {'verdict':>20}")
    print("-" * 100)
    for v in ['M', 'H', 'X']:
        ds = deltas[v]
        if not ds['test']:
            print(f"{v:<8}  NO DATA")
            continue
        n = len(ds['test'])
        mean = statistics.fmean(ds['test'])
        std = statistics.stdev(ds['test']) if n > 1 else 0.0
        sem = std / (n ** 0.5)
        ci_lo = mean - 1.96 * sem
        ci_hi = mean + 1.96 * sem
        sig = abs(mean) > 2 * sem
        # Single-seed N=8 SEM ~0.05-0.07 dB → threshold +0.10 minimum for signal.
        # Per memory project_3dgs_variance_floor.md: min detectable Δ ≈ ±0.10 multi-seed.
        # Per memory feedback_full_8scene_ablation.md: KHÔNG commit single-seed.
        if mean >= 0.10:
            verdict = "🎯 MULTI-SEED VERIFY (≥+0.10 single-seed signal)"
        elif mean >= 0.05:
            verdict = "🟡 MARGINAL — multi-seed needed"
        elif abs(mean) < 0.03:
            verdict = "⚪ neutral — A3 base optimal"
        else:
            verdict = "📉 worse"
        print(f"{v:<8}  {n:>3}  {mean:>+11.4f}  {sem:>7.4f}  "
              f"[{ci_lo:>+7.3f}, {ci_hi:>+7.3f}]  {verdict:>20}")

    print("\n=== Train + Gap Δ summary ===")
    for v in ['M', 'H', 'X']:
        ds = deltas[v]
        if not ds['train']:
            continue
        mean_train = statistics.fmean(ds['train'])
        mean_gap = statistics.fmean(ds['gap'])
        print(f"  {v}: Δtrain_mean={mean_train:+.3f}  Δgap_mean={mean_gap:+.3f}  "
              f"({'less overfit' if mean_gap < -0.05 else 'more overfit' if mean_gap > 0.05 else 'neutral overfit'})")

    # ── Best variant decision ──
    print("\n=== Decision ===")
    valid_variants = [(v, deltas[v]['test']) for v in ['M', 'H', 'X'] if deltas[v]['test']]
    if not valid_variants:
        print("  NO sweep data — run scripts/p13_1_sweep.sh first")
        return

    best_v, best_dts = max(valid_variants, key=lambda x: statistics.fmean(x[1]))
    best_mean = statistics.fmean(best_dts)
    print(f"  Best variant: {best_v}  Δtest mean = {best_mean:+.4f}")
    if best_mean >= 0.10:
        print(f"  🎯 STRONG single-seed signal: {best_v} Δtest mean = {best_mean:+.4f}")
        print(f"  → REQUIRED: multi-seed verify (seeds 137 + 9999) trước khi adopt")
        print(f"  → Nếu pooled N=24 Δ ≥ +0.10 → replace A3 as Phase 13 FULL recipe")
    elif best_mean >= 0.05:
        print(f"  🟡 Best variant {best_v} marginal (+0.05..+0.10)")
        print(f"  → Multi-seed verify trên best variant ({best_v}) recommended")
        print(f"  → Nếu pooled Δ < +0.10 → A3 base remains optimal")
    else:
        print(f"  ⚪ All sweeps < +0.05 over A3 — A3 base optimal")
        print(f"  → Lock A3 21.330 as Phase 13 FULL recipe (no further sweep)")


if __name__ == "__main__":
    main()
