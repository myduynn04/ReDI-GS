#!/usr/bin/env python3
"""[CRSGaussian Phase 13] Train-test gap analyzer — measure overfit reduction.

Parse logs/p13_lfcf/{cfg}_seed{seed}_{scene}.log để extract train + test PSNR
ở iter cuối (10000). Compute train-test gap per config.

Insight quan trọng:
  - Train PSNR ↑  = model fit training views tốt hơn
  - Test PSNR ↑   = model generalize tốt hơn (mục tiêu)
  - Gap = Train - Test (cao = overfit nặng)
  - Δ_gap (A3 - A0) < 0  →  A3 GIẢM overfit (good!)
  - Δ_gap (A3 - A0) ≈ 0  →  A3 add capacity, không giảm overfit
  - Δ_gap > 0           →  A3 tăng overfit (bad)

Args (env):
    SEEDS: comma/space-separated (default "42 137 9999")
    SCENES: default 8 LLFF
    LOG_DIR: default logs/p13_lfcf
    CONFIGS: default "A0 A1 A2 A3 A4"

Run:
    python scripts/p13_overfit_analyze.py
    SEEDS="42 137 9999" python scripts/p13_overfit_analyze.py
"""

import os
import re
import statistics


SCENES = os.environ.get("SCENES", "fern flower fortress horns leaves orchids room trex").split()
SEEDS = os.environ.get("SEEDS", "42 137 9999").split()
LOG_DIR = os.environ.get("LOG_DIR", "logs/p13_lfcf")
CONFIGS = os.environ.get("CONFIGS", "A0 A1 A2 A3 A4").split()

# Pattern khớp dòng SUMMARY table — e.g. "10000 |  test |  21.3274 |  0.7653 |..."
# Group: iter, split (test|train), psnr
TABLE_PAT = re.compile(
    r"\b(\d{2,5})\s*\|\s*(test|train)\s*\|\s*([0-9]+\.[0-9]+)\s*\|"
)
# Fallback pattern: "[ITER 10000] Evaluating test: L1 ... PSNR 21.32..."
EVAL_PAT = re.compile(
    r"\[ITER\s+(\d+)\]\s+Evaluating\s+(test|train):\s+L1\s+\S+\s+PSNR\s+([0-9]+\.[0-9]+)"
)


def parse_train_test(log_path):
    """Returns (test_psnr, train_psnr) at largest iter found, or (None, None)."""
    if not os.path.isfile(log_path):
        return None, None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Try SUMMARY table first (cleaner)
    matches = TABLE_PAT.findall(text)
    if not matches:
        # Fallback to [ITER X] Evaluating
        matches = EVAL_PAT.findall(text)
    if not matches:
        return None, None

    # Find largest iter
    iter_to_split = {}  # iter -> {split: psnr}
    for it, split, psnr in matches:
        it = int(it)
        iter_to_split.setdefault(it, {})[split] = float(psnr)
    if not iter_to_split:
        return None, None
    max_iter = max(iter_to_split.keys())
    pair = iter_to_split[max_iter]
    return pair.get("test"), pair.get("train")


def summarize_config(cfg, data):
    """Compute mean train, test, gap, and N for config across seeds×scenes."""
    train_vals = []
    test_vals = []
    gaps = []
    for seed in SEEDS:
        for sc in SCENES:
            entry = data[cfg].get((seed, sc))
            if entry is None:
                continue
            test, train = entry
            if test is None or train is None:
                continue
            train_vals.append(train)
            test_vals.append(test)
            gaps.append(train - test)
    if not gaps:
        return None
    return {
        "N": len(gaps),
        "train_mean": statistics.fmean(train_vals),
        "test_mean": statistics.fmean(test_vals),
        "gap_mean": statistics.fmean(gaps),
        "gap_std": statistics.stdev(gaps) if len(gaps) > 1 else 0.0,
    }


def paired_delta_gap(data, ref_cfg, cmp_cfg):
    """Paired Δ_gap (cmp - ref) per (seed, scene). Returns list of deltas."""
    deltas = []
    for seed in SEEDS:
        for sc in SCENES:
            ref = data[ref_cfg].get((seed, sc))
            cmp = data[cmp_cfg].get((seed, sc))
            if ref is None or cmp is None:
                continue
            r_test, r_train = ref
            c_test, c_train = cmp
            if None in (r_test, r_train, c_test, c_train):
                continue
            ref_gap = r_train - r_test
            cmp_gap = c_train - c_test
            deltas.append(cmp_gap - ref_gap)
    return deltas


def fmt_stat(deltas, label):
    """Print mean ± SEM ± 95% CI for paired deltas."""
    if not deltas:
        print(f"  {label}: NO DATA")
        return
    n = len(deltas)
    mean = statistics.fmean(deltas)
    std = statistics.stdev(deltas) if n > 1 else 0.0
    sem = std / (n ** 0.5)
    ci_lo = mean - 1.96 * sem
    ci_hi = mean + 1.96 * sem
    # Negative = LESS overfit (good)
    direction = ("🎯 LESS OVERFIT" if mean < -0.10
                 else "📉 MORE OVERFIT" if mean > 0.10
                 else "🔇 NEUTRAL")
    print(f"  {label:<26}  N={n:<3}  Δ_gap={mean:+.4f}  SEM={sem:.4f}  "
          f"95% CI=[{ci_lo:+.3f}, {ci_hi:+.3f}]  {direction}")


def main():
    print("=== Phase 13 LFCF Train-Test Gap Analyzer (overfit reduction check) ===")
    print(f"LOG_DIR={LOG_DIR}  SEEDS={SEEDS}  SCENES={SCENES}  CONFIGS={CONFIGS}")

    # Collect: data[cfg][(seed, scene)] = (test, train)
    data = {cfg: {} for cfg in CONFIGS}
    n_total = n_parsed = 0
    for cfg in CONFIGS:
        for seed in SEEDS:
            for sc in SCENES:
                n_total += 1
                p = f"{LOG_DIR}/{cfg}_seed{seed}_{sc}.log"
                test, train = parse_train_test(p)
                data[cfg][(seed, sc)] = (test, train) if (test or train) else None
                if test is not None and train is not None:
                    n_parsed += 1
    print(f"\nParsed {n_parsed}/{n_total} (cfg, seed, scene) cells with full train+test data")

    # ── Per-config summary ──
    print("\n=== Per-config train/test/gap means ===")
    print(f"{'config':<8}  {'N':<3}  {'train':>8}  {'test':>8}  {'gap':>8}  {'gap_std':>8}")
    print("-" * 60)
    cfg_stats = {}
    for cfg in CONFIGS:
        s = summarize_config(cfg, data)
        if s is None:
            print(f"{cfg:<8}  NO DATA")
            continue
        cfg_stats[cfg] = s
        print(f"{cfg:<8}  {s['N']:<3}  {s['train_mean']:>8.3f}  {s['test_mean']:>8.3f}  "
              f"{s['gap_mean']:>8.3f}  {s['gap_std']:>8.3f}")

    # ── Paired Δ_gap (each cfg vs A0) ──
    print("\n=== Paired Δ_gap vs A0 (negative = LESS overfit) ===")
    if "A0" in cfg_stats:
        for cfg in CONFIGS:
            if cfg == "A0":
                continue
            deltas = paired_delta_gap(data, "A0", cfg)
            fmt_stat(deltas, f"Δ_gap ({cfg} − A0)")
    else:
        print("  A0 NO DATA — skip paired comparison")

    # ── Per-scene gap breakdown (for top configs A0, A3, A4) ──
    print("\n=== Per-scene train-test gap (across 3 seeds, A0 vs A3 vs A4) ===")
    print(f"{'scene':<10}  {'A0 gap':>8}  {'A3 gap':>8}  {'A4 gap':>8}  "
          f"{'A3−A0':>8}  {'A4−A0':>8}")
    print("-" * 70)
    for sc in SCENES:
        row = {}
        for cfg in ["A0", "A3", "A4"]:
            gaps_sc = []
            for seed in SEEDS:
                entry = data.get(cfg, {}).get((seed, sc))
                if entry and entry[0] is not None and entry[1] is not None:
                    gaps_sc.append(entry[1] - entry[0])
            row[cfg] = statistics.fmean(gaps_sc) if gaps_sc else None
        if all(row[c] is not None for c in ["A0", "A3", "A4"]):
            d_a3 = row["A3"] - row["A0"]
            d_a4 = row["A4"] - row["A0"]
            print(f"{sc:<10}  {row['A0']:>8.3f}  {row['A3']:>8.3f}  {row['A4']:>8.3f}  "
                  f"{d_a3:>+8.3f}  {d_a4:>+8.3f}")

    # ── Interpretation ──
    print("\n=== Interpretation ===")
    if "A0" in cfg_stats and "A3" in cfg_stats:
        a0_gap = cfg_stats["A0"]["gap_mean"]
        a3_gap = cfg_stats["A3"]["gap_mean"]
        a0_train = cfg_stats["A0"]["train_mean"]
        a3_train = cfg_stats["A3"]["train_mean"]
        a0_test = cfg_stats["A0"]["test_mean"]
        a3_test = cfg_stats["A3"]["test_mean"]
        d_train = a3_train - a0_train
        d_test = a3_test - a0_test
        d_gap = a3_gap - a0_gap
        print(f"  A3 vs A0:")
        print(f"    Δ train PSNR     = {d_train:+.3f}  ({'higher' if d_train > 0 else 'lower'})")
        print(f"    Δ test  PSNR     = {d_test:+.3f}   ({'higher' if d_test > 0 else 'lower'})")
        print(f"    Δ gap (overfit)  = {d_gap:+.3f}   ({'LESS overfit' if d_gap < 0 else 'MORE overfit'})")
        print()
        if d_test > 0 and d_train < 0:
            print("  ✓ Pattern: A3 GIẢM train PSNR + TĂNG test PSNR")
            print("    → A3 thật sự ANTI-OVERFIT (model less memorize, better generalize)")
        elif d_test > 0 and d_train > 0:
            if d_test > d_train:
                print("  ✓ Pattern: A3 TĂNG cả train+test, NHƯNG test tăng nhiều hơn")
                print("    → A3 add capacity HỢP LÝ (test gain > train gain)")
            else:
                print("  ⚠ Pattern: A3 TĂNG train nhiều hơn test")
                print("    → A3 mostly add capacity, gain ít chống overfit")
        elif d_test < 0 and d_train < 0:
            print("  ⚠ Pattern: A3 GIẢM cả train+test")
            print("    → A3 under-capacity, not good direction")
        else:
            print("  ⚠ Pattern non-standard — check per-scene breakdown")


if __name__ == "__main__":
    main()
