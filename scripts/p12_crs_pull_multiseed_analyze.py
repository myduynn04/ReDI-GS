#!/usr/bin/env python3
# ============================================================
# DEPRECATED 2026-05-12: Phase 12 CRS-pull REJECTED — Δ_A1=−0.027, Δ_A2=−0.033, Δ_A3=−0.096 N=8
# Flags removed from arguments/__init__.py — script no longer runnable.
# Module utils/loss/crs_pull.py deleted. Kept for paper reproducibility evidence.
# ============================================================
"""[CRSGaussian Phase 12] Multi-seed analyzer — CRS-pull (4 configs).

Configs:
  A0 = Phase 8 FULL baseline
  A1 = + CRS-pull (full: pull + scale + opacity)
  A2 = + CRS-pull − Phase 4 CRS pruning (replace prune)
  A3 = + CRS-pull pull-only (no scale, no opacity)

3 paired Δ comparisons vs A0:
  Δ_A1 = A1 − A0   (Full CRS-pull effect)
  Δ_A2 = A2 − A0   (CRS-pull replace prune effect)
  Δ_A3 = A3 − A0   (Pull-only isolation)

Round 1 (adaptive): seed 42 only → N=8 paired per Δ.
Round 2 (if winner): seeds 137+9999 with best config → N=24 total.

Run:
    python scripts/p12_crs_pull_multiseed_analyze.py
    SEEDS="42 137 9999" SCENES="fern ..." python scripts/p12_crs_pull_multiseed_analyze.py
"""

import os
import re
import statistics

SCENES = os.environ.get("SCENES", "fern flower fortress horns leaves orchids room trex").split()
SEEDS = os.environ.get("SEEDS", "42").split()
LOG_DIR = os.environ.get("LOG_DIR", "logs/p12_crs_pull")

PSNR_PAT = re.compile(r"Best test PSNR:\s*([\d\.eE\-\+]+)")
PHASE_8_PAPER_AVG = 21.335
PHASE_8_FAIR_BASELINE = 21.16     # multi-seed N=24
PHASE_8_PAPER_PER_SCENE = {
    'fern':     23.1267,
    'flower':   21.0358,
    'fortress': 24.5264,
    'horns':    20.3325,
    'leaves':   18.5195,
    'orchids':  16.7605,
    'room':     23.0706,
    'trex':     23.3058,
}

CONFIGS = ['A0', 'A1', 'A2', 'A3']


def parse_psnr(path):
    if not os.path.isfile(path):
        return None
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    m = PSNR_PAT.search(text)
    return float(m.group(1)) if m else None


def fmt(v, prec=3):
    return f"{v:.{prec}f}" if v is not None else "    -"


def summarize_paired(label, deltas):
    """Stats cho 1 paired comparison."""
    if not deltas:
        print(f"  {label}: NO DATA")
        return None, None, None
    n = len(deltas)
    mean = statistics.fmean(deltas)
    std = statistics.stdev(deltas) if n > 1 else 0.0
    sem = std / (n ** 0.5)
    ci_lo = mean - 1.96 * sem
    ci_hi = mean + 1.96 * sem
    sig = abs(mean) > 2 * sem
    sig_label = ("🎯 SIGNIFICANT" if mean > 0 else "📉 SIG NEG") if sig else "🔇 NOT SIG"
    print(f"  {label:<24}  N={n:<3}  mean={mean:+.4f}  SEM={sem:.4f}  "
          f"95% CI=[{ci_lo:+.3f}, {ci_hi:+.3f}]  {sig_label}")
    if mean >= 0.20:
        decision = "🎯 STRONG WINNER (≥+0.20)"
    elif mean >= 0.10:
        decision = "🎯 WINNER (≥+0.10)"
    elif mean >= 0.05:
        decision = "🟡 MARGINAL (+0.05..+0.10)"
    else:
        decision = "❌ REJECT (<+0.05)"
    print(f"  {'':24}  Decision: {decision}")
    return mean, sem, sig


def main():
    print(f"=== Phase 12 CRS-pull multi-seed analyzer ===")
    print(f"LOG_DIR={LOG_DIR}  SEEDS={SEEDS}  SCENES={SCENES}")

    # Collect: data[cfg][scene][seed] = psnr
    data = {cfg: {} for cfg in CONFIGS}
    for cfg in CONFIGS:
        for sc in SCENES:
            data[cfg][sc] = {}
            for seed in SEEDS:
                p = f"{LOG_DIR}/{cfg}_seed{seed}_{sc}.log"
                data[cfg][sc][seed] = parse_psnr(p)

    # ── Per-scene per-seed raw table ──
    print("\n=== Per-scene PSNR by seed (A0/A1/A2/A3) ===")
    header = f"{'scene':<10}"
    for seed in SEEDS:
        for cfg in CONFIGS:
            header += f"  {cfg}_s{seed:<5}"
    print(header)
    print("-" * len(header))
    for sc in SCENES:
        row = f"{sc:<10}"
        for seed in SEEDS:
            for cfg in CONFIGS:
                row += f"  {fmt(data[cfg][sc].get(seed))}   "
        print(row)

    # ── Per-scene mean across seeds ──
    print("\n=== Per-scene mean (across seeds) + Δ vs A0 ===")
    print(f"{'scene':<10}  {'A0':>7}  {'A1':>7}  {'A2':>7}  {'A3':>7}  "
          f"{'Δ_A1':>8}  {'Δ_A2':>8}  {'Δ_A3':>8}  {'P8 paper':>9}")
    print("-" * 90)
    for sc in SCENES:
        means = {}
        for cfg in CONFIGS:
            vals = [v for v in data[cfg][sc].values() if v is not None]
            means[cfg] = statistics.fmean(vals) if vals else None
        if all(means[c] is not None for c in CONFIGS):
            d_a1 = means['A1'] - means['A0']
            d_a2 = means['A2'] - means['A0']
            d_a3 = means['A3'] - means['A0']
            ref = PHASE_8_PAPER_PER_SCENE.get(sc)
            print(f"{sc:<10}  {means['A0']:>7.3f}  {means['A1']:>7.3f}  "
                  f"{means['A2']:>7.3f}  {means['A3']:>7.3f}  "
                  f"{d_a1:>+8.3f}  {d_a2:>+8.3f}  {d_a3:>+8.3f}  {fmt(ref):>9}")

    # ── 8-scene avg per seed ──
    print("\n=== 8-scene avg PSNR per seed ===")
    print(f"{'seed':<8}  {'A0':>8}  {'A1':>8}  {'A2':>8}  {'A3':>8}  "
          f"{'Δ_A1':>8}  {'Δ_A2':>8}  {'Δ_A3':>8}")
    print("-" * 80)
    seed_avgs = {cfg: [] for cfg in CONFIGS}
    for seed in SEEDS:
        avgs = {}
        for cfg in CONFIGS:
            psnrs = [data[cfg][sc].get(seed) for sc in SCENES]
            psnrs = [v for v in psnrs if v is not None]
            avgs[cfg] = statistics.fmean(psnrs) if psnrs else None
            if avgs[cfg] is not None:
                seed_avgs[cfg].append(avgs[cfg])
        if all(avgs[c] is not None for c in CONFIGS):
            d_a1 = avgs['A1'] - avgs['A0']
            d_a2 = avgs['A2'] - avgs['A0']
            d_a3 = avgs['A3'] - avgs['A0']
            print(f"{seed:<8}  {avgs['A0']:>8.3f}  {avgs['A1']:>8.3f}  "
                  f"{avgs['A2']:>8.3f}  {avgs['A3']:>8.3f}  "
                  f"{d_a1:>+8.3f}  {d_a2:>+8.3f}  {d_a3:>+8.3f}")

    # ── Paired Δ summary — 3 comparisons ──
    print("\n=== Paired Δ summary — 3 comparisons (N = seeds × scenes) ===")
    deltas_a1, deltas_a2, deltas_a3 = [], [], []
    for seed in SEEDS:
        for sc in SCENES:
            a0 = data['A0'][sc].get(seed)
            a1 = data['A1'][sc].get(seed)
            a2 = data['A2'][sc].get(seed)
            a3 = data['A3'][sc].get(seed)
            if a0 is not None and a1 is not None:
                deltas_a1.append(a1 - a0)
            if a0 is not None and a2 is not None:
                deltas_a2.append(a2 - a0)
            if a0 is not None and a3 is not None:
                deltas_a3.append(a3 - a0)

    print(f"  {'comparison':<24}  {'N':<3}  {'mean':>9}  {'SEM':>7}  {'95% CI':<22}  verdict")
    print("-" * 100)
    a1_mean, a1_sem, a1_sig = summarize_paired("Δ_A1 (full pull)", deltas_a1)
    a2_mean, a2_sem, a2_sig = summarize_paired("Δ_A2 (pull − Phase4)", deltas_a2)
    a3_mean, a3_sem, a3_sig = summarize_paired("Δ_A3 (pull-only)", deltas_a3)

    # ── Hypothesis interpretation ──
    print("\n=== Hypothesis verdict ===")
    if a1_mean is not None and a3_mean is not None:
        scale_op_contribution = a1_mean - a3_mean
        print(f"  Scale+Opacity contribution = Δ_A1 − Δ_A3 = {scale_op_contribution:+.4f}")
        if scale_op_contribution > 0.05:
            print(f"  → Scale + Opacity giúp ngoài pull")
        elif abs(scale_op_contribution) <= 0.05:
            print(f"  → Scale + Opacity neutral, pull-only đủ")
        else:
            print(f"  → Scale + Opacity HẠI, pull-only tốt hơn")
    if a1_mean is not None and a2_mean is not None:
        replace_vs_full = a2_mean - a1_mean
        print(f"  Replace vs full = Δ_A2 − Δ_A1 = {replace_vs_full:+.4f}")
        if replace_vs_full > 0.05:
            print(f"  → CRS-pull CÓ THỂ replace Phase 4 prune")
        elif abs(replace_vs_full) <= 0.05:
            print(f"  → Replace tương đương full — Phase 4 dispensable")
        else:
            print(f"  → CRS-pull KHÔNG thay được prune (cần cả 2)")

    # ── Baseline reproducibility ──
    if seed_avgs['A0']:
        a0_overall = statistics.fmean(seed_avgs['A0'])
        print("\n=== Baseline + best ===")
        print(f"  Phase 8 paper (1 sample):     {PHASE_8_PAPER_AVG:.3f} dB")
        print(f"  Phase 8 multi-seed fair:      {PHASE_8_FAIR_BASELINE:.3f} dB")
        print(f"  A0 (this batch, N={len(seed_avgs['A0'])} seeds): {a0_overall:.3f} dB  "
              f"({a0_overall - PHASE_8_FAIR_BASELINE:+.3f} vs multi-seed fair)")
        for cfg in ['A1', 'A2', 'A3']:
            if seed_avgs[cfg]:
                m = statistics.fmean(seed_avgs[cfg])
                print(f"  {cfg} mean:                    {m:.3f} dB  ({m - a0_overall:+.3f} vs A0)")
        candidates = [(cfg, statistics.fmean(seed_avgs[cfg]))
                      for cfg in CONFIGS if seed_avgs[cfg]]
        if candidates:
            best_cfg, best_psnr = max(candidates, key=lambda x: x[1])
            print(f"  Best config:                  {best_cfg} ({best_psnr:.3f} dB)")


if __name__ == "__main__":
    main()
