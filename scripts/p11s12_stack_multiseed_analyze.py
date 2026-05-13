#!/usr/bin/env python3
# ============================================================
# DEPRECATED 2026-05-11: Phase 11 Stack (S1+S2) REJECTED — Δ_Synergy=−0.063 N=24 no synergy
# Flags removed from arguments/__init__.py — script no longer runnable.
# Kept for paper reproducibility evidence. See CLAUDE.md Phase 11 results.
# ============================================================
"""[CRSGaussian Phase 11 Step 1+Stack] Multi-seed analyzer — 3 configs same batch.

Configs:
  A0 = Phase 8 FULL baseline
  A1 = + Step 1 covisibility reweight only
  A2 = + Step 1 AND Step 2 perceptual DINO (stack)

3 paired comparisons:
  Δ_S1     = A1 − A0   (Step 1 alone effect, re-run same batch)
  Δ_Stack  = A2 − A0   (Stack total effect)
  Δ_Synergy = A2 − A1  (Step 2 added on top of Step 1)

Verdict logic:
  Δ_Stack significant AND > Δ_S1 + 0.05 → 🎯 SYNERGIZE
  Δ_Stack significant AND ≈ Δ_S1       → Step 2 redundant on Step 1
  Δ_Stack significant AND < Δ_S1       → Step 2 cancels Step 1
  Δ_Stack not significant              → 🔇 stack KHÔNG break ceiling

Run:
    python scripts/p11s12_stack_multiseed_analyze.py
    SEEDS="42 137 9999" SCENES="fern flower ..." python scripts/p11s12_stack_multiseed_analyze.py
"""

import os
import re
import statistics

SCENES = os.environ.get("SCENES", "fern flower fortress horns leaves orchids room trex").split()
SEEDS = os.environ.get("SEEDS", "42 137 9999").split()
LOG_DIR = os.environ.get("LOG_DIR", "logs/p11s12_ms")

PSNR_PAT = re.compile(r"Best test PSNR:\s*([\d\.eE\-\+]+)")
PHASE_8_PAPER_AVG = 21.335
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

CONFIGS = ['A0', 'A1', 'A2']


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
    """In statistics cho 1 paired comparison."""
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
    print(f"  {label:<20}  N={n:<3}  mean={mean:+.4f}  SEM={sem:.4f}  "
          f"95% CI=[{ci_lo:+.3f}, {ci_hi:+.3f}]  {sig_label}")
    return mean, sem, sig


def main():
    print(f"=== Phase 11 Step 1+Stack — Re-run same batch ===")
    print(f"LOG_DIR={LOG_DIR}  SEEDS={SEEDS}  SCENES={SCENES}")

    # Collect: data[cfg][scene][seed] = psnr
    data = {cfg: {} for cfg in CONFIGS}
    for cfg in CONFIGS:
        for sc in SCENES:
            data[cfg][sc] = {}
            for seed in SEEDS:
                p = f"{LOG_DIR}/{cfg}_seed{seed}_{sc}.log"
                data[cfg][sc][seed] = parse_psnr(p)

    # ── Per-scene per-seed table ──
    print("\n=== Per-scene PSNR by seed (A0/A1/A2) ===")
    header = f"{'scene':<10}"
    for seed in SEEDS:
        header += f"  A0_s{seed:<5}  A1_s{seed:<5}  A2_s{seed:<5}"
    print(header)
    print("-" * len(header))
    for sc in SCENES:
        row = f"{sc:<10}"
        for seed in SEEDS:
            for cfg in CONFIGS:
                row += f"  {fmt(data[cfg][sc].get(seed))}   "
        print(row)

    # ── Per-scene mean ± std across seeds ──
    print("\n=== Per-scene mean ± std (across seeds) ===")
    print(f"{'scene':<10}  {'A0 mean':>8}  {'A1 mean':>8}  {'A2 mean':>8}  "
          f"{'Δ_S1':>8}  {'Δ_Stack':>8}  {'Δ_Syn':>8}  {'P8 paper':>9}")
    print("-" * 90)
    for sc in SCENES:
        means = {}
        for cfg in CONFIGS:
            vals = [v for v in data[cfg][sc].values() if v is not None]
            means[cfg] = statistics.fmean(vals) if vals else None
        if all(means[c] is not None for c in CONFIGS):
            d_s1 = means['A1'] - means['A0']
            d_stack = means['A2'] - means['A0']
            d_syn = means['A2'] - means['A1']
            ref = PHASE_8_PAPER_PER_SCENE.get(sc)
            print(f"{sc:<10}  {means['A0']:>8.3f}  {means['A1']:>8.3f}  {means['A2']:>8.3f}  "
                  f"{d_s1:>+8.3f}  {d_stack:>+8.3f}  {d_syn:>+8.3f}  {fmt(ref):>9}")

    # ── 8-scene avg per seed ──
    print("\n=== 8-scene avg PSNR per seed ===")
    print(f"{'seed':<8}  {'A0 8-avg':>9}  {'A1 8-avg':>9}  {'A2 8-avg':>9}  "
          f"{'Δ_S1':>8}  {'Δ_Stack':>8}  {'Δ_Syn':>8}")
    print("-" * 75)
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
            d_s1 = avgs['A1'] - avgs['A0']
            d_stack = avgs['A2'] - avgs['A0']
            d_syn = avgs['A2'] - avgs['A1']
            print(f"{seed:<8}  {avgs['A0']:>9.3f}  {avgs['A1']:>9.3f}  {avgs['A2']:>9.3f}  "
                  f"{d_s1:>+8.3f}  {d_stack:>+8.3f}  {d_syn:>+8.3f}")

    # ── Paired Δ summary (3 comparisons) ──
    print("\n=== Paired Δ summary — 3 comparisons (N = seeds × scenes) ===")
    deltas_s1, deltas_stack, deltas_syn = [], [], []
    for seed in SEEDS:
        for sc in SCENES:
            a0 = data['A0'][sc].get(seed)
            a1 = data['A1'][sc].get(seed)
            a2 = data['A2'][sc].get(seed)
            if a0 is not None and a1 is not None:
                deltas_s1.append(a1 - a0)
            if a0 is not None and a2 is not None:
                deltas_stack.append(a2 - a0)
            if a1 is not None and a2 is not None:
                deltas_syn.append(a2 - a1)

    print(f"  {'comparison':<20}  {'N':<3}  {'mean':>9}  {'SEM':>7}  {'95% CI':<22}  verdict")
    print("-" * 90)
    s1_mean, s1_sem, s1_sig = summarize_paired("Δ_S1     (A1-A0)", deltas_s1)
    stk_mean, stk_sem, stk_sig = summarize_paired("Δ_Stack  (A2-A0)", deltas_stack)
    syn_mean, syn_sem, syn_sig = summarize_paired("Δ_Synergy(A2-A1)", deltas_syn)

    # ── Synergy interpretation ──
    print("\n=== Synergy verdict ===")
    if stk_mean is not None and s1_mean is not None:
        if stk_sig and stk_mean > 0:
            if stk_mean > s1_mean + 0.05:
                verdict = "🎯 SYNERGIZE — Stack > Step 1 alone, Step 2 contributes"
            elif abs(stk_mean - s1_mean) <= 0.05:
                verdict = "🟡 REDUNDANT — Stack ≈ Step 1, Step 2 không thêm signal"
            else:
                verdict = "📉 CANCEL — Step 2 trừ đi Step 1 effect"
        elif stk_sig and stk_mean < 0:
            verdict = "📉 STACK HURTS — combined effect negative"
        else:
            verdict = "🔇 STACK KHÔNG SIGNIFICANT — variance dominate (như Step 1/2 alone)"
        print(f"  {verdict}")
        print(f"  Δ_Stack = {stk_mean:+.4f}  vs  Δ_S1 = {s1_mean:+.4f}  → "
              f"diff = {stk_mean - s1_mean:+.4f} ({'Step 2 helps' if stk_mean > s1_mean else 'Step 2 hurts'})")

    # ── Baseline reproducibility ──
    if seed_avgs['A0']:
        a0_overall = statistics.fmean(seed_avgs['A0'])
        a1_overall = statistics.fmean(seed_avgs['A1']) if seed_avgs['A1'] else None
        a2_overall = statistics.fmean(seed_avgs['A2']) if seed_avgs['A2'] else None
        print("\n=== Baseline + best ===")
        print(f"  Phase 8 paper avg:    {PHASE_8_PAPER_AVG:.3f} dB")
        print(f"  A0 mean (this batch): {a0_overall:.3f} dB  ({a0_overall - PHASE_8_PAPER_AVG:+.3f} vs paper)")
        if a1_overall is not None:
            print(f"  A1 mean:              {a1_overall:.3f} dB  ({a1_overall - PHASE_8_PAPER_AVG:+.3f} vs paper)")
        if a2_overall is not None:
            print(f"  A2 mean:              {a2_overall:.3f} dB  ({a2_overall - PHASE_8_PAPER_AVG:+.3f} vs paper)")
        candidates = [(cfg, statistics.fmean(seed_avgs[cfg])) for cfg in CONFIGS if seed_avgs[cfg]]
        if candidates:
            best_cfg, best_psnr = max(candidates, key=lambda x: x[1])
            print(f"  Best config:          {best_cfg} ({best_psnr:.3f} dB)")


if __name__ == "__main__":
    main()
