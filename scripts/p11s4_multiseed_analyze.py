#!/usr/bin/env python3
"""[CRSGaussian Phase 11 Step 4] Multi-seed analyzer — Cross-view Feature MPC.

Parse logs/p11s4_ms/{A0,A1}_seed{42,137,9999}_<scene>.log → table với:
- Per-scene per-seed PSNR
- Mean ± std across seeds (for each config × scene)
- Δ_paired (A1 − A0) per (seed, scene) — paired by both
- Mean Δ over (3 seeds × 8 scenes = 24 samples) ± std
- Compare with Phase 8 paper avg 21.335

Run:
    python scripts/p11s4_multiseed_analyze.py
    SEEDS="42 137 9999" SCENES="fern flower ..." python scripts/p11s4_multiseed_analyze.py
"""

import os
import re
import statistics

SCENES = os.environ.get("SCENES", "fern flower fortress horns leaves orchids room trex").split()
SEEDS = os.environ.get("SEEDS", "42 137 9999").split()
LOG_DIR = os.environ.get("LOG_DIR", "logs/p11s4_ms")

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


def parse_psnr(path):
    if not os.path.isfile(path):
        return None
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    m = PSNR_PAT.search(text)
    return float(m.group(1)) if m else None


def fmt(v, prec=3):
    return f"{v:.{prec}f}" if v is not None else "    -"


def main():
    print(f"=== Phase 11 Step 4 — Cross-view Feature MPC multi-seed ===")
    print(f"LOG_DIR={LOG_DIR}  SEEDS={SEEDS}  SCENES={SCENES}")

    # Collect: data[cfg][scene][seed] = psnr
    data = {'A0': {}, 'A1': {}}
    for cfg in ['A0', 'A1']:
        for sc in SCENES:
            data[cfg][sc] = {}
            for seed in SEEDS:
                p = f"{LOG_DIR}/{cfg}_seed{seed}_{sc}.log"
                data[cfg][sc][seed] = parse_psnr(p)

    # ── Per-scene per-seed table ──
    print("\n=== Per-scene PSNR by seed ===")
    header = f"{'scene':<10}"
    for seed in SEEDS:
        header += f"  A0_s{seed:<5}  A1_s{seed:<5}"
    print(header)
    print("-" * len(header))
    for sc in SCENES:
        row = f"{sc:<10}"
        for seed in SEEDS:
            a0 = data['A0'][sc].get(seed)
            a1 = data['A1'][sc].get(seed)
            row += f"  {fmt(a0)}    {fmt(a1)}   "
        print(row)

    # ── Per-scene mean ± std across seeds ──
    print("\n=== Per-scene mean ± std (across seeds) ===")
    print(f"{'scene':<10}  {'A0 mean':>8}  {'A0 std':>7}  {'A1 mean':>8}  {'A1 std':>7}  {'Δ mean':>8}  {'P8 paper':>9}")
    print("-" * 70)
    for sc in SCENES:
        a0_vals = [v for v in data['A0'][sc].values() if v is not None]
        a1_vals = [v for v in data['A1'][sc].values() if v is not None]
        if len(a0_vals) >= 1 and len(a1_vals) >= 1:
            a0_mean = statistics.fmean(a0_vals)
            a1_mean = statistics.fmean(a1_vals)
            a0_std = statistics.stdev(a0_vals) if len(a0_vals) > 1 else 0.0
            a1_std = statistics.stdev(a1_vals) if len(a1_vals) > 1 else 0.0
            delta = a1_mean - a0_mean
            ref = PHASE_8_PAPER_PER_SCENE.get(sc)
            print(f"{sc:<10}  {a0_mean:>8.3f}  {a0_std:>7.3f}  "
                  f"{a1_mean:>8.3f}  {a1_std:>7.3f}  "
                  f"{delta:>+8.3f}  {fmt(ref):>9}")

    # ── 8-scene avg per seed ──
    print("\n=== 8-scene avg PSNR per seed ===")
    print(f"{'seed':<8}  {'A0 8-avg':>9}  {'A1 8-avg':>9}  {'Δ paired':>9}")
    print("-" * 45)
    seed_avgs = []
    for seed in SEEDS:
        a0_psnrs = [data['A0'][sc].get(seed) for sc in SCENES]
        a1_psnrs = [data['A1'][sc].get(seed) for sc in SCENES]
        a0_psnrs = [v for v in a0_psnrs if v is not None]
        a1_psnrs = [v for v in a1_psnrs if v is not None]
        if len(a0_psnrs) >= 1 and len(a1_psnrs) >= 1:
            a0_avg = statistics.fmean(a0_psnrs)
            a1_avg = statistics.fmean(a1_psnrs)
            delta = a1_avg - a0_avg
            seed_avgs.append((seed, a0_avg, a1_avg, delta))
            print(f"{seed:<8}  {a0_avg:>9.3f}  {a1_avg:>9.3f}  {delta:>+9.3f}")

    # ── Summary across all (seed × scene) paired diffs ──
    print("\n=== Summary — all paired Δ (A1 − A0) ===")
    all_deltas = []
    for seed in SEEDS:
        for sc in SCENES:
            a0 = data['A0'][sc].get(seed)
            a1 = data['A1'][sc].get(seed)
            if a0 is not None and a1 is not None:
                all_deltas.append(a1 - a0)
    if all_deltas:
        n = len(all_deltas)
        mean = statistics.fmean(all_deltas)
        std = statistics.stdev(all_deltas) if n > 1 else 0.0
        sem = std / (n ** 0.5)
        print(f"  N samples:     {n}")
        print(f"  Δ mean:        {mean:+.4f} dB")
        print(f"  Δ std:         {std:.4f} dB")
        print(f"  Δ SEM:         {sem:.4f} dB  (mean ± SEM)")
        print(f"  95% CI:        [{mean - 1.96*sem:+.3f}, {mean + 1.96*sem:+.3f}]")

        # ── Verdict — is signal robust? ──
        if abs(mean) > 2 * sem:
            sig = "🎯 SIGNIFICANT" if mean > 0 else "📉 SIGNIFICANT NEGATIVE"
        else:
            sig = "🔇 NOT SIGNIFICANT (within noise)"
        print(f"  Verdict:       {sig}")
        if mean >= 0.10:
            print(f"  Decision:      🎯 WINNER CANDIDATE (Δ ≥ +0.10)")
        elif mean >= 0.05:
            print(f"  Decision:      🟡 MARGINAL (+0.05 ≤ Δ < +0.10)")
        else:
            print(f"  Decision:      ❌ REJECT (Δ < +0.05)")

    # ── Compare A0 mean with Phase 8 paper ──
    if seed_avgs:
        a0_overall = statistics.fmean([s[1] for s in seed_avgs])
        a1_overall = statistics.fmean([s[2] for s in seed_avgs])
        print("\n=== Baseline reproducibility ===")
        print(f"  Phase 8 paper avg (May 5):    {PHASE_8_PAPER_AVG:.3f} dB")
        print(f"  A0 mean (3 seeds × 8 scenes): {a0_overall:.3f} dB")
        print(f"  A1 mean (3 seeds × 8 scenes): {a1_overall:.3f} dB")
        print(f"  A0 drift vs paper:            {a0_overall - PHASE_8_PAPER_AVG:+.3f} dB")
        print(f"  Best mean (A1):               {a1_overall:.3f} dB  "
              f"({a1_overall - PHASE_8_PAPER_AVG:+.3f} vs paper)")


if __name__ == "__main__":
    main()
