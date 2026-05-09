#!/usr/bin/env python3
"""[CRSGaussian Phase 9] Simplification + cross-backbone analyzer.

Test 1 — D1-O999 backbone simplification:
  FULL (reuse Phase 8) | FULL_NoS | D_ONLY_FREEZE | D_ONLY_GATE
  Attribution:
    Δ_NoS    = FULL_NoS      - FULL          (drop S effect — H2)
    Δ_NoR    = D_ONLY_FREEZE - FULL_NoS      (drop R effect — H1)
    Δ_freeze = D_ONLY_FREEZE - D_ONLY_GATE   (SH freeze mechanism alone)

Test 2 — A1+B1β cross-backbone:
  A1B1_BASELINE | A1B1_BEST
  Attribution:
    Δ_Phase9 = A1B1_BEST - A1B1_BASELINE     (Phase 8 components on A1+B1β — H3)

Hypotheses:
  H1: drop R helps (D_ONLY > FULL_NoS by +0.05-0.20)
  H2: S is redundant (FULL_NoS ≈ FULL within ±0.05)
  H3: SH freeze universal (Δ_Phase9 ≥ +0.10 on A1+B1β)

Run: python scripts/p9_analyze.py
     LOG_DIR=logs/p9 P8_LOG_DIR=logs/p8 python scripts/p9_analyze.py
"""

import os
import re
import statistics

SCENES = ['fern', 'flower', 'fortress', 'horns',
          'leaves', 'orchids', 'room', 'trex']

# Test 1 configs (D1-O999 backbone)
T1_CONFIGS = ['FULL', 'FULL_NoS', 'D_ONLY_FREEZE', 'D_ONLY_GATE']
# Test 2 configs (A1+B1β backbone)
T2_CONFIGS = ['A1B1_BASELINE', 'A1B1_BEST']

LOG_DIR = os.environ.get("LOG_DIR", "logs/p9")
P8_LOG_DIR = os.environ.get("P8_LOG_DIR", "logs/p8")

PSNR_PAT = re.compile(r"Best test PSNR:\s*([\d\.eE\-\+]+)")
N_GAUSS_PAT = re.compile(r"\bN=(\d+)\b")
TIME_PAT = re.compile(r"elapsed=([\d\.]+)s")

PHASE_8_FULL_REF = 21.335     # Phase 8 FULL AVG (TIER1_DAV2_GATE + 0.157)
NO_CRS_REF = 21.210           # D1-noCRS-O999 (current project best, no CRS)
A1B1_REF = 20.96              # Track A+B reference
DOC_GS_REF = 21.380


def parse_psnr(path):
    if not os.path.isfile(path):
        return None
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    m = PSNR_PAT.search(text)
    return float(m.group(1)) if m else None


def parse_n_gauss(path):
    if not os.path.isfile(path):
        return None
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    matches = N_GAUSS_PAT.findall(text)
    return int(matches[-1]) if matches else None


def parse_train_time(path):
    if not os.path.isfile(path):
        return None
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    matches = TIME_PAT.findall(text)
    return float(matches[-1]) if matches else None


def get_path(cfg, scene):
    """FULL reuse từ Phase 8; còn lại từ Phase 9 logs."""
    if cfg == 'FULL':
        return f"{P8_LOG_DIR}/FULL_{scene}.log"
    return f"{LOG_DIR}/{cfg}_{scene}.log"


def analyze_test1():
    """Test 1 — D1-O999 simplification."""
    print("=" * 150)
    print("PHASE 9 TEST 1 — D1-O999 backbone simplification")
    print("=" * 150)
    fmt = "{:>10} | {:>7} | {:>9} | {:>14} | {:>12} | {:>7} | {:>7} | {:>9}"
    print(fmt.format('Scene', 'FULL', 'FULL_NoS', 'D_ONLY_FREEZE', 'D_ONLY_GATE',
                     'Δ_NoS', 'Δ_NoR', 'Δ_freeze'))
    print("-" * 150)

    cols = {k: [] for k in T1_CONFIGS}
    for s in SCENES:
        psnrs = {k: parse_psnr(get_path(k, s)) for k in T1_CONFIGS}
        if any(v is None for v in psnrs.values()):
            miss = [k for k, v in psnrs.items() if v is None]
            print(f"{s:>10} | INCOMPLETE — missing {miss}")
            continue
        d_nos    = psnrs['FULL_NoS']      - psnrs['FULL']
        d_nor    = psnrs['D_ONLY_FREEZE'] - psnrs['FULL_NoS']
        d_freeze = psnrs['D_ONLY_FREEZE'] - psnrs['D_ONLY_GATE']
        print(fmt.format(
            s,
            f"{psnrs['FULL']:.3f}",
            f"{psnrs['FULL_NoS']:.3f}",
            f"{psnrs['D_ONLY_FREEZE']:.3f}",
            f"{psnrs['D_ONLY_GATE']:.3f}",
            f"{d_nos:+.3f}", f"{d_nor:+.3f}", f"{d_freeze:+.3f}"
        ))
        for k, v in psnrs.items():
            cols[k].append(v)

    print("-" * 150)
    if not all(len(cols[k]) == len(SCENES) for k in cols):
        print("[INCOMPLETE]")
        return None

    avgs = {k: statistics.mean(v) for k, v in cols.items()}
    d_nos_avg    = avgs['FULL_NoS']      - avgs['FULL']
    d_nor_avg    = avgs['D_ONLY_FREEZE'] - avgs['FULL_NoS']
    d_freeze_avg = avgs['D_ONLY_FREEZE'] - avgs['D_ONLY_GATE']

    print(fmt.format('AVG',
        f"{avgs['FULL']:.3f}",
        f"{avgs['FULL_NoS']:.3f}",
        f"{avgs['D_ONLY_FREEZE']:.3f}",
        f"{avgs['D_ONLY_GATE']:.3f}",
        f"{d_nos_avg:+.3f}", f"{d_nor_avg:+.3f}", f"{d_freeze_avg:+.3f}"
    ))
    return avgs


def analyze_test2():
    """Test 2 — A1+B1β cross-backbone."""
    print()
    print("=" * 150)
    print("PHASE 9 TEST 2 — A1+B1β cross-backbone (SH freeze mechanism universality)")
    print("=" * 150)
    fmt = "{:>10} | {:>14} | {:>10} | {:>10}"
    print(fmt.format('Scene', 'A1B1_BASELINE', 'A1B1_BEST', 'Δ_Phase9'))
    print("-" * 150)

    cols = {k: [] for k in T2_CONFIGS}
    for s in SCENES:
        psnrs = {k: parse_psnr(get_path(k, s)) for k in T2_CONFIGS}
        if any(v is None for v in psnrs.values()):
            miss = [k for k, v in psnrs.items() if v is None]
            print(f"{s:>10} | INCOMPLETE — missing {miss}")
            continue
        d_phase9 = psnrs['A1B1_BEST'] - psnrs['A1B1_BASELINE']
        print(fmt.format(
            s,
            f"{psnrs['A1B1_BASELINE']:.3f}",
            f"{psnrs['A1B1_BEST']:.3f}",
            f"{d_phase9:+.3f}"
        ))
        for k, v in psnrs.items():
            cols[k].append(v)

    print("-" * 150)
    if not all(len(cols[k]) == len(SCENES) for k in cols):
        print("[INCOMPLETE]")
        return None

    avgs = {k: statistics.mean(v) for k, v in cols.items()}
    d_phase9_avg = avgs['A1B1_BEST'] - avgs['A1B1_BASELINE']
    print(fmt.format('AVG',
        f"{avgs['A1B1_BASELINE']:.3f}",
        f"{avgs['A1B1_BEST']:.3f}",
        f"{d_phase9_avg:+.3f}"
    ))
    return avgs


def compute_table(test1_cfgs, test2_cfgs):
    """Compute & efficiency per memory rule 'always measure compute cost'."""
    print()
    print("=" * 150)
    print("COMPUTE & EFFICIENCY")
    print("=" * 150)
    print(f"{'Config':>15} | {'avg N_gauss':>13} | {'avg train (s)':>15} | {'backbone':>15}")
    print("-" * 150)
    for cfg, backbone in (
        [(c, "D1-O999") for c in test1_cfgs]
        + [(c, "A1+B1β") for c in test2_cfgs]
    ):
        ng, tt = [], []
        for s in SCENES:
            n = parse_n_gauss(get_path(cfg, s))
            t = parse_train_time(get_path(cfg, s))
            if n is not None: ng.append(n)
            if t is not None: tt.append(t)
        avg_ng = statistics.mean(ng) if ng else None
        avg_tt = statistics.mean(tt) if tt else None
        print(f"{cfg:>15} | "
              f"{(f'{avg_ng:.0f}' if avg_ng else 'N/A'):>13} | "
              f"{(f'{avg_tt:.1f}' if avg_tt else 'N/A'):>15} | "
              f"{backbone:>15}")


def verdict(avgs1, avgs2):
    """Hypothesis verification + final verdict."""
    print()
    print("=" * 150)
    print("VERDICT — Phase 9 hypothesis verification")
    print("=" * 150)

    # Test 1 verdicts
    if avgs1 is not None:
        d_nos    = avgs1['FULL_NoS']      - avgs1['FULL']
        d_nor    = avgs1['D_ONLY_FREEZE'] - avgs1['FULL_NoS']
        d_freeze = avgs1['D_ONLY_FREEZE'] - avgs1['D_ONLY_GATE']
        best1 = max(avgs1['FULL'], avgs1['FULL_NoS'],
                    avgs1['D_ONLY_FREEZE'], avgs1['D_ONLY_GATE'])

        print(f"\n  TEST 1 (D1-O999 simplification):")
        print(f"    FULL avg:           {avgs1['FULL']:.3f}")
        print(f"    FULL_NoS avg:       {avgs1['FULL_NoS']:.3f}   (Δ_NoS = {d_nos:+.3f})")
        print(f"    D_ONLY_FREEZE avg:  {avgs1['D_ONLY_FREEZE']:.3f}   (Δ_NoR = {d_nor:+.3f})")
        print(f"    D_ONLY_GATE avg:    {avgs1['D_ONLY_GATE']:.3f}   (Δ_freeze = {d_freeze:+.3f})")
        print(f"    BEST: {best1:.3f}")

        # H1: drop R helps
        if d_nor >= 0.10:
            print(f"  ✅ H1 CONFIRMED: drop R helps significantly ({d_nor:+.3f})")
        elif d_nor >= 0.0:
            print(f"  🟡 H1 partial: drop R neutral/slight help ({d_nor:+.3f})")
        else:
            print(f"  ❌ H1 REJECTED: R contributes ({d_nor:+.3f})")

        # H2: S redundant
        if abs(d_nos) <= 0.05:
            print(f"  ✅ H2 CONFIRMED: S adds nothing (|Δ_NoS|={abs(d_nos):.3f} ≤ 0.05)")
        elif d_nos > 0.05:
            print(f"  ✅✅ H2 STRONG: dropping S actively helps ({d_nos:+.3f})")
        else:
            print(f"  ❌ H2 REJECTED: S contributes ({d_nos:+.3f})")

        # SH freeze isolation
        if d_freeze >= 0.10:
            print(f"  → SH freeze mechanism alone delivers {d_freeze:+.3f} (matches Phase 8 +0.189)")
        else:
            print(f"  → SH freeze marginal alone ({d_freeze:+.3f}) — relies on R/S synergy")

    # Test 2 verdict
    if avgs2 is not None:
        d_phase9 = avgs2['A1B1_BEST'] - avgs2['A1B1_BASELINE']
        print(f"\n  TEST 2 (A1+B1β cross-backbone):")
        print(f"    A1B1_BASELINE avg:  {avgs2['A1B1_BASELINE']:.3f}   (Track A+B ref ~{A1B1_REF:.2f})")
        print(f"    A1B1_BEST avg:      {avgs2['A1B1_BEST']:.3f}")
        print(f"    Δ_Phase9 on A1+B1β: {d_phase9:+.3f}")

        if d_phase9 >= 0.15:
            print(f"  ✅ H3 CONFIRMED: SH freeze universal ({d_phase9:+.3f} ≥ 0.15)")
        elif d_phase9 >= 0.05:
            print(f"  🟡 H3 partial: works smaller gain on A1+B1β ({d_phase9:+.3f})")
        else:
            print(f"  ❌ H3 REJECTED: SH freeze backbone-specific (D1-O999 only) ({d_phase9:+.3f})")

    # Final paper claim assessment
    print()
    print("=" * 150)
    print("FINAL PAPER ASSESSMENT")
    print("=" * 150)
    if avgs1 is not None:
        best1 = max(avgs1['FULL'], avgs1['FULL_NoS'],
                    avgs1['D_ONLY_FREEZE'], avgs1['D_ONLY_GATE'])
        delta_no_crs = best1 - NO_CRS_REF
        print(f"  Best Phase 9 config (Test 1):  {best1:.3f} dB")
        print(f"  Δ vs No-CRS (21.21):            {delta_no_crs:+.3f}")
        print(f"  Δ vs DOC-GS (21.38):            {best1 - DOC_GS_REF:+.3f}")
        if delta_no_crs >= 0.20:
            print(f"  🟢 STRONG paper claim: simplification + SH freeze beat no-CRS by {delta_no_crs:+.3f}")
        elif delta_no_crs >= 0.10:
            print(f"  🟡 SOLID paper claim: meaningful CRS contribution")
        else:
            print(f"  ⚪ Marginal — Phase 8 FULL ({PHASE_8_FULL_REF:.3f}) was already at this level")

    print()
    print("  Reference targets:")
    print(f"    Phase 8 FULL:        {PHASE_8_FULL_REF:.3f} dB")
    print(f"    No-CRS Tier1:        {NO_CRS_REF:.3f} dB")
    print(f"    A1+B1β Track:        ~{A1B1_REF:.2f} dB")
    print(f"    DOC-GS:              {DOC_GS_REF:.3f} dB")
    print(f"    BinocularGS:         21.44 dB")
    print(f"    ICO-GS (SOTA):       22.20 dB")


def main():
    print(f"[p9_analyze] LOG_DIR={LOG_DIR} | P8_LOG_DIR={P8_LOG_DIR}")
    print()
    avgs1 = analyze_test1()
    avgs2 = analyze_test2()
    compute_table(T1_CONFIGS, T2_CONFIGS)
    verdict(avgs1, avgs2)


if __name__ == "__main__":
    main()
