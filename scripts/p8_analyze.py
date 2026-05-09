#!/usr/bin/env python3
"""[CRSGaussian Phase 8] 5-config attribution analyzer.

5 configs trên D1-O999 backbone:
  OLD         — D_DAV2 + R_old + gate prune  (= TIER1_DAV2_GATE từ Phase 7 Stage 1)
  FIX_R_DAV2  — D_DAV2 + R_visible + gate
  FIX_R_DC    — D_cycle + R_visible + gate
  FIX_RS      — D_cycle + R_visible + S + gate
  FULL        — D_cycle + R_visible + S + CRS-modulated SH freeze

Attribution diagnostic:
  Δ_R = FIX_R_DAV2 - OLD          (R_visible alone)
  Δ_D = FIX_R_DC   - FIX_R_DAV2   (D_cycle on clean R)
  Δ_S = FIX_RS     - FIX_R_DC     (S signal addition)
  Δ_M = FULL       - FIX_RS       (SH freeze mechanism)
  Δ_FULL = FULL    - OLD          (combined Phase 8)

Verdict:
  FULL > 21.51 (>+0.30 vs no-CRS 21.21)  → 🟢 BREAKTHROUGH
  +0.15 ~ +0.30                           → 🟡 SOLID
  < +0.15                                  → 🔴 STOP, CRS axis exhausted

Run: python scripts/p8_analyze.py
     LOG_DIR=logs/p8 python scripts/p8_analyze.py
"""

import os
import re
import statistics

SCENES = ['fern', 'flower', 'fortress', 'horns',
          'leaves', 'orchids', 'room', 'trex']
CONFIGS = ['OLD', 'FIX_R_DAV2', 'FIX_R_DC', 'FIX_RS', 'FULL']

LOG_DIR = os.environ.get("LOG_DIR", "logs/p8")
PSNR_PAT = re.compile(r"Best test PSNR:\s*([\d\.eE\-\+]+)")
N_GAUSS_PAT = re.compile(r"\bN=(\d+)\b")
TIME_PAT = re.compile(r"elapsed=([\d\.]+)s")

NO_CRS_REF = 21.210  # D1-noCRS-O999 (current project best, no CRS)


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


def main():
    print(f"[p8_analyze] LOG_DIR={LOG_DIR}")
    print()
    print("=" * 150)
    print("PHASE 8 — 5-CONFIG ABLATION (D1-O999 strong backbone)")
    print("=" * 150)
    fmt = "{:>10} | {:>7} | {:>10} | {:>9} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7}"
    print(fmt.format('Scene', 'OLD', 'FIX_R_DAV2', 'FIX_R_DC', 'FIX_RS', 'FULL',
                     'Δ_R', 'Δ_D', 'Δ_S', 'Δ_M'))
    print("-" * 150)

    cols = {k: [] for k in CONFIGS}
    delta_full_per_scene = []

    for s in SCENES:
        psnrs = {k: parse_psnr(f"{LOG_DIR}/{k}_{s}.log") for k in CONFIGS}
        if any(v is None for v in psnrs.values()):
            miss = [k for k, v in psnrs.items() if v is None]
            print(f"{s:>10} | INCOMPLETE — missing {miss}")
            continue

        d_R = psnrs['FIX_R_DAV2'] - psnrs['OLD']
        d_D = psnrs['FIX_R_DC']   - psnrs['FIX_R_DAV2']
        d_S = psnrs['FIX_RS']     - psnrs['FIX_R_DC']
        d_M = psnrs['FULL']       - psnrs['FIX_RS']
        d_full = psnrs['FULL']    - psnrs['OLD']

        delta_full_per_scene.append((s, d_full))

        print(fmt.format(
            s,
            f"{psnrs['OLD']:.3f}",
            f"{psnrs['FIX_R_DAV2']:.3f}",
            f"{psnrs['FIX_R_DC']:.3f}",
            f"{psnrs['FIX_RS']:.3f}",
            f"{psnrs['FULL']:.3f}",
            f"{d_R:+.3f}", f"{d_D:+.3f}", f"{d_S:+.3f}", f"{d_M:+.3f}"
        ))

        for k, v in psnrs.items():
            cols[k].append(v)

    print("-" * 150)
    if not all(len(cols[k]) == len(SCENES) for k in cols):
        print("\n[INCOMPLETE]")
        return

    avgs = {k: statistics.mean(v) for k, v in cols.items()}
    d_R_avg = avgs['FIX_R_DAV2'] - avgs['OLD']
    d_D_avg = avgs['FIX_R_DC']   - avgs['FIX_R_DAV2']
    d_S_avg = avgs['FIX_RS']     - avgs['FIX_R_DC']
    d_M_avg = avgs['FULL']       - avgs['FIX_RS']
    d_full_avg = avgs['FULL']    - avgs['OLD']

    print(fmt.format('AVG',
        f"{avgs['OLD']:.3f}",
        f"{avgs['FIX_R_DAV2']:.3f}",
        f"{avgs['FIX_R_DC']:.3f}",
        f"{avgs['FIX_RS']:.3f}",
        f"{avgs['FULL']:.3f}",
        f"{d_R_avg:+.3f}", f"{d_D_avg:+.3f}", f"{d_S_avg:+.3f}", f"{d_M_avg:+.3f}"
    ))

    # ── Compute & efficiency table ──
    print()
    print("=" * 150)
    print("COMPUTE & EFFICIENCY (per memory rule 'always measure compute cost')")
    print("=" * 150)
    print(f"{'Config':>15} | {'avg N_gauss':>13} | {'avg train (s)':>15} | {'slowdown vs OLD':>18}")
    print("-" * 150)
    times = {}
    for cfg in CONFIGS:
        ng = []
        tt = []
        for s in SCENES:
            n = parse_n_gauss(f"{LOG_DIR}/{cfg}_{s}.log")
            t = parse_train_time(f"{LOG_DIR}/{cfg}_{s}.log")
            if n is not None:
                ng.append(n)
            if t is not None:
                tt.append(t)
        avg_ng = statistics.mean(ng) if ng else None
        avg_tt = statistics.mean(tt) if tt else None
        times[cfg] = avg_tt
        slowdown_str = "—"
        if cfg != 'OLD' and avg_tt and times.get('OLD'):
            slowdown = (avg_tt / times['OLD'] - 1) * 100
            slowdown_str = f"{slowdown:+.1f}%"
        print(f"{cfg:>15} | "
              f"{(f'{avg_ng:.0f}' if avg_ng else 'N/A'):>13} | "
              f"{(f'{avg_tt:.1f}' if avg_tt else 'N/A'):>15} | "
              f"{slowdown_str:>18}")

    # ── Attribution summary ──
    print()
    print("=" * 150)
    print("ATTRIBUTION SUMMARY")
    print("=" * 150)
    print(f"  Δ_R    (R_visible fix alone):         {d_R_avg:+.3f} dB")
    print(f"  Δ_D    (D_cycle on clean R):          {d_D_avg:+.3f} dB")
    print(f"  Δ_S    (S signal addition):           {d_S_avg:+.3f} dB")
    print(f"  Δ_M    (SH freeze mechanism):         {d_M_avg:+.3f} dB")
    print(f"  Δ_FULL (Phase 8 combined vs OLD):     {d_full_avg:+.3f} dB")
    print()

    # Hypothesis attribution
    contributors = []
    if d_R_avg > 0.05: contributors.append(f"R_visible({d_R_avg:+.2f})")
    if d_D_avg > 0.05: contributors.append(f"D_cycle({d_D_avg:+.2f})")
    if d_S_avg > 0.05: contributors.append(f"S({d_S_avg:+.2f})")
    if d_M_avg > 0.05: contributors.append(f"SH_freeze({d_M_avg:+.2f})")
    if contributors:
        print(f"  → Positive contributors: {', '.join(contributors)}")
    else:
        print(f"  → No component > +0.05 dB — formula redesign saturated")

    # ── Verdict ──
    n_positive = sum(1 for _, d in delta_full_per_scene if d > 0)
    delta_vs_no_crs = avgs['FULL'] - NO_CRS_REF

    print()
    print("=" * 150)
    print("VERDICT — Phase 8 vs Reference Targets")
    print("=" * 150)
    print(f"  OLD avg (= TIER1_DAV2_GATE):  {avgs['OLD']:.3f} dB (expected ~21.18 from Phase 7 Stage 1)")
    print(f"  FULL avg:                     {avgs['FULL']:.3f} dB")
    print(f"  No-CRS reference (Tier1):     {NO_CRS_REF:.3f} dB")
    print(f"  Δ_FULL vs OLD:                {d_full_avg:+.3f} dB | {n_positive}/{len(SCENES)} positive")
    print(f"  Δ vs No-CRS:                  {delta_vs_no_crs:+.3f} dB")
    print()

    if d_full_avg >= 0.30 and delta_vs_no_crs >= 0.10:
        verdict = "🟢 BREAKTHROUGH — formula redesign + SH path delivers"
        action = "→ Defendable signal+mechanism contribution. Ship paper với Phase 8 ablation."
    elif d_full_avg >= 0.15:
        verdict = "🟡 SOLID — meaningful improvement"
        action = "→ Recipe paper + Phase 8 row in ablation table."
    else:
        verdict = "🔴 STOP — formula redesign không break ceiling"
        action = "→ CRS axis exhausted across signal redesign. Pivot recipe paper definitively."
    print(f"  {verdict}")
    print(f"  {action}")

    # Reference targets
    print()
    print("  Reference targets:")
    print(f"    No-CRS Tier1 (D1-noCRS-O999):  21.21 dB")
    print(f"    DOC-GS:                         21.38 dB")
    print(f"    BinocularGS:                    21.44 dB")
    print(f"    ICO-GS (SOTA):                  22.20 dB")


if __name__ == "__main__":
    main()
