#!/usr/bin/env python3
"""
[CRSGaussian Tier 2-min + Phase 7-FINAL] Ablation analyzer

Modes (qua env STAGE):
  STAGE=phase7  (default): Phase 7 original (B0/DCYCLE/LWEIGHT/TIER2MIN) — Phase 5 weak backbone.
  STAGE=1               : Phase 7-FINAL Stage 1 — D1-O999 family (TIER1_*).
  STAGE=2               : Phase 7-FINAL Stage 2 — A1+B1β family (TIER2_*).
  STAGE=both            : Stage 1 + Stage 2 cross-backbone analysis.

Env vars khác:
  LOG_DIR (default logs/t2min): dir chứa logs *.log của configs cần phân tích.
  B0_LOG_DIR (default = LOG_DIR): nếu B0 logs ở dir riêng (vd reuse từ logs/c1_ablation).

Run examples:
  python scripts/t2min_final_analyze.py                                 # Phase 7 original
  LOG_DIR=logs/t2min_final STAGE=1 python scripts/t2min_final_analyze.py
  LOG_DIR=logs/t2min_final STAGE=both python scripts/t2min_final_analyze.py
"""

import os
import re
import statistics

SCENES = ['fern', 'flower', 'fortress', 'horns',
          'leaves', 'orchids', 'room', 'trex']

PSNR_PAT = re.compile(r"Best test PSNR:\s*([\d\.eE\-\+]+)")
N_GAUSS_PAT = re.compile(r"\bN=(\d+)\b")
TIME_PAT = re.compile(r"elapsed=([\d\.]+)s")

STAGE = os.environ.get("STAGE", "phase7").lower()
LOG_DIR = os.environ.get("LOG_DIR", "logs/t2min")
B0_LOG_DIR = os.environ.get("B0_LOG_DIR", LOG_DIR)


def parse_psnr(path):
    if not os.path.isfile(path):
        return None
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    m = PSNR_PAT.search(text)
    return float(m.group(1)) if m else None


def parse_n_gauss_final(path):
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


def collect_psnrs(sources):
    """Returns (cols dict {cfg: [psnrs]}, missing list, per_scene list)."""
    cols = {k: [] for k in sources}
    missing = []
    per_scene = []
    for s in SCENES:
        psnrs = {k: parse_psnr(f(s)) for k, f in sources.items()}
        miss = [k for k, v in psnrs.items() if v is None]
        if miss:
            for m in miss:
                missing.append(f"{m}/{s}")
            per_scene.append((s, None))
            continue
        per_scene.append((s, psnrs))
        for k, v in psnrs.items():
            cols[k].append(v)
    return cols, missing, per_scene


def compute_compute_table(sources, cols):
    """Print compute & efficiency table per memory rule 'always measure compute cost'."""
    print()
    print("=" * 140)
    print("COMPUTE & EFFICIENCY")
    print("=" * 140)
    print(f"{'Config':>20} | {'avg N_gauss':>13} | {'avg train (s)':>15} | {'slowdown vs ref':>18}")
    print("-" * 140)
    ref_cfg = list(sources.keys())[0]  # first config = reference
    times = {}
    for cfg in sources:
        ng = []
        tt = []
        for s in SCENES:
            n = parse_n_gauss_final(sources[cfg](s))
            t = parse_train_time(sources[cfg](s))
            if n is not None:
                ng.append(n)
            if t is not None:
                tt.append(t)
        avg_ng = statistics.mean(ng) if ng else None
        avg_tt = statistics.mean(tt) if tt else None
        times[cfg] = avg_tt
        slowdown_str = "—"
        if cfg != ref_cfg and avg_tt and times.get(ref_cfg):
            slowdown = (avg_tt / times[ref_cfg] - 1) * 100
            slowdown_str = f"{slowdown:+.1f}%"
        print(f"{cfg:>20} | "
              f"{(f'{avg_ng:.0f}' if avg_ng else 'N/A'):>13} | "
              f"{(f'{avg_tt:.1f}' if avg_tt else 'N/A'):>15} | "
              f"{slowdown_str:>18}")


def analyze_phase7_original():
    """Phase 7 original — B0/DCYCLE/LWEIGHT/TIER2MIN trên Phase 5 weak backbone."""
    sources = {
        'B0':       lambda s: f'{B0_LOG_DIR}/B0_{s}.log',
        'DCYCLE':   lambda s: f'{LOG_DIR}/DCYCLE_{s}.log',
        'LWEIGHT':  lambda s: f'{LOG_DIR}/LWEIGHT_{s}.log',
        'TIER2MIN': lambda s: f'{LOG_DIR}/TIER2MIN_{s}.log',
    }
    print("=" * 140)
    print("PHASE 7 ORIGINAL — Tier 2-min trên Phase 5 weak backbone")
    print("=" * 140)
    print(f"{'Scene':>10} | {'B0':>7} | {'DCYCLE':>7} | {'LWEIGHT':>7} | {'TIER2MIN':>8} | "
          f"{'Δ_DC':>7} | {'Δ_LW':>7} | {'Δ_T2M':>7} | {'Synergy':>8}")
    print("-" * 140)

    cols, missing, per_scene = collect_psnrs(sources)
    delta_t2m_per_scene = []
    for s, psnrs in per_scene:
        if psnrs is None:
            print(f"{s:>10} | INCOMPLETE")
            continue
        b0, dc, lw, t2m = psnrs['B0'], psnrs['DCYCLE'], psnrs['LWEIGHT'], psnrs['TIER2MIN']
        d_dc, d_lw, d_t2m = dc - b0, lw - b0, t2m - b0
        synergy = d_t2m - (d_dc + d_lw)
        delta_t2m_per_scene.append((s, d_t2m))
        print(f"{s:>10} | {b0:>7.3f} | {dc:>7.3f} | {lw:>7.3f} | {t2m:>8.3f} | "
              f"{d_dc:>+7.3f} | {d_lw:>+7.3f} | {d_t2m:>+7.3f} | {synergy:>+8.3f}")

    print("-" * 140)
    if not all(len(cols[k]) == len(SCENES) for k in cols):
        print("\n[INCOMPLETE]")
        if missing:
            print(f"Missing logs: {missing[:20]}")
        return

    avgs = {k: statistics.mean(v) for k, v in cols.items()}
    d_dc_avg = avgs['DCYCLE'] - avgs['B0']
    d_lw_avg = avgs['LWEIGHT'] - avgs['B0']
    d_t2m_avg = avgs['TIER2MIN'] - avgs['B0']
    synergy_avg = d_t2m_avg - (d_dc_avg + d_lw_avg)

    print(f"{'AVG':>10} | "
          f"{avgs['B0']:>7.3f} | {avgs['DCYCLE']:>7.3f} | {avgs['LWEIGHT']:>7.3f} | "
          f"{avgs['TIER2MIN']:>8.3f} | "
          f"{d_dc_avg:>+7.3f} | {d_lw_avg:>+7.3f} | {d_t2m_avg:>+7.3f} | {synergy_avg:>+8.3f}")

    n_positive = sum(1 for _, d in delta_t2m_per_scene if d > 0)
    n_breakthrough = sum(1 for _, d in delta_t2m_per_scene if d >= 0.30)

    compute_compute_table(sources, cols)

    print()
    print("=" * 140)
    print("VERDICT — Phase 7 original (Phase 5 weak backbone)")
    print("=" * 140)
    print(f"  Δ_T2M AVG: {d_t2m_avg:+.3f} dB | {n_positive}/{len(SCENES)} positive | {n_breakthrough}/{len(SCENES)} ≥+0.30")

    if d_t2m_avg >= 0.30 and n_positive >= 6:
        verdict, action = "🟢 BREAKTHROUGH", "→ Run Phase 7-FINAL Stage 1 (TIER1) for strong-backbone confirmation"
    elif 0.15 <= d_t2m_avg < 0.30:
        verdict, action = "🟡 SOLID", "→ Run Phase 7-FINAL Stage 1 to verify scale-up; ship recipe paper concurrently"
    else:
        verdict, action = "🔴 STOP — below +0.15 dB", "→ Pivot recipe paper, KHÔNG variant #8"
    print(f"  {verdict}")
    print(f"  {action}")
    if missing:
        print(f"\n[NOTE] Missing logs: {missing[:10]}")


def analyze_stage1():
    """Phase 7-FINAL Stage 1 — D1-O999 family."""
    sources = {
        'TIER1_DAV2_GATE': lambda s: f'{LOG_DIR}/TIER1_DAV2_GATE_{s}.log',
        'TIER1_DC_GATE':   lambda s: f'{LOG_DIR}/TIER1_DC_GATE_{s}.log',
        'TIER1_DC_LW':     lambda s: f'{LOG_DIR}/TIER1_DC_LW_{s}.log',
    }
    print("=" * 140)
    print("PHASE 7-FINAL STAGE 1 — D1-O999 family (DropAnSH + DECAY 0.999)")
    print("=" * 140)
    print(f"{'Scene':>10} | {'DAV2_GATE':>10} | {'DC_GATE':>10} | {'DC_LW':>10} | "
          f"{'Δ_DC':>7} | {'Δ_LW_DC':>8} | {'Δ_combo':>8}")
    print("-" * 140)

    cols, missing, per_scene = collect_psnrs(sources)
    for s, psnrs in per_scene:
        if psnrs is None:
            print(f"{s:>10} | INCOMPLETE")
            continue
        ref = psnrs['TIER1_DAV2_GATE']
        dc_g = psnrs['TIER1_DC_GATE']
        dc_lw = psnrs['TIER1_DC_LW']
        d_dc = dc_g - ref           # signal effect on gate mechanism
        d_lw_dc = dc_lw - dc_g       # mechanism effect with D_cycle
        d_combo = dc_lw - ref        # signal + mechanism combined
        print(f"{s:>10} | {ref:>10.3f} | {dc_g:>10.3f} | {dc_lw:>10.3f} | "
              f"{d_dc:>+7.3f} | {d_lw_dc:>+8.3f} | {d_combo:>+8.3f}")

    print("-" * 140)
    if not all(len(cols[k]) == len(SCENES) for k in cols):
        print("\n[INCOMPLETE]")
        if missing:
            print(f"Missing logs: {missing[:20]}")
        return None

    avgs = {k: statistics.mean(v) for k, v in cols.items()}
    d_dc_avg = avgs['TIER1_DC_GATE'] - avgs['TIER1_DAV2_GATE']
    d_lw_dc_avg = avgs['TIER1_DC_LW'] - avgs['TIER1_DC_GATE']
    d_combo_avg = avgs['TIER1_DC_LW'] - avgs['TIER1_DAV2_GATE']

    print(f"{'AVG':>10} | "
          f"{avgs['TIER1_DAV2_GATE']:>10.3f} | {avgs['TIER1_DC_GATE']:>10.3f} | "
          f"{avgs['TIER1_DC_LW']:>10.3f} | "
          f"{d_dc_avg:>+7.3f} | {d_lw_dc_avg:>+8.3f} | {d_combo_avg:>+8.3f}")

    compute_compute_table(sources, cols)

    # ── Attribution diagnostic ──
    print()
    print("=" * 140)
    print("ATTRIBUTION DIAGNOSTIC — Stage 1")
    print("=" * 140)
    print(f"  Δ_DC      (signal effect, gate kept):       {d_dc_avg:+.3f} dB")
    print(f"  Δ_LW_DC   (mechanism effect, D_cycle kept): {d_lw_dc_avg:+.3f} dB")
    print(f"  Δ_combo   (signal + mechanism):             {d_combo_avg:+.3f} dB")
    print()
    if d_dc_avg > 0.10 and abs(d_lw_dc_avg) <= 0.05:
        hyp = "H_signal pure: signal là chính, mechanism marginal"
    elif abs(d_dc_avg) <= 0.05 and d_lw_dc_avg > 0.10:
        hyp = "H_mechanism pure: mechanism là chính, signal marginal"
    elif d_dc_avg > 0.05 and d_lw_dc_avg > 0.05:
        hyp = "H_dual: cả 2 cần upgrade"
    elif d_dc_avg < -0.05 and d_lw_dc_avg > 0.10:
        hyp = "H_mech_rescue: signal alone hại, nhưng mechanism cứu được khi stack"
    else:
        hyp = "H_both_dead: signal + mechanism đều saturated → pivot"
    print(f"  → {hyp}")

    # ── Verdict ──
    tier1_best = max(avgs['TIER1_DC_GATE'], avgs['TIER1_DC_LW'])
    NO_CRS_REF = 21.21  # current best (D1-noCRS-O999)

    print()
    print("=" * 140)
    print("VERDICT — Stage 1")
    print("=" * 140)
    print(f"  TIER1_DAV2_GATE (reference):  {avgs['TIER1_DAV2_GATE']:.3f} dB (expected ~21.13)")
    print(f"  TIER1_BEST (DC_GATE / DC_LW): {tier1_best:.3f} dB")
    print(f"  No-CRS reference (Tier1):     {NO_CRS_REF:.3f} dB")
    print()
    if tier1_best > NO_CRS_REF + 0.10:
        v, action = "🟢 BREAKTHROUGH", "→ Run Stage 2 confirm cross-backbone universality"
    elif NO_CRS_REF - 0.10 <= tier1_best <= NO_CRS_REF + 0.10:
        v, action = "🟡 Marginal", "→ Run Stage 2 for paper data"
    else:
        v, action = "🔴 STOP", "→ Tier1 saturated; pivot recipe paper. KHÔNG run Stage 2."
    print(f"  {v}")
    print(f"  {action}")
    if missing:
        print(f"\n[NOTE] Missing logs: {missing[:10]}")
    return {
        'd_dc_avg': d_dc_avg,
        'd_lw_dc_avg': d_lw_dc_avg,
        'd_combo_avg': d_combo_avg,
        'tier1_best': tier1_best,
        'tier1_dav2': avgs['TIER1_DAV2_GATE'],
    }


def analyze_stage2():
    """Phase 7-FINAL Stage 2 — A1+B1β family."""
    sources = {
        'TIER2_DAV2_GATE': lambda s: f'{LOG_DIR}/TIER2_DAV2_GATE_{s}.log',
        'TIER2_DC_GATE':   lambda s: f'{LOG_DIR}/TIER2_DC_GATE_{s}.log',
    }
    print("=" * 140)
    print("PHASE 7-FINAL STAGE 2 — A1+B1β family (sh1 + freeze_sh + dropout)")
    print("=" * 140)
    print(f"{'Scene':>10} | {'DAV2_GATE':>10} | {'DC_GATE':>10} | {'Δ_DC_T2':>8}")
    print("-" * 140)

    cols, missing, per_scene = collect_psnrs(sources)
    for s, psnrs in per_scene:
        if psnrs is None:
            print(f"{s:>10} | INCOMPLETE")
            continue
        ref = psnrs['TIER2_DAV2_GATE']
        dc_g = psnrs['TIER2_DC_GATE']
        d_dc = dc_g - ref
        print(f"{s:>10} | {ref:>10.3f} | {dc_g:>10.3f} | {d_dc:>+8.3f}")

    print("-" * 140)
    if not all(len(cols[k]) == len(SCENES) for k in cols):
        print("\n[INCOMPLETE]")
        if missing:
            print(f"Missing logs: {missing[:20]}")
        return None

    avgs = {k: statistics.mean(v) for k, v in cols.items()}
    d_dc_t2_avg = avgs['TIER2_DC_GATE'] - avgs['TIER2_DAV2_GATE']

    print(f"{'AVG':>10} | "
          f"{avgs['TIER2_DAV2_GATE']:>10.3f} | {avgs['TIER2_DC_GATE']:>10.3f} | "
          f"{d_dc_t2_avg:>+8.3f}")

    compute_compute_table(sources, cols)
    if missing:
        print(f"\n[NOTE] Missing logs: {missing[:10]}")
    return {
        'd_dc_t2_avg': d_dc_t2_avg,
        'tier2_dav2': avgs['TIER2_DAV2_GATE'],
        'tier2_dc': avgs['TIER2_DC_GATE'],
    }


def cross_backbone_verdict(stage1, stage2):
    """Cross-backbone universality: D_cycle hoạt động cả 2 backbones?"""
    if stage1 is None or stage2 is None:
        print("\n[SKIP cross-backbone] — Stage 1 or 2 incomplete")
        return
    print()
    print("=" * 140)
    print("CROSS-BACKBONE VERDICT — D_cycle universality test")
    print("=" * 140)
    d1 = stage1['d_dc_avg']
    d2 = stage2['d_dc_t2_avg']
    print(f"  Δ_DC trên D1-O999 backbone (Stage 1): {d1:+.3f} dB")
    print(f"  Δ_DC trên A1+B1β backbone (Stage 2):  {d2:+.3f} dB")
    print()
    if d1 >= 0.10 and d2 >= 0.10:
        v = "🟢 D_cycle UNIVERSAL — works across multiple strong backbones → defendable signal upgrade"
        action = "→ Ship paper claim: 'first signal upgrade beating consistency-based ceiling'"
    elif (d1 >= 0.10) != (d2 >= 0.10):  # XOR
        v = "🟡 D_cycle BACKBONE-SPECIFIC — chỉ work với 1 backbone class"
        action = "→ Recipe paper với 1 ablation row + caveat về backbone dependence"
    else:
        v = "🔴 D_cycle DOESN'T CARRY across strong backbones"
        action = "→ Definitive pivot recipe paper. Phase 7 Δ_DC=+0.125 = artifact của weak backbone."
    print(f"  {v}")
    print(f"  {action}")


def main():
    print(f"[t2min_final_analyze] STAGE={STAGE} | LOG_DIR={LOG_DIR} | B0_LOG_DIR={B0_LOG_DIR}")
    print()
    if STAGE == "phase7":
        analyze_phase7_original()
    elif STAGE == "1":
        analyze_stage1()
    elif STAGE == "2":
        analyze_stage2()
    elif STAGE == "both":
        s1 = analyze_stage1()
        print()
        s2 = analyze_stage2()
        cross_backbone_verdict(s1, s2)
    else:
        print(f"Unknown STAGE={STAGE}. Valid: phase7|1|2|both")
        return

    # Reference targets — luôn print cuối
    print()
    print("=" * 140)
    print("REFERENCE TARGETS (LLFF 3-view)")
    print("=" * 140)
    print(f"  No-CRS Tier1 (D1-noCRS-O999):       21.21 dB  (current project best, no CRS)")
    print(f"  D1-O999 with CRS (Phase 6 ref):     21.13 dB")
    print(f"  A1+B1β reference (Track A+B):       ~20.96 dB")
    print(f"  DOC-GS reported:                    21.38 dB")
    print(f"  BinocularGS:                        21.44 dB")
    print(f"  ICO-GS (SOTA):                      22.20 dB")


if __name__ == "__main__":
    main()
