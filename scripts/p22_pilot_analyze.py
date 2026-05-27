#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 22 — Bước 5] Pilot N=24 3-way analyzer
# File: scripts/p22_pilot_analyze.py  (KEEP LOCAL — server-only)
#
# Pair RoMa v1 (Phase 22 pilot) vs:
#   - MVS (Phase 20 A3-TRIM `no_crsprune_rvis` paired N=24) — primary baseline
#   - RoMa v2 (Phase 21 pilot N=24) — direct v1-vs-v2 comparison
#   - PDCNet+ (Phase 20 dense `trim_full` N=8 seed42 only) — informational
#
# Pre-registered verdict vs MVS (mirror Phase 21):
#   C1: 8-scene avg Δ_PSNR ≥ +0.10 (paired t, CI excludes 0)
#   C2: ≥6/8 scenes mean Δ ≥ 0
#   C3: horns + trex Δ ≥ −0.05 ← key test cho WxBS hypothesis
#   C4: N_gauss avg ≤ MVS × 1.1
#
# v1-vs-v2 direct: per-scene Δ → biết v1 thắng/thua v2 ở scene nào (horns/trex critical)
# ============================================================
"""Phase 22 RoMa v1 pilot — 3-way paired analyzer."""

import os
import re
import math
from pathlib import Path

import numpy as np

SCENES = os.environ.get(
    "SCENES", "fern flower fortress horns leaves orchids room trex").split()
SEEDS = os.environ.get("SEEDS", "42 137 9999").split()
PILOT_DIR = Path(os.environ.get("PILOT_DIR", "logs/p22_pilot"))
MVS_DIR   = Path(os.environ.get("MVS_DIR",   "logs/p20_ablation/no_crsprune_rvis"))
V2_DIR    = Path(os.environ.get("V2_DIR",    "logs/p21_pilot"))
PDC_DIR   = Path(os.environ.get("PDC_DIR",   "logs/p20_ablation_dense/trim_full"))

C1_DELTA_MIN = 0.10
C3_DELTA_MIN = -0.05

PSNR_PAT = re.compile(r"Best test PSNR:\s*([\d.eE+\-]+)")
ROW_PAT  = re.compile(
    r"\|\s*test\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
    r"[^|]*\|\s*(\d+)")


def parse_log(path: Path):
    if not path.is_file():
        return None
    txt = path.read_text(encoding="utf-8", errors="ignore")
    rows = ROW_PAT.findall(txt)
    if rows:
        psnr, ssim, lpips, n = rows[-1]
        return float(psnr), float(ssim), float(lpips), int(n)
    mp = PSNR_PAT.search(txt)
    return (float(mp.group(1)), None, None, None) if mp else None


def collect(root: Path):
    out = {}
    for sc in SCENES:
        for sd in SEEDS:
            f = root / f"A3_seed{sd}_{sc}.log"
            v = parse_log(f)
            if v is not None:
                out[(sc, sd)] = v
    return out


def paired_ci(deltas, conf=0.95):
    arr = np.array(deltas)
    n = len(arr)
    if n < 2:
        return float(arr.mean()) if n else float("nan"), float("nan"), float("nan")
    m, sd = arr.mean(), arr.std(ddof=1)
    sem = sd / math.sqrt(n)
    z = 1.96 if conf == 0.95 else 2.576
    return float(m), float(m - z * sem), float(m + z * sem)


def fmt(v, w=8, prec=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return f"{'?':>{w}}"
    return f"{v:{w}.{prec}f}"


def main():
    print("=" * 100)
    print(f"[P22 v1 pilot] PILOT_DIR={PILOT_DIR}")
    print(f"              MVS_DIR ={MVS_DIR}  (primary baseline, N=24)")
    print(f"              V2_DIR  ={V2_DIR}  (RoMa v2 Phase 21 comparison, N=24)")
    print(f"              PDC_DIR ={PDC_DIR}  (PDCNet+ Phase 20 dense, N=8 seed42)")
    print(f"              SCENES={SCENES}  SEEDS={SEEDS}")
    print("=" * 100)

    pilot = collect(PILOT_DIR)
    mvs   = collect(MVS_DIR)
    v2    = collect(V2_DIR)
    pdc   = collect(PDC_DIR)

    print(f"  found: pilot(v1)={len(pilot)}/24  mvs={len(mvs)}/24  v2={len(v2)}/24  pdc={len(pdc)}/24")
    print(f"  paired: v1∩MVS={len(set(pilot)&set(mvs))}  v1∩v2={len(set(pilot)&set(v2))}  v1∩PDC={len(set(pilot)&set(pdc))}")
    print()

    # ── PER-SEED v1 detail ──
    print("─" * 88)
    print(f"  Per-seed RoMa v1 PSNR (1 row / scene, 3 cols / seed):")
    print(f"  {'Scene':10s}  {'seed42':>9s}  {'seed137':>9s}  {'seed9999':>9s}  "
          f"{'mean':>8s}  {'std':>6s}")
    print("─" * 88)
    per_scene_v1 = {}
    for sc in SCENES:
        vals, cells = [], []
        for sd in SEEDS:
            if (sc, sd) in pilot:
                v = pilot[(sc, sd)][0]
                vals.append(v); cells.append(f"{v:9.3f}")
            else:
                cells.append("        ?")
        if vals:
            m = float(np.mean(vals))
            s = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
            per_scene_v1[sc] = (m, s, vals)
            print(f"  {sc:10s}  {cells[0]}  {cells[1]}  {cells[2]}  {m:8.3f}  {s:6.3f}")
        else:
            print(f"  {sc:10s}  (no data)")
    print("─" * 88)
    if per_scene_v1:
        all_means = [v[0] for v in per_scene_v1.values()]
        all_stds = [v[1] for v in per_scene_v1.values() if not math.isnan(v[1])]
        print(f"  8-scene avg: PSNR mean={np.mean(all_means):.3f}  avg seed-std={np.mean(all_stds):.3f}")
    print()

    # ── 3-WAY per-scene ──
    print("─" * 110)
    print(f"  3-WAY per-scene (RoMa v1 PSNR | Δ-vs-MVS(3s) | Δ-vs-V2(3s) | Δ-vs-PDC(n) | SSIM-vs-MVS | LPIPS-vs-MVS | N R1/v2/mvs)")
    print("─" * 110)
    per_scene_d_mvs = {}
    per_scene_d_v2 = {}
    per_scene_d_pdc = {}
    rows_mvs, rows_v2 = [], []  # 24-pair lists
    for sc in SCENES:
        rm = per_scene_v1.get(sc)
        p_avg = rm[0] if rm else float("nan")

        # vs MVS paired (3 seeds)
        mvs_pairs = [(sd, pilot[(sc, sd)], mvs[(sc, sd)])
                     for sd in SEEDS if (sc, sd) in pilot and (sc, sd) in mvs]
        if mvs_pairs:
            d_mvs = float(np.mean([x[1][0] - x[2][0] for x in mvs_pairs]))
            ssim_pairs = [x[1][1] - x[2][1] for x in mvs_pairs
                          if x[1][1] is not None and x[2][1] is not None]
            lpips_pairs = [x[1][2] - x[2][2] for x in mvs_pairs
                           if x[1][2] is not None and x[2][2] is not None]
            d_ssim_mvs = float(np.mean(ssim_pairs)) if ssim_pairs else float("nan")
            d_lpips_mvs = float(np.mean(lpips_pairs)) if lpips_pairs else float("nan")
            mvs_n = int(np.mean([x[2][3] for x in mvs_pairs if x[2][3]]))
            for sd, pp, mm in mvs_pairs:
                rows_mvs.append(pp[0] - mm[0])
        else:
            d_mvs = float("nan"); d_ssim_mvs = float("nan"); d_lpips_mvs = float("nan"); mvs_n = 0

        # vs v2 paired (3 seeds)
        v2_pairs = [(sd, pilot[(sc, sd)], v2[(sc, sd)])
                    for sd in SEEDS if (sc, sd) in pilot and (sc, sd) in v2]
        if v2_pairs:
            d_v2 = float(np.mean([x[1][0] - x[2][0] for x in v2_pairs]))
            v2_n = int(np.mean([x[2][3] for x in v2_pairs if x[2][3]]))
            for sd, pp, vv in v2_pairs:
                rows_v2.append(pp[0] - vv[0])
        else:
            d_v2 = float("nan"); v2_n = 0

        # vs PDC paired (often only seed42)
        pdc_pairs = [(sd, pilot[(sc, sd)], pdc[(sc, sd)])
                     for sd in SEEDS if (sc, sd) in pilot and (sc, sd) in pdc]
        if pdc_pairs:
            d_pdc = float(np.mean([x[1][0] - x[2][0] for x in pdc_pairs]))
            n_pdc = len(pdc_pairs)
        else:
            d_pdc = float("nan"); n_pdc = 0

        p_n = int(np.mean([pilot[(sc, sd)][3] for sd in SEEDS
                           if (sc, sd) in pilot and pilot[(sc, sd)][3]])) if rm else 0

        mark_mvs = " ⭐" if d_mvs > 0.1 else (" ✗" if d_mvs < -0.05 else "")
        mark_v2 = "★" if d_v2 > 0.1 else ("✗" if d_v2 < -0.1 else "≈")
        d_pdc_str = f"{d_pdc:+.3f}({n_pdc})" if not math.isnan(d_pdc) else "    ?(0)"

        print(f"  {sc:10s} {fmt(p_avg)} {fmt(d_mvs, prec=3)}{mark_mvs} "
              f"{fmt(d_v2, prec=3):>8s}{mark_v2} "
              f"{d_pdc_str:>10s}  "
              f"{fmt(d_ssim_mvs, prec=4):>9s}  {fmt(d_lpips_mvs, prec=4):>10s}   "
              f"{p_n}/{v2_n}/{mvs_n}")

        per_scene_d_mvs[sc] = d_mvs
        per_scene_d_v2[sc] = d_v2
        per_scene_d_pdc[sc] = d_pdc
    print("─" * 110)

    # ── 8-scene aggregates ──
    if rows_mvs:
        m, lo, hi = paired_ci(rows_mvs)
        sig = "SIG" if (lo > 0 or hi < 0) else "ns"
        print(f"\n  8-scene Δ vs MVS : mean={m:+.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]  {sig}  (N={len(rows_mvs)})")
    if rows_v2:
        m2, lo2, hi2 = paired_ci(rows_v2)
        sig2 = "SIG" if (lo2 > 0 or hi2 < 0) else "ns"
        print(f"  8-scene Δ vs v2  : mean={m2:+.4f}  95% CI=[{lo2:+.4f}, {hi2:+.4f}]  {sig2}  (N={len(rows_v2)})")

    # N_gauss ratios
    all_n_p = [pilot[k][3] for k in pilot if pilot[k][3]]
    all_n_m = [mvs[k][3] for k in mvs if mvs[k][3]]
    all_n_v2 = [v2[k][3] for k in v2 if v2[k][3]]
    if all_n_p and all_n_m:
        print(f"  8-scene N_gauss: v1/MVS = {np.mean(all_n_p)/np.mean(all_n_m):.3f}  ({int(np.mean(all_n_p))}/{int(np.mean(all_n_m))})")
    if all_n_p and all_n_v2:
        print(f"  8-scene N_gauss: v1/v2  = {np.mean(all_n_p)/np.mean(all_n_v2):.3f}  ({int(np.mean(all_n_p))}/{int(np.mean(all_n_v2))})")

    # ── PRE-REGISTERED VERDICT vs MVS (primary) ──
    if rows_mvs:
        print("\n" + "=" * 80)
        print(f"PRE-REGISTERED VERDICT vs MVS (N_paired={len(rows_mvs)}):")
        print("=" * 80)
        c1 = (m >= C1_DELTA_MIN) and (lo > 0)
        n_pos = sum(1 for d in per_scene_d_mvs.values() if not math.isnan(d) and d >= 0)
        c2 = n_pos >= 6
        c3_h = per_scene_d_mvs.get("horns", float("-inf"))
        c3_t = per_scene_d_mvs.get("trex", float("-inf"))
        c3 = (c3_h >= C3_DELTA_MIN) and (c3_t >= C3_DELTA_MIN)
        c4 = bool(all_n_p and all_n_m) and (np.mean(all_n_p) <= np.mean(all_n_m) * 1.1)

        print(f"  C1  Δ ≥ +{C1_DELTA_MIN:.2f} AND CI>0 : {'✓ PASS' if c1 else '✗ FAIL'}  (Δ={m:+.4f}, CI lo={lo:+.4f})")
        print(f"  C2  ≥6/8 scenes Δ≥0       : {'✓ PASS' if c2 else '✗ FAIL'}  ({n_pos}/8 positive)")
        print(f"  C3  horns+trex Δ ≥ {C3_DELTA_MIN:+.2f}    : {'✓ PASS' if c3 else '✗ FAIL'}  "
              f"(horns={c3_h:+.3f}, trex={c3_t:+.3f})")
        print(f"  C4  N_gauss ≤ MVS×1.1     : {'✓ PASS' if c4 else '✗ FAIL'}")

        print(f"\nWxBS hypothesis test (Phase 22 raison d'être):")
        c3_h_v2 = -0.636  # Phase 21 v2 horns Δ vs MVS (locked reference)
        c3_t_v2 = -0.177
        h_imp = c3_h - c3_h_v2
        t_imp = c3_t - c3_t_v2
        print(f"  horns: v1 Δ_MVS = {c3_h:+.3f}  vs  v2 Δ_MVS = {c3_h_v2:+.3f}  → v1 improves horns by {h_imp:+.3f}")
        print(f"  trex : v1 Δ_MVS = {c3_t:+.3f}  vs  v2 Δ_MVS = {c3_t_v2:+.3f}  → v1 improves trex  by {t_imp:+.3f}")
        if h_imp > 0.2 and t_imp > 0.1:
            print(f"  → 🎯 WxBS hypothesis CONFIRMED — v1 fix thin-structure rõ rệt")
        elif h_imp > 0 and t_imp > 0:
            print(f"  → 🟡 partial WxBS confirm — v1 hơi better, không decisive")
        else:
            print(f"  → ❌ WxBS hypothesis REFUTED — v1 không fix horns/trex (structural)")

        print(f"\nOverall vs-MVS verdict:")
        if c1 and c2 and c3 and c4:
            print("  🎯 ALL PASS → commit RoMa v1")
        elif c1 and c2 and not c3:
            print("  🟡 C1+C2 PASS, C3 FAIL (cùng Phase 21 v2 pattern)")
        elif not c1:
            print("  ❌ C1 FAIL → RoMa v1 không win")
        else:
            print("  ⚠ partial — đọc chi tiết")

    # ── v1 vs v2 direct ──
    if rows_v2:
        print("\n" + "=" * 80)
        print(f"v1 vs v2 DIRECT (N_paired={len(rows_v2)}):")
        print("=" * 80)
        n_pos_v2 = sum(1 for d in per_scene_d_v2.values() if not math.isnan(d) and d >= 0)
        print(f"  Mean Δ (v1 − v2)    : {m2:+.4f}  CI[{lo2:+.4f}, {hi2:+.4f}]  {sig2}")
        print(f"  Per-scene wins      : v1 beats v2 in {n_pos_v2}/8 scenes")
        win_scenes = [sc for sc, d in per_scene_d_v2.items() if not math.isnan(d) and d > 0.05]
        lose_scenes = [sc for sc, d in per_scene_d_v2.items() if not math.isnan(d) and d < -0.05]
        if win_scenes: print(f"  v1 wins (Δ>+0.05)   : {win_scenes}")
        if lose_scenes: print(f"  v1 loses (Δ<-0.05)  : {lose_scenes}")
        if abs(m2) < 0.05 and len(win_scenes) <= 2 and len(lose_scenes) <= 2:
            print(f"  → v1 ≈ v2 overall — KHÔNG worth swap nếu C3 cùng fail")
        elif m2 > 0.05:
            print(f"  → v1 BETTER than v2 overall — consider commit v1 thay v2")
        else:
            print(f"  → v1 WORSE than v2 overall — KHÔNG commit v1")


if __name__ == "__main__":
    main()
