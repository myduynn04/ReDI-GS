#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 21 — Bước 4] Pilot N=24 paired analyzer
# File: scripts/p21_pilot_analyze.py  (KEEP LOCAL — server-only)
#
# Pair RoMa (Phase 21 pilot) vs PDCNet+ (Phase 20 dense trim_full) per
# (scene, seed). 24 paired Δ → mean / 95% CI / per-scene / per-seed.
#
# Pre-registered verdict (PHẢI lock TRƯỚC khi xem số):
#   C1: 8-scene avg Δ_PSNR ≥ +0.10 (paired t, CI excludes 0)
#   C2: ≥6/8 scenes mean Δ ≥ 0
#   C3: horns + trex Δ ≥ −0.05 (no thin-structure regress — Phase 18 failure case)
#   C4: N_gauss avg ≤ PDCNet+ N_gauss (free compute win)
#
# Verdict combinations:
#   C1+C2+C3+C4 PASS    → GO commit RoMa init
#   C1+C2 PASS, C3 fail → CONDITIONAL (như Phase 18) — case-by-case scenes
#   C1 fail             → NO commit, axis dead
#
# Optional override (default = Phase 20 dense trim_full baseline):
#   PILOT_DIR=logs/p21_pilot  BASE_DIR=logs/p20_ablation_dense/trim_full \
#       python scripts/p21_pilot_analyze.py
# ============================================================
"""Phase 21 RoMa pilot — paired Δ vs PDCNet+ dense baseline."""

import os
import re
import math
from pathlib import Path

import numpy as np

SCENES = os.environ.get(
    "SCENES", "fern flower fortress horns leaves orchids room trex").split()
SEEDS = os.environ.get("SEEDS", "42 137 9999").split()
PILOT_DIR = Path(os.environ.get("PILOT_DIR", "logs/p21_pilot"))
BASE_DIR = Path(os.environ.get("BASE_DIR", "logs/p20_ablation_dense/trim_full"))
MVS_DIR  = Path(os.environ.get("MVS_DIR",  "logs/p20_ablation/no_crsprune_rvis"))

# Pre-registered thresholds (do not tune)
C1_DELTA_MIN = 0.10
C3_DELTA_MIN = -0.05
C3_SCENES = ["horns", "trex"]

# Regex patterns
PSNR_PAT = re.compile(r"Best test PSNR:\s*([\d.eE+\-]+)")
ROW_PAT  = re.compile(
    r"\|\s*test\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
    r"[^|]*\|\s*(\d+)")     # PSNR | SSIM | LPIPS | L1 | N_gauss


def parse_log(path: Path):
    """Return (psnr, ssim, lpips, n_gauss) or None."""
    if not path.is_file():
        return None
    txt = path.read_text(encoding="utf-8", errors="ignore")
    # Prefer summary table last test row
    rows = ROW_PAT.findall(txt)
    if rows:
        psnr, ssim, lpips, n = rows[-1]
        return float(psnr), float(ssim), float(lpips), int(n)
    # Fallback to "Best test PSNR"
    mp = PSNR_PAT.search(txt)
    return (float(mp.group(1)), None, None, None) if mp else None


def collect(root: Path):
    """Return dict (scene, seed) → (psnr, ssim, lpips, n_gauss)."""
    out = {}
    for sc in SCENES:
        for sd in SEEDS:
            f = root / f"A3_seed{sd}_{sc}.log"
            v = parse_log(f)
            if v is not None:
                out[(sc, sd)] = v
    return out


def paired_ci(deltas: list, conf: float = 0.95) -> tuple:
    """Return (mean, lo, hi) — paired-Δ mean + 95% CI using normal-approx (N=24 fine)."""
    arr = np.array(deltas)
    n = len(arr)
    if n < 2:
        return float(arr.mean()) if n else float("nan"), float("nan"), float("nan")
    m = arr.mean()
    sd = arr.std(ddof=1)
    sem = sd / math.sqrt(n)
    z = 1.96 if conf == 0.95 else 2.576
    return float(m), float(m - z * sem), float(m + z * sem)


def fmt(v, w=8, prec=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return f"{'?':>{w}}"
    return f"{v:{w}.{prec}f}"


def main():
    print("=" * 88)
    print(f"[P21 pilot] PILOT_DIR={PILOT_DIR}")
    print(f"           BASE_DIR ={BASE_DIR} (Phase 20 PDCNet+ A3-TRIM — primary baseline cho C1-C4)")
    print(f"           MVS_DIR  ={MVS_DIR}  (Phase 20 MVS A3-TRIM — bonus reference)")
    print(f"           SCENES={SCENES}  SEEDS={SEEDS}")
    print("=" * 88)

    pilot = collect(PILOT_DIR)
    base = collect(BASE_DIR)
    mvs  = collect(MVS_DIR)

    n_p, n_b, n_m = len(pilot), len(base), len(mvs)
    print(f"  found: pilot={n_p}/24  base={n_b}/24  mvs={n_m}/24")
    if n_b < 24:
        print(f"  ⚠ PDCNet+ baseline (Phase 20 dense) THIẾU {24-n_b} runs — "
              f"có thể chỉ chạy 1 seed lúc đó. Δ-vs-PDCNet+ giới hạn N={n_b}.")
    paired_pdc = sorted(set(pilot.keys()) & set(base.keys()))
    paired_mvs = sorted(set(pilot.keys()) & set(mvs.keys()))
    print(f"  paired runs: vs PDCNet+ = {len(paired_pdc)}/24,  vs MVS = {len(paired_mvs)}/24")
    print()

    # ── PER-SEED RoMa detail (user-requested) ──
    print("─" * 88)
    print(f"  Per-seed RoMa PSNR (1 row / scene, 3 cols / seed):")
    print(f"  {'Scene':10s}  {'seed42':>9s}  {'seed137':>9s}  {'seed9999':>9s}  "
          f"{'mean':>8s}  {'std':>6s}")
    print("─" * 88)
    per_scene_pilot = {}
    for sc in SCENES:
        vals = []
        cells = []
        for sd in SEEDS:
            if (sc, sd) in pilot:
                v = pilot[(sc, sd)][0]
                vals.append(v)
                cells.append(f"{v:9.3f}")
            else:
                cells.append("        ?")
        if vals:
            m = float(np.mean(vals))
            s = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
            per_scene_pilot[sc] = (m, s, vals)
            print(f"  {sc:10s}  {cells[0]}  {cells[1]}  {cells[2]}  "
                  f"{m:8.3f}  {s:6.3f}")
        else:
            print(f"  {sc:10s}  (no data)")
    print("─" * 88)
    # 8-scene avg of 3-seed-means
    if per_scene_pilot:
        all_means = [v[0] for v in per_scene_pilot.values()]
        all_stds  = [v[1] for v in per_scene_pilot.values() if not math.isnan(v[1])]
        print(f"  {'8-scene avg':10s}  ({len(all_means)} scenes)  "
              f"PSNR mean={np.mean(all_means):.3f}  "
              f"avg seed-std={np.mean(all_stds):.3f}")
    print()

    # Per (scene, seed) Δ — vs PDCNet+ (paired set)
    rows = []
    for sc, sd in paired_pdc:
        p_psnr, p_ssim, p_lpips, p_n = pilot[(sc, sd)]
        b_psnr, b_ssim, b_lpips, b_n = base[(sc, sd)]
        rows.append({
            "scene": sc, "seed": sd,
            "p_psnr": p_psnr, "b_psnr": b_psnr, "d_psnr": p_psnr - b_psnr,
            "p_ssim": p_ssim, "b_ssim": b_ssim,
            "d_ssim": (p_ssim - b_ssim) if (p_ssim and b_ssim) else None,
            "p_lpips": p_lpips, "b_lpips": b_lpips,
            "d_lpips": (p_lpips - b_lpips) if (p_lpips and b_lpips) else None,
            "p_n": p_n, "b_n": b_n,
        })

    # Per-scene aggregation — 3-seed RoMa vs PDC (paired = limited N) vs MVS (paired)
    print("─" * 110)
    print(f"  {'Scene':10s}  RoMa(3s)  PDC+(p)   MVS(3s)   Δ-PDC(n)   Δ-MVS(3s)  "
          f"ΔSSIM-MVS  ΔLPIPS-MVS  N (R/P/M)")
    print("─" * 110)
    per_scene_delta_pdc = {}    # vs PDCNet+ (limited by paired-PDC, often N=1 per scene)
    per_scene_delta_mvs = {}    # vs MVS (full N=3 per scene)
    per_scene_n = {}
    for sc in SCENES:
        # RoMa 3-seed avg (always full)
        rm = per_scene_pilot.get(sc)
        p_avg = rm[0] if rm else float("nan")
        p_n = int(np.mean([pilot[(sc, sd)][3] for sd in SEEDS
                           if (sc, sd) in pilot and pilot[(sc, sd)][3]])) if rm else 0

        # PDC paired (often only seed42)
        pdc_pairs = [(sd, pilot[(sc, sd)][0], base[(sc, sd)][0])
                     for sd in SEEDS if (sc, sd) in pilot and (sc, sd) in base]
        if pdc_pairs:
            b_avg = float(np.mean([x[2] for x in pdc_pairs]))
            d_pdc = float(np.mean([x[1] - x[2] for x in pdc_pairs]))
            b_n = int(np.mean([base[(sc, sd)][3] for sd, _, _ in pdc_pairs
                               if base[(sc, sd)][3]]))
            n_pdc_pairs = len(pdc_pairs)
        else:
            b_avg = float("nan"); d_pdc = float("nan"); b_n = 0; n_pdc_pairs = 0

        # MVS paired (full 3 seeds)
        mvs_pairs = [(sd, pilot[(sc, sd)], mvs[(sc, sd)])
                     for sd in SEEDS if (sc, sd) in pilot and (sc, sd) in mvs]
        if mvs_pairs:
            m_avg = float(np.mean([x[2][0] for x in mvs_pairs]))
            d_mvs = float(np.mean([x[1][0] - x[2][0] for x in mvs_pairs]))
            m_n = int(np.mean([x[2][3] for x in mvs_pairs if x[2][3]]))
            # SSIM/LPIPS Δ vs MVS (paired)
            ssim_pairs = [x[1][1] - x[2][1] for x in mvs_pairs
                          if x[1][1] is not None and x[2][1] is not None]
            lpips_pairs = [x[1][2] - x[2][2] for x in mvs_pairs
                           if x[1][2] is not None and x[2][2] is not None]
            d_ssim_mvs = float(np.mean(ssim_pairs)) if ssim_pairs else float("nan")
            d_lpips_mvs = float(np.mean(lpips_pairs)) if lpips_pairs else float("nan")
        else:
            m_avg = float("nan"); d_mvs = float("nan"); m_n = 0
            d_ssim_mvs = float("nan"); d_lpips_mvs = float("nan")

        mark = " ⭐" if d_mvs > 0.1 else (" ✗" if d_mvs < -0.05 else "")
        d_pdc_str = f"{d_pdc:+.3f}({n_pdc_pairs})" if not math.isnan(d_pdc) else "    ?(0)"
        print(f"  {sc:10s}  {fmt(p_avg)}  {fmt(b_avg)}  {fmt(m_avg)}  "
              f"{d_pdc_str:>10s}  {fmt(d_mvs, prec=3):>9s}  "
              f"{fmt(d_ssim_mvs, prec=4):>9s}  {fmt(d_lpips_mvs, prec=4):>10s}   "
              f"{p_n}/{b_n}/{m_n}{mark}")
        per_scene_delta_pdc[sc] = d_pdc
        per_scene_delta_mvs[sc] = d_mvs
        per_scene_n[sc] = (p_n, b_n, m_n)
    # backward compat — verdict block dùng tên cũ
    per_scene_delta = per_scene_delta_pdc
    print("─" * 110)

    # 8-scene overall vs PDCNet+ (primary, for C1)
    all_d_psnr = [r["d_psnr"] for r in rows]
    m, lo, hi = paired_ci(all_d_psnr)
    sig = "SIG" if (lo > 0 or hi < 0) else "ns"
    print(f"\n  8-scene paired Δ vs PDCNet+ : mean={m:+.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]  {sig}")

    # 8-scene overall vs MVS (bonus reference)
    mvs_paired = [(sc, sd) for sc in SCENES for sd in SEEDS
                  if (sc, sd) in pilot and (sc, sd) in mvs]
    if mvs_paired:
        all_d_mvs = [pilot[k][0] - mvs[k][0] for k in mvs_paired]
        mm, mlo, mhi = paired_ci(all_d_mvs)
        msig = "SIG" if (mlo > 0 or mhi < 0) else "ns"
        print(f"  8-scene paired Δ vs MVS     : mean={mm:+.4f}  "
              f"95% CI=[{mlo:+.4f}, {mhi:+.4f}]  {msig}  (N={len(all_d_mvs)})")

    all_n_p = [r["p_n"] for r in rows if r["p_n"]]
    all_n_b = [r["b_n"] for r in rows if r["b_n"]]
    all_n_m = [mvs[k][3] for k in mvs_paired if mvs[k][3]]
    if all_n_p and all_n_b:
        n_ratio_pdc = np.mean(all_n_p) / np.mean(all_n_b)
        print(f"  8-scene N_gauss: RoMa/PDCNet+ = {n_ratio_pdc:.3f}  "
              f"({int(np.mean(all_n_p))} / {int(np.mean(all_n_b))})")
    if all_n_p and all_n_m:
        n_ratio_mvs = np.mean(all_n_p) / np.mean(all_n_m)
        print(f"  8-scene N_gauss: RoMa/MVS     = {n_ratio_mvs:.3f}  "
              f"({int(np.mean(all_n_p))} / {int(np.mean(all_n_m))})")

    # ── PRE-REGISTERED VERDICT — primary vs PDCNet+ (như đã lock) ──
    print("\n" + "=" * 80)
    print(f"PRE-REGISTERED VERDICT vs PDCNet+ (N_paired={len(rows)}):")
    print("=" * 80)
    c1 = (m >= C1_DELTA_MIN) and (lo > 0)
    n_pos = sum(1 for d in per_scene_delta.values()
                if not math.isnan(d) and d >= 0)
    n_scenes_with_pdc = sum(1 for d in per_scene_delta.values() if not math.isnan(d))
    c2 = n_pos >= 6
    c3_horns = per_scene_delta.get("horns", float("-inf"))
    c3_trex = per_scene_delta.get("trex", float("-inf"))
    c3 = (c3_horns >= C3_DELTA_MIN) and (c3_trex >= C3_DELTA_MIN)
    c4 = bool(all_n_p and all_n_b) and (np.mean(all_n_p) <= np.mean(all_n_b))

    print(f"  C1  Δ ≥ +{C1_DELTA_MIN:.2f} AND CI>0 : {'✓ PASS' if c1 else '✗ FAIL'}  "
          f"(Δ={m:+.4f}, CI lo={lo:+.4f})")
    print(f"  C2  ≥6/8 scenes Δ≥0       : {'✓ PASS' if c2 else '✗ FAIL'}  "
          f"({n_pos}/{n_scenes_with_pdc} positive — chỉ {n_scenes_with_pdc} scene có PDC paired)")
    print(f"  C3  horns+trex Δ ≥ {C3_DELTA_MIN:+.2f}    : {'✓ PASS' if c3 else '✗ FAIL'}  "
          f"(horns={c3_horns:+.3f}, trex={c3_trex:+.3f})")
    print(f"  C4  N_gauss ≤ PDCNet+      : {'✓ PASS' if c4 else '✗ FAIL'}")

    if len(rows) < 24:
        print(f"\n  ⚠ vs-PDCNet+ verdict N={len(rows)}<24 — PDC baseline THIẾU "
              f"{24-len(rows)} runs (Phase 20 dense 1-seed). Verdict UNDERPOWERED.")
        print(f"  → cần chạy bổ sung PDC baseline seed 137 + 9999 (16 runs ~30 phút) "
              f"để vs-PDC verdict đầy đủ N=24.")

    # ── BONUS VERDICT vs MVS (full N=24 paired) ──
    if mvs_paired and len(mvs_paired) >= 20:
        print("\n" + "=" * 80)
        print(f"BONUS VERDICT vs MVS (N_paired={len(mvs_paired)}):")
        print("=" * 80)
        c1_mvs = (mm >= C1_DELTA_MIN) and (mlo > 0)
        n_pos_mvs = sum(1 for d in per_scene_delta_mvs.values()
                        if not math.isnan(d) and d >= 0)
        c2_mvs = n_pos_mvs >= 6
        c3h_mvs = per_scene_delta_mvs.get("horns", float("-inf"))
        c3t_mvs = per_scene_delta_mvs.get("trex", float("-inf"))
        c3_mvs = (c3h_mvs >= C3_DELTA_MIN) and (c3t_mvs >= C3_DELTA_MIN)
        c4_mvs = bool(all_n_p and all_n_m) and (np.mean(all_n_p) <= np.mean(all_n_m) * 1.1)

        print(f"  C1  Δ ≥ +{C1_DELTA_MIN:.2f} AND CI>0 : {'✓ PASS' if c1_mvs else '✗ FAIL'}  "
              f"(Δ={mm:+.4f}, CI lo={mlo:+.4f})")
        print(f"  C2  ≥6/8 scenes Δ≥0       : {'✓ PASS' if c2_mvs else '✗ FAIL'}  "
              f"({n_pos_mvs}/8 positive)")
        print(f"  C3  horns+trex Δ ≥ {C3_DELTA_MIN:+.2f}    : {'✓ PASS' if c3_mvs else '✗ FAIL'}  "
              f"(horns={c3h_mvs:+.3f}, trex={c3t_mvs:+.3f})")
        print(f"  C4  N_gauss ≈ MVS (≤1.1×)  : {'✓ PASS' if c4_mvs else '✗ FAIL'}")

        print("\nOverall vs-MVS verdict:")
        if c1_mvs and c2_mvs and c3_mvs and c4_mvs:
            print("  🎯 ALL PASS → RoMa beats MVS clean, GO commit")
        elif c1_mvs and c2_mvs and not c3_mvs:
            print("  🟡 C1+C2 PASS, C3 FAIL → real overall gain, regress thin-structure")
        elif c1_mvs and not c2_mvs:
            print("  🟡 C1 PASS, C2 FAIL → mean win driven by few scenes")
        elif not c1_mvs:
            print("  ❌ C1 FAIL vs MVS → RoMa = MVS effectively")
        else:
            print("  ⚠ partial — đọc chi tiết từng C trên")


if __name__ == "__main__":
    main()
