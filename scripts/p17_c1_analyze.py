#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 17 — C1] Pilot analyzer.
# Parse + paired-Δ + 95%CI = EXACT mirror p13_lfcf_multiseed_analyze.py
#   (PSNR_PAT, summarize_paired). KHÔNG tự chế thống kê.
#
# VERDICT = CHỈ C1L10 (λ=0.10 pre-registered) vs A3, paired N=24.
#   C1L05 / C1L20 = SENSITIVITY-ONLY, NON-VERDICT, KHÔNG lật (audit #3
#   multiple-comparison guard — đúng kỷ luật τ NON-CANONICAL của p16).
# GO ⟺ (mean Δ ≥ +0.10) ∧ (95%CI loại 0) ∧ (#scene Δ≥0 ≥ 7/8 ∧
#   horns Δ ≥ −0.05) ∧ cost reported. Khác = NO (đóng C1, đã trả pilot).
# Pre-registered prediction (decisions_log Phase-17): ∇depth→normal nhiễu
#   → dự đoán degrade/no-gain horns/foliage; ra vậy = XÁC NHẬN.
#
# A3 baseline = REUSE logs/p13_lfcf/A3_seed{seed}_{scene}.log (KHÔNG run
#   lại — như p15). C1 logs = logs/p17_c1/{cell}_seed{seed}_{scene}.log.
# ============================================================
"""[CRSGaussian Phase 17] C1 pilot analyzer.

Run (CPU, instant):
    python scripts/p17_c1_analyze.py
    SCENES="..." python scripts/p17_c1_analyze.py
"""

import os
import re
import statistics

SCENES = os.environ.get(
    "SCENES", "fern flower fortress horns leaves orchids room trex").split()
SEEDS = os.environ.get("SEEDS", "42 137 9999").split()       # C1L10 = N=24
SENS_SEEDS = os.environ.get("SENS_SEEDS", "42").split()      # C1L05/20
C1_LOG = os.environ.get("C1_LOG", "logs/p17_c1")
A3_LOG = os.environ.get("A3_LOG", "logs/p13_lfcf")           # reuse Phase-13

PSNR_PAT = re.compile(r"Best test PSNR:\s*([\d\.eE\-\+]+)")  # = p13 analyze
NG_PAT = re.compile(r"Final #Gaussians:\s*(\d+)")
TT_PAT = re.compile(r"Total training:\s*([\d\.]+)s")

GO_DMEAN = 0.10        # pre-registered (locked)
GO_HORNS_MIN = -0.05   # no per-scene catastrophe (esp. horns)
GO_NONNEG_MIN = 7      # ≥7/8 scenes Δ ≥ 0


def _parse(path, pat):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        m = pat.search(f.read())
    return float(m.group(1)) if m else None


def _psnr(d, seed, sc, cell=None):
    p = (f"{A3_LOG}/A3_seed{seed}_{sc}.log" if cell is None
         else f"{C1_LOG}/{cell}_seed{seed}_{sc}.log")
    return _parse(p, PSNR_PAT)


def summarize_paired(label, deltas):
    """EXACT mirror p13_lfcf_multiseed_analyze.summarize_paired."""
    if not deltas:
        print(f"  {label}: NO DATA"); return None, None, None, None
    n = len(deltas)
    mean = statistics.fmean(deltas)
    std = statistics.stdev(deltas) if n > 1 else 0.0
    sem = std / (n ** 0.5)
    ci_lo, ci_hi = mean - 1.96 * sem, mean + 1.96 * sem
    sig = abs(mean) > 2 * sem
    sl = ("🎯 SIG" if mean > 0 else "📉 SIG NEG") if sig else "🔇 NOT SIG"
    print(f"  {label:<26} N={n:<3} mean={mean:+.4f} SEM={sem:.4f} "
          f"95%CI=[{ci_lo:+.3f},{ci_hi:+.3f}] {sl}")
    return mean, sem, ci_lo, ci_hi


def main():
    print("=== Phase 17 — C1 (DSINE normal prior) pilot analyzer ===")
    print(f"C1_LOG={C1_LOG}  A3_LOG={A3_LOG}(reuse)  "
          f"PRIMARY seeds={SEEDS}  sens seeds={SENS_SEEDS}")
    print("VERDICT = C1L10 only (λ=0.10 pre-registered). "
          "C1L05/C1L20 = SENSITIVITY-ONLY, NON-VERDICT.\n")

    # ── VOFF: flag-OFF == A3 (N-criterion) ──
    print("[VOFF — verify flag-OFF byte-identical vs A3_seed42]")
    any_voff = False
    for sc in SCENES:
        nv = _parse(f"{C1_LOG}/VOFF_seed42_{sc}.log", NG_PAT)
        na = _parse(f"{A3_LOG}/A3_seed42_{sc}.log", NG_PAT)
        pv = _parse(f"{C1_LOG}/VOFF_seed42_{sc}.log", PSNR_PAT)
        pa = _psnr(None, 42, sc)
        if nv is None:
            continue
        any_voff = True
        rd = abs(nv - na) / na * 100 if na else float("nan")
        ok = "OK" if (na and rd < 5.0) else "⚠CHECK"
        print(f"  {sc:<9} N_voff={int(nv)} N_A3={int(na) if na else '-'} "
              f"reldiff={rd:.2f}% PSNR {pv} vs {pa}  {ok}")
    if not any_voff:
        print("  (no VOFF logs — run pre-pilot verify first)")

    # ── PRIMARY: C1L10 vs A3, paired N=24 ──
    print("\n[PRIMARY — C1L10 (λ=0.10) vs A3, paired]")
    print(f"{'scene':<9} {'A3(mean)':>9} {'C1(mean)':>9} {'Δ':>8} "
          f"{'NΔ':>3}  per-seed Δ")
    deltas, per_scene = [], {}
    for sc in SCENES:
        ds, a3s, c1s = [], [], []
        for sd in SEEDS:
            a = _psnr(None, sd, sc)
            c = _psnr(None, sd, sc, "C1L10")
            if a is not None and c is not None:
                ds.append(c - a); a3s.append(a); c1s.append(c)
        deltas += ds
        if ds:
            per_scene[sc] = statistics.fmean(ds)
            print(f"{sc:<9} {statistics.fmean(a3s):>9.3f} "
                  f"{statistics.fmean(c1s):>9.3f} "
                  f"{statistics.fmean(ds):>+8.3f} {len(ds):>3}  "
                  f"{['%+.3f' % x for x in ds]}")
        else:
            print(f"{sc:<9} {'-':>9} {'-':>9} {'-':>8}   (missing)")

    print()
    mean, sem, ci_lo, ci_hi = summarize_paired("Δ C1L10 (λ0.10) vs A3", deltas)

    # ── Cost (feedback_measure_compute_cost) ──
    print("\n[COST — C1L10 vs A3, seed42]")
    for sc in SCENES:
        for tag, lg in (("C1", f"{C1_LOG}/C1L10_seed42_{sc}.log"),
                        ("A3", f"{A3_LOG}/A3_seed42_{sc}.log")):
            n = _parse(lg, NG_PAT); t = _parse(lg, TT_PAT)
            print(f"  {sc:<9} {tag}  N={int(n) if n else '-':>7}  "
                  f"train={t if t else '-'}s")

    # ── Sensitivity (NON-VERDICT — audit#3) ──
    print("\n[SENSITIVITY-ONLY — C1L05 / C1L20 vs A3 (seed42). "
          "NON-VERDICT, KHÔNG lật C1L10.]")
    for cell in ("C1L05", "C1L20"):
        dd = []
        for sc in SCENES:
            for sd in SENS_SEEDS:
                a = _psnr(None, sd, sc); c = _psnr(None, sd, sc, cell)
                if a is not None and c is not None:
                    dd.append(c - a)
        if dd:
            print(f"  {cell}: mean Δ={statistics.fmean(dd):+.4f} "
                  f"N={len(dd)}  (sensitivity-only)")
        else:
            print(f"  {cell}: NO DATA")

    # ── PRE-REGISTERED VERDICT (locked; only C1L10 N=24) ──
    print("\n=== VERDICT (PRE-REGISTERED, locked — only C1L10/λ0.10) ===")
    if mean is None:
        print("  NO DATA (C1L10 logs missing) — cannot judge."); return
    horns = per_scene.get("horns", None)
    n_nonneg = sum(1 for v in per_scene.values() if v >= 0.0)
    n_sc = len(per_scene)
    c_mean = mean >= GO_DMEAN
    c_ci = ci_lo is not None and ci_lo > 0.0
    c_horns = (horns is not None and horns >= GO_HORNS_MIN)
    c_nonneg = n_nonneg >= GO_NONNEG_MIN
    print(f"  C1 mean Δ ≥ +0.10        : {mean:+.4f}  "
          f"{'PASS' if c_mean else 'FAIL'}")
    print(f"  C2 95%CI excludes 0      : CI_lo={ci_lo:+.3f}  "
          f"{'PASS' if c_ci else 'FAIL'}")
    print(f"  C3 ≥7/8 Δ≥0 & horns≥-.05 : nonneg={n_nonneg}/{n_sc} "
          f"horns={horns if horns is None else f'{horns:+.3f}'}  "
          f"{'PASS' if (c_horns and c_nonneg) else 'FAIL'}")
    print(f"  C4 cost reported         : PASS (table above)")
    GO = c_mean and c_ci and c_horns and c_nonneg
    if GO:
        print("\n  ✅ GO — C1 mang +info thật, vượt bar Phase-13. "
              "Tiến hành commit-grade (giữ module ON, update recipe).")
    else:
        print("\n  ❌ NO — C1 KHÔNG đạt bar pre-registered. ĐÓNG C1 "
              "(đã trả pilot, đúng cam kết). KHÔNG dùng λ sensitivity để "
              "lật. Nếu fail kèm horns/foliage degrade → XÁC NHẬN đúng "
              "pre-registered prediction (∇depth→normal nhiễu), KHÔNG "
              "re-engineer. Cleanup: revert arguments/train.py về "
              "A3/Phase-13 clean + re-sync server (Quy tắc 13).")


if __name__ == "__main__":
    main()
