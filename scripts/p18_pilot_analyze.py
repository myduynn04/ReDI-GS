#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 18] Pilot analyzer — A3-PDCNet+ vs A3-MVS paired N=24
# File: scripts/p18_pilot_analyze.py  (TẠO MỚI — keep local)
#
# Parse PSNR + train-time + N_gauss; paired Δ + 95%CI = mirror
# p17_c1_analyze.py (summarize_paired). KHÔNG tự chế thống kê.
#
# So: logs/p18_pilot (A3-PDCNet+ init) vs logs/p13_lfcf (A3-MVS init,
# reuse — KHÔNG train lại). Paired per (seed, scene).
#
# PRE-REGISTERED VERDICT (locked, mirror Phase 13/17):
#   GO ⟺ Δ_mean ≥ +0.10 ∧ 95%CI loại 0 ∧ (≥7/8 scene Δ≥0 ∧ horns Δ≥−0.05)
#        ∧ cost reported.  Khác = NO.
#   Cost (train-time + N_gauss) = trục RIÊNG: report bắt buộc, kể cả
#   PSNR-GO vẫn phải cân cost 2-7× cho quyết định commit recipe.
# ============================================================
"""[CRSGaussian Phase 18] pilot analyzer.

Run (CPU, instant):
    python scripts/p18_pilot_analyze.py
"""

import os
import re
import statistics

SCENES = os.environ.get(
    "SCENES", "fern flower fortress horns leaves orchids room trex").split()
SEEDS = os.environ.get("SEEDS", "42 137 9999").split()
PDC_LOG = os.environ.get("PDC_LOG", "logs/p18_pilot")     # A3-PDCNet+ init
MVS_LOG = os.environ.get("MVS_LOG", "logs/p13_lfcf")      # A3-MVS init (reuse)

PSNR_PAT = re.compile(r"Best test PSNR:\s*([\d\.eE\-\+]+)")
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


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def summarize_paired(label, deltas):
    """EXACT mirror p17_c1_analyze.summarize_paired."""
    if not deltas:
        print(f"  {label}: NO DATA"); return None, None, None, None
    n = len(deltas)
    mean = statistics.fmean(deltas)
    std = statistics.stdev(deltas) if n > 1 else 0.0
    sem = std / (n ** 0.5)
    ci_lo, ci_hi = mean - 1.96 * sem, mean + 1.96 * sem
    sig = abs(mean) > 2 * sem
    sl = ("🎯 SIG" if mean > 0 else "📉 SIG NEG") if sig else "🔇 NOT SIG"
    print(f"  {label:<28} N={n:<3} mean={mean:+.4f} SEM={sem:.4f} "
          f"95%CI=[{ci_lo:+.3f},{ci_hi:+.3f}] {sl}")
    return mean, sem, ci_lo, ci_hi


def main():
    print("=== Phase 18 pilot — A3-PDCNet+ dense init vs A3-MVS, paired N=24 ===")
    print(f"PDC_LOG={PDC_LOG}  MVS_LOG={MVS_LOG}(reuse)  seeds={SEEDS}\n")

    # ── PRIMARY: PSNR paired Δ ──
    print(f"{'scene':<9} {'MVS(mean)':>10} {'PDC(mean)':>10} {'Δ':>9} "
          f"{'NΔ':>3}  per-seed Δ")
    deltas, per_scene = [], {}
    for sc in SCENES:
        ds, mv, pd = [], [], []
        for sd in SEEDS:
            m = _parse(f"{MVS_LOG}/A3_seed{sd}_{sc}.log", PSNR_PAT)
            p = _parse(f"{PDC_LOG}/A3_seed{sd}_{sc}.log", PSNR_PAT)
            if m is not None and p is not None:
                ds.append(p - m); mv.append(m); pd.append(p)
        deltas += ds
        if ds:
            per_scene[sc] = statistics.fmean(ds)
            print(f"{sc:<9} {statistics.fmean(mv):>10.3f} "
                  f"{statistics.fmean(pd):>10.3f} {statistics.fmean(ds):>+9.3f} "
                  f"{len(ds):>3}  {['%+.3f' % x for x in ds]}")
        else:
            print(f"{sc:<9} {'-':>10} {'-':>10} {'-':>9}   (missing)")

    print()
    mean, sem, ci_lo, ci_hi = summarize_paired("Δ PDCNet+ vs MVS", deltas)

    # ── COST: train-time + N_gauss (feedback_measure_compute_cost) ──
    print("\n[COST — mean over seeds; slowx/Nx = PDCNet+ / MVS]")
    print(f"{'scene':<9} {'t_MVS(s)':>9} {'t_PDC(s)':>9} {'slowx':>7}  "
          f"{'N_MVS':>9} {'N_PDC':>9} {'Nx':>6}")
    tot_tm, tot_tp = [], []
    for sc in SCENES:
        tm = _mean([_parse(f"{MVS_LOG}/A3_seed{sd}_{sc}.log", TT_PAT) for sd in SEEDS])
        tp = _mean([_parse(f"{PDC_LOG}/A3_seed{sd}_{sc}.log", TT_PAT) for sd in SEEDS])
        nm = _mean([_parse(f"{MVS_LOG}/A3_seed{sd}_{sc}.log", NG_PAT) for sd in SEEDS])
        npd = _mean([_parse(f"{PDC_LOG}/A3_seed{sd}_{sc}.log", NG_PAT) for sd in SEEDS])
        sx = (tp / tm) if (tm and tp) else None
        nx = (npd / nm) if (nm and npd) else None
        if tm and tp:
            tot_tm.append(tm); tot_tp.append(tp)
        print(f"{sc:<9} "
              f"{tm if tm else '-':>9.1f}" if tm else f"{sc:<9} {'-':>9}", end="")
        # second half (avoid format crash on None)
        print(f" {tp:>9.1f} {sx:>6.2f}x  {nm:>9.0f} {npd:>9.0f} {nx:>5.2f}x"
              if (tm and tp and nm and npd) else "  (cost data missing)")
    if tot_tm:
        agg_sx = statistics.fmean(tot_tp) / statistics.fmean(tot_tm)
        print(f"  → aggregate train slowdown ≈ {agg_sx:.2f}× "
              f"(mean t_PDC {statistics.fmean(tot_tp):.0f}s / t_MVS "
              f"{statistics.fmean(tot_tm):.0f}s)")

    # ── PRE-REGISTERED VERDICT ──
    print("\n=== VERDICT (PRE-REGISTERED, locked) ===")
    if mean is None:
        print("  NO DATA — cannot judge."); return
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
    print(f"  C4 cost reported         : PASS (bảng COST trên)")
    GO = c_mean and c_ci and c_horns and c_nonneg
    if GO:
        print("\n  ✅ GO — PDCNet+ dense init thắng MVS init về PSNR (N=24, "
              "pre-registered). NHƯNG: cân COST (slowdown ×) cho quyết định "
              "commit recipe — PSNR-GO ≠ tự động commit nếu cost quá cao.")
    else:
        print("\n  ❌ NO — PDCNet+ init KHÔNG đạt bar pre-registered. "
              "Restore backup (RESTORE=1 p18_gate2_place_dense_init.py) → "
              "data về MVS. Ghi decisions_log + giữ logs evidence.")


if __name__ == "__main__":
    main()
