#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 19] SH-degree curriculum pilot analyzer
# File: scripts/p19_curriculum_analyze.py  (TẠO MỚI — keep local)
#
# So A3 + SH-curriculum vs A3 baseline (curriculum OFF), paired per
# (seed, scene). Chạy 1 lần / backbone (env BACKBONE).
#   BACKBONE=mvs   : curric logs/p19_curriculum_mvs   vs base logs/p13_lfcf
#   BACKBONE=dense : curric logs/p19_curriculum_dense vs base logs/p18_pilot
# Baseline = REUSE (KHÔNG train lại).
#
# VERDICT PRE-REGISTERED (locked, mirror Phase 13/18):
#   GO ⟺ Δ_mean ≥ +0.10 ∧ 95%CI loại 0 ∧ (≥7/8 scene Δ≥0 ∧ horns ≥ −0.05)
#        ∧ cost reported.  Khác = NO.
# Ngoài ra in: curric 8-scene avg tuyệt đối + Δ vs committed A3 21.330.
# ============================================================
"""[CRSGaussian Phase 19] curriculum pilot analyzer.

Run (CPU, instant):
    BACKBONE=mvs   python scripts/p19_curriculum_analyze.py
    BACKBONE=dense python scripts/p19_curriculum_analyze.py
"""

import os
import re
import statistics

BACKBONE = os.environ.get("BACKBONE", "mvs")
SCENES = os.environ.get(
    "SCENES", "fern flower fortress horns leaves orchids room trex").split()
SEEDS = os.environ.get("SEEDS", "42 137 9999").split()

# curric log + same-backbone baseline log
CURRIC_LOG = os.environ.get("CURRIC_LOG", f"logs/p19_curriculum_{BACKBONE}")
_BASE_DEFAULT = {"mvs": "logs/p13_lfcf", "dense": "logs/p18_pilot"}
BASE_LOG = os.environ.get("BASE_LOG", _BASE_DEFAULT.get(BACKBONE, "logs/p13_lfcf"))

A3_COMMITTED = 21.330   # A3-MVS committed (decisions_log Phase 13)

PSNR_PAT = re.compile(r"Best test PSNR:\s*([\d\.eE\-\+]+)")
NG_PAT = re.compile(r"Final #Gaussians:\s*(\d+)")
TT_PAT = re.compile(r"Total training:\s*([\d\.]+)s")

GO_DMEAN = 0.10
GO_HORNS_MIN = -0.05
GO_NONNEG_MIN = 7


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
    if not deltas:
        print(f"  {label}: NO DATA"); return None, None, None
    n = len(deltas)
    mean = statistics.fmean(deltas)
    sem = (statistics.stdev(deltas) / (n ** 0.5)) if n > 1 else 0.0
    ci_lo, ci_hi = mean - 1.96 * sem, mean + 1.96 * sem
    sig = abs(mean) > 2 * sem
    sl = ("🎯 SIG" if mean > 0 else "📉 SIG NEG") if sig else "🔇 NOT SIG"
    print(f"  {label:<30} N={n:<3} mean={mean:+.4f} SEM={sem:.4f} "
          f"95%CI=[{ci_lo:+.3f},{ci_hi:+.3f}] {sl}")
    return mean, ci_lo, ci_hi


def main():
    print(f"=== Phase 19 SH-curriculum — backbone={BACKBONE} ===")
    print(f"curric={CURRIC_LOG}  base={BASE_LOG}(reuse)  seeds={SEEDS}\n")

    # ── PRIMARY: PSNR paired Δ (curric − base) ──
    print(f"{'scene':<9} {'base':>9} {'curric':>9} {'Δ':>9} {'NΔ':>3}  per-seed Δ")
    deltas, per_scene, curric_means = [], {}, {}
    for sc in SCENES:
        ds, bv, cv = [], [], []
        for sd in SEEDS:
            b = _parse(f"{BASE_LOG}/A3_seed{sd}_{sc}.log", PSNR_PAT)
            c = _parse(f"{CURRIC_LOG}/A3_seed{sd}_{sc}.log", PSNR_PAT)
            if b is not None and c is not None:
                ds.append(c - b); bv.append(b); cv.append(c)
        deltas += ds
        if ds:
            per_scene[sc] = statistics.fmean(ds)
            curric_means[sc] = statistics.fmean(cv)
            print(f"{sc:<9} {statistics.fmean(bv):>9.3f} "
                  f"{statistics.fmean(cv):>9.3f} {statistics.fmean(ds):>+9.3f} "
                  f"{len(ds):>3}  {['%+.3f' % x for x in ds]}")
        else:
            print(f"{sc:<9} {'-':>9} {'-':>9} {'-':>9}   (missing)")

    print()
    mean, ci_lo, ci_hi = summarize_paired("Δ curriculum vs baseline", deltas)

    # ── COST ──
    print("\n[COST — mean over seeds; slowx/Nx = curric / baseline]")
    print(f"{'scene':<9} {'t_base':>9} {'t_curr':>9} {'slowx':>7}  "
          f"{'N_base':>9} {'N_curr':>9} {'Nx':>6}")
    tot_tb, tot_tc = [], []
    for sc in SCENES:
        tb = _mean([_parse(f"{BASE_LOG}/A3_seed{sd}_{sc}.log", TT_PAT) for sd in SEEDS])
        tc = _mean([_parse(f"{CURRIC_LOG}/A3_seed{sd}_{sc}.log", TT_PAT) for sd in SEEDS])
        nb = _mean([_parse(f"{BASE_LOG}/A3_seed{sd}_{sc}.log", NG_PAT) for sd in SEEDS])
        nc = _mean([_parse(f"{CURRIC_LOG}/A3_seed{sd}_{sc}.log", NG_PAT) for sd in SEEDS])
        if tb and tc and nb and nc:
            tot_tb.append(tb); tot_tc.append(tc)
            print(f"{sc:<9} {tb:>9.1f} {tc:>9.1f} {tc/tb:>6.2f}x  "
                  f"{nb:>9.0f} {nc:>9.0f} {nc/nb:>5.2f}x")
        else:
            print(f"{sc:<9}  (cost data missing)")
    if tot_tb:
        print(f"  → aggregate train slowdown ≈ "
              f"{statistics.fmean(tot_tc)/statistics.fmean(tot_tb):.2f}×")

    # ── Absolute headline: curric 8-scene avg vs committed A3 21.330 ──
    if len(curric_means) == 8:
        c_avg = statistics.fmean(curric_means.values())
        print(f"\n[ABSOLUTE] curriculum({BACKBONE}) 8-scene avg = {c_avg:.3f}  "
              f"(committed A3 = {A3_COMMITTED:.3f}, Δ = {c_avg - A3_COMMITTED:+.3f})")
    else:
        print(f"\n[ABSOLUTE] chỉ {len(curric_means)}/8 scene — chưa đủ cho avg.")

    # ── PRE-REGISTERED VERDICT ──
    print("\n=== VERDICT (PRE-REGISTERED, locked) ===")
    if mean is None:
        print("  NO DATA."); return
    horns = per_scene.get("horns")
    n_nonneg = sum(1 for v in per_scene.values() if v >= 0.0)
    n_sc = len(per_scene)
    c_mean = mean >= GO_DMEAN
    c_ci = ci_lo is not None and ci_lo > 0.0
    c_horns = horns is not None and horns >= GO_HORNS_MIN
    c_nonneg = n_nonneg >= GO_NONNEG_MIN
    print(f"  C1 mean Δ ≥ +0.10       : {mean:+.4f}  {'PASS' if c_mean else 'FAIL'}")
    print(f"  C2 95%CI loại 0         : CI_lo={ci_lo:+.3f}  {'PASS' if c_ci else 'FAIL'}")
    print(f"  C3 ≥7/8 Δ≥0 & horns≥-.05: nonneg={n_nonneg}/{n_sc} "
          f"horns={horns if horns is None else f'{horns:+.3f}'}  "
          f"{'PASS' if (c_horns and c_nonneg) else 'FAIL'}")
    print(f"  C4 cost reported        : PASS (bảng COST trên)")
    if c_mean and c_ci and c_horns and c_nonneg:
        print(f"\n  ✅ GO — SH-curriculum thắng baseline ({BACKBONE}) pre-registered.")
    else:
        print(f"\n  ❌ NO — SH-curriculum KHÔNG đạt bar pre-registered ({BACKBONE}).")


if __name__ == "__main__":
    main()
