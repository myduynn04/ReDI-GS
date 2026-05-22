#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 18b] Cyclic-gate triage analyzer
# File: scripts/p18b_triage_analyze.py  (TẠO MỚI — keep local)
#
# So A3 với init PDCNet+ GATED (cyclic τ=1.0) vs init PDCNet+ RAW.
# Paired per (seed, scene). 4 scene × 3 seed = 12 run.
#   GATED = logs/p18b_triage     (train mới)
#   RAW   = logs/p18_pilot       (REUSE — KHÔNG train lại)
#   MVS   = logs/p13_lfcf        (informational — Δ vs baseline gốc)
#
# VERDICT TRIAGE (pre-registered, locked — decisions_log [2026-05-22]):
#   GO ⟺ horns Δ(gated−raw) ≥ +0.15 ∧ trex ≥ +0.15
#        ∧ fortress Δ ≥ −0.10 ∧ orchids Δ ≥ −0.10
#   Else = NO → dừng Phase 18b, raw-PDCNet+ verdict NO chốt lại.
#   GO → mở full N=24 gated, verdict C1–C4 như Phase 18.
# Cost (train-time + N_gauss) = report bắt buộc (feedback_measure_compute_cost).
# ============================================================
"""[CRSGaussian Phase 18b] cyclic-gate triage analyzer.

Run (CPU, instant):
    python scripts/p18b_triage_analyze.py
"""

import os
import re
import statistics

SCENES = os.environ.get("SCENES", "horns trex fortress orchids").split()
SEEDS = os.environ.get("SEEDS", "42 137 9999").split()
GATED_LOG = os.environ.get("GATED_LOG", "logs/p18b_triage")   # gated init
RAW_LOG = os.environ.get("RAW_LOG", "logs/p18_pilot")          # raw PDCNet+
MVS_LOG = os.environ.get("MVS_LOG", "logs/p13_lfcf")           # MVS (info)

PSNR_PAT = re.compile(r"Best test PSNR:\s*([\d\.eE\-\+]+)")
NG_PAT = re.compile(r"Final #Gaussians:\s*(\d+)")
TT_PAT = re.compile(r"Total training:\s*([\d\.]+)s")

# Pre-registered triage thresholds (locked)
RESCUE_SCENES = {"horns", "trex"}
CONTROL_SCENES = {"fortress", "orchids"}
GO_RESCUE_MIN = 0.15      # must-rescue: Δ(gated−raw) ≥ +0.15
GO_CONTROL_MIN = -0.10    # control: gate KHÔNG được hại winner quá −0.10


def _parse(path, pat):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        m = pat.search(f.read())
    return float(m.group(1)) if m else None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def main():
    print("=== Phase 18b cyclic-gate triage — A3 gated vs raw-PDCNet+, paired ===")
    print(f"GATED={GATED_LOG}  RAW={RAW_LOG}(reuse)  MVS={MVS_LOG}(info)  "
          f"seeds={SEEDS}\n")

    # ── PRIMARY: Δ(gated − raw) per scene ──
    print(f"{'scene':<10} {'raw(mean)':>10} {'gated(mean)':>12} "
          f"{'Δg-raw':>9} {'NΔ':>3}  per-seed Δ(g-raw)        {'MVS':>8} {'Δg-MVS':>9}")
    per_scene_graw = {}
    for sc in SCENES:
        ds, raw_v, gat_v, mvs_v = [], [], [], []
        for sd in SEEDS:
            r = _parse(f"{RAW_LOG}/A3_seed{sd}_{sc}.log", PSNR_PAT)
            g = _parse(f"{GATED_LOG}/A3_seed{sd}_{sc}.log", PSNR_PAT)
            m = _parse(f"{MVS_LOG}/A3_seed{sd}_{sc}.log", PSNR_PAT)
            if r is not None and g is not None:
                ds.append(g - r); raw_v.append(r); gat_v.append(g)
            if m is not None:
                mvs_v.append(m)
        if ds:
            per_scene_graw[sc] = statistics.fmean(ds)
            mvs_m = statistics.fmean(mvs_v) if mvs_v else None
            gmvs = (statistics.fmean(gat_v) - mvs_m) if mvs_m else None
            print(f"{sc:<10} {statistics.fmean(raw_v):>10.3f} "
                  f"{statistics.fmean(gat_v):>12.3f} {statistics.fmean(ds):>+9.3f} "
                  f"{len(ds):>3}  {str(['%+.3f' % x for x in ds]):<24} "
                  f"{(f'{mvs_m:.3f}' if mvs_m else '-'):>8} "
                  f"{(f'{gmvs:+.3f}' if gmvs is not None else '-'):>9}")
        else:
            print(f"{sc:<10} {'-':>10} {'-':>12} {'-':>9}   (missing)")

    # ── COST ──
    print("\n[COST — mean over seeds]")
    print(f"{'scene':<10} {'t_raw(s)':>9} {'t_gat(s)':>9} {'slowx':>7}  "
          f"{'N_raw':>9} {'N_gat':>9} {'Nx':>6}")
    for sc in SCENES:
        tr = _mean([_parse(f"{RAW_LOG}/A3_seed{sd}_{sc}.log", TT_PAT) for sd in SEEDS])
        tg = _mean([_parse(f"{GATED_LOG}/A3_seed{sd}_{sc}.log", TT_PAT) for sd in SEEDS])
        nr = _mean([_parse(f"{RAW_LOG}/A3_seed{sd}_{sc}.log", NG_PAT) for sd in SEEDS])
        ng = _mean([_parse(f"{GATED_LOG}/A3_seed{sd}_{sc}.log", NG_PAT) for sd in SEEDS])
        if tr and tg and nr and ng:
            print(f"{sc:<10} {tr:>9.1f} {tg:>9.1f} {tg/tr:>6.2f}x  "
                  f"{nr:>9.0f} {ng:>9.0f} {ng/nr:>5.2f}x")
        else:
            print(f"{sc:<10}  (cost data missing)")

    # ── PRE-REGISTERED TRIAGE VERDICT ──
    print("\n=== TRIAGE VERDICT (PRE-REGISTERED, locked) ===")
    missing = [s for s in SCENES if s not in per_scene_graw]
    if missing:
        print(f"  ⚠️  THIẾU scene: {missing} — chưa đủ data, không kết luận.")
        return
    checks, ok = [], True
    for sc in SCENES:
        d = per_scene_graw[sc]
        if sc in RESCUE_SCENES:
            p = d >= GO_RESCUE_MIN
            checks.append(f"  {sc:<9} (rescue) Δ={d:+.3f}  cần ≥+{GO_RESCUE_MIN:.2f}"
                          f"  {'PASS' if p else 'FAIL'}")
        elif sc in CONTROL_SCENES:
            p = d >= GO_CONTROL_MIN
            checks.append(f"  {sc:<9} (control) Δ={d:+.3f}  cần ≥{GO_CONTROL_MIN:.2f}"
                          f"   {'PASS' if p else 'FAIL'}")
        else:
            p = True
        ok = ok and p
    for c in checks:
        print(c)
    if ok:
        print("\n  ✅ GO — cyclic-gate dịch horns+trex đúng hướng, control KHÔNG vỡ.")
        print("  → Mở FULL N=24 gated (8 scene × 3 seed), verdict C1–C4 như Phase 18.")
    else:
        print("\n  ❌ NO — cyclic-gate KHÔNG đạt bar triage. Dừng Phase 18b.")
        print("  → raw-PDCNet+ verdict NO chốt lại. Restore 8 backup MVS "
              "(RESTORE=1 p18_gate2_place_dense_init.py). Ghi decisions_log.")


if __name__ == "__main__":
    main()
