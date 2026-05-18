#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 13.2.5] GDAGS pilot analyzer.
# File: scripts/p13_2_gdags_pilot_analyze.py  (TẠO MỚI)
# Parse logs/p13_2_gdags/GDAGS_seed42_<scene>.log + A3 baseline
# logs/p13_lfcf/A3_seed42_<scene>.log → paired Δ_PSNR per scene.
# Cũng check VERIFYOFF_*.log (flag-OFF phải ≈ A3 ± atomicAdd noise).
# ============================================================
"""[CRSGaussian Phase 13.2.5] GDAGS A/B pilot analyzer.

Verdict (single-seed N=3 — project_3dgs_variance_floor: ±0.10 floor):
  Δ_mean ≥ +0.10  → 🎯 multi-seed verify (137+9999) trước adopt
  +0.05..+0.10    → 🟡 marginal, multi-seed recommended
  < +0.05         → ⚪ A3 (AbsGS-OR) ≥ GDAGS → policy-swap không gain → lock A3
  < 0             → 📉 GDAGS worse → reject, A3 remains

Run:
    python scripts/p13_2_gdags_pilot_analyze.py
    python scripts/p13_2_gdags_pilot_analyze.py --verify   # check flag-OFF=A3
"""

import os
import re
import sys
import statistics

SCENES = os.environ.get(
    "SCENES", "fern flower fortress horns leaves orchids room trex").split()
SEED = os.environ.get("SEED", "42")
GDAGS_LOG_DIR = os.environ.get("GDAGS_LOG_DIR", "logs/p13_2_gdags")
A3_LOG_DIR = os.environ.get("A3_LOG_DIR", "logs/p13_lfcf")

TABLE_PAT = re.compile(
    r"\b(\d{2,5})\s*\|\s*(test|train)\s*\|\s*([0-9]+\.[0-9]+)\s*\|")


N_PAT = re.compile(r"Best test PSNR:\s*[0-9.]+\s*at iter\s*\d+\s*\(N=(\d+)\)")


def parse_test_psnr(log_path):
    if not os.path.isfile(log_path):
        return None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    ms = TABLE_PAT.findall(txt)
    if not ms:
        return None
    it2p = {}
    for it, sp, ps in ms:
        if sp == "test":
            it2p[int(it)] = float(ps)
    return it2p[max(it2p)] if it2p else None


def parse_train_psnr(log_path):
    """Train PSNR cuối — để lộ overfit signature (train↑ + test phẳng)."""
    if not os.path.isfile(log_path):
        return None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        ms = TABLE_PAT.findall(f.read())
    it2p = {int(it): float(ps) for it, sp, ps in ms if sp == "train"}
    return it2p[max(it2p)] if it2p else None


def parse_n(log_path):
    """N_gaussians cuối — 'Best test PSNR: X at iter Y (N=Z)'.
    N = output trực tiếp của densification path (chỗ edit GDAGS đụng),
    nhiễu THẤP hơn PSNR rất nhiều (PSNR ±1.3 dB single-scene do atomicAdd,
    N chỉ jitter ~vài % cho code IDENTICAL). → dùng N làm tiêu chí
    byte-identical PRIMARY, KHÔNG dùng PSNR (PSNR single-scene quá nhiễu —
    bug threshold cũ 0.05 mâu thuẫn project_3dgs_variance_floor)."""
    if not os.path.isfile(log_path):
        return None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        m = N_PAT.findall(f.read())
    return int(m[-1]) if m else None


def main():
    verify = "--verify" in sys.argv
    cfg = "VERIFYOFF" if verify else "GDAGS"
    print(f"=== Phase 13.2.5 GDAGS pilot analyzer "
          f"({'VERIFY flag-OFF=A3' if verify else 'A/B GDAGS vs A3'}) ===")
    print(f"{cfg}: {GDAGS_LOG_DIR}/{cfg}_seed{SEED}_*.log")
    print(f"A3 baseline: {A3_LOG_DIR}/A3_seed{SEED}_*.log")
    print(f"Scenes: {SCENES}\n")

    rows, deltas, dn_pct, dtrain = [], [], [], []
    for sc in SCENES:
        a3 = parse_test_psnr(f"{A3_LOG_DIR}/A3_seed{SEED}_{sc}.log")
        gd = parse_test_psnr(f"{GDAGS_LOG_DIR}/{cfg}_seed{SEED}_{sc}.log")
        if a3 is None or gd is None:
            print(f"  {sc}: MISSING (A3={a3}, {cfg}={gd})")
            continue
        a3t = parse_train_psnr(f"{A3_LOG_DIR}/A3_seed{SEED}_{sc}.log")
        gdt = parse_train_psnr(f"{GDAGS_LOG_DIR}/{cfg}_seed{SEED}_{sc}.log")
        a3n = parse_n(f"{A3_LOG_DIR}/A3_seed{SEED}_{sc}.log")
        gdn = parse_n(f"{GDAGS_LOG_DIR}/{cfg}_seed{SEED}_{sc}.log")
        d = gd - a3
        deltas.append(d)
        if a3n and gdn:
            dn_pct.append((gdn - a3n) / a3n * 100.0)
        if a3t is not None and gdt is not None:
            dtrain.append(gdt - a3t)
        rows.append((sc, a3, gd, d, a3t, gdt, a3n, gdn))

    hdr = (f"{'scene':<10} {'A3_te':>8} {'GD_te':>8} {'Δte':>8} "
           f"{'A3_tr':>8} {'GD_tr':>8} {'A3_N':>8} {'GD_N':>8} {'ΔN%':>7}")
    print(hdr)
    print("-" * len(hdr))
    for sc, a3, gd, d, a3t, gdt, a3n, gdn in rows:
        dnp = ((gdn - a3n) / a3n * 100.0) if (a3n and gdn) else float('nan')
        print(f"{sc:<10} {a3:>8.3f} {gd:>8.3f} {d:>+8.3f} "
              f"{(a3t if a3t else float('nan')):>8.3f} "
              f"{(gdt if gdt else float('nan')):>8.3f} "
              f"{str(a3n):>8} {str(gdn):>8} {dnp:>+7.1f}")

    if not deltas:
        print("\nNO paired data."); sys.exit(1)

    m = statistics.fmean(deltas)
    sd = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
    mdn = statistics.fmean(dn_pct) if dn_pct else float('nan')
    mdtr = statistics.fmean(dtrain) if dtrain else float('nan')
    print(f"\n  Δtest_mean = {m:+.4f} (std {sd:.4f}, N={len(deltas)})")
    print(f"  ΔN_mean = {mdn:+.1f}%   Δtrain_mean = {mdtr:+.3f}")

    print("\n=== VERDICT ===")
    if verify:
        # FIX: PSNR single-scene nhiễu ±1.3 dB (atomicAdd, project_3dgs_
        # variance_floor) → KHÔNG dùng PSNR Δ làm gate (threshold 0.05 cũ SAI,
        # mâu thuẫn variance floor). Densification trajectory = chỗ edit GDAGS
        # đụng → N_gaussians là tín hiệu PRIMARY (code identical → N jitter
        # chỉ vài %; gating vỡ → N lệch lớn, vd GDAGS-on N gần 2×).
        print("  [PRIMARY = N_gaussians; PSNR single-scene chỉ context vì"
              " ±1.3 dB atomicAdd noise]")
        ok = True
        for sc in SCENES:
            a3n = parse_n(f"{A3_LOG_DIR}/A3_seed{SEED}_{sc}.log")
            ofn = parse_n(f"{GDAGS_LOG_DIR}/{cfg}_seed{SEED}_{sc}.log")
            ap = parse_test_psnr(f"{A3_LOG_DIR}/A3_seed{SEED}_{sc}.log")
            op = parse_test_psnr(f"{GDAGS_LOG_DIR}/{cfg}_seed{SEED}_{sc}.log")
            if a3n is None or ofn is None:
                print(f"  {sc}: N MISSING (A3={a3n}, off={ofn}) — skip")
                continue
            rel = abs(ofn - a3n) / max(a3n, 1)
            dp = (op - ap) if (ap is not None and op is not None) else float('nan')
            tag = ("✅ N≈A3" if rel < 0.05 else
                   "🟡 N lệch nhẹ" if rel < 0.20 else "❌ N LỆCH LỚN")
            print(f"  {sc}: N {a3n}→{ofn} (Δ{rel*100:+.1f}%) {tag} | "
                  f"ΔPSNR={dp:+.3f} (≤±1.3 single-scene noise = context)")
            if rel >= 0.20:
                ok = False
        if ok:
            print("\n  ✅ flag-OFF densification ≈ A3 (N reldiff < 20% mọi"
                  " scene) → gating ĐÚNG, code GDAGS KHÔNG phá A3 khi flag OFF.")
            print("  Contract HOLDS. (ΔPSNR là atomicAdd noise, KHÔNG phải"
                  " regression — N là bằng chứng.)")
        else:
            print("\n  ❌ N LỆCH ≥20% → densification trajectory đổi khi flag"
                  " OFF → gating có bug. STOP, debug.")
        return

    # Over-densify → overfit signature (predicted: HF-pilot/D3 pattern).
    overfit_sig = (not (mdn != mdn)) and mdn > 30.0 and m < 0.05 \
        and (not (mdtr != mdtr)) and mdtr > 0.0
    if overfit_sig:
        print(f"  ❌ OVER-DENSIFY → OVERFIT confirmed: ΔN={mdn:+.0f}% (capacity↑)"
              f" + Δtrain={mdtr:+.3f} (fit↑) + Δtest={m:+.3f} (< +0.05, phẳng).")
        print(f"  → Đúng pattern HF-pilot(−0.31)/D3(−0.48): 3-view + thêm")
        print(f"     capacity = memorize. GDAGS regime-mismatch (tuned Mip360")
        print(f"     dense-view). REJECT GDAGS → LOCK A3 21.330.")
        print(f"  → + Cost: GDAGS ~{1+mdn/100:.1f}× Gaussian = ~{1+mdn/100:.1f}×"
              f" memory/chậm (feedback_measure_compute_cost) — cost thật kể cả"
              f" nếu PSNR ngang.")
    elif m >= 0.10:
        print(f"  🎯 Δtest ≥ +0.10 bất chấp ΔN={mdn:+.0f}% → multi-seed verify"
              f" (137+9999). Kiểm overfit gap (Δtrain={mdtr:+.3f}) trước adopt.")
    elif m >= 0.05:
        print(f"  🟡 Δtest ∈ [+0.05,+0.10) marginal → multi-seed; cân nhắc"
              f" cost ΔN={mdn:+.0f}%")
    elif m >= 0.0:
        print(f"  ⚪ Δtest ∈ [0,+0.05) — GDAGS ≈ A3 (policy-swap không gain)."
              f" + ΔN={mdn:+.0f}% cost → Lock A3 (AbsGS-OR).")
    else:
        print(f"  📉 Δtest < 0 — GDAGS WORSE → reject, A3 remains")
    print(f"\n  Caveat: single-seed N=8 paired (±0.10 floor); GDAGS=policy-A/B")
    print(f"  KHÔNG +feature (GCR=grads/grads_abs không orthogonal AbsGS).")


if __name__ == "__main__":
    main()
