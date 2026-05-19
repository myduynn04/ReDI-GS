#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 15] shape-reg pilot analyzer.
# Parse logs/p15_shape/{Ablunt,Bexc3,Bexc2}_seed42_<scene>.log vs
# A3 baseline logs/p13_lfcf/A3_seed42_<scene>.log → paired Δ.
#
# Ablunt = CONTROL (Q4 dự đoán catastrophe vì 77% high-aniso=
#   legitimate-flat). Nếu Ablunt catastrophe → Q4-diagnostic ĐÁNG TIN.
#   Nếu Ablunt KHÔNG hại → diagnostic discipline phải xem lại.
# Bexc* = form Q4-data CHỈ vào (chỉ phạt s_max-excess, né flat).
#
# Verdict (single-seed N=8 paired, ±0.10 floor) + per-scene
#   CATASTROPHE-GUARD (Δ<−0.20 → flag, KHÔNG để mean che — GDAGS horns).
# N primary cho VOFF byte-identical (PSNR ±1.3 single-scene noise).
#
# Run: python scripts/p15_shape_analyze.py
#      python scripts/p15_shape_analyze.py --verify   # VOFF=A3
# ============================================================
"""[CRSGaussian Phase 15] shape-reg pilot analyzer (PSNR/SSIM/LPIPS/time)."""

import os
import re
import sys
import statistics

SCENES = os.environ.get(
    "SCENES", "fern flower fortress horns leaves orchids room trex").split()
SEED = os.environ.get("SEED", "42")
SR_DIR = os.environ.get("SR_LOG_DIR", "logs/p15_shape")
A_DIR = os.environ.get("A_LOG_DIR", "logs/p13_lfcf")
CELLS = os.environ.get("CELLS", "Ablunt Bexc3 Bexc2").split()
CATA = float(os.environ.get("CATASTROPHE", "0.20"))

METRIC_PAT = re.compile(
    r"\b(\d{2,5})\s*\|\s*(test|train)\s*\|\s*"
    r"([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|")
N_PAT = re.compile(r"Best test PSNR:\s*[0-9.]+\s*at iter\s*\d+\s*\(N=(\d+)\)")
TIME_PAT = re.compile(r"Total training:\s*([0-9.]+)\s*s")


def _read(p):
    if not os.path.isfile(p):
        return None
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def metrics(p):
    t = _read(p)
    if t is None:
        return None
    d = {int(i): (float(ps), float(ss), float(lp))
         for i, sp, ps, ss, lp in METRIC_PAT.findall(t) if sp == "test"}
    if not d:
        return None
    ps, ss, lp = d[max(d)]
    return dict(psnr=ps, ssim=ss, lpips=lp)


def pn(p):
    t = _read(p)
    m = N_PAT.findall(t) if t else []
    return int(m[-1]) if m else None


def pt(p):
    t = _read(p)
    m = TIME_PAT.findall(t) if t else []
    return float(m[-1]) if m else None


def main():
    verify = "--verify" in sys.argv
    print(f"=== Phase 15 shape-reg analyzer "
          f"({'VERIFY VOFF=A3' if verify else 'pilot vs A3'}) ===")
    print(f"A3 baseline: {A_DIR}/A3_seed{SEED}_*  | {SR_DIR}/<cell>_*\n")

    if verify:
        print("  [PRIMARY=N; PSNR ±1.3 single-scene noise]")
        ok = True
        for sc in SCENES:
            a = pn(f"{A_DIR}/A3_seed{SEED}_{sc}.log")
            v = pn(f"{SR_DIR}/VOFF_seed{SEED}_{sc}.log")
            if a is None or v is None:
                print(f"  {sc}: N MISSING (A={a},VOFF={v}) skip"); continue
            rel = abs(v - a) / max(a, 1)
            tag = "✅≈A3" if rel < .05 else "🟡" if rel < .20 else "❌LỆCH"
            print(f"  {sc}: N {a}→{v} ({rel*100:+.1f}%) {tag}")
            if rel >= .20:
                ok = False
        print("\n  " + ("✅ flag-OFF≈A3 → code Phase-15 KHÔNG phá A3, baseline"
              " reuse hợp lệ." if ok else "❌ N LỆCH≥20% → STOP debug."))
        return

    deltas = {c: {"psnr": [], "ssim": [], "lpips": []} for c in CELLS}
    tc = {c: [] for c in CELLS}
    tA = []
    cata = []
    rows = []
    for sc in SCENES:
        mA = metrics(f"{A_DIR}/A3_seed{SEED}_{sc}.log")
        ta = pt(f"{A_DIR}/A3_seed{SEED}_{sc}.log")
        if ta:
            tA.append(ta)
        if mA is None:
            print(f"  {sc}: A3 MISSING skip"); continue
        row = [sc, mA]
        for c in CELLS:
            mc = metrics(f"{SR_DIR}/{c}_seed{SEED}_{sc}.log")
            t = pt(f"{SR_DIR}/{c}_seed{SEED}_{sc}.log")
            if t:
                tc[c].append(t)
            row.append(mc)
            if mc:
                d = mc["psnr"] - mA["psnr"]
                deltas[c]["psnr"].append(d)
                deltas[c]["ssim"].append(mc["ssim"] - mA["ssim"])
                deltas[c]["lpips"].append(mc["lpips"] - mA["lpips"])
                if d < -CATA:
                    cata.append((sc, c, d))
        rows.append(row)

    hdr = f"{'scene':<9} {'A3':>7} " + " ".join(f"{c:>8}" for c in CELLS)
    print(hdr); print("-" * len(hdr))
    for row in rows:
        sc, mA = row[0], row[1]
        cells = " ".join(
            (f"{row[2+i]['psnr']-mA['psnr']:+8.3f}" if row[2+i] else f"{'-':>8}")
            for i in range(len(CELLS)))
        print(f"{sc:<9} {mA['psnr']:7.3f} {cells}   (Δ vs A3)")

    print("\n=== Aggregate Δ (mean, ±0.10 floor) ===")
    mp = {}
    for c in CELLS:
        dp = deltas[c]["psnr"]
        if not dp:
            print(f"  {c}: NO data"); continue
        m = statistics.fmean(dp)
        sd = statistics.pstdev(dp) if len(dp) > 1 else 0.0
        ms = statistics.fmean(deltas[c]["ssim"])
        ml = statistics.fmean(deltas[c]["lpips"])
        mp[c] = m
        print(f"  {c}: ΔPSNR={m:+.4f}(std {sd:.3f},N={len(dp)})  "
              f"ΔSSIM={ms:+.4f}  ΔLPIPS={ml:+.4f}(↓tốt)")

    print("\n=== Train-time / cost ===")
    mtA = statistics.fmean(tA) if tA else None
    print(f"  A3: {f'{mtA:.0f}s' if mtA else 'N/A'}")
    for c in CELLS:
        mt = statistics.fmean(tc[c]) if tc[c] else None
        ov = (f" ({(mt/mtA-1)*100:+.0f}% vs A3)" if mt and mtA else "")
        print(f"  {c}: {f'{mt:.0f}s' if mt else 'N/A'}{ov}")

    print("\n=== PER-SCENE CATASTROPHE GUARD ===")
    if cata:
        print(f"  ❌ {len(cata)} scene-cell Δ<−{CATA} (mean CHE):")
        for sc, c, d in cata:
            print(f"     {c}/{sc}: {d:+.3f}")
        print("  → cell có catastrophe = REJECT cell đó dù mean đẹp.")
    else:
        print(f"  ✅ Không scene nào Δ<−{CATA}.")

    print("\n=== VERDICT ===")
    if "Ablunt" in mp:
        a = mp["Ablunt"]
        has_cat = any(c == "Ablunt" for _, c, _ in cata)
        if a <= -0.10 or has_cat:
            print(f"  Ablunt={a:+.3f}{' +catastrophe' if has_cat else ''} → "
                  f"Q4-diagnostic ĐÁNG TIN (blunt hại đúng dự đoán 77%-flat). "
                  f"Discipline diagnostic verified.")
        elif a >= 0.05:
            print(f"  ⚠️ Ablunt={a:+.3f} ≥+0.05 → Q4-diagnostic DỰ ĐOÁN SAI "
                  f"(blunt KHÔNG hại) → phải xem lại toàn bộ diagnostic-"
                  f"discipline (đã misfire nhiều). Quan trọng meta.")
        else:
            print(f"  Ablunt={a:+.3f}≈0 → blunt neutral (Q4 'hại' hơi quá; "
                  f"không catastrophe nhưng cũng không gain).")
    for c in CELLS:
        if c == "Ablunt" or c not in mp:
            continue
        m = mp[c]
        hc = any(cc == c for _, cc, _ in cata)
        if m >= 0.10 and not hc:
            print(f"  🎯 {c}={m:+.3f}≥+0.10 no-catastrophe → multi-seed "
                  f"137+9999 (cân cost).")
        elif m >= 0.05 and not hc:
            print(f"  🟡 {c}={m:+.3f} marginal → cân multi-seed + cost.")
        elif hc:
            print(f"  ❌ {c}={m:+.3f} có per-scene catastrophe → REJECT cell.")
        else:
            print(f"  ⚪ {c}={m:+.3f}≈0/neg → saturate/hại → reject.")
    print(f"\n  Caveat: single-seed N=8 (±0.10 floor); per-scene guard "
          f"override mean; λ no-natural-scale (B sweep 1e-3/1e-2); "
          f"multi-seed bắt buộc trước adopt.")


if __name__ == "__main__":
    main()
