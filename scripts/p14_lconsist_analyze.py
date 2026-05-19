#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 14] L_consist 2×2 pilot analyzer.
# File: scripts/p14_lconsist_analyze.py  (TẠO MỚI)
# Parse logs/p14_lconsist/{B,C,D}_seed42_<scene>.log + Cell A baseline
# logs/p13_lfcf/A3_seed42_<scene>.log → paired Δ per scene.
#
# 2×2: A=A3(reuse) B=A3+Lc C=A3−Dcyc D=A3−Dcyc+Lc
#   ΔB=B−A : Lc THÊM value trên A3 ?
#   ΔC=C−A : D_cycle còn giúp A3 ? (≥0 → D_cycle dead-weight, Phase-7 flip)
#   ΔD=D−A , D vs B : Lc THAY được D_cycle ?
#
# Metrics: PSNR(↑) + SSIM(↑) + LPIPS(↓) + train-time(s) + Lc cost%
#   (feedback_measure_compute_cost: KHÔNG chỉ PSNR — kèm SSIM/LPIPS+speed).
# Verdict (single-seed N=8 paired — project_3dgs_variance_floor ±0.10):
#   + per-scene CATASTROPHE GUARD: scene nào ΔPSNR < −0.20 → flag ĐẬM,
#     KHÔNG để mean che (bài học GDAGS horns −0.605 / HF horns −0.31).
#   N primary cho VOFF byte-identical (PSNR ±1.3 single-scene noise).
#
# Run:
#   python scripts/p14_lconsist_analyze.py
#   python scripts/p14_lconsist_analyze.py --verify   # check VOFF=A3
# ============================================================
"""[CRSGaussian Phase 14] L_consist 2×2 pilot analyzer (PSNR/SSIM/LPIPS/time)."""

import os
import re
import sys
import statistics

SCENES = os.environ.get(
    "SCENES", "fern flower fortress horns leaves orchids room trex").split()
SEED = os.environ.get("SEED", "42")
LC_LOG_DIR = os.environ.get("LC_LOG_DIR", "logs/p14_lconsist")
A_LOG_DIR = os.environ.get("A_LOG_DIR", "logs/p13_lfcf")
CATA = float(os.environ.get("CATASTROPHE", "0.20"))   # per-scene PSNR guard

# SUMMARY row: "<it> | test | <PSNR> | <SSIM> | <LPIPS> | <L1> | <N>"
METRIC_PAT = re.compile(
    r"\b(\d{2,5})\s*\|\s*(test|train)\s*\|\s*"
    r"([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|")
N_PAT = re.compile(r"Best test PSNR:\s*[0-9.]+\s*at iter\s*\d+\s*\(N=(\d+)\)")
TIME_PAT = re.compile(r"Total training:\s*([0-9.]+)\s*s")


def _read(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def parse_metrics(log_path):
    """→ dict {psnr, ssim, lpips} của test split iter cuối, hoặc None."""
    txt = _read(log_path)
    if txt is None:
        return None
    test = {}
    for it, sp, ps, ss, lp in METRIC_PAT.findall(txt):
        if sp == "test":
            test[int(it)] = (float(ps), float(ss), float(lp))
    if not test:
        return None
    ps, ss, lp = test[max(test)]
    return {"psnr": ps, "ssim": ss, "lpips": lp}


def parse_n(log_path):
    txt = _read(log_path)
    if txt is None:
        return None
    m = N_PAT.findall(txt)
    return int(m[-1]) if m else None


def parse_time(log_path):
    txt = _read(log_path)
    if txt is None:
        return None
    m = TIME_PAT.findall(txt)
    return float(m[-1]) if m else None


def main():
    verify = "--verify" in sys.argv
    print(f"=== Phase 14 L_consist 2×2 analyzer "
          f"({'VERIFY VOFF=A3' if verify else 'pilot B/C/D vs A'}) ===")
    print(f"A baseline: {A_LOG_DIR}/A3_seed{SEED}_*  | "
          f"{LC_LOG_DIR}/{{B,C,D,VOFF}}_seed{SEED}_*\n")

    if verify:
        print("  [PRIMARY = N_gaussians; PSNR single-scene ±1.3 noise]")
        ok = True
        for sc in SCENES:
            an = parse_n(f"{A_LOG_DIR}/A3_seed{SEED}_{sc}.log")
            vn = parse_n(f"{LC_LOG_DIR}/VOFF_seed{SEED}_{sc}.log")
            if an is None or vn is None:
                print(f"  {sc}: N MISSING (A={an}, VOFF={vn}) — skip")
                continue
            rel = abs(vn - an) / max(an, 1)
            tag = ("✅ ≈A3" if rel < 0.05 else
                   "🟡 nhẹ" if rel < 0.20 else "❌ LỆCH LỚN")
            print(f"  {sc}: N {an}→{vn} ({rel*100:+.1f}%) {tag}")
            if rel >= 0.20:
                ok = False
        print("\n  " + ("✅ flag-OFF ≈ A3 (N<5% mọi scene) → code Phase-14 "
                         "KHÔNG phá A3 khi tắt. Pilot baseline hợp lệ."
                         if ok else
                         "❌ N LỆCH ≥20% → code phá A3 khi flag OFF. STOP."))
        return

    rows = []
    dB = {"psnr": [], "ssim": [], "lpips": []}
    dC = {"psnr": [], "ssim": [], "lpips": []}
    dD = {"psnr": [], "ssim": [], "lpips": []}
    tcell = {"A": [], "B": [], "C": [], "D": []}
    cata = []   # (scene, cell, ΔPSNR) — per-scene catastrophe

    for sc in SCENES:
        mA = parse_metrics(f"{A_LOG_DIR}/A3_seed{SEED}_{sc}.log")
        mB = parse_metrics(f"{LC_LOG_DIR}/B_seed{SEED}_{sc}.log")
        mC = parse_metrics(f"{LC_LOG_DIR}/C_seed{SEED}_{sc}.log")
        mD = parse_metrics(f"{LC_LOG_DIR}/D_seed{SEED}_{sc}.log")
        tA = parse_time(f"{A_LOG_DIR}/A3_seed{SEED}_{sc}.log")
        tB = parse_time(f"{LC_LOG_DIR}/B_seed{SEED}_{sc}.log")
        tC = parse_time(f"{LC_LOG_DIR}/C_seed{SEED}_{sc}.log")
        tD = parse_time(f"{LC_LOG_DIR}/D_seed{SEED}_{sc}.log")
        for k, t in (("A", tA), ("B", tB), ("C", tC), ("D", tD)):
            if t is not None:
                tcell[k].append(t)
        if mA is None:
            print(f"  {sc}: A MISSING — skip"); continue
        for cell, m, acc in (("B", mB, dB), ("C", mC, dC), ("D", mD, dD)):
            if m is not None:
                acc["psnr"].append(m["psnr"] - mA["psnr"])
                acc["ssim"].append(m["ssim"] - mA["ssim"])
                acc["lpips"].append(m["lpips"] - mA["lpips"])
                if (m["psnr"] - mA["psnr"]) < -CATA:
                    cata.append((sc, cell, m["psnr"] - mA["psnr"]))
        rows.append((sc, mA, mB, mC, mD))

    # ── PSNR table (primary, per-scene + catastrophe) ──
    print("=== PSNR (↑) per-scene + Δ vs A ===")
    hdr = (f"{'scene':<9} {'A':>7} {'B':>7} {'C':>7} {'D':>7} "
           f"{'ΔB':>7} {'ΔC':>7} {'ΔD':>7}")
    print(hdr); print("-" * len(hdr))
    for (sc, mA, mB, mC, mD) in rows:
        def pv(m):
            return f"{m['psnr']:7.3f}" if m else "   -   "
        def dv(m):
            return f"{m['psnr']-mA['psnr']:+.3f}" if m else "  -  "
        print(f"{sc:<9} {mA['psnr']:7.3f} {pv(mB)} {pv(mC)} {pv(mD)} "
              f"{dv(mB):>7} {dv(mC):>7} {dv(mD):>7}")

    def stat(acc, name, lower_better=False):
        v = acc
        if not v:
            print(f"  Δ{name}: NO data"); return None
        m = statistics.fmean(v)
        sd = statistics.pstdev(v) if len(v) > 1 else 0.0
        arrow = "(↓ tốt)" if lower_better else "(↑ tốt)"
        print(f"  Δ{name} = {m:+.4f} {arrow} (std {sd:.4f}, N={len(v)})")
        return m

    print("\n=== Aggregate Δ (mean over scenes) ===")
    print(" [B = A3+Lc]")
    mBp = stat(dB["psnr"], "B_PSNR")
    stat(dB["ssim"], "B_SSIM")
    stat(dB["lpips"], "B_LPIPS", lower_better=True)
    print(" [C = A3−Dcyc]")
    mCp = stat(dC["psnr"], "C_PSNR")
    stat(dC["ssim"], "C_SSIM")
    stat(dC["lpips"], "C_LPIPS", lower_better=True)
    print(" [D = A3−Dcyc+Lc]")
    mDp = stat(dD["psnr"], "D_PSNR")
    stat(dD["ssim"], "D_SSIM")
    stat(dD["lpips"], "D_LPIPS", lower_better=True)
    if dB["psnr"] and dD["psnr"]:
        print(f"  D−B (PSNR) = {statistics.fmean(dD['psnr'])-statistics.fmean(dB['psnr']):+.4f}"
              f"  (Lc THAY D_cycle tốt hơn nếu > 0)")

    # ── TIMING / cost (feedback_measure_compute_cost) ──
    print("\n=== TRAIN-TIME / cost ===")
    def mt(k):
        return statistics.fmean(tcell[k]) if tcell[k] else None
    for k in ("A", "B", "C", "D"):
        t = mt(k)
        print(f"  {k}: mean train-time = "
              f"{(f'{t:.0f}s ({t/60:.1f}min)' if t else 'N/A (log thiếu [TIMING])')}"
              f"  (N={len(tcell[k])})")
    tB, tC, tA = mt("B"), mt("C"), mt("A")
    base = tC if tC else tA   # C = Lc-off ≈ A3-speed (cùng đợt run); fallback A
    if tB and base:
        print(f"  → L_consist overhead = {tB-base:+.0f}s "
              f"({(tB/base-1)*100:+.1f}%) vs {'C(Lc-off)' if tC else 'A'}")
        print(f"     (cost THẬT — cân ở verdict kể cả khi PSNR tăng)")

    # ── per-scene catastrophe guard ──
    print("\n=== PER-SCENE CATASTROPHE GUARD (PSNR) ===")
    if cata:
        print(f"  ❌ {len(cata)} scene-cell ΔPSNR < −{CATA} (mean CHE — "
              f"bài học GDAGS horns/HF horns):")
        for (sc, cl, e) in cata:
            print(f"     {cl}/{sc}: ΔPSNR={e:+.3f}")
        print("  → cell có catastrophe = REJECT cell đó DÙ mean đẹp.")
    else:
        print(f"  ✅ Không scene nào ΔPSNR < −{CATA} ở B/C/D.")

    # ── verdict ──
    print("\n=== VERDICT (single-seed N=8 paired, ±0.10 floor) ===")
    if mCp is not None:
        if mCp >= -0.05:
            print(f"  ΔC_PSNR={mCp:+.3f} ≥ −0.05 → **D_cycle ≈ dead-weight "
                  f"trên A3** (đúng nghi vấn Phase-7 flip). Finding độc lập.")
        else:
            print(f"  ΔC_PSNR={mCp:+.3f} < −0.05 → D_cycle VẪN giúp A3 (giữ).")
    if mBp is not None:
        if mBp >= 0.10:
            print(f"  ΔB_PSNR={mBp:+.3f} ≥ +0.10 → Lc THÊM value → multi-seed "
                  f"137+9999 (nếu no-catastrophe + cost chấp nhận được).")
        elif mBp >= 0.05:
            print(f"  ΔB_PSNR={mBp:+.3f} ∈[+0.05,+0.10) marginal → cân cost "
                  f"(+~35% time) + multi-seed.")
        elif mBp >= -0.05:
            print(f"  ΔB_PSNR={mBp:+.3f} ≈ 0 → Lc BÃO HOÀ trên A3 → reject "
                  f"(D_cycle+CRS+pearson đã dọn; +35% time vô ích).")
        else:
            print(f"  ΔB_PSNR={mBp:+.3f} < −0.05 → Lc HẠI → reject.")
    if mBp is not None and mDp is not None and mDp >= 0.10 and mDp >= mBp:
        print(f"  ΔD_PSNR={mDp:+.3f} ≥ +0.10 & ≥ ΔB → Lc THAY D_cycle "
              f"(bỏ D_cycle + dùng Lc) — kịch bản giá trị cao nhất.")
    print(f"\n  Caveat: single-seed N=8 (±0.10 floor); per-scene guard "
          f"override mean; SSIM/LPIPS = phụ; cost cân ở adopt; multi-seed "
          f"bắt buộc trước adopt.")


if __name__ == "__main__":
    main()
