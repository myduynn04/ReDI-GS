#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 18b — Step 1] PDCNet+ gate-signal diagnostic
# File: scripts/p18b_confidence_histogram.py  (keep local)
#
# KHÔNG train. Đọc các file <scene>_confidence.npy và <scene>_cyclic.npy
# do triangulate.py (bản [P18b diag]) dump → so 2 tín hiệu gate ứng viên:
#   - confidence (p_r)            : cao = tốt   (mạng tự ước xác suất)
#   - cyclic_consistency_error    : THẤP = tốt  (kiểm hình học fwd-bwd, pixel)
#
# Câu hỏi quyết định (Phase 18b chọn trục gate):
#   H1  scene-thua (horns/trex) có đuôi "xấu" dày hơn scene-thắng không?
#   H2  histogram pooled có bimodal → valley = τ a-priori?
#   D   DISCRIMINATION — per-scene "bad fraction" có tương quan với pilot Δ
#       N=24 không? Tín hiệu TỐT ⟺ |r| cao ∧ KHÔNG dính phản-ví-dụ orchids
#       (orchids đuôi-confidence-thấp dày nhất NHƯNG pilot THẮNG +0.139).
#
# Chạy (env corgs, dir CoR-GS — chỉ cần numpy):
#   python scripts/p18b_confidence_histogram.py
#   CONF_DIR=/tmp/p18_gate1 python scripts/p18b_confidence_histogram.py
# ============================================================
"""[CRSGaussian Phase 18b Step 1] PDCNet+ gate-signal diagnostic (confidence + cyclic)."""

import os
import numpy as np

SCENES = os.environ.get(
    "SCENES", "fern flower fortress horns leaves orchids room trex").split()
CONF_DIR = os.environ.get("CONF_DIR", "/tmp/p18_gate1")
GOOD = os.environ.get("GOOD_SCENES", "fortress leaves flower").split()
BAD = os.environ.get("BAD_SCENES", "horns trex").split()

# Pilot N=24 paired Δ (PDCNet+ raw init − MVS init) — để test DISCRIMINATION.
# Nguồn: decisions_log [2026-05-21] Phase 18.
PILOT_DELTA = {
    "fern": +0.077, "flower": +0.512, "fortress": +0.996, "horns": -0.318,
    "leaves": +0.839, "orchids": +0.139, "room": +0.268, "trex": -0.357,
}
NBINS = 40


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xm, ym = x - x.mean(), y - y.mean()
    d = np.sqrt((xm ** 2).sum() * (ym ** 2).sum())
    return float((xm * ym).sum() / d) if d > 0 else float("nan")


def spearman(x, y):
    def rk(v):
        order = np.argsort(np.argsort(np.asarray(v, float)))
        return order.astype(float)
    return pearson(rk(x), rk(y))


def load_signal(suffix):
    """Trả {scene: np.array}. suffix = 'confidence' hoặc 'cyclic'."""
    data = {}
    for sc in SCENES:
        p = os.path.join(CONF_DIR, f"{sc}_{suffix}.npy")
        if not os.path.isfile(p):
            continue
        arr = np.load(p).astype(np.float64).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if len(arr):
            data[sc] = arr
    return data


def text_histogram(pooled, label, lo=None, hi=None):
    print(f"\n--- histogram POOLED — {label} ---")
    if lo is None:
        lo = np.percentile(pooled, 0.5)
    if hi is None:
        hi = np.percentile(pooled, 99.5)
    counts, edges = np.histogram(pooled, bins=NBINS, range=(lo, hi))
    peak = max(counts.max(), 1)
    for i, c in enumerate(counts):
        bar = "#" * int(round(50 * c / peak))
        print(f"  [{edges[i]:8.3f},{edges[i+1]:8.3f})  {c:>9}  {bar}")


def analyze(data, label, better):
    """better='high' (confidence) hoặc 'low' (cyclic error).
    Trả per-scene 'bad fraction' dict + median dict để test discrimination."""
    if not data:
        print(f"\n⚠️  KHÔNG có dữ liệu '{label}' — bỏ qua.")
        return None, None
    print("\n" + "=" * 88)
    print(f"TÍN HIỆU = {label}   (better = {better};  "
          f"{'cao' if better == 'high' else 'thấp'} = match tốt)")
    print("=" * 88)

    pooled = np.concatenate([data[s] for s in SCENES if s in data])
    # ngưỡng "xấu" để tính bad-fraction: dùng percentile của pooled (PSNR-independent)
    if better == "high":
        taus = [round(np.percentile(pooled, q), 3) for q in (10, 20, 30, 40, 50)]
        bad_of = lambda a, t: float((a < t).mean())          # thấp = xấu
    else:
        taus = [round(np.percentile(pooled, q), 3) for q in (50, 60, 70, 80, 90)]
        bad_of = lambda a, t: float((a > t).mean())          # cao = xấu

    # per-scene table
    print(f"\n{'scene':<10} {'N':>9} {'min':>8} {'med':>8} {'mean':>8} "
          f"{'max':>8}  " + " ".join(f'bad@{t:g}'.rjust(9) for t in taus) + "   pilotΔ")
    print("-" * 88)
    bad_mid = {}        # bad-fraction ở tau giữa — dùng cho discrimination
    med = {}
    tau_mid = taus[len(taus) // 2]
    for sc in SCENES:
        if sc not in data:
            continue
        a = data[sc]
        bfr = " ".join(f"{100*bad_of(a,t):>8.1f}%" for t in taus)
        bad_mid[sc] = bad_of(a, tau_mid)
        med[sc] = float(np.median(a))
        tag = ("  <--BAD" if sc in BAD else
               ("  <--good" if sc in GOOD else
                ("  <--orchids?" if sc == "orchids" else "")))
        print(f"{sc:<10} {len(a):>9} {a.min():>8.3f} {np.median(a):>8.3f} "
              f"{a.mean():>8.3f} {a.max():>8.3f}  {bfr}  {PILOT_DELTA.get(sc,0):+6.3f}{tag}")

    text_histogram(pooled, label)

    # H1 — good vs bad nhóm
    print(f"\n[H1] đuôi-xấu good vs bad  (bad-fraction @ tau_mid={tau_mid:g})")
    for grp, names in (("good", GOOD), ("bad", BAD)):
        arrs = [data[s] for s in names if s in data]
        if arrs:
            pl = np.concatenate(arrs)
            bf = bad_of(pl, tau_mid)
            print(f"   {grp:<5}: bad-fraction = {100*bf:.1f}%")

    # D — DISCRIMINATION vs pilot Δ
    scs = [s for s in SCENES if s in data]
    dl = [PILOT_DELTA[s] for s in scs]
    bf = [bad_mid[s] for s in scs]
    mv = [med[s] for s in scs]
    print(f"\n[D] DISCRIMINATION — per-scene vs pilot Δ N=24  (N_scene={len(scs)})")
    print(f"   r(bad-fraction, Δ)  Pearson={pearson(bf,dl):+.3f}  "
          f"Spearman={spearman(bf,dl):+.3f}   (tín hiệu tốt ⟹ ÂM mạnh)")
    print(f"   r(median,       Δ)  Pearson={pearson(mv,dl):+.3f}  "
          f"Spearman={spearman(mv,dl):+.3f}   (tín hiệu tốt ⟹ "
          f"{'DƯƠNG' if better=='high' else 'ÂM'} mạnh)")
    # bảng sắp theo pilot Δ → nhìn pattern + orchids
    print(f"\n   scene xếp theo pilot Δ (xem bad-fraction có đi cùng chiều không):")
    for s in sorted(scs, key=lambda z: PILOT_DELTA[z]):
        flag = " <-- orchids (phản-ví-dụ confidence)" if s == "orchids" else ""
        print(f"     {s:<10} Δ={PILOT_DELTA[s]:+.3f}   "
              f"bad-frac={100*bad_mid[s]:5.1f}%   med={med[s]:.3f}{flag}")
    return bad_mid, med


def keep_fraction(data, label, better):
    if not data:
        return
    pooled = np.concatenate([data[s] for s in SCENES if s in data])
    if better == "high":
        taus = [round(np.percentile(pooled, q), 3)
                for q in (5, 10, 20, 30, 40, 50, 60)]
        keep_of = lambda a, t: float((a >= t).mean())
    else:
        taus = [round(np.percentile(pooled, q), 3)
                for q in (95, 90, 80, 70, 60, 50, 40)]
        keep_of = lambda a, t: float((a <= t).mean())
    print(f"\n--- KEEP-FRACTION — {label} (một τ chung, mỗi scene giữ %) ---")
    print(f"{'τ':>9}  " + " ".join(f"{s[:7]:>8}" for s in SCENES if s in data))
    print("-" * 88)
    for t in taus:
        row = f"{t:>9.3f}  "
        for sc in SCENES:
            if sc in data:
                row += f"{100*keep_of(data[sc],t):>7.1f}%"
        print(row)


def main():
    print("=" * 88)
    print("Phase 18b Step 1 — PDCNet+ gate-signal diagnostic (confidence + cyclic)")
    print(f"CONF_DIR={CONF_DIR}  scenes={SCENES}")
    print("=" * 88)

    conf = load_signal("confidence")
    cyc = load_signal("cyclic")
    if not conf and not cyc:
        print("\n❌ Không có file .npy nào. Chạy triangulate.py [P18b diag] trước.")
        return

    bc, _ = analyze(conf, "confidence (p_r)", better="high")
    keep_fraction(conf, "confidence (p_r)", better="high")
    by, _ = analyze(cyc, "cyclic_consistency_error", better="low")
    keep_fraction(cyc, "cyclic_consistency_error", better="low")

    print("\n" + "=" * 88)
    print("ĐỌC KẾT QUẢ:")
    print("  - So [D] r(bad-fraction,Δ) của 2 tín hiệu: |r| lớn hơn = discriminate tốt hơn.")
    print("  - Nhìn dòng orchids: tín hiệu TỐT thì orchids bad-frac PHẢI thấp")
    print("    (giống scene-thắng), KHÔNG cao như horns/trex.")
    print("  - cyclic |r| > confidence |r| ∧ orchids OK ⟹ chốt gate theo cyclic.")
    print("    cả hai |r| yếu / orchids vẫn dính ⟹ substrate yếu → cân nhắc dừng.")
    print("=" * 88)


if __name__ == "__main__":
    main()
