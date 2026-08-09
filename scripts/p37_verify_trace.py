#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 37] So QUỸ ĐẠO N_gauss giữa các nhánh verify
# File: scripts/p37_verify_trace.py  (KEEP LOCAL — server-only)
#
# VÌ SAO CẦN
#   Verify lần 1 cho A2 − A1 = −953 Gaussian và script kết luận "buffer
#   KHÔNG trơ". Kết luận đó VƯỢT BẰNG CHỨNG: nó dựa trên giả định
#   "cùng seed ⇒ N_gauss trùng khít", mà giả định đó CHƯA AI ĐO.
#   Dự án đã biết atomicAdd gây ±1.3 dB PSNR trên 1 scene — gradient không
#   tất định thì quyết định densify cũng không tất định.
#
# PHÉP CHẨN ĐOÁN NÀY PHÂN BIỆT ĐƯỢC HAI KHẢ NĂNG, MIỄN PHÍ
#   Đọc dòng "[STATS] iter=… N=…" đã có sẵn trong log, dựng quỹ đạo N theo
#   iteration cho từng nhánh, rồi xem CHÚNG TÁCH NHAU TỪ LÚC NÀO.
#
#     Tách ngay từ lần densify ĐẦU TIÊN
#        → BUG THẬT. Code P37 đổi quyết định densify ngay lập tức.
#
#     Trùng khít vài nghìn iter rồi mới tách dần
#        → NHIỄU TÍCH LUỸ. atomicAdd làm gradient lệch chút, vượt ngưỡng
#          densify khác đi, rồi khuếch đại. Không phải bug.
#
#     Tách ngay nhưng chỉ vài Gaussian rồi phình
#        → nghi ngờ, cần đối chứng A1 chạy hai lần.
#
# KHÔNG CẦN GPU, KHÔNG CHẠY LẠI GÌ.
#
# Usage:
#   python scripts/p37_verify_trace.py 2>&1 | tee logs/p37/verify_trace.log
# ============================================================
"""Phase 37 — compare N_gauss trajectories across verify arms."""

import os
import re
import sys

LOG_DIR = os.environ.get("LOG_DIR", "logs/p37")
SCENE = os.environ.get("SCENE", "fern")

ARMS = [
    ("A0_orig_off",  "init gốc,  flag OFF"),
    ("A1_p37_off",   "init _p37, flag OFF"),
    ("A2_p37_on_w0", "init _p37, flag ON w_Q=0"),
]

# "[STATS] iter=500 | N=12345 | elapsed=..." — in mỗi 500 iter ở train.py
RE_STATS = re.compile(r"\[STATS\]\s*iter=(\d+)\s*\|\s*N=(\d+)")
# "[P37] iter=2000 w_Q=0.0 | freeze_frac(...)=0.4213 | q_all mean=... std=..."
RE_P37 = re.compile(
    r"\[P37\]\s*iter=(\d+)\s*w_Q=([\d.]+)\s*\|\s*freeze_frac\([^)]*\)=([\d.]+)"
    r"\s*\|\s*q_all mean=([\d.-]+) std=([\d.]+)"
    r"\s*\|\s*q_orig\(spawn=0\) n=(\d+) \(([\d.]+)%\) std=([\d.]+)"
)


def parse(arm: str):
    p = os.path.join(LOG_DIR, f"verify_{arm}_{SCENE}.log")
    if not os.path.isfile(p):
        return None, None, p
    txt = open(p, encoding="utf-8", errors="ignore").read()
    traj = {int(i): int(n) for i, n in RE_STATS.findall(txt)}
    p37 = [
        dict(iter=int(m[0]), w_q=float(m[1]), freeze=float(m[2]),
             q_mean=float(m[3]), q_std=float(m[4]),
             n_orig=int(m[5]), frac_orig=float(m[6]), std_orig=float(m[7]))
        for m in RE_P37.findall(txt)
    ]
    return traj, p37, p


def main():
    print("=" * 92)
    print(f"[P37 TRACE] quỹ đạo N_gauss — scene={SCENE}, log={LOG_DIR}")
    print("=" * 92)

    data = {}
    for arm, desc in ARMS:
        traj, p37, path = parse(arm)
        if traj is None:
            print(f"  ⚠ thiếu log: {path}")
            continue
        if not traj:
            print(f"  ⚠ {arm}: không tìm thấy dòng [STATS] nào trong log")
            continue
        data[arm] = (traj, p37)
        print(f"  ✓ {arm:<16} {desc:<26} {len(traj)} mốc")

    if "A1_p37_off" not in data or "A2_p37_on_w0" not in data:
        print("\n❌ Cần cả A1 và A2 để so. Dừng.")
        return

    t1 = data["A1_p37_off"][0]
    t2 = data["A2_p37_on_w0"][0]
    t0 = data.get("A0_orig_off", (None,))[0]
    iters = sorted(set(t1) & set(t2))

    # ── Bảng quỹ đạo ──
    print()
    print("=" * 92)
    print("[TRACE] A2 vs A1 — CÙNG init _p37, CÙNG seed, chỉ khác cờ P37")
    print("=" * 92)
    hdr = f"{'iter':>7} {'A1 (OFF)':>10} {'A2 (ON w0)':>11} {'ΔN':>8} {'Δ%':>8}"
    if t0:
        hdr += f"   {'A0 (init gốc)':>14}"
    print(hdr)
    print("-" * len(hdr))

    first_div = None
    for it in iters:
        d = t2[it] - t1[it]
        pct = 100.0 * d / max(t1[it], 1)
        line = f"{it:>7} {t1[it]:>10} {t2[it]:>11} {d:>+8} {pct:>+7.2f}%"
        if t0 and it in t0:
            line += f"   {t0[it]:>14}"
        print(line)
        if first_div is None and d != 0:
            first_div = (it, d)

    # ── Log P37 — buffer có sống đúng không ──
    p37_2 = data["A2_p37_on_w0"][1]
    if p37_2:
        print()
        print("=" * 92)
        print("[TRACE] Log [P37] trong nhánh A2 — buffer có hoạt động đúng không")
        print("=" * 92)
        print(f"{'iter':>7} {'w_Q':>6} {'freeze_frac':>12} {'q_all mean':>11} "
              f"{'q_all std':>10} {'n_orig':>8} {'%orig':>7} {'std_orig':>9}")
        print("-" * 76)
        for r in p37_2:
            print(f"{r['iter']:>7} {r['w_q']:>6.2f} {r['freeze']:>12.4f} "
                  f"{r['q_mean']:>11.4f} {r['q_std']:>10.4f} "
                  f"{r['n_orig']:>8} {r['frac_orig']:>6.1f}% {r['std_orig']:>9.4f}")
        print("  q_all std tụt dần trong khi std_orig giữ nguyên = con cháu pha")
        print("  loãng tín hiệu (docs/37 §13.7c). Đây là dữ liệu, không phải lỗi.")

    # ── Phán quyết ──
    print()
    print("=" * 92)
    print("[TRACE] PHÁN QUYẾT — bug hay nhiễu?")
    print("=" * 92)

    if first_div is None:
        print("  ✅ HAI QUỸ ĐẠO TRÙNG KHÍT TOÀN BỘ.")
        print("     Buffer trơ tuyệt đối. Chênh lệch cuối (nếu có) chỉ ở dòng")
        print("     TIMING, không phải ở quỹ đạo → đọc lại parse.")
        return

    it_div, d_div = first_div
    densify_from = 500     # opt.densify_from_iter mặc định
    print(f"  Tách nhau LẦN ĐẦU tại iter = {it_div}, ΔN = {d_div:+d}")
    print(f"  (densify bắt đầu quanh iter {densify_from}, log mỗi 500 iter)")
    print()

    if it_div <= densify_from + 500:
        print("  🔴 NGHI BUG THẬT — tách ngay từ lần densify đầu tiên.")
        print("     Code P37 đổi quyết định densify ngay lập tức, không phải")
        print("     nhiễu tích luỹ. Soi lại densification_postfix + prune_points.")
    else:
        print("  🟡 TÁCH MUỘN — dấu hiệu của NHIỄU TÍCH LUỸ, không phải bug.")
        print(f"     Hai nhánh chạy giống hệt tới iter {it_div} rồi mới lệch.")
        print("     Nếu là bug thì phải lệch ngay từ lần densify đầu.")
        print()
        print("     ⚠ NHƯNG CHƯA KẾT LUẬN ĐƯỢC. Phải có ĐỐI CHỨNG:")
        print("       chạy A1 HAI LẦN với cùng seed, đo |N_a − N_b|.")
        print("       Đó mới là phân phối null. Nếu |A2−A1| nằm trong đó → trơ.")
        print("       → bash scripts/p37_verify_null.sh")

    # Độ lớn tương đối để tham chiếu nhanh
    last = iters[-1]
    print()
    print(f"  Cuối cùng (iter {last}): ΔN = {t2[last]-t1[last]:+d} "
          f"({100.0*(t2[last]-t1[last])/max(t1[last],1):+.2f}%)")
    if t0 and last in t0:
        print(f"  Tham chiếu — A0 dùng init KHÁC: N={t0[last]} "
              f"(lệch {t0[last]-t1[last]:+d} so A1). Cùng cỡ ⇒ 1% là biên độ")
        print(f"  bình thường của quá trình này, không phải chữ ký của bug.")


if __name__ == "__main__":
    main()
