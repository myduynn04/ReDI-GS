#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 37 — S1 GATE] Tổng hợp + phán quyết
# File: scripts/p37_s1_qinit_stats.py  (KEEP LOCAL — server-only)
#
# Đọc 8 sidecar .npz do p37_romav1_preprocess_qinit.py sinh ra.
# KHÔNG cần GPU, KHÔNG chạy lại RoMa — chỉ numpy + plyfile.
#
# Làm hai việc, TÁCH BẠCH:
#
#   A. REPRODUCIBILITY — số điểm p37 có trùng fused.ply.romav1 gốc (Phase 22)?
#      Đây là cửa chặn quan trọng hơn cả H1. Lệch = env dựng lại KHÔNG tương
#      đương → Phase 22 không reproduce được.
#
#   B. S1 DATA GATE — Q_init có phương sai để khai thác không?
#      ⚠ XÉT RIÊNG TỪNG NỬA (sửa 2026-08-07 sau smoke fern).
#      Lý do: fern cho q_cert std=0.072 nhưng p05=p50=p95=1.000 — certainty
#      BÃO HOÀ (sampler RoMa đã lọc sẵn), trong khi q_rep std=0.251 sống khoẻ.
#      Bản gate cũ chỉ xét std(q_init) tổng → sẽ báo PASS mà che mất việc
#      một nửa tín hiệu đã chết. Phải nói rõ nửa nào sống.
#
# Tiêu chí giết (pre-registered docs/37 §5, áp cho TỪNG nửa):
#   CHẾT NẾU  std < 0.05  HOẶC  >90% điểm trong một bin rộng 0.05
#
# Usage:
#   python scripts/p37_s1_qinit_stats.py
#   SCENES="fern horns" python scripts/p37_s1_qinit_stats.py
# ============================================================
"""Phase 37 S1 gate — reproducibility check + per-half Q_init distribution."""

import os
from pathlib import Path

import numpy as np

SCENES = os.environ.get(
    "SCENES", "fern flower fortress horns leaves orchids room trex"
).split()
DATA_ROOT = os.environ.get("DATA_ROOT", "data/nerf_llff_data")
N_VIEWS = os.environ.get("N_VIEWS", "3")

# ── Ngưỡng pre-registered — KHÔNG đổi sau khi thấy kết quả ──
STD_KILL  = 0.05
BIN_WIDTH = 0.05
CONC_KILL = 0.90


def dense_dir(scene: str) -> Path:
    return Path(DATA_ROOT) / scene / f"{N_VIEWS}_views/dense"


# ============================================================
# [CRSGaussian P37] max_bin_fraction
# Tỉ lệ điểm rơi vào bin đông nhất. Bổ sung cho std: phân phối
# bimodal có std lớn vẫn dùng được, còn dồn 1 chỗ thì không.
# ============================================================
def max_bin_fraction(a: np.ndarray, width: float = BIN_WIDTH) -> float:
    if a.size == 0:
        return 0.0
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-9:
        return 1.0                      # tất cả bằng nhau → suy biến hoàn toàn
    n_bins = max(1, int(np.ceil((hi - lo) / width)))
    counts, _ = np.histogram(a, bins=n_bins, range=(lo, hi))
    return float(counts.max()) / a.size


def judge(a: np.ndarray) -> tuple:
    """Trả về (std, maxbin_frac, is_dead)."""
    s = float(a.std())
    f = max_bin_fraction(a)
    return s, f, (s < STD_KILL or f > CONC_KILL)


def count_ply(path: Path):
    """Đếm vertex trong .ply. None nếu không đọc được."""
    try:
        from plyfile import PlyData
        return len(PlyData.read(str(path))["vertex"])
    except Exception:
        return None


def main():
    print("=" * 100)
    print(f"[P37 S1 GATE] SCENES={SCENES}")
    print(f"  Ngưỡng: std < {STD_KILL} HOẶC >{CONC_KILL:.0%} trong 1 bin rộng {BIN_WIDTH}")
    print("=" * 100)

    rows, missing = [], []
    for sc in SCENES:
        p = dense_dir(sc) / "fused.romav1.qinit.npz"
        if not p.is_file():
            missing.append((sc, p))
            continue
        d = np.load(str(p))
        rows.append(dict(
            scene=sc, d=d,
            n_p37_ply=count_ply(dense_dir(sc) / "fused.ply.romav1_p37"),
            n_p22_ply=count_ply(dense_dir(sc) / "fused.ply.romav1"),
        ))

    if not rows:
        print("\n❌ Không đọc được sidecar nào.")
        print("   ⚠ ĐÂY CÓ THỂ LÀ LỖI HẠ TẦNG, KHÔNG PHẢI PHÁN QUYẾT S1.")
        print("   Kiểm tra: grep -inE 'error|traceback' logs/p37/s1_*.log")
        for sc, p in missing:
            print(f"     thiếu {sc:<10} {p}")
        return

    # ══════════════════════════════════════════════════════════════
    # PHẦN A — REPRODUCIBILITY (quan trọng hơn cả H1)
    # ══════════════════════════════════════════════════════════════
    print()
    print("=" * 100)
    print("[A] REPRODUCIBILITY — số điểm p37 vs Phase 22 gốc")
    print("=" * 100)
    print(f"{'scene':<10} {'N sidecar':>10} {'N p37.ply':>10} {'N P22 gốc':>10} "
          f"{'Δ':>8} {'Δ%':>8}  verdict")
    print("-" * 76)

    n_exact, n_close, n_bad, n_nocmp = 0, 0, 0, 0
    tot_p37, tot_p22 = 0, 0
    for r in rows:
        n_side = int(r["d"]["q_init"].size)
        n37, n22 = r["n_p37_ply"], r["n_p22_ply"]

        if n37 is not None and n_side != n37:
            # Sidecar phải khớp TỪNG PHẦN TỬ với vertex trong ply cùng lượt chạy
            v = "❌ LỆCH SIDECAR"
            n_bad += 1
            print(f"{r['scene']:<10} {n_side:>10} {n37:>10} "
                  f"{(n22 if n22 is not None else '—'):>10} {'':>8} {'':>8}  {v}")
            continue

        if n22 is None:
            print(f"{r['scene']:<10} {n_side:>10} "
                  f"{(n37 if n37 is not None else '—'):>10} {'—':>10} "
                  f"{'':>8} {'':>8}  ⚠ không đọc được ply gốc")
            n_nocmp += 1
            continue

        n_now = n37 if n37 is not None else n_side
        delta = n_now - n22
        pct = 100.0 * delta / max(n22, 1)
        tot_p37 += n_now
        tot_p22 += n22

        if delta == 0:
            v, = ("✅ TRÙNG KHỚP",); n_exact += 1
        elif abs(pct) < 1.0:
            v = "🟡 lệch <1%"; n_close += 1
        else:
            v = "❌ LỆCH"; n_bad += 1
        print(f"{r['scene']:<10} {n_side:>10} {n_now:>10} {n22:>10} "
              f"{delta:>+8} {pct:>+7.2f}%  {v}")

    if tot_p22 > 0:
        print("-" * 76)
        dt = tot_p37 - tot_p22
        print(f"{'TỔNG':<10} {'':>10} {tot_p37:>10} {tot_p22:>10} "
              f"{dt:>+8} {100.0*dt/tot_p22:>+7.2f}%")
        print(f"  (Phase 22 ghi nhận tổng ~178k điểm trên 8 scene)")

    print()
    if n_bad > 0:
        print("  🔴 REPRODUCIBILITY FAIL — env dựng lại KHÔNG tương đương Phase 22.")
        print("     DỪNG. Đây là vấn đề của con số 21.89, ưu tiên hơn H1.")
    elif n_exact == len(rows):
        print("  ✅ REPRODUCIBILITY PASS — trùng khớp TUYỆT ĐỐI mọi scene.")
        print("     Đã lấy lại khả năng reproduce Phase 22.")
    elif n_nocmp == len(rows):
        print("  ⚠ Không so được — thiếu fused.ply.romav1 gốc.")
    else:
        print(f"  🟡 PASS có sai lệch nhỏ — {n_exact} trùng tuyệt đối, {n_close} lệch <1%.")
        print("     RoMa v1 không hoàn toàn deterministic là chấp nhận được ở mức này.")

    # ══════════════════════════════════════════════════════════════
    # PHẦN B — S1 DATA GATE, XÉT RIÊNG TỪNG NỬA
    # ══════════════════════════════════════════════════════════════
    print()
    print("=" * 100)
    print("[B] S1 DATA GATE — xét RIÊNG q_cert và q_rep")
    print("=" * 100)
    print(f"{'scene':<10} | {'q_cert std':>10} {'maxbin':>7} {'':>6} | "
          f"{'q_rep std':>10} {'maxbin':>7} {'':>6} | {'q_init std':>10} {'':>6}")
    print("-" * 92)

    dead_cert, dead_rep, dead_init = 0, 0, 0
    for r in rows:
        d = r["d"]
        sc_, fc, kc = judge(d["q_cert"])
        sr_, fr, kr = judge(d["q_rep"])
        si_, fi, ki = judge(d["q_init"])
        dead_cert += kc; dead_rep += kr; dead_init += ki
        m = lambda k: "CHẾT" if k else "sống"
        print(f"{r['scene']:<10} | {sc_:>10.4f} {fc:>6.1%} {m(kc):>6} | "
              f"{sr_:>10.4f} {fr:>6.1%} {m(kr):>6} | {si_:>10.4f} {m(ki):>6}")

    n = len(rows)
    print()
    print("[B] Phân tích nguyên nhân — sampler hay mask triangulation giết phương sai?")
    print("-" * 92)
    print(f"{'scene':<10} {'cert TRƯỚC mask':>24} {'cert SAU mask':>24} {'p05 sau':>10}")
    for r in rows:
        pre, post = r["d"]["cert_all"], r["d"]["q_cert"]
        print(f"{r['scene']:<10} {pre.mean():>10.4f}±{pre.std():.4f}      "
              f"{post.mean():>10.4f}±{post.std():.4f}      "
              f"{np.percentile(post, 5):>9.3f}")
    print("  → p05 ≈ 1.000 nghĩa là >95% điểm certainty bão hoà.")
    print("    cert TRƯỚC ≈ cert SAU → thủ phạm là SAMPLER RoMa (không cứu được bằng τ).")

    # ══════════════════════════════════════════════════════════════
    print()
    print("=" * 100)
    print("[B] PHÁN QUYẾT S1")
    print("=" * 100)
    print(f"  scene đọc được : {n}/{len(SCENES)}")
    print(f"  q_cert CHẾT    : {dead_cert}/{n}")
    print(f"  q_rep  CHẾT    : {dead_rep}/{n}")
    print(f"  q_init CHẾT    : {dead_init}/{n}")
    print()

    half = n / 2
    if dead_cert > half and dead_rep > half:
        print("  🔪 KILL TOÀN BỘ — cả hai nửa suy biến. DỪNG H1, không viết code training.")
        print("     Ghi verdict vào docs/37 + decisions_log.")
    elif dead_cert > half:
        print("  🟡 CERTAINTY CHẾT, REPROJ SỐNG → đổi thiết kế, KHÔNG bỏ hướng.")
        print("     → Q_init dùng REPROJ-ONLY (W_CERT=0, W_REPROJ=1).")
        print("     → Novelty MẠNH HƠN, không yếu đi: S0 xác nhận InstantSplat chiếm")
        print("       phạm trù 'matcher CERTAINTY vào training'. Bỏ certainty là")
        print("       tránh hẳn nó. Còn 'reproj error làm trạng thái per-Gaussian")
        print("       thường trực' thì CHƯA AI chiếm (CoMapGS/Dense-SfM chỉ dùng làm")
        print("       ngưỡng lọc lúc init; RGS-SLAM tiêm 1 lần vào axial scale).")
        print("     → Đi tiếp S2 với Q_init = q_rep.")
    elif dead_rep > half:
        print("  🟡 REPROJ CHẾT, CERTAINTY SỐNG → Q_init dùng CERT-ONLY.")
        print("     ⚠ NHƯNG cảnh báo: hướng này đâm thẳng vào InstantSplat (S0 VERIFIED")
        print("       OCCUPIED ở cấp phạm trù). Novelty phải cãi ở mức cơ chế.")
    else:
        print("  ✅ CẢ HAI NỬA SỐNG — giữ nguyên thiết kế 0.5/0.5.")
        print("     → Đi tiếp S2 (cửa tín hiệu). Mốc so: detector CRS hiện tại ρ=0.30, 8/8.")

    if missing:
        print()
        print(f"  ⚠ {len(missing)} scene thiếu sidecar:")
        for sc, p in missing:
            print(f"      {sc:<10} {p}")


if __name__ == "__main__":
    main()
