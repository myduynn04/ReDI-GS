#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 37 — R1 v2] So hai point cloud RoMa v1
# File: scripts/p37_r1_compare_pointclouds.py  (KEEP LOCAL — server-only)
#
# BỐI CẢNH
#   Env `roma_v1` từng bị xoá, đã dựng lại 2026-08-07. Preprocess chạy lại
#   cho số điểm lệch −0.14% so với Phase 22 gốc. Số điểm bằng nhau KHÔNG
#   chứng minh điểm nằm cùng chỗ → script này hỏi "cùng bề mặt không".
#
# 🔴 v1 SAI — SỬA 2026-08-07
#   v1 đo r = median(NN chéo) / median(NN trong đám gốc), ngưỡng "tốt" r<0.10.
#   NGƯỠNG ĐÓ SAI VỀ TOÁN HỌC. Hai đám lấy mẫu CÙNG bề mặt với CÙNG mật độ
#   nhưng ở vị trí ngẫu nhiên khác nhau thì NN chéo ≈ NN nội tại (với Poisson,
#   cả hai đều ~0.5/√λ) → r ≈ 1 LÀ GIÁ TRỊ CỦA TRƯỜNG HỢP GIỐNG NHAU.
#   Đòi r<0.10 là đòi hai đám trùng TỪNG ĐIỂM — bất khả thi khi sampler ngẫu
#   nhiên. v1 chạy ra r≈0.98 trên cả 8 scene và báo "❌ LỆCH" = BÁO ĐỘNG GIẢ.
#
# CÁCH ĐO v2 — ĐỐI CHỨNG THỰC NGHIỆM, không dựa ngưỡng lý thuyết
#   Chia đám GỐC làm đôi ngẫu nhiên: B1, B2. Chia đám MỚI làm đôi: A1, A2.
#     d_null = median NN( B1 → B2 )   ← hai mẫu độc lập của CÙNG MỘT đám mây
#                                       = chuẩn vàng "hình học giống hệt"
#     d_real = median NN( A1 → B2 )   ← đám mới so đám gốc, CÙNG mật độ
#     ratio  = d_real / d_null
#   ratio ≈ 1  → không phân biệt được với "cùng hình học" ✅
#   ratio >> 1 → đám mới nằm xa bề mặt gốc ❌
#   Mọi thứ so ở NỬA mật độ cho cả hai vế → công bằng, không cần hiệu chỉnh.
#
#   ⚠ GIỚI HẠN PHÂN GIẢI: phép này KHÔNG phát hiện được dịch chuyển nhỏ hơn
#     khoảng cách giữa các điểm. Kết luận mạnh nhất nó cho được là
#     "cùng bề mặt trong phạm vi một spacing".
#
# BỔ SUNG v2 — thống kê không phụ thuộc mật độ
#   - Extent ROBUST theo percentile p1..p99 mỗi trục (thay min/max, vì min/max
#     do vài điểm biên quyết định → v1 báo room 47.9% chỉ vì 1 outlier)
#   - Trị riêng hiệp phương sai (hình dạng phân bố điểm) — bắt lệch scale/xoay
#
# ⚠ R1 VẪN CHỈ LÀ BỘ LỌC RẺ, KHÔNG PHẢI BẰNG CHỨNG REPRODUCIBILITY.
#   Bằng chứng thật = R2, train N=24 trên init mới và so 21.89 ± 0.10.
#
# KHÔNG CẦN GPU. Cần numpy + plyfile (scipy tuỳ chọn, nhanh hơn).
#
# Usage:
#   python scripts/p37_r1_compare_pointclouds.py 2>&1 | tee logs/p37/r1_compare.log
# ============================================================
"""Phase 37 R1 v2 — same-surface test with an empirical split-half control."""

import os
from pathlib import Path

import numpy as np

SCENES = os.environ.get(
    "SCENES", "fern flower fortress horns leaves orchids room trex"
).split()
DATA_ROOT = os.environ.get("DATA_ROOT", "data/nerf_llff_data")
N_VIEWS = os.environ.get("N_VIEWS", "3")

SAMPLE_N = int(os.environ.get("SAMPLE_N", "8000"))   # điểm mỗi nửa
RNG_SEED = 0

# ── Ngưỡng đăng ký trước, đặt trên RATIO so với đối chứng ──
RATIO_GOOD = 1.15   # ratio < 1.15 → không phân biệt được với "cùng hình học"
RATIO_OK   = 1.50   # ratio < 1.50 → chấp nhận
EXT_WARN   = 0.05   # lệch extent robust > 5% scene diag → cảnh báo


def dense_dir(scene: str) -> Path:
    return Path(DATA_ROOT) / scene / f"{N_VIEWS}_views/dense"


def load_ply(path: Path):
    try:
        from plyfile import PlyData
        v = PlyData.read(str(path))["vertex"]
        return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    except Exception as e:
        print(f"    ⚠ không đọc được {path.name}: {type(e).__name__} {e}")
        return None


# ============================================================
# [CRSGaussian P37 R1] nn_dist — NN distance từ query sang ref
# scipy cKDTree nếu có, không thì brute-force theo lô numpy.
# ============================================================
def nn_dist(query: np.ndarray, ref: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree
        d, _ = cKDTree(ref).query(query, k=1)
        return d
    except ImportError:
        pass
    out = np.empty(len(query), dtype=np.float64)
    CHUNK = 512
    for i in range(0, len(query), CHUNK):
        q = query[i:i + CHUNK]
        d2 = ((q[:, None, :] - ref[None, :, :]) ** 2).sum(-1)
        out[i:i + CHUNK] = np.sqrt(d2.min(axis=1))
    return out


def split_half(a: np.ndarray, n: int, rng):
    """Chia ngẫu nhiên thành 2 phần RỜI NHAU, mỗi phần tối đa n điểm."""
    idx = rng.permutation(len(a))
    h = len(a) // 2
    i1, i2 = idx[:h], idx[h:2 * h]
    if len(i1) > n:
        i1, i2 = i1[:n], i2[:n]
    return a[i1], a[i2]


def robust_extent(a: np.ndarray):
    """Extent p1..p99 mỗi trục — bỏ qua outlier biên."""
    lo = np.percentile(a, 1, axis=0)
    hi = np.percentile(a, 99, axis=0)
    return lo, hi, hi - lo


def cov_eigs(a: np.ndarray):
    """Trị riêng hiệp phương sai, giảm dần. Bắt lệch scale/xoay tổng thể."""
    c = np.cov((a - a.mean(0)).T)
    w = np.linalg.eigvalsh(c)
    return np.sort(w)[::-1]


def main():
    print("=" * 108)
    print("[P37 R1 v2] Kiểm định CÙNG BỀ MẶT — có đối chứng split-half")
    print(f"  SCENES={SCENES}  SAMPLE_N={SAMPLE_N} điểm/nửa")
    print()
    print("  🔴 v1 đã SAI: ngưỡng r<0.10 đòi hai đám trùng từng điểm — bất khả thi")
    print("     khi sampler ngẫu nhiên. r≈1 LÀ giá trị của trường hợp GIỐNG NHAU.")
    print("     v2 thay bằng đối chứng thực nghiệm: chia đám GỐC làm đôi và đo")
    print("     chính nó với chính nó, lấy đó làm chuẩn vàng.")
    print()
    print("  ⚠ R1 là BỘ LỌC RẺ, không phải bằng chứng reproducibility.")
    print("    Bằng chứng thật = R2 (train N=24, so 21.89 ± 0.10).")
    print("=" * 108)
    print()

    hdr = (f"{'scene':<10} {'N mới':>7} {'N gốc':>7} | "
           f"{'d_null (B1→B2)':>15} {'d_real (A1→B2)':>15} | "
           f"{'ratio':>7}  verdict")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for sc in SCENES:
        rng = np.random.default_rng(RNG_SEED)
        A = load_ply(dense_dir(sc) / "fused.ply.romav1_p37")   # dựng lại
        B = load_ply(dense_dir(sc) / "fused.ply.romav1")       # gốc Phase 22
        if A is None or B is None:
            print(f"{sc:<10} thiếu file — bỏ qua")
            continue

        B1, B2 = split_half(B, SAMPLE_N, rng)
        A1, _  = split_half(A, SAMPLE_N, rng)
        # Ép cùng cỡ để mật độ hai vế bằng nhau tuyệt đối
        m = min(len(A1), len(B1), len(B2))
        A1, B1, B2 = A1[:m], B1[:m], B2[:m]

        d_null = float(np.median(nn_dist(B1, B2)))   # chuẩn vàng: gốc vs gốc
        d_real = float(np.median(nn_dist(A1, B2)))   # mới vs gốc
        ratio = d_real / max(d_null, 1e-12)

        if ratio < RATIO_GOOD:
            v = "✅ CÙNG BỀ MẶT"
        elif ratio < RATIO_OK:
            v = "🟡 gần"
        else:
            v = "❌ LỆCH"

        print(f"{sc:<10} {len(A):>7} {len(B):>7} | {d_null:>15.6f} "
              f"{d_real:>15.6f} | {ratio:>7.3f}  {v}")
        rows.append(dict(scene=sc, ratio=ratio, d_null=d_null, d_real=d_real,
                         A=A, B=B))

    if not rows:
        print("\n❌ Không so được scene nào.")
        return

    # ── Extent robust + trị riêng — không phụ thuộc mật độ ──
    print()
    print("=" * 108)
    print("[R1] Thống kê không phụ thuộc mật độ")
    print("  extent p1..p99 mỗi trục (KHÔNG dùng min/max — v1 báo room 47.9% chỉ vì 1 outlier)")
    print("=" * 108)
    print(f"{'scene':<10} {'|Δextent| / diag':>18} {'Δ√λ1':>10} {'Δ√λ2':>10} {'Δ√λ3':>10}")
    print("-" * 62)
    for r_ in rows:
        A, B = r_["A"], r_["B"]
        _, _, extA = robust_extent(A)
        _, _, extB = robust_extent(B)
        diag = float(np.linalg.norm(extB))
        d_ext = float(np.linalg.norm(extA - extB)) / max(diag, 1e-12)
        ea, eb = np.sqrt(np.maximum(cov_eigs(A), 0)), np.sqrt(np.maximum(cov_eigs(B), 0))
        rel = (ea - eb) / np.maximum(eb, 1e-12)
        flag = "" if d_ext < EXT_WARN else "  ⚠"
        print(f"{r_['scene']:<10} {d_ext:>17.2%} "
              f"{rel[0]:>+9.2%} {rel[1]:>+9.2%} {rel[2]:>+9.2%}{flag}")
        r_["d_ext"] = d_ext
    print("  Δ√λ = lệch tương đối độ lệch chuẩn theo 3 trục chính.")
    print("  Vài % là nhiễu lấy mẫu. Chục % nghĩa là scale hoặc hệ toạ độ khác.")

    # ── Phán quyết ──
    n = len(rows)
    n_good = sum(1 for r_ in rows if r_["ratio"] < RATIO_GOOD)
    n_ok   = sum(1 for r_ in rows if RATIO_GOOD <= r_["ratio"] < RATIO_OK)
    n_bad  = sum(1 for r_ in rows if r_["ratio"] >= RATIO_OK)
    ratio_mean = float(np.mean([r_["ratio"] for r_ in rows]))
    ext_bad = sum(1 for r_ in rows if r_.get("d_ext", 0) >= EXT_WARN)

    print()
    print("=" * 108)
    print("[R1] PHÁN QUYẾT")
    print("=" * 108)
    print(f"  scene so được   : {n}/{len(SCENES)}")
    print(f"  ratio trung bình: {ratio_mean:.3f}   (tốt <{RATIO_GOOD}, chấp nhận <{RATIO_OK})")
    print(f"  ✅ {n_good} | 🟡 {n_ok} | ❌ {n_bad}    extent lệch >{EXT_WARN:.0%}: {ext_bad}")
    print()

    if n_bad > n / 2:
        print("  🔴 R1 FAIL — đám mới nằm xa bề mặt gốc hơn hẳn đối chứng.")
        print("     Nghi sai cấu hình (weights / upsample_res / coarse_res / commit repo).")
        print("     DỪNG, đừng tốn R2. Đối chiếu lại log Phase 22 gốc.")
    elif n_good == n and ext_bad == 0:
        print("  ✅ R1 PASS — mọi scene KHÔNG phân biệt được với 'cùng hình học'.")
        print("     Đám dựng lại và đám gốc là hai mẫu độc lập của cùng bề mặt;")
        print("     khác biệt duy nhất là sampler RoMa chọn điểm nào (khớp với")
        print("     dấu lệch số điểm hai chiều ở S1, và 'trùng khít' chỉ ~1%).")
        print()
        print("  ⚠ GIỚI HẠN — phép này mù với dịch chuyển nhỏ hơn spacing giữa")
        print("    các điểm. Nó nói 'cùng bề mặt trong phạm vi một spacing',")
        print("    KHÔNG nói 'train ra 21.89'. Muốn kết luận phải chạy R2.")
    else:
        print(f"  🟡 R1 PASS một phần — {n_good} tốt, {n_ok} gần, {n_bad} lệch, "
              f"{ext_bad} scene lệch extent.")
        print("     Đáng chạy R2, nhưng theo dõi riêng scene có ratio hoặc extent cao.")


if __name__ == "__main__":
    main()
