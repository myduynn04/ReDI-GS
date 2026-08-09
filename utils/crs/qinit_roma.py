# ============================================================
# [CRSGaussian Phase 37 — H1] Q_init từ RoMa triangulation
# File: utils/crs/qinit_roma.py  (TẠO MỚI)
#
# MỤC ĐÍCH
#   Nạp sidecar .npz do scripts/p37_romav1_preprocess_qinit.py sinh ra,
#   dựng per-point score Q_init, đưa vào logit CRS.
#
#   Q_init đo ĐỘ TIN CẬY VỊ TRÍ 3D của điểm khởi tạo, lấy từ chính
#   phép triangulate đã dựng ra nó — thông tin RoMa tính rồi vứt đi.
#
# BỐI CẢNH THỰC NGHIỆM (docs/37 §12)
#   S1 8/8 scene: q_cert CHẾT (sampler RoMa bão hoà, p05 = 1.000 mọi scene),
#   q_rep SỐNG (std 0.245-0.279). → mặc định REPROJ-ONLY, W_CERT = 0.
#
# 🔴 KHÔNG NHẦM VỚI informed_crs_init CŨ (đã WASH ở Phase 24)
#   Tín hiệu cũ lấy reproj error của COLMAP rồi gán cho điểm RoMa bằng
#   nearest-neighbor (utils/crs/crs_init.py:348) — trên backbone dense thì
#   22-26k điểm dùng chung ~3k giá trị của MỘT TẬP ĐIỂM KHÁC. Đó là phép
#   bôi thô, không phải tín hiệu per-point.
#   Q_init ở đây là sai số THẬT của chính phép triangulate ra điểm đó.
#
# Được gọi từ: scene/__init__.py (nạp) và utils/crs/crs_module.py (dùng)
# ============================================================
"""Phase 37 H1 — per-Gaussian init reliability from RoMa triangulation."""

import os
from typing import Optional

import numpy as np
import torch

# Tên sidecar do p37_romav1_preprocess_qinit.py ghi ra
SIDECAR_NAME = "fused.romav1.qinit.npz"


# ============================================================
# [CRSGaussian P37] find_qinit_sidecar
# Mục đích: tìm sidecar .npz cạnh fused.ply của scene đang train.
# Lý do tách hàm: đường dẫn dense dir phụ thuộc n_views, và scene/__init__
#                 không nên biết chi tiết layout của Phase 22.
# Trả None nếu không có → caller tự quyết định fail hay bỏ qua.
# ============================================================
def find_qinit_sidecar(source_path: str, n_views: int) -> Optional[str]:
    p = os.path.join(source_path, f"{n_views}_views", "dense", SIDECAR_NAME)
    return p if os.path.isfile(p) else None


# ============================================================
# [CRSGaussian P37] load_q_init
# Mục đích: đọc sidecar → vector Q_init (N,) đã chuẩn hoá về [0,1].
#
# Args:
#   path        : đường dẫn .npz
#   w_cert      : trọng số certainty. MẶC ĐỊNH 0.0 — S1 chứng minh nó bão hoà
#   w_reproj    : trọng số reprojection quality. Mặc định 1.0
#   expect_n    : số điểm kỳ vọng (= số Gaussian init). None thì bỏ qua check
#
# Trả (N,) float32 numpy, hoặc raise nếu lệch chiều dài.
#   Lệch chiều dài KHÔNG được bỏ qua im lặng — index lệch thì mọi thứ
#   sau đó sai âm thầm, đúng kiểu lỗi khó nhất để phát hiện.
# ============================================================
def load_q_init(path: str,
                w_cert: float = 0.0,
                w_reproj: float = 1.0,
                expect_n: Optional[int] = None) -> np.ndarray:
    d = np.load(path)

    q_cert = d["q_cert"].astype(np.float32).reshape(-1)
    q_rep = d["q_rep"].astype(np.float32).reshape(-1)

    if q_cert.shape[0] != q_rep.shape[0]:
        raise ValueError(
            f"[P37] sidecar hỏng: q_cert {q_cert.shape[0]} != q_rep {q_rep.shape[0]}"
        )

    if expect_n is not None and q_rep.shape[0] != expect_n:
        raise ValueError(
            f"[P37] Q_init lệch chiều dài — sidecar {q_rep.shape[0]} vs "
            f"point cloud {expect_n}.\n"
            f"  Nguyên nhân thường gặp: fused.ply đang dùng KHÔNG PHẢI bản "
            f"fused.ply.romav1_p37 sinh cùng lượt với sidecar này.\n"
            f"  Sidecar: {path}\n"
            f"  → Đặt đúng init trước khi train, hoặc chạy lại preprocess."
        )

    # Tổ hợp tuyến tính. Trọng số tự chuẩn hoá để tổng = 1, tránh việc
    # đổi w làm dịch thang đo (và qua đó dịch cả phân phối CRS).
    s = w_cert + w_reproj
    if s <= 0:
        raise ValueError(f"[P37] w_cert + w_reproj phải > 0, đang là {s}")
    q = (w_cert * q_cert + w_reproj * q_rep) / s
    return np.clip(q, 0.0, 1.0).astype(np.float32)


# ============================================================
# [CRSGaussian P37] to_gpu_buffer
# Mục đích: (N,) numpy → (N,1) tensor GPU, đúng shape của _crs_score
#           để mọi phép slice/cat dùng chung một khuôn.
# ============================================================
def to_gpu_buffer(q: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(q)).float().unsqueeze(-1).cuda()


# ============================================================
# [CRSGaussian P37] center_q — CHỐNG CONFOUND, phần quan trọng nhất file
#
# VẤN ĐỀ. Cộng thẳng w_Q·Q_init vào crs_logit làm DỊCH CẢ PHÂN PHỐI CRS.
#   Ở tau = 0.65 cố định, tỉ lệ Gaussian bị đóng băng SH sẽ khác đi.
#   Khi đó ΔPSNR đo được trộn lẫn hai thứ:
#     (1) tín hiệu xếp hạng tốt hơn        ← thứ ta muốn đo
#     (2) số Gaussian bị freeze thay đổi   ← hiệu ứng ngưỡng, vô can
#   Không tách được thì thí nghiệm vô nghĩa.
#
# CÁCH XỬ LÝ. Trừ trung bình trước khi cộng. Thứ hạng giữ nguyên,
#   trung bình logit không đổi → tỉ lệ đóng băng xấp xỉ giữ nguyên.
#
# ⚠ "Xấp xỉ" chứ không phải "đúng bằng" — phân phối CRS không đối xứng
#   nên dịch trung bình về 0 không đảm bảo tỉ lệ vượt ngưỡng y hệt.
#   → train.py PHẢI log tỉ lệ đóng băng ở cả hai nhánh để kiểm chứng,
#     đừng chỉ tin vào lập luận này.
# ============================================================
def center_q(q: torch.Tensor) -> torch.Tensor:
    return q - q.mean()


# ============================================================
# [CRSGaussian P37] qinit_logit_term
# Mục đích: số hạng cộng vào crs_logit. Tách hàm để chỗ gọi trong
#           crs_module.py chỉ còn 1 dòng, và để bật/tắt centering ở 1 nơi.
#
# Trả về 0.0 (scalar) khi tắt → phép cộng thành no-op, không cần rẽ nhánh
# ở chỗ gọi.
# ============================================================
def qinit_logit_term(q_init: Optional[torch.Tensor],
                     w_q: float,
                     center: bool = True):
    if q_init is None or w_q == 0.0:
        return 0.0
    q = center_q(q_init) if center else q_init
    return w_q * q


# ============================================================
# [CRSGaussian P37] inherit_child_q
# Mục đích: giá trị Q_init cho Gaussian con sinh ra từ densification.
#
# "inherit" (mặc định) — con giữ nguyên giá trị của cha.
#   Lý do: Q_init là THUỘC TÍNH XUẤT XỨ, không phải trạng thái phải kiếm.
#   Nó nói "hậu duệ của một điểm triangulate với sai số X". Sai số đó
#   không thay đổi vì Gaussian được nhân bản.
#   KHÁC luật của _crs_score (clip(η·CRS_parent, 0, 0.5)) — CRS là trạng
#   thái động nên con phải tự kiếm, còn Q_init là dữ kiện tĩnh.
#
# "neutral" — con nhận trung bình toàn cục (ablate ở S5).
#
# ⚠ RỦI RO ĐÃ BIẾT (docs/37 §13.7c): nếu đa số Gaussian cuối là con cháu
#   kế thừa cùng giá trị thì phân phối Q_init bị làm phẳng → tín hiệu pha
#   loãng dù q_rep gốc tốt. S2 phải đo std trên Gaussian CUỐI, không chỉ
#   trên điểm init.
# ============================================================
def inherit_child_q(parent_q: Optional[torch.Tensor],
                    n_new: int,
                    mode: str = "inherit",
                    global_mean: float = 0.5) -> Optional[torch.Tensor]:
    if parent_q is None:
        return None
    if mode == "neutral":
        return torch.full((n_new, 1), float(global_mean),
                          device=parent_q.device, dtype=parent_q.dtype)
    # "inherit" — parent_q đã được caller lấy đúng theo selected_pts_mask
    return parent_q


# ============================================================
# [CRSGaussian P37] describe — thống kê gọn để log
# Dùng ở train.py mỗi N iter, và ở S2 để trả lời câu §13.7c
# (tín hiệu có sống qua densification không).
# ============================================================
def describe(q: Optional[torch.Tensor], tag: str = "q_init") -> str:
    if q is None or q.numel() == 0:
        return f"{tag}=<none>"
    f = q.float().flatten()
    return (f"{tag} n={f.numel()} mean={f.mean():.4f} std={f.std():.4f} "
            f"p05={torch.quantile(f, 0.05):.3f} p50={torch.quantile(f, 0.5):.3f} "
            f"p95={torch.quantile(f, 0.95):.3f}")
