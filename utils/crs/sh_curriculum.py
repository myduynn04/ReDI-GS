# ============================================================
# [CRSGaussian Phase 19] SH-degree coarse-to-fine curriculum
# File: utils/crs/sh_curriculum.py  (TẠO MỚI)
# Mục đích: Làm CHẬM lịch mở khóa SH degree (= màu phụ-thuộc-góc-nhìn).
#           Warmup 3DGS gốc đạt full degree 3 @ iter 1500 (15% của 10k iter)
#           → 85% training có toàn quyền SH → gần như KHÔNG curriculum.
#           Giãn lịch ra → giữ appearance THÔ trong cửa sổ densification →
#           chống overfit memorize 3 train view (nặng nhất ở scene cấu
#           trúc mảnh horns/trex).
# Được gọi từ: train.py — mỗi iteration, quyết định có gọi oneupSHdegree().
#
# Default OFF (use_sh_curriculum=False) → step=500 → điều kiện hệt
# train.py gốc (`iteration % 500 == 0`) → BYTE-IDENTICAL baseline.
# ============================================================


def should_step_sh(iteration, use_curriculum, curriculum_interval,
                    default_interval=500):
    """[CRSGaussian Phase 19] Quyết định có tăng SH degree ở iter này không.

    Args:
        iteration: int — iter hiện tại trong training loop.
        use_curriculum: bool — master switch. False = warmup 3DGS gốc.
        curriculum_interval: int — số iter giữa 2 lần +1 degree khi curriculum
            ON (vd 1500 → degree 0→1→2→3 đạt @ iter 1500 / 3000 / 4500).
        default_interval: int = 500 — khoảng warmup 3DGS gốc. KHÔNG đổi —
            giữ để nhánh OFF byte-identical code cũ.

    Returns:
        bool — True nếu nên gọi oneupSHdegree() ở iter này.

    Lưu ý: oneupSHdegree() tự cap ở max_sh_degree → gọi dư sau khi đạt max
    là no-op (an toàn). use_curriculum=False trả `iteration % 500 == 0`
    đúng y train.py:241 cũ → 0 thay đổi baseline khi flag tắt.
    """
    # Curriculum OFF: step=500 → điều kiện hệt code gốc.
    # Curriculum ON: step lớn hơn → degree mở chậm hơn → appearance giữ thô lâu.
    step = curriculum_interval if use_curriculum else default_interval
    return iteration % step == 0
