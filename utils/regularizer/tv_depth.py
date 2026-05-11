# ============================================================
# [CRSGaussian Phase 11 Step 5] TV depth regularizer (edge-preserving).
# File: utils/regularizer/tv_depth.py (NEW)
# Mục đích:
#   - Penalize depth jumps trong vùng image smooth (= floater có thể
#     drift về sai depth khi không có RGB constraint mạnh).
#   - Giữ depth jumps tại object boundary (= edge RGB thật) để tránh
#     blurring depth field — đặc trưng "edge-preserving".
#   - Anti-overfit: smooth depth field → giảm test blur do floater drift.
#
# Cost: ~1-2ms/iter (chỉ tensor diff). Gated bởi start_iter để skip
# giai đoạn geometry chưa ổn định.
# ============================================================

import torch


def compute_tv_depth_loss(rendered_depth, gt_image,
                           mode='edge_preserving', alpha=10.0):
    """Edge-preserving Total Variation loss trên rendered depth.

    Cơ chế:
      - Tính depth gradients |∂d/∂x|, |∂d/∂y| (forward differences).
      - Weight bằng exp(-α·|∂I/∂x|) (và y tương tự): tại smooth region
        (image gradient nhỏ) → weight ≈ 1 → penalty mạnh; tại edge
        (image gradient lớn) → weight ≈ 0 → penalty yếu → cho phép
        depth discontinuity tại object boundary.

    Args:
        rendered_depth: (1, H, W) hoặc (H, W) tensor với grad enabled.
        gt_image:       (3, H, W) tensor target image — dùng đo edge weight.
        mode:           'plain' (TV naive) | 'edge_preserving'.
        alpha:          edge weight strength (default 10.0). Lớn hơn →
                        edge preservation mạnh hơn (nhiều regions cho phép
                        depth jump). Mặc định 10 dựa trên LoopSparseGS/
                        SPIDR depth regularizer literature.

    Returns:
        scalar loss tensor (gradient-enabled).
    """
    # Squeeze về (H, W) nếu input là (1, H, W).
    d = rendered_depth.squeeze(0) if rendered_depth.dim() == 3 else rendered_depth

    # Depth gradients forward-difference.
    dx = torch.abs(d[:, 1:] - d[:, :-1])   # (H, W-1)
    dy = torch.abs(d[1:, :] - d[:-1, :])   # (H-1, W)

    if mode == 'edge_preserving':
        # Image luminance (gray) làm reference cho edge detection.
        img = gt_image.mean(dim=0)         # (H, W)
        ix = torch.abs(img[:, 1:] - img[:, :-1])
        iy = torch.abs(img[1:, :] - img[:-1, :])
        # Weight: lớn ở smooth region, nhỏ ở edge region.
        wx = torch.exp(-alpha * ix)
        wy = torch.exp(-alpha * iy)
        return (dx * wx).mean() + (dy * wy).mean()
    elif mode == 'plain':
        return dx.mean() + dy.mean()
    else:
        raise ValueError(f"[Phase 11 Step 5] tv_depth_mode invalid: {mode}")
