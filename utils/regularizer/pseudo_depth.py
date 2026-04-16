# ============================================================
# [CRSGaussian Pseudo-depth] Pseudo-View Depth Consistency Loss
# File: utils/regularizer/pseudo_depth.py  (TẠO MỚI — refactored from train.py)
# Mục đích:
#   Forward warp aligned DAV2 depth từ training cam → pseudo cam,
#   so với rendered depth tại pseudo cam (Pearson loss).
#
# Status: KHÔNG hoạt động trong thực nghiệm (xem decisions_log 2026-04).
#   Root cause: pseudo cams chỉ cách training cams 0.3-3.68° (LLFF
#   forward-facing + bbox sampling). Reference duplicate training depth.
#   → Switch sang pseudo_photo.py (Approach 2). File này giữ lại để
#   re-enable nếu cần test variant trong tương lai.
#
# Được gọi từ: train.py khi --use_pseudo_depth_loss (default OFF)
# ============================================================

import torch
from utils.depth.depth_warping import (
    find_nearest_training_cam,
    forward_warp_depth,
    compute_warp_coverage,
)
from utils.loss_utils import pearson_depth_loss


def compute_pseudo_depth_loss(rendered_depth_P, pseudo_cam, train_cameras,
                              aligned_depth_dict, lambda_weight, ramp,
                              min_valid_pixels=100):
    """[CRSGaussian Pseudo-depth] Pearson loss giữa rendered depth và warped DAV2 depth.

    Args:
        rendered_depth_P: (1,H,W) hoặc (H,W) tensor — depth render tại pseudo (có grad)
        pseudo_cam: PseudoCamera object
        train_cameras: list of training Camera objects
        aligned_depth_dict: {cam.uid: (H,W) tensor}
        lambda_weight: float — loss weight
        ramp: float [0,1] — linear ramp tránh shock
        min_valid_pixels: int — skip if fewer valid pixels than this

    Returns:
        L: torch.Tensor scalar loss (có gradient) — None nếu skip
        coverage: float — % pixels valid sau warp
    """
    nearest_cam = find_nearest_training_cam(pseudo_cam, train_cameras)
    if nearest_cam is None or nearest_cam.uid not in aligned_depth_dict:
        return None, 0.0

    # Forward warp depth (no_grad bên trong)
    depth_ref_P, valid_P = forward_warp_depth(
        aligned_depth_dict[nearest_cam.uid],
        nearest_cam,
        pseudo_cam,
    )

    coverage = compute_warp_coverage(valid_P)

    n_valid = int(valid_P.sum().item())
    if n_valid < min_valid_pixels:
        return None, coverage

    # Squeeze depth shape (1,H,W) → (H,W) nếu cần
    if rendered_depth_P.dim() == 3:
        rendered_depth_P = rendered_depth_P.squeeze(0)

    L = ramp * lambda_weight * pearson_depth_loss(
        rendered_depth_P[valid_P].unsqueeze(0),
        depth_ref_P[valid_P],
    )
    return L, coverage
