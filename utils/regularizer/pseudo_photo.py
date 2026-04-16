# ============================================================
# [CRSGaussian Pseudo-photo] Pseudo-View Photometric Consistency Loss
# File: utils/regularizer/pseudo_photo.py  (TẠO MỚI)
# Mục đích:
#   Ép Gaussian model render đúng IMAGE ở pseudo views (không chỉ training).
#   Detect floater qua PARALLAX: floater 3D ở vị trí sai → khi nhìn từ
#   pseudo cam (dù chỉ 3°), parallax shift ~ depth × tan(3°) đủ để L1 detect.
#
# Khác với pseudo depth (Approach cũ — duplicated training depth signal):
#   - Reference = WARPED GT IMAGE (3 channels), KHÔNG phải DAV2 depth
#   - Loss = L1 per-pixel masked → giữ absolute color, signal mạnh
#   - Signal độc lập với depth loss (không redundant)
#
# Được gọi từ: train.py, mỗi pseudo-view iteration sau pseudo_photo_start_iter
# ============================================================

import torch
from utils.depth.depth_warping import (
    find_nearest_training_cam,
    warp_image_forward,
)
from utils.loss_utils import l1_loss_mask


def compute_pseudo_photo_loss(rendered_img_P, pseudo_cam, train_cameras,
                              aligned_depth_dict, lambda_weight,
                              min_valid_pixels=100):
    """[CRSGaussian Pseudo-photo] Compute pseudo-view photometric consistency loss.

    Args:
        rendered_img_P: (3,H,W) tensor — image render tại pseudo cam (có gradient)
        pseudo_cam: PseudoCamera object
        train_cameras: list of training Camera objects
        aligned_depth_dict: {cam.uid: (H,W) tensor} — aligned DAV2 depth
        lambda_weight: float — loss weight (typically 0.005-0.05)
        min_valid_pixels: int — skip if fewer valid pixels than this

    Returns:
        L: torch.Tensor scalar loss (có gradient) — None nếu skip
        coverage: float — % pixels valid trong warped GT (cho TB log)
    """
    # Tìm training cam gần nhất pseudo cam (angular forward distance)
    nearest_cam = find_nearest_training_cam(pseudo_cam, train_cameras)
    if nearest_cam is None or nearest_cam.uid not in aligned_depth_dict:
        return None, 0.0

    # Forward warp GT image: A → P (no_grad bên trong warp_image_forward)
    warped_gt, valid_gt = warp_image_forward(
        nearest_cam.original_image,
        aligned_depth_dict[nearest_cam.uid],
        nearest_cam,
        pseudo_cam,
    )

    # Cần đủ pixels valid để L1 có ý nghĩa
    n_valid = int(valid_gt.sum().item())
    if n_valid < min_valid_pixels:
        return None, float(valid_gt.float().mean().item())

    # Mask broadcast (H,W) → (3,H,W) cho L1 masked loss
    mask3 = valid_gt.unsqueeze(0).expand(3, -1, -1).float()

    # L1 masked: rendered_img_P còn gradient, warped_gt là external reference
    L = lambda_weight * l1_loss_mask(rendered_img_P, warped_gt, mask3)

    coverage = float(valid_gt.float().mean().item())
    return L, coverage
