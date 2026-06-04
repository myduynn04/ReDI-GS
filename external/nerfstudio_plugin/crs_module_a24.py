# ============================================================
# [CRSGaussian Plug-in A2.4] CRS module adapted cho splatfacto
# File: crsgaussian_plugin/crs_module_a24.py
#
# Port D_i + R_i + EMA update từ CRSGaussian utils/crs/crs_module.py,
# adapt cho splatfacto:
#   - means thay xyz
#   - gauss_params["features_rest"] thay _features_rest
#   - CRS score stored as buffer (KHÔNG Parameter, không qua optimizer)
#
# A2.4-minimal scope:
#   - CRS score computed mỗi N iter sau warmup
#   - Log diagnostic (median, distribution)
#   - SKIP CRS-modulated SH freeze (cần BEFORE_OPTIMIZER_STEP callback,
#     densify replace features_rest tensor → hook không persist).
#     Workaround: scale features_rest data per-step (post optim) bằng
#     CRS-weighted mask. Hiệu ứng nhẹ hơn freeze nhưng work without hook.
# ============================================================
"""[CRSGaussian Plug-in A2.4] CRS module adapted cho splatfacto."""

from typing import Dict, List, Optional

import torch


@torch.no_grad()
def compute_depth_consistency(
    means: torch.Tensor,
    cameras: List,
    aligned_depth_dict: Dict[str, torch.Tensor],
    idx_to_stem: Dict[int, str],
    depth_range: float,
) -> torch.Tensor:
    """[A2.4] Port D_i computation cho splatfacto.

    Args:
        means: (N, 3) gauss_params["means"]
        cameras: list of Cameras object từ splatfacto datamanager
        aligned_depth_dict: {stem: (H, W) tensor} từ A2.3 preprocess
        idx_to_stem: {camera_idx: stem} mapping
        depth_range: float
    Returns:
        D: (N, 1) tensor [0,1], 0.5 nếu Gaussian không visible
    """
    N = means.shape[0]
    device = means.device

    D_sum = torch.zeros(N, 1, device=device)
    count = torch.zeros(N, 1, device=device)

    ones = torch.ones(N, 1, device=device, dtype=means.dtype)
    xyz_hom = torch.cat([means.detach(), ones], dim=1)  # (N, 4)

    depth_range_safe = max(float(depth_range), 1e-6)

    for cam_idx in range(len(cameras)):
        stem = idx_to_stem.get(cam_idx)
        if stem is None or stem not in aligned_depth_dict:
            continue

        depth_prior_map = aligned_depth_dict[stem]  # (H, W) GPU

        # Camera intrinsics + extrinsics từ Cameras object
        # Nerfstudio Cameras format: camera_to_worlds (3,4)
        cam = cameras[cam_idx:cam_idx + 1]   # single camera
        c2w = cam.camera_to_worlds[0]        # (3, 4)
        # World→Camera: inverse of c2w
        R = c2w[:3, :3]                       # (3, 3)
        t = c2w[:3, 3]                        # (3,)
        # W2C: R_inv = R.T (rotation), t_inv = -R.T @ t
        R_w2c = R.T
        t_w2c = -R.T @ t

        # World → Camera coords
        pts_cam = (R_w2c @ means.detach().T).T + t_w2c  # (N, 3)
        depth = pts_cam[:, 2]  # depth in cam

        # Intrinsics
        fx = float(cam.fx[0])
        fy = float(cam.fy[0])
        cx = float(cam.cx[0])
        cy = float(cam.cy[0])
        H_full = int(cam.height[0])
        W_full = int(cam.width[0])

        # Match resolution của depth_prior (downscale)
        H = depth_prior_map.shape[0]
        W = depth_prior_map.shape[1]
        scale_w = W / float(W_full)
        scale_h = H / float(H_full)
        fx_s = fx * scale_w
        fy_s = fy * scale_h
        cx_s = cx * scale_w
        cy_s = cy * scale_h

        # Convention nerfstudio camera: -z forward
        # Flip depth sign nếu cần
        depth = -depth if depth.mean() < 0 else depth
        pixel_x = pts_cam[:, 0] / (-pts_cam[:, 2] + 1e-8) * fx_s + cx_s
        pixel_y = pts_cam[:, 1] / (-pts_cam[:, 2] + 1e-8) * fy_s + cy_s

        valid = (
            (depth > 0)
            & (pixel_x >= 0) & (pixel_x < W)
            & (pixel_y >= 0) & (pixel_y < H)
        )
        if valid.sum() == 0:
            continue

        px = pixel_x[valid].long().clamp(0, W - 1)
        py = pixel_y[valid].long().clamp(0, H - 1)
        d_prior = depth_prior_map[py, px]
        d_proj = depth[valid]

        D_i = 1.0 - torch.abs(d_proj - d_prior) / depth_range_safe
        D_i = D_i.clamp(0.0, 1.0)

        D_sum[valid, 0] += D_i
        count[valid, 0] += 1.0

    visible_any = (count > 0).squeeze(1)
    D = torch.full((N, 1), 0.5, device=device)
    D[visible_any] = D_sum[visible_any] / count[visible_any]
    return D


@torch.no_grad()
def update_crs_ema(
    crs_score: torch.Tensor,
    D: torch.Tensor,
    R: Optional[torch.Tensor],
    w1: float = 0.5,
    w2: float = 0.5,
    scale: float = 5.0,
    ema: float = 0.9,
) -> torch.Tensor:
    """[A2.4] Update _crs_score in logit space via EMA.

    Formula:
        score = w1 * D + w2 * R (or w1 * D nếu R=None)
        crs_logit_new = scale * (score - 0.5)
        crs_score = ema * crs_score + (1 - ema) * crs_logit_new

    Returns: updated crs_score (logit) of shape (N, 1).
    """
    if R is None:
        score = D
    else:
        score = w1 * D + w2 * R
    crs_logit_new = scale * (score - 0.5)
    return ema * crs_score + (1.0 - ema) * crs_logit_new
