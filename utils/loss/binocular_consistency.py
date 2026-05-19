# ============================================================
# [CRSGaussian Phase 14] Binocular Stereo Consistency loss.
# File: utils/loss/binocular_consistency.py  (TẠO MỚI)
# Mục đích: Port L_consist từ Binocular3DGS (NeurIPS 2024) — self-
#   supervised single-view stereo consistency. Mỗi iter (sau
#   start_iter): dịch ngang camera baseline B → render shifted →
#   inverse-warp về cam gốc qua disparity (=f·B/depth) → phạt
#   L1(warped, GT) + Godard edge-smooth. Differentiable: gradient
#   chảy vào CẢ geometry (qua depth→disparity) LẪN appearance/SH
#   (qua shifted render). Self-supervised single-view → thoát
#   cross-view-structurally-dead; 0 Gaussian thêm → thoát capacity-
#   ceiling. Orthogonal vs d_cycle (no_grad-score ≠ differentiable-loss).
#
# Được gọi từ: train.py (gated `opt.use_lconsist`, sau
#   `opt.lconsist_start_iter`). Default OFF → A3 byte-identical.
#
# Verified-from-code (KHÔNG đoán — kỷ luật GDAGS):
#   - shifted-cam: PseudoCamera (scene/cameras.py:66-87) + W2C=wvt.T
#     (utils/crs/d_cycle.py:31). CRS Camera KHÔNG có get_camera_matrix/
#     get_focal (Bino-specific) → dựng bằng wvt (#1 fix).
#   - depth ACCUMULATED ΣαT· → /(alpha+1e-6) differentiable trước
#     disparity (bottleneck:203-208, crs_module:697-708 — #2 silent-bug).
#   - trans_dist = FRACTION × cameras_extent (#3: Bino 0.4 là world-
#     scale RIÊNG → KHÔNG copy; scale theo extent).
#   - fov2focal (graphics_utils.py:100); render(...,disable_dropout=True)
#     tồn tại (train.py:436) — render CLEAN (DropAnSH nhiễu signal stereo).
#
# Faithful Binocular semantics (train.py:123-136, loss_utils.py:68-91):
#   random magnitude ∈ (0, B] × random sign (multi-baseline stochastic);
#   inverse_warp_images column-bilinear; SmoothLoss edge-aware Godard.
# ============================================================
"""[CRSGaussian Phase 14] Binocular L_consist (differentiable loss)."""

import random

import numpy as np
import torch
import torch.nn.functional as F

# Godard edge kernels (verified Binocular loss_utils.py:80-84)
_K_X = torch.tensor([[0., 0., 0.], [-0.5, 0., 0.5], [0., 0., 0.]])
_K_Y = torch.tensor([[0., -0.5, 0.], [0., 0., 0.], [0., 0.5, 0.]])
_KERN = {}  # device-cached conv weights


def _kernels(device):
    """Lazy per-device cache: (wx_im 1x3x3x3, wy_im, wx_d 1x1x3x3, wy_d)."""
    if device not in _KERN:
        kx = _K_X.to(device)
        ky = _K_Y.to(device)
        _KERN[device] = (
            kx.view(1, 1, 3, 3).repeat(1, 3, 1, 1),   # image x (3→1)
            ky.view(1, 1, 3, 3).repeat(1, 3, 1, 1),   # image y
            kx.view(1, 1, 3, 3),                       # disparity x (1→1)
            ky.view(1, 1, 3, 3),                       # disparity y
        )
    return _KERN[device]


def _smooth_loss(disparity, image):
    """Edge-aware disparity smoothness — EXACT Binocular SmoothLoss
    (loss_utils.py:86-91): phạt ∇disparity TRỪ tại edge ảnh.
    Args: disparity (1,1,H,W), image (1,3,H,W). Returns scalar."""
    wx_im, wy_im, wx_d, wy_d = _kernels(image.device)
    edge_x_im = torch.exp(F.conv2d(image, wx_im).abs() * -0.33)
    edge_y_im = torch.exp(F.conv2d(image, wy_im).abs() * -0.33)
    edge_x_d = F.conv2d(disparity, wx_d)
    edge_y_d = F.conv2d(disparity, wy_d)
    return (edge_x_im * edge_x_d).abs().mean() + \
           (edge_y_im * edge_y_d).abs().mean()


def _make_shifted_cam(cam, B):
    """PseudoCamera dịch camera-center +B trục X-local (R giữ → rectified
    stereo). #1 fix: dùng W2C=wvt.T thay get_camera_matrix (CRS thiếu).
    Camera là hằng hình học (KHÔNG learnable) → trans detach OK; gradient
    chảy qua gaussians trong render, KHÔNG qua camera."""
    from scene.cameras import PseudoCamera
    W2C = cam.world_view_transform.T                       # (4,4) actual W2C
    pt_cam = torch.tensor([float(B), 0.0, 0.0, 1.0],
                          device=W2C.device, dtype=W2C.dtype)
    pt_world = torch.inverse(W2C) @ pt_cam                  # C2W @ pt_cam
    trans = (pt_world[:3] - cam.camera_center).detach().cpu().numpy()
    return PseudoCamera(R=cam.R, T=cam.T, FoVx=cam.FoVx, FoVy=cam.FoVy,
                        width=cam.image_width, height=cam.image_height,
                        trans=trans, scale=float(getattr(cam, "scale", 1.0)))


def _inverse_warp_cols(img, disparity):
    """warped[:,r,c] = bilinear(img[:,r,c+disparity[r,c]]) — column-only
    inverse warp (rectified stereo), faithful Binocular inverse_warp_images
    (graphics_utils.py:80-118) qua grid_sample (differentiable cả img LẪN
    disparity → grad vào appearance + geometry).
    Args: img (3,H,W), disparity (H,W). Returns warped (3,H,W), valid (H,W)."""
    C, H, W = img.shape
    ys, xs = torch.meshgrid(
        torch.arange(H, device=img.device, dtype=img.dtype),
        torch.arange(W, device=img.device, dtype=img.dtype),
        indexing="ij")
    xsrc = xs + disparity
    valid = (xsrc >= 0) & (xsrc <= (W - 1))
    gx = 2.0 * xsrc / max(W - 1, 1) - 1.0
    gy = 2.0 * ys / max(H - 1, 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)       # (1,H,W,2)
    out = F.grid_sample(img.unsqueeze(0), grid, mode="bilinear",
                        padding_mode="zeros", align_corners=True).squeeze(0)
    return out, valid


def _render_clean(render_func, cam, gaussians, pipe, bg):
    """Render disable_dropout=True (signal stereo sạch — DropAnSH ngẫu
    nhiên sẽ nhiễu disparity/warp). Fallback nếu signature khác."""
    try:
        return render_func(cam, gaussians, pipe, bg, disable_dropout=True)
    except TypeError:
        return render_func(cam, gaussians, pipe, bg)


def compute_lconsist_loss(viewpoint_cam, gaussians, render_func, pipe, bg,
                          gt_image, trans_dist, lam, smooth_w,
                          cameras_extent):
    """Binocular stereo consistency loss (differentiable).

    Args:
        viewpoint_cam: train camera (đang optimize).
        gaussians: GaussianModel (gs0).
        render_func, pipe, bg: render context (bg = bg CÙNG iter của
            train loop — random_background → phải nhất quán 2 render).
        gt_image: (3,H,W) GT của viewpoint_cam.
        trans_dist: FRACTION — baseline B = trans_dist × cameras_extent.
        lam: trọng số L1-warp. smooth_w: trọng số Godard-smooth.
        cameras_extent: scene.cameras_extent (#3 scale).
    Returns:
        (L tensor scalar, stats dict)  hoặc  (None, stats) nếu skip.
    """
    from utils.graphics_utils import fov2focal

    H = viewpoint_cam.image_height
    W = viewpoint_cam.image_width
    # Faithful Binocular: random magnitude ∈ (0,Bmax] × random sign
    # (multi-baseline stochastic regularization, train.py:125-126).
    Bmax = float(trans_dist) * float(cameras_extent)
    B = float(torch.rand(1).item()) * Bmax * random.choice([-1.0, 1.0])
    if abs(B) < 1e-9:
        return None, {"skip": "B≈0"}

    # Render gốc CLEAN → depth/alpha (gradient-carrying). #2: depth
    # accumulated → /(alpha+1e-6) trước disparity (hard per-pixel).
    pkg_o = _render_clean(render_func, viewpoint_cam, gaussians, pipe, bg)
    depth = pkg_o["depth"].squeeze(0)                       # (H,W)
    alpha = pkg_o["alpha"].squeeze(0)                       # (H,W)
    depth_n = depth / (alpha + 1e-6)

    # Shifted view CLEAN
    shifted_cam = _make_shifted_cam(viewpoint_cam, B)
    img_s = _render_clean(render_func, shifted_cam, gaussians, pipe, bg)
    img_s = img_s["render"]                                 # (3,H,W)

    fx = fov2focal(viewpoint_cam.FoVx, W)
    disparity = fx * (-B) / (depth_n + 1e-5)                # (H,W) — Bino sign
    warped, valid = _inverse_warp_cols(img_s, disparity)    # (3,H,W),(H,W)

    n_valid = int(valid.sum())
    if n_valid < 100:
        return None, {"skip": "valid<100", "B": B}

    m3 = valid.unsqueeze(0).expand(3, -1, -1)
    l1 = (warped[m3] - gt_image[:3][m3]).abs().mean()
    L = lam * l1
    if smooth_w > 0:
        sm = _smooth_loss(
            (disparity * valid).unsqueeze(0).unsqueeze(0),
            gt_image[:3].unsqueeze(0))
        L = L + smooth_w * sm
    stats = {"B": B, "l1": float(l1.detach()),
             "cov": n_valid / float(H * W)}
    return L, stats
