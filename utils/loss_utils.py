#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch.autograd import Variable
from math import exp
import torch.nn.functional as F


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l1_loss_mask(network_output, gt, mask = None):
    if mask is None:
        return l1_loss(network_output, gt)
    else:
        return torch.abs((network_output - gt) * mask).sum() / mask.sum()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, mask=None, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if mask is not None:
        img1 = img1 * mask + (1 - mask)
        img2 = img2 * mask + (1 - mask)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def loss_photometric(image, gt_image, opt, valid=None):
    Ll1 =  l1_loss_mask(image, gt_image, mask=valid)
    loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image, mask=valid)))
    return loss


# ============================================================
# [CRSGaussian] Task: T3.1 — pearson_depth_loss
# File: CRSGaussian/utils/loss_utils.py  (THÊM VÀO)
# Mục đích: Pearson correlation loss giữa rendered depth và
#           aligned depth prior. Dùng correlation thay L1/L2
#           vì robust với scale/shift ambiguity còn sót.
# Được gọi từ: train.py, trong loss computation block
# ============================================================


def pearson_depth_loss(rendered_depth, depth_prior, mask=None):
    """Pearson correlation depth loss.

    Loss = 1 - pearson_corr(rendered, prior).
    Pearson correlation đo tương quan tuyến tính, invariant với
    scale và shift → robust khi depth alignment chưa hoàn hảo.
    Loss = 0 khi hai depth maps hoàn toàn tương quan (tốt nhất).
    Loss = 2 khi hoàn toàn ngược chiều (tệ nhất).

    Args:
        rendered_depth: (1, H, W) tensor GPU — từ rasterizer.
        depth_prior: (H, W) tensor — aligned depth prior.
            Có thể trên CPU (sẽ chuyển GPU) hoặc GPU.
        mask: (1, H, W) hoặc (H, W) tensor boolean, optional.
            True = pixel hợp lệ. None = dùng tất cả pixels.

    Returns:
        loss: scalar tensor — 1 - pearson_corr, range [0, 2].
    """
    # Đảm bảo cùng device
    if depth_prior.device != rendered_depth.device:
        depth_prior = depth_prior.to(rendered_depth.device)

    # Flatten về 1D — pearson_corr cần vectors
    rd = rendered_depth.squeeze()  # (H, W)
    dp = depth_prior.squeeze()     # (H, W)

    if mask is not None:
        m = mask.squeeze().bool()  # (H, W)
        rd = rd[m]
        dp = dp[m]
    else:
        rd = rd.reshape(-1)
        dp = dp.reshape(-1)

    # Bỏ pixels depth_prior = 0 (vùng không có thông tin)
    valid = dp > 0
    if valid.sum() < 10:
        # Quá ít pixels hợp lệ → trả loss 0 (không penalize)
        return torch.tensor(0.0, device=rendered_depth.device)
    rd = rd[valid]
    dp = dp[valid]

    # Pearson correlation thủ công — tránh dependency torchmetrics
    # và kiểm soát edge cases tốt hơn.
    # corr = cov(x,y) / (std(x) * std(y))
    rd_mean = rd.mean()
    dp_mean = dp.mean()
    rd_centered = rd - rd_mean
    dp_centered = dp - dp_mean

    cov = (rd_centered * dp_centered).mean()
    rd_std = rd_centered.pow(2).mean().sqrt()
    dp_std = dp_centered.pow(2).mean().sqrt()

    # Clamp std tối thiểu tránh chia 0 (depth map phẳng)
    eps = 1e-6
    corr = cov / (rd_std * dp_std + eps)

    # Loss = 1 - corr: corr=1 → loss=0 (hoàn hảo),
    #                   corr=-1 → loss=2 (ngược chiều)
    return 1.0 - corr




