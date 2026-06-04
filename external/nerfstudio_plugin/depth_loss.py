# ============================================================
# [CRSGaussian Plug-in A2.3] DAV2 depth loss cho splatfacto
# File: crsgaussian_plugin/depth_loss.py
#
# Port pearson_depth_loss từ CRSGaussian utils/loss_utils.py + load
# pre-computed aligned depth .npy (chạy preprocess_depth_a23.py riêng).
#
# Mechanism (Phase 22):
#   L_depth = λ × (1 - pearson_corr(rendered_depth, aligned_depth))
#   Pearson scale/shift invariant → robust khi alignment chưa hoàn hảo.
# ============================================================
"""[CRSGaussian Plug-in A2.3] DAV2 depth loss helpers."""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch


def load_aligned_depth_dict(data_root: Path):
    """Load tất cả aligned depth .npy → (dict[image_stem] → tensor, depth_range).

    Path convention: <data_root>/aligned_depth_a23/<stem>.npy + _meta.json

    Returns:
        (depth_dict, depth_range) — depth_dict={} + depth_range=1.0 nếu folder không tồn tại.
    """
    depth_dir = data_root / "aligned_depth_a23"
    if not depth_dir.is_dir():
        return {}, 1.0

    depth_dict = {}
    for npy_path in depth_dir.glob("*.npy"):
        stem = npy_path.stem
        if stem.startswith("_"):
            continue
        try:
            depth = np.load(str(npy_path)).astype(np.float32)
            depth_dict[stem] = torch.from_numpy(depth).cuda()
        except Exception as e:
            print(f"[A2.3] Skip {npy_path.name}: {e}")

    # Load depth_range từ _meta.json
    import json as _json
    meta_path = depth_dir / "_meta.json"
    depth_range = 1.0
    if meta_path.is_file():
        try:
            with open(meta_path) as f:
                meta = _json.load(f)
            depth_range = float(meta.get("depth_range", 1.0))
        except Exception:
            pass
    return depth_dict, depth_range


def pearson_depth_loss(
    rendered_depth: torch.Tensor,
    depth_prior: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """[CRSGaussian Plug-in A2.3] Pearson correlation depth loss.

    Verbatim port từ utils/loss_utils.py:93 của CRSGaussian.

    Args:
        rendered_depth: (1, H, W) hoặc (H, W) — từ splatfacto rasterizer output.
        depth_prior: (H, W) — aligned DAV2 depth (cùng cam).
        mask: optional (H, W) bool — True = pixel hợp lệ.

    Returns:
        loss: scalar tensor — 1 - pearson_corr, range [0, 2].
              0 = tốt nhất (correlated), 2 = ngược chiều.
    """
    if depth_prior.device != rendered_depth.device:
        depth_prior = depth_prior.to(rendered_depth.device)

    rd = rendered_depth.squeeze()
    dp = depth_prior.squeeze()

    # Resize depth_prior nếu shape khác (splatfacto downscale ảnh)
    if rd.shape != dp.shape:
        # Bilinear resize depth_prior để match rendered_depth
        dp_4d = dp.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        dp_resized = torch.nn.functional.interpolate(
            dp_4d, size=rd.shape, mode="bilinear", align_corners=False
        )
        dp = dp_resized.squeeze()

    if mask is not None:
        m = mask.squeeze().bool()
        rd = rd[m]
        dp = dp[m]
    else:
        rd = rd.reshape(-1)
        dp = dp.reshape(-1)

    # Bỏ pixels depth_prior = 0 (vùng không có info)
    valid = dp > 0
    if valid.sum() < 10:
        return torch.tensor(0.0, device=rendered_depth.device)
    rd = rd[valid]
    dp = dp[valid]

    # Pearson correlation
    rd_centered = rd - rd.mean()
    dp_centered = dp - dp.mean()
    num = (rd_centered * dp_centered).sum()
    den = torch.sqrt((rd_centered ** 2).sum() * (dp_centered ** 2).sum() + 1e-8)
    corr = num / den

    return 1.0 - corr
