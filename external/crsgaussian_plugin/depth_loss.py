# ============================================================
# [CRSGaussian Path A B2] DAV2 depth loss helpers
# File: crsgaussian_plugin/depth_loss.py  (KEEP LOCAL — upload server)
#
# Port verbatim từ Phase A `nerfstudio_plugin/depth_loss.py` (đã verify A2.3 PASS).
#
# Mechanism (Phase 22 contrib #4):
#   L_depth = λ × (1 - pearson_corr(rendered_depth, aligned_depth))
#   λ = 0.05 (Phase 3 default)
#   Pearson scale/shift invariant → robust khi DAV2 alignment chưa hoàn hảo.
#
# Aligned depth tạo bằng `nerfstudio_plugin/preprocess_depth_a23.py`:
#   conda activate corgs   # vì cần DAV2 + align_depth_to_colmap
#   SCENE=fern python preprocess_depth_a23.py
#   → output: <data>/<scene>/3_views/aligned_depth_a23/<image_stem>.npy
# ============================================================
"""[Path A B2] DAV2 depth loss helpers — verbatim từ Phase A."""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch


def load_aligned_depth_dict(depth_dir: Path) -> Tuple[Dict[str, torch.Tensor], float]:
    """Load tất cả aligned depth .npy → (dict[image_stem] → tensor cuda, depth_range).

    [B3 split-fix 2026-06-04] depth_dir là FULL PATH tới folder (không phải data_root).
    Caller phải compute full path: data_root/aligned_depth_relpath.

    Returns:
        (depth_dict, depth_range) — depth_dict={} + depth_range=1.0 nếu folder không tồn tại.
    """
    depth_dir = Path(depth_dir)
    if not depth_dir.is_dir():
        print(f"[Path A B2] aligned_depth folder không tồn tại: {depth_dir}")
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
            print(f"[Path A B2] Skip {npy_path.name}: {e}")

    meta_path = depth_dir / "_meta.json"
    depth_range = 1.0
    if meta_path.is_file():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            depth_range = float(meta.get("depth_range", 1.0))
        except Exception:
            pass

    print(f"[Path A B2] Loaded {len(depth_dict)} aligned depth maps, depth_range={depth_range:.3f}")
    return depth_dict, depth_range


def pearson_depth_loss(
    rendered_depth: torch.Tensor,
    depth_prior: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """[Path A B2] Pearson correlation depth loss.

    Verbatim port từ CRSGaussian utils/loss_utils.py:93.

    Args:
        rendered_depth: (1, H, W) hoặc (H, W) — từ CoR-GS render output.
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

    # Resize depth_prior nếu shape khác
    if rd.shape != dp.shape:
        dp_4d = dp.unsqueeze(0).unsqueeze(0)
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


__all__ = ["load_aligned_depth_dict", "pearson_depth_loss"]
