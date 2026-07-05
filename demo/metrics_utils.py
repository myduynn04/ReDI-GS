"""Wrapper PSNR / SSIM / LPIPS cho demo Gradio.

Reuse code CRSGaussian có sẵn:
- PSNR: `utils.image_utils.psnr`
- SSIM: `utils.loss_utils.ssim`
- LPIPS: `lpipsPyTorch.lpips`

Chấp nhận input dạng
- HxWx3 uint8 numpy (ảnh đã render / GT PIL đã convert),
- 3xHxW float torch tensor trên CUDA (path native của renderer).

Trả về dict `{psnr, ssim, lpips}` mỗi giá trị là float.
"""

from __future__ import annotations

from typing import Dict, Union

import numpy as np
import torch

from lpipsPyTorch import lpips as _lpips
from utils.image_utils import psnr as _psnr
from utils.loss_utils import ssim as _ssim


Array = Union[np.ndarray, torch.Tensor]


# ---------------------------------------------------------------------------
# Adapter — chuẩn hoá input về 3xHxW float [0,1] CUDA
# ---------------------------------------------------------------------------


def _to_chw_float(x: Array) -> torch.Tensor:
    """Chuẩn hoá về 3xHxW float [0,1] CUDA tensor."""
    if isinstance(x, np.ndarray):
        if x.dtype != np.uint8 and x.max() > 1.5:
            # Có vẻ float [0,255] range, scale xuống.
            x = x.astype(np.float32) / 255.0
        elif x.dtype == np.uint8:
            x = x.astype(np.float32) / 255.0
        if x.ndim == 3 and x.shape[-1] == 3:
            # HxWx3 -> 3xHxW
            x = np.transpose(x, (2, 0, 1))
        t = torch.from_numpy(x).float().cuda()
    elif isinstance(x, torch.Tensor):
        t = x.detach().float()
        if t.max() > 1.5:
            t = t / 255.0
        if t.ndim == 3 and t.shape[0] != 3 and t.shape[-1] == 3:
            t = t.permute(2, 0, 1)
        t = t.cuda()
    else:
        raise TypeError(f"Unsupported input type: {type(x)}")
    return t.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_psnr(pred: Array, gt: Array) -> float:
    """PSNR (dB, float) giữa pred và gt."""
    p = _to_chw_float(pred).unsqueeze(0)
    g = _to_chw_float(gt).unsqueeze(0)
    with torch.no_grad():
        val = _psnr(p, g)
    if val.ndim > 0:
        val = val.mean()
    return float(val.item())


def compute_ssim(pred: Array, gt: Array) -> float:
    """SSIM (float trong [-1, 1]) giữa pred và gt."""
    p = _to_chw_float(pred).unsqueeze(0)
    g = _to_chw_float(gt).unsqueeze(0)
    with torch.no_grad():
        val = _ssim(p, g)
    return float(val.item())


def compute_lpips(pred: Array, gt: Array, net_type: str = "vgg") -> float:
    """LPIPS (float, càng thấp càng tốt) giữa pred và gt."""
    p = _to_chw_float(pred).unsqueeze(0)
    g = _to_chw_float(gt).unsqueeze(0)
    with torch.no_grad():
        val = _lpips(p, g, net_type=net_type)
    return float(val.item())


def compute_all_metrics(pred: Array, gt: Array,
                        lpips_net: str = "vgg") -> Dict[str, float]:
    """Trả về dict tất cả 3 metric cùng lúc, tiết kiệm 1 lần chuẩn hoá."""
    p = _to_chw_float(pred).unsqueeze(0)
    g = _to_chw_float(gt).unsqueeze(0)
    with torch.no_grad():
        psnr_val = _psnr(p, g)
        if psnr_val.ndim > 0:
            psnr_val = psnr_val.mean()
        ssim_val = _ssim(p, g)
        lpips_val = _lpips(p, g, net_type=lpips_net)
    return {
        "psnr": float(psnr_val.item()),
        "ssim": float(ssim_val.item()),
        "lpips": float(lpips_val.item()),
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("[demo] metrics_utils smoke test — 2 ảnh dummy giống nhau")
    a = np.random.randint(0, 255, size=(378, 504, 3), dtype=np.uint8)
    b = a.copy()
    m = compute_all_metrics(a, b)
    print(f"  identical: {m}")

    b_noisy = np.clip(a.astype(np.int32) + np.random.randint(
        -20, 20, size=a.shape), 0, 255).astype(np.uint8)
    m2 = compute_all_metrics(a, b_noisy)
    print(f"  noisy:     {m2}")
