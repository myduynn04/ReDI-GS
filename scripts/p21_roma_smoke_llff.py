#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 21 — Bước 0b] RoMa v2 smoke test on 1 LLFF pair
# File: scripts/p21_roma_smoke_llff.py  (KEEP LOCAL — server-only run)
#
# Pre-requisite (server):
#   conda activate romav2
#   cd ~/workspace/representation-3d/duyen/CoR-GS
#
# Usage:
#   python scripts/p21_roma_smoke_llff.py                    # fern, images_4
#   SCENE=horns IMG_DIR=images_8 python scripts/p21_roma_smoke_llff.py
#
# Mục đích:
#   1. Verify RoMa v2 import + GPU path OK trên LLFF data thực tế
#   2. Measure DENSE per-pixel distributions: overlap (covisibility) + std (pixels)
#   3. Measure SAMPLED N=10000 distributions (cái sẽ feed vào triangulate ở Bước 1)
#   4. Đo VRAM peak + match wall-time
#
# KHÔNG triangulate, KHÔNG ghi ply, KHÔNG đụng CRSGaussian code.
# Pure read-only smoke → decide τ_std threshold cho preprocess (Bước 1+).
#
# Output stats để compare với Phase 18b PDCNet+ cyclic distribution:
#   - PDCNet+: cyclic distribution near-degenerate (Phase 18b 5/5 fix REFUTED)
#   - RoMa v2: kỳ vọng pixel-std bimodal hoặc long-tail → precision filter có signal
# ============================================================
"""Smoke test 1-pair RoMa v2 trên LLFF — measure precision/overlap distributions.

Decide thresholds:
  std_50 (median pixel-std) ≤ 1.5 px → precision filter sẽ work
  std_50 ≥ 5 px              → no discrimination (similar to Phase 18b fail)
  overlap distribution bimodal → covisibility filter có signal
"""

import os
import glob
import time
from pathlib import Path

import torch
import numpy as np

# Env vars
SCENE = os.environ.get("SCENE", "fern")
DATA_ROOT = os.environ.get("DATA_ROOT", "data/nerf_llff_data")
IMG_DIR = os.environ.get("IMG_DIR", "images_4")   # LLFF downsample 4x (~504×378)
N_SAMPLE = int(os.environ.get("N_SAMPLE", "10000"))


def quantile_line(name: str, t: torch.Tensor) -> None:
    """Print quantile summary one line."""
    a = t.detach().flatten().cpu().numpy().astype(np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        print(f"  {name:22s} (empty / all NaN)")
        return
    q = np.quantile(a, [0.05, 0.25, 0.5, 0.75, 0.95])
    print(f"  {name:22s} min={a.min():.3f} max={a.max():.3f} mean={a.mean():.3f} | "
          f"q5={q[0]:.3f} q25={q[1]:.3f} q50={q[2]:.3f} q75={q[3]:.3f} q95={q[4]:.3f}")


def main() -> None:
    # Locate 2 images (any 2 — smoke only, không cần đúng training subset)
    img_dir = Path(DATA_ROOT) / SCENE / IMG_DIR
    if not img_dir.is_dir():
        raise SystemExit(f"[smoke] img dir KHÔNG tồn tại: {img_dir}\n"
                         f"  → check DATA_ROOT + SCENE + IMG_DIR env vars")
    patterns = ("*.JPG", "*.jpg", "*.JPEG", "*.jpeg", "*.PNG", "*.png")
    imgs = sorted(p for pat in patterns for p in glob.glob(str(img_dir / pat)))
    if len(imgs) < 2:
        raise SystemExit(f"[smoke] cần ≥2 ảnh, có {len(imgs)} trong {img_dir}")
    imA, imB = imgs[0], imgs[1]
    print(f"[smoke] scene={SCENE}  pair=({Path(imA).name}, {Path(imB).name})")

    # Import RoMa v2 (env romav2 phải active)
    from romav2 import RoMaV2

    t0 = time.time()
    model = RoMaV2()
    model.apply_setting("precise")
    print(f"[smoke] init: {time.time()-t0:.1f}s | "
          f"H_lr={model.H_lr} W_lr={model.W_lr} H_hr={model.H_hr} W_hr={model.W_hr}")

    # Match
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    preds = model.match(imA, imB)
    torch.cuda.synchronize()
    t_match = time.time() - t0
    print(f"[smoke] match: {t_match:.2f}s")

    # Inspect preds keys + shapes
    print(f"\n[smoke] preds dict:")
    for k, v in preds.items():
        if torch.is_tensor(v):
            print(f"  {k:20s} shape={tuple(v.shape)} dtype={v.dtype} device={v.device}")
        else:
            print(f"  {k:20s} type={type(v).__name__}")

    # DENSE per-pixel distributions
    # precision_AB[0] shape (H, W, 2, 2) → det (H, W) → std = det^(-1/4) pixel
    # overlap_AB[0] shape (H, W, 1) → covisibility prob ∈ [0,1]
    print(f"\n[smoke] DENSE per-pixel distributions:")
    overlap_AB = preds["overlap_AB"][0, ..., 0]
    overlap_BA = preds["overlap_BA"][0, ..., 0]
    prec_AB = preds["precision_AB"][0]
    prec_BA = preds["precision_BA"][0]
    std_AB = torch.linalg.det(prec_AB) ** (-1.0 / 4.0)
    std_BA = torch.linalg.det(prec_BA) ** (-1.0 / 4.0)
    quantile_line("overlap_AB", overlap_AB)
    quantile_line("overlap_BA", overlap_BA)
    quantile_line("std_AB (pixel)", std_AB)
    quantile_line("std_BA (pixel)", std_BA)

    # SAMPLED — cái thực sự ta sẽ feed triangulate (Bước 1+)
    t0 = time.time()
    m, ov, pAB_s, pBA_s = model.sample(preds, N_SAMPLE)
    torch.cuda.synchronize()
    t_sample = time.time() - t0
    print(f"\n[smoke] sample(N={N_SAMPLE}): {t_sample:.3f}s | "
          f"matches={tuple(m.shape)} overlaps={tuple(ov.shape)} "
          f"pAB={tuple(pAB_s.shape)} pBA={tuple(pBA_s.shape)}")

    std_AB_s = torch.linalg.det(pAB_s) ** (-1.0 / 4.0)
    std_BA_s = torch.linalg.det(pBA_s) ** (-1.0 / 4.0)
    print(f"\n[smoke] SAMPLED distributions (input cho triangulate):")
    quantile_line("sampled overlap", ov)
    quantile_line("sampled std_AB (px)", std_AB_s)
    quantile_line("sampled std_BA (px)", std_BA_s)

    # Pixel-std threshold sweep — cho biết bao nhiêu correspondences sống sót
    print(f"\n[smoke] N matches sống sót theo τ_std (px) — std_AB sampled:")
    for tau in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]:
        n_keep = int((std_AB_s <= tau).sum().item())
        pct = 100.0 * n_keep / std_AB_s.numel()
        print(f"  τ ≤ {tau:5.1f} px : {n_keep:6d} / {N_SAMPLE} ({pct:5.1f}%)")

    # VRAM
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    print(f"\n[smoke] VRAM peak: {peak_mb:.0f} MB")
    print(f"[smoke] DONE — match {t_match:.2f}s + sample {t_sample:.3f}s")


if __name__ == "__main__":
    main()
