#!/usr/bin/env python
"""
[Verify script 2026-07-02 REWRITE v2] Compute PSNR + SSIM + LPIPS từ checkpoint plug-in crsgaussian.

Cần thiết vì ns-eval mặc định chỉ compute PSNR (CrsGaussianModel.get_image_metrics_and_images
chỉ return {"psnr": ...}). Script này iterate qua `fixed_indices_eval_dataloader` (đúng flow của
ns-eval) và compute 3 metric qua torchmetrics.

v2 FIX (2026-07-02): v1 iterate qua `dm.eval_dataset[i]["image"]` trả về DISTORTED image,
trong khi camera bị in-place modified thành UNDISTORTED intrinsics → mismatch → PSNR sụp 3.58 dB.
v2 dùng `pipeline.datamanager.fixed_indices_eval_dataloader` — đúng flow ns-eval, trả về cached
undistorted image + matching camera.

Cách dùng:
    cd /home/aidev/workspace/representation-3d/duyen/nerfstudio
    conda activate nerfstudio

    python crsgaussian_plugin/scripts/eval_full_metrics.py \\
        --load-config outputs/.../config.yml \\
        --output-path outputs/.../eval_full_metrics.json

Output JSON structure:
    {
      "psnr": 25.71,
      "ssim": 0.82,
      "lpips": 0.15,
      "n_test": 3,
      "per_image": [{"psnr": ..., "ssim": ..., "lpips": ...}, ...]
    }
"""
import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    args = parser.parse_args()

    from nerfstudio.utils.eval_utils import eval_setup
    from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    print(f"[Load] {args.load_config}")
    config, pipeline, _, _ = eval_setup(args.load_config)
    pipeline.eval()

    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")

    # ── [v2 FIX] Dùng fixed_indices_eval_dataloader — đúng flow ns-eval ──
    # Trả về list of (camera, batch) pairs với batch["image"] = cached undistorted image
    # matching camera intrinsics (đã undistort in-place trong _load_images).
    dataloader = pipeline.datamanager.fixed_indices_eval_dataloader
    n_test = len(dataloader)
    print(f"[Eval] {n_test} test cameras (via fixed_indices_eval_dataloader)")

    per_image = []
    with torch.no_grad():
        for i, (camera, batch) in enumerate(dataloader):
            # ── Render ──
            outputs = pipeline.model.get_outputs_for_camera(camera=camera)

            # ── GT từ batch (undistorted + on device) ──
            gt = batch["image"].to("cuda")
            if gt.dtype == torch.uint8:
                gt = gt.float() / 255.0

            rgb = outputs["rgb"].clamp(0, 1)
            gt = gt.clamp(0, 1)

            # Metrics cần (N, C, H, W)
            rgb_bchw = rgb.permute(2, 0, 1).unsqueeze(0).contiguous()
            gt_bchw = gt.permute(2, 0, 1).unsqueeze(0).contiguous()

            psnr = peak_signal_noise_ratio(rgb_bchw, gt_bchw, data_range=1.0).item()
            ssim = structural_similarity_index_measure(rgb_bchw, gt_bchw, data_range=1.0).item()
            lpips = lpips_metric(rgb_bchw, gt_bchw).item()

            per_image.append({"psnr": psnr, "ssim": ssim, "lpips": lpips})
            print(f"  cam {i} — PSNR {psnr:.3f}  SSIM {ssim:.4f}  LPIPS {lpips:.4f}")

    mean_psnr = sum(p["psnr"] for p in per_image) / n_test
    mean_ssim = sum(p["ssim"] for p in per_image) / n_test
    mean_lpips = sum(p["lpips"] for p in per_image) / n_test

    result = {
        "psnr": mean_psnr,
        "ssim": mean_ssim,
        "lpips": mean_lpips,
        "n_test": n_test,
        "per_image": per_image,
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(result, f, indent=2)

    print()
    print("═════════════════════════════════════════════")
    print(f"  PSNR  = {mean_psnr:.3f} dB")
    print(f"  SSIM  = {mean_ssim:.4f}")
    print(f"  LPIPS = {mean_lpips:.4f}  (lower is better)")
    print(f"  n_test = {n_test}")
    print("═════════════════════════════════════════════")
    print(f"  Saved to {args.output_path}")


if __name__ == "__main__":
    main()
