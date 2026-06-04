#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Plug-in A2.3] Pre-compute aligned DAV2 depth → .npy
# File: crsgaussian_plugin/preprocess_depth_a23.py
#
# Run 1 lần per scene TRONG ENV `corgs` (CRSGaussian) — vì cần DAV2 model
# + align_depth_to_colmap. Output .npy load được trong env `nerfstudio`.
#
# Usage:
#   conda activate corgs
#   cd ~/workspace/representation-3d/duyen/CoR-GS
#   SCENE=fern python crsgaussian_plugin/preprocess_depth_a23.py
#
#   # Hoặc all scenes
#   for SC in fern horns fortress flower; do
#       SCENE=$SC python crsgaussian_plugin/preprocess_depth_a23.py
#   done
#
# Output:
#   <data>/<scene>/3_views/aligned_depth_a23/<image_stem>.npy  (float32, H, W)
#   <data>/<scene>/3_views/aligned_depth_a23/_meta.json         (depth_range)
# ============================================================
"""[CRSGaussian Plug-in A2.3] Pre-compute aligned depth per scene."""

import os
import sys
import json
from pathlib import Path

import numpy as np

# Allow run từ anywhere — add CRSGaussian root to path
CRSG_ROOT = Path(__file__).resolve().parent.parent  # crsgaussian_plugin/ → root
sys.path.insert(0, str(CRSG_ROOT))

from utils.depth import precompute_depth_priors, align_depth_to_colmap  # noqa: E402
from scene import Scene  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402
from arguments import ModelParams, OptimizationParams, PipelineParams  # noqa: E402
from argparse import ArgumentParser  # noqa: E402


SCENE = os.environ.get("SCENE", "fern")
DATA_ROOT = os.environ.get("DATA_ROOT", "data/nerf_llff_data")
N_VIEWS = int(os.environ.get("N_VIEWS", "3"))
DAV2_PATH = os.environ.get("DAV2_PATH", "../Depth-Anything-V2")
DAV2_ENCODER = os.environ.get("DAV2_ENCODER", "vitl")


class ArgsWrapper:
    """Auto-default False cho mọi attr missing.

    Lý do: CRSGaussian Scene/GaussianModel access ~150 attr của args. Build
    Namespace manual sẽ thiếu, lỗi AttributeError lặp lại. Wrapper này
    fallback False cho bất kỳ attr nào không có trong real args.
    """
    def __init__(self, real_args):
        self.__dict__["_real"] = real_args

    def __getattr__(self, name):
        if hasattr(self._real, name):
            return getattr(self._real, name)
        return False   # safe default — code path preprocess không cần value cụ thể

    def __setattr__(self, name, value):
        setattr(self._real, name, value)


def main():
    print("=" * 70)
    print(f"[A2.3 preprocess] scene={SCENE} n_views={N_VIEWS}")
    print("=" * 70)

    source_path = str(Path(DATA_ROOT) / SCENE)
    out_dir = Path(source_path) / f"{N_VIEWS}_views" / "aligned_depth_a23"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Build full args giống train.py để có ĐỦ field cho GaussianModel ──
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    # TẤT CẢ custom args từ train.py main (verbatim)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[10000])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint2", type=str, default=None)
    parser.add_argument("--train_bg", action="store_true")
    parser.add_argument('--gaussiansN', type=int, default=1)
    parser.add_argument("--onlyrgb", action='store_true', default=False)
    parser.add_argument("--coreg", action='store_true', default=False)
    parser.add_argument("--coprune", action='store_true', default=False)
    parser.add_argument('--coprune_threshold', type=int, default=5)
    parser.add_argument("--save_log_images", action="store_true")
    parser.add_argument('--seed', type=int, default=42)

    real_args = parser.parse_args([])
    # Override fields cần thiết cho preprocess
    real_args.source_path = source_path
    real_args.model_path = ""
    real_args.images = "images"
    real_args.resolution = 8
    real_args.eval = True
    real_args.n_views = N_VIEWS
    real_args.use_depth_prior = True
    real_args.dav2_path = DAV2_PATH
    real_args.dav2_encoder = DAV2_ENCODER

    # Wrap để fallback False cho mọi attr missing (vd train_bg, use_color, etc.)
    args = ArgsWrapper(real_args)

    # Load scene để có train cameras list
    print(f"[A2.3] Loading Scene from {source_path}...")
    dummy_gaussians = GaussianModel(args)
    scene = Scene(args, dummy_gaussians, shuffle=False)
    train_cams = scene.getTrainCameras().copy()
    print(f"[A2.3] {len(train_cams)} training cameras")

    # ── Compute DAV2 depth priors ──
    print(f"[A2.3] Running DAV2 ({DAV2_ENCODER})...")
    depth_prior_dict = precompute_depth_priors(
        train_cams, DAV2_PATH, encoder=DAV2_ENCODER
    )

    # ── Align với COLMAP ──
    print(f"[A2.3] Aligning depth với COLMAP sparse points...")
    aligned_depth_dict, depth_range = align_depth_to_colmap(
        depth_prior_dict, train_cams, source_path, N_VIEWS
    )
    print(f"[A2.3] depth_range = {depth_range:.3f}")

    # ── Save mỗi cam .npy theo image stem (KHÔNG dùng cam.uid vì khác process) ──
    n_saved = 0
    for cam in train_cams:
        if cam.uid not in aligned_depth_dict:
            print(f"  ⚠ {cam.image_name} → KHÔNG có aligned depth")
            continue
        stem = Path(cam.image_name).stem
        out_path = out_dir / f"{stem}.npy"
        depth = aligned_depth_dict[cam.uid].cpu().numpy().astype(np.float32)
        np.save(out_path, depth)
        n_saved += 1

    # Save meta
    meta = {
        "scene": SCENE,
        "n_views": N_VIEWS,
        "n_saved": n_saved,
        "depth_range": float(depth_range),
        "dav2_encoder": DAV2_ENCODER,
        "image_stems": [Path(c.image_name).stem for c in train_cams],
    }
    with open(out_dir / "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[A2.3] Saved {n_saved} aligned depth .npy + _meta.json to {out_dir}")


if __name__ == "__main__":
    main()
