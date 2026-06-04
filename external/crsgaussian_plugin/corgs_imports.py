# ============================================================
# [CRSGaussian Path A B1] Centralized CoR-GS import wrapper
# File: crsgaussian_plugin/corgs_imports.py  (KEEP LOCAL — upload server)
#
# Mục đích: CoR-GS source ở `~/workspace/representation-3d/duyen/CoR-GS/` KHÔNG
# được pip install. Plug-in cần manipulate sys.path để import GaussianModel,
# render, Camera, MiniCam, etc.
#
# Quan trọng: KHÔNG sửa CoR-GS source — chỉ import as-is. Path A contract.
#
# Usage:
#   from crsgaussian_plugin.corgs_imports import (
#       CorGsGaussianModel, corgs_render, CorGsCamera, CorGsMiniCam,
#       corgs_l1_loss, corgs_ssim, ModelParams, OptimizationParams, PipelineParams,
#   )
# ============================================================
"""[Path A B1] Centralized CoR-GS imports với sys.path setup."""

import os
import sys
from pathlib import Path


# ── Default CoR-GS source path trên server ──
# Override qua env var CORGS_SOURCE_PATH nếu cần
CORGS_SOURCE_PATH = os.environ.get(
    "CORGS_SOURCE_PATH",
    "/home/aidev/workspace/representation-3d/duyen/CoR-GS",
)


def _setup_corgs_path():
    """Insert CoR-GS source path vào sys.path nếu chưa có."""
    p = str(Path(CORGS_SOURCE_PATH).resolve())
    if not Path(p).exists():
        raise ImportError(
            f"[Path A] CoR-GS source path không tồn tại: {p}\n"
            f"Set env CORGS_SOURCE_PATH hoặc clone CoR-GS về {CORGS_SOURCE_PATH}"
        )
    if p not in sys.path:
        sys.path.insert(0, p)


_setup_corgs_path()

# ── Lazy import — chỉ chạy sau _setup_corgs_path() ──
from scene.gaussian_model import GaussianModel as CorGsGaussianModel  # noqa: E402
from scene.cameras import Camera as CorGsCamera  # noqa: E402
from scene.cameras import MiniCam as CorGsMiniCam  # noqa: E402
from gaussian_renderer import render as corgs_render  # noqa: E402
from utils.loss_utils import l1_loss as corgs_l1_loss, ssim as corgs_ssim  # noqa: E402
from arguments import ModelParams, OptimizationParams, PipelineParams  # noqa: E402
from utils.graphics_utils import getWorld2View2, getProjectionMatrix  # noqa: E402


__all__ = [
    "CORGS_SOURCE_PATH",
    "CorGsGaussianModel",
    "CorGsCamera",
    "CorGsMiniCam",
    "corgs_render",
    "corgs_l1_loss",
    "corgs_ssim",
    "ModelParams",
    "OptimizationParams",
    "PipelineParams",
    "getWorld2View2",
    "getProjectionMatrix",
]
