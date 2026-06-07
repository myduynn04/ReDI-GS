# ============================================================
# [CRSGaussian Path A B2] Camera adapter — Nerfstudio Cameras → CoR-GS Camera/MiniCam
# File: crsgaussian_plugin/camera_adapter.py  (KEEP LOCAL — upload server)
#
# 2 builders:
#   - ns_camera_to_corgs_minicam: dùng cho render (eval cam không có image file)
#   - ns_camera_to_corgs_camera: dùng cho TRAIN cam Phase 22 callbacks
#     (LFCF cần R/T numpy, CRS update R compute cần original_image)
#
# Source-of-truth:
#   - CRSGaussian/scene/cameras.py:18 Camera.__init__ (full)
#   - CRSGaussian/scene/cameras.py:90 MiniCam.__init__ (render-only)
#   - CRSGaussian/utils/camera_utils.py:21 loadCam (Scene loader pattern)
#   - CRSGaussian/utils/general_utils.py:25 PILtoTorch (image format)
#
# CONVENTION NOTE:
#   - Nerfstudio: OpenGL (Y up, -Z forward) trong camera_to_worlds
#   - CoR-GS/Inria: COLMAP (Y down, +Z forward) trong R/T
#   - → Flip Y + Z khi convert
#
# Resolution alignment (Path A B2 FIX 1):
#   - CrsGaussianDataParserConfig.downscale_factor=8 → ns Cameras image_width/height
#     đã match Phase 22 args.resolution=8
#   - Plug-in PIL load resize theo ns_cam.image_width/height direct (KHÔNG hardcode)
# ============================================================
"""[Path A B2] Bridge Nerfstudio Cameras → CoR-GS Camera + MiniCam."""

import math
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from .corgs_imports import (
    CorGsMiniCam,
    CorGsCamera,
    getWorld2View2,
    getProjectionMatrix,
)


def _fov_from_focal(focal: float, length: int) -> float:
    """FoV (radians) = 2 * atan(L / (2 * f)).

    [B6 viewer fix 2026-06-06] Guard against focal=0 (ns viewer pass invalid
    intrinsics ở first frame trước khi user move camera). Fallback default 60° FoV.
    """
    if focal <= 1e-6 or length <= 0:
        return math.radians(60.0)  # safe default
    return 2.0 * math.atan(length / (2.0 * focal))


def _extract_RT_FoV(camera) -> Tuple[np.ndarray, np.ndarray, float, float, int, int]:
    """Shared helper — extract R (3,3 np), T (3, np), FoVx, FoVy, width, height từ ns Cameras.

    PORTED FROM convention notes:
      - ns camera_to_worlds: OpenGL (Y up, -Z forward)
      - CoR-GS R/T: COLMAP (Y down, +Z forward)
      - Flip Y + Z columns: c2w_colmap = c2w_ns @ diag(1, -1, -1, 1)
      - R = w2c[:3,:3].T = c2w[:3,:3] (per Inria getWorld2View2 expects R transposed)
      - T = w2c[:3, 3]
    """
    # Intrinsics
    fx = float(camera.fx[0].item() if camera.fx.numel() > 1 else camera.fx.item())
    fy = float(camera.fy[0].item() if camera.fy.numel() > 1 else camera.fy.item())
    width = int(camera.width[0].item() if camera.width.numel() > 1 else camera.width.item())
    height = int(camera.height[0].item() if camera.height.numel() > 1 else camera.height.item())

    # [B6 viewer fix 2026-06-06] Guard zero width/height (ns viewer init frame
    # có thể pass camera với dim=0 → rasterizer launch grid (0,0) → CUDA error
    # "invalid configuration argument").
    # Fallback Phase 22 res: 504×378 (= fern 4032×3024 / 8).
    if width <= 0:
        width = 504
    if height <= 0:
        height = 378
    if fx <= 1e-6:
        fx = width  # arbitrary positive focal
    if fy <= 1e-6:
        fy = height

    fov_x = _fov_from_focal(fx, width)
    fov_y = _fov_from_focal(fy, height)

    # Pose: camera_to_worlds (3x4)
    c2w_ns = camera.camera_to_worlds[0] if camera.camera_to_worlds.ndim == 3 else camera.camera_to_worlds
    c2w_ns = c2w_ns.cpu().numpy().astype(np.float64)

    c2w_4x4 = np.eye(4)
    c2w_4x4[:3, :4] = c2w_ns

    # Flip Y, Z (OpenGL → COLMAP)
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    c2w_colmap = c2w_4x4 @ flip

    w2c = np.linalg.inv(c2w_colmap)
    R = w2c[:3, :3].T   # CoR-GS expects R transposed
    T = w2c[:3, 3]

    return R, T, fov_x, fov_y, width, height


def ns_camera_to_corgs_minicam(
    camera,
    znear: float = 0.01,
    zfar: float = 100.0,
    device: str = "cuda",
) -> CorGsMiniCam:
    """[Path A B1] Render-only camera adapter (no image needed).

    Dùng cho eval cam + get_outputs() — chỉ cần pose matrices.
    """
    R, T, fov_x, fov_y, width, height = _extract_RT_FoV(camera)

    world_view = torch.tensor(
        getWorld2View2(R, T, np.zeros(3), 1.0)
    ).transpose(0, 1).to(device, dtype=torch.float32)

    proj_matrix = (
        getProjectionMatrix(znear=znear, zfar=zfar, fovX=fov_x, fovY=fov_y)
        .transpose(0, 1)
        .to(device, dtype=torch.float32)
    )

    full_proj = (
        world_view.unsqueeze(0).bmm(proj_matrix.unsqueeze(0))
    ).squeeze(0)

    return CorGsMiniCam(
        width=width,
        height=height,
        fovy=fov_y,
        fovx=fov_x,
        znear=znear,
        zfar=zfar,
        world_view_transform=world_view,
        full_proj_transform=full_proj,
    )


def ns_camera_to_corgs_camera(
    camera,
    image_filename,
    idx: int,
    data_device: str = "cuda",
) -> CorGsCamera:
    """[Path A B2] Full CoR-GS Camera builder cho TRAIN cam.

    PORTED FROM: CRSGaussian/utils/camera_utils.py:21 loadCam()
    Differences (em adapt cho plug-in context):
      - R, T extract từ ns Cameras (qua convention flip) thay vì cam_info COLMAP
      - FoVx/y compute từ ns fx/fy + width/height
      - Image load PIL match ns_cam.image_width/height (đã align downscale_factor=8)
      - gt_alpha_mask = None (Phase 22 LLFF không dùng alpha)
      - depth_image = None (Phase 22 dùng aligned_depth_dict riêng)
      - mask, bounds = None (Phase 22 LLFF context)

    Args:
        camera: nerfstudio Cameras object (single, .ndim=2 hoặc batched first)
        image_filename: Path/str đến image file (đã downscale theo downscale_factor=8)
        idx: image index (= uid + colmap_id)
        data_device: "cuda" or "cpu"

    Returns:
        CorGsCamera với .R, .T, .original_image, .uid, .world_view_transform, etc.
    """
    R, T, fov_x, fov_y, width, height = _extract_RT_FoV(camera)

    # ── Load image PIL → torch (3, H, W) float [0, 1] CPU ──
    # PORTED FROM: utils/general_utils.py:25 PILtoTorch + utils/camera_utils.py:43-45
    # [B3 split-fix 2026-06-04] Fallback fullres → resize in-memory (như Phase 22 standalone).
    # Lý do: ns ColmapDataParser hardcode build path images_{factor}/ với COLMAP naming.
    # Nếu folder tồn tại NHƯNG khác naming (vd CRSGaussian preprocess image000.png),
    # em fallback về images/ + resize 1/8 — KHÔNG cần user pre-generate folder.
    from PIL import Image
    import re
    img_path = Path(str(image_filename))
    if not img_path.exists():
        fullres = Path(re.sub(r'images_\d+/', 'images/', str(img_path)))
        if fullres.exists():
            img_path = fullres
        else:
            raise FileNotFoundError(
                f"[Path A] Image not found:\n"
                f"  expected: {image_filename}\n"
                f"  fallback: {fullres}\n"
                f"  Check folder structure or COLMAP naming."
            )
    pil = Image.open(str(img_path))
    # Match exactly ns_cam.image_width/height (resolution-aligned via downscale_factor=8)
    pil_resized = pil.resize((width, height))
    arr = np.array(pil_resized).astype(np.float32) / 255.0
    if arr.ndim == 3:
        # HWC → CHW, lấy 3 channel đầu (filter alpha nếu RGBA)
        img_tensor = torch.from_numpy(arr).permute(2, 0, 1)[:3]
    else:
        # Grayscale → broadcast 3 channels
        img_tensor = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1)

    # ── Build Camera ──
    # PORTED FROM: scene/cameras.py:18 Camera.__init__
    # Camera tự build: world_view_transform, projection_matrix, full_proj_transform, camera_center
    # Camera tự .clamp(0, 1).to(data_device) on image
    return CorGsCamera(
        colmap_id=idx,
        R=R,
        T=T,
        FoVx=fov_x,
        FoVy=fov_y,
        image=img_tensor,
        gt_alpha_mask=None,
        image_name=Path(str(image_filename)).stem,
        uid=idx,
        data_device=data_device,
    )


__all__ = [
    "ns_camera_to_corgs_minicam",
    "ns_camera_to_corgs_camera",
]
