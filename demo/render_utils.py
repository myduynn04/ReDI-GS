"""Utilities to load a trained Gaussian checkpoint and render single views.

Design decisions verified against real code (2026-07-05 audit):

- GaussianModel: constructor là `GaussianModel(args)`, args cần có
  `sh_degree` (số nguyên, max degree) và optional `absdensify` (bool).
  Xem `scene/gaussian_model.py:150`.

- Scene: constructor `Scene(args, gaussians, load_iteration=None,
  shuffle=True, resolution_scales=[1.0])`. Không có `no_prior`. Scene
  đọc `args.model_path, args.source_path, args.images, args.eval,
  args.n_views, args.rand_pcd, args.resolution, args.data_device,
  args.white_background`. Xem `scene/__init__.py:49`.

- Scene ghi `input.ply` và `cameras.json` vào `args.model_path` nên
  cần point tới 1 temp dir để không rác cwd.

- Scene tự động gọi `gaussians.create_from_pcd(...)` khi `load_iteration=None`.
  Ta gọi Scene chỉ để lấy cameras, sau đó tạo GaussianModel MỚI và
  load .ply user chỉ định (khác với Gaussians dummy trong Scene).

- Sau `load_ply(...)`, `pc.confidence` vẫn là `torch.empty(0)` (chỉ
  `create_from_pcd` mới init confidence). Nếu gọi thẳng `render()`
  rasterizer sẽ nhận confidence rỗng và crash. Ta phải init tay bằng
  `torch.ones_like(pc._opacity)` cho khớp pattern `create_from_pcd`
  ở `scene/gaussian_model.py:378`.

- render() truy cập trực tiếp các thuộc tính `pipe.use_confidence`,
  `pipe.debug`, `pipe.compute_cov3D_python`, `pipe.convert_SHs_python`.
  Tất cả dropout/dropansh fields được đọc qua `getattr(pipe, ...,
  default)` nên chỉ cần 4 field trên là đủ. Xem
  `gaussian_renderer/__init__.py:120,135,162,173`.

- render() signature: `render(viewpoint_camera, pc, pipe, bg_color,
  scaling_modifier=1.0, override_color=None, white_bg=False,
  disable_dropout=False)`.

Reuse hoàn toàn code CRSGaussian có sẵn, không sửa gì trong scene/
hay gaussian_renderer/.
"""

from __future__ import annotations

import sys
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
from PIL import Image

# Cho phép import module CRSGaussian từ root repo.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gaussian_renderer import render as _render  # noqa: E402
from scene import Scene  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402


# ---------------------------------------------------------------------------
# GaussianModel loader
# ---------------------------------------------------------------------------


def load_gaussians_from_ply(ply_path: Union[str, Path],
                            sh_degree: int = 3) -> GaussianModel:
    """Load 1 file `point_cloud.ply` thành GaussianModel sẵn sàng render.

    Trả về GaussianModel đã ở CUDA. Không setup optimizer (inference-only).

    Args:
        ply_path: đường dẫn tới file .ply, thường là
            `output/<scene>/point_cloud/iteration_<N>/point_cloud.ply`.
        sh_degree: max SH degree của checkpoint. Recipe ReDI-GS lock ở 3.
    """
    ply_path = Path(ply_path)
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY not found: {ply_path}")

    # Path này chỉ gọi load_ply, không đi qua create_from_pcd,
    # nên chỉ cần các field mà __init__ trực tiếp đọc.
    args_ns = Namespace(sh_degree=sh_degree, absdensify=False, train_bg=False,
                        use_color=True)
    gaussians = GaussianModel(args_ns)
    gaussians.load_ply(str(ply_path))

    # `load_ply` không set `confidence` — mà rasterizer trong render() lại
    # truy cập `pc.confidence` (dù pipe.use_confidence=False vẫn gọi
    # torch.ones_like(pc.confidence)). Manual init cho khớp
    # create_from_pcd() line 378 trong scene/gaussian_model.py.
    if gaussians.confidence.numel() == 0:
        gaussians.confidence = torch.ones_like(gaussians._opacity, device="cuda")

    return gaussians


# ---------------------------------------------------------------------------
# Scene loader — lấy cameras (train + test) của 1 LLFF scene
# ---------------------------------------------------------------------------


def _build_scene_args(source_path: str, model_path: str,
                      n_views: int = 3, resolution: int = 8) -> Namespace:
    """Namespace tối thiểu đủ cho Scene() + create_from_pcd() chạy qua path LLFF.

    Trong train.py production, args là parser namespace hợp nhất
    ModelParams + PipelineParams + extra CLI. GaussianModel truy cập
    args qua `self.args.<field>` — cần bao đủ mọi field mà chuỗi gọi
    `Scene → create_from_pcd → __init__` đọc.

    Field cần thiết theo đường code inference-only (không training):
    - Scene args (đọc trực tiếp trong scene/__init__.py:49-90):
      source_path, model_path, images, resolution, white_background,
      data_device, eval, n_views, rand_pcd.
    - GaussianModel + create_from_pcd (scene/gaussian_model.py):
      sh_degree (line 154), absdensify (line 183),
      use_color (line 345), train_bg (line 392).
    """
    return Namespace(
        # ── Scene reads these directly ─────────────────────────
        source_path=source_path,
        model_path=model_path,
        images="images",
        resolution=resolution,
        white_background=False,
        data_device="cuda",
        eval=True,
        n_views=n_views,
        rand_pcd=False,
        # ── GaussianModel + create_from_pcd read these ────────
        sh_degree=3,
        absdensify=False,
        train_bg=False,
        use_color=True,   # default cho production; init màu Gaussian từ point cloud
    )


def load_llff_scene(source_path: str, n_views: int = 3,
                    resolution: int = 8) -> Scene:
    """Load LLFF scene, trả về Scene có train/test cameras đúng convention.

    Scene sẽ tự init 1 GaussianModel dummy qua `create_from_pcd(...)` —
    ta bỏ qua các Gaussians này và load riêng .ply mình cần
    ở phía trên (`load_gaussians_from_ply`).

    Scene ghi `input.ply` và `cameras.json` vào `model_path`; ta dùng
    tempfile để không rác cwd. Temp dir tồn tại đến khi process exit.

    Args:
        source_path: `data/nerf_llff_data/fern` chẳng hạn.
        n_views: 3 cho sparse-view recipe.
        resolution: 8 tương ứng flag `-r 8` trong training.
    """
    # NamedTempDir sẽ auto-clean; giữ reference để đừng bị GC lúc đang dùng.
    tmpdir = tempfile.mkdtemp(prefix="redigs_demo_")
    dummy_args = _build_scene_args(source_path, tmpdir, n_views, resolution)
    dummy_gaussians = GaussianModel(dummy_args)
    scene = Scene(dummy_args, dummy_gaussians, load_iteration=None,
                  shuffle=False, resolution_scales=[1.0])
    return scene


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------


# Chỉ cần 4 field pipe được truy cập trực tiếp trong render().
# Mọi dropout/dropansh field khác được đọc qua getattr với default.
_DEFAULT_PIPELINE = Namespace(
    convert_SHs_python=False,
    compute_cov3D_python=False,
    debug=False,
    use_confidence=False,
)


def render_view(gaussians: GaussianModel, viewpoint_camera,
                bg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                scaling_modifier: float = 1.0) -> np.ndarray:
    """Render 1 view, trả về HxWx3 uint8 numpy RGB.

    Truyền `disable_dropout=True` để chắc chắn không mất Gaussian nào
    (dropout chỉ có ý nghĩa lúc train).
    """
    with torch.no_grad():
        bg = torch.tensor(list(bg_color), dtype=torch.float32, device="cuda")
        render_pkg = _render(
            viewpoint_camera,
            gaussians,
            _DEFAULT_PIPELINE,
            bg,
            scaling_modifier=scaling_modifier,
            disable_dropout=True,
        )
    image = render_pkg["render"].clamp(0.0, 1.0)  # (3, H, W)
    image_np = (image.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return image_np


def render_view_pil(gaussians: GaussianModel, viewpoint_camera,
                    bg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                    scaling_modifier: float = 1.0) -> Image.Image:
    return Image.fromarray(
        render_view(gaussians, viewpoint_camera, bg_color, scaling_modifier)
    )


# ---------------------------------------------------------------------------
# Ground truth helpers
# ---------------------------------------------------------------------------


def camera_gt_image(viewpoint_camera) -> np.ndarray:
    """Ground truth ảnh của 1 Camera, HxWx3 uint8 numpy RGB."""
    gt = viewpoint_camera.original_image.clamp(0.0, 1.0)
    return (gt.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)


def camera_gt_pil(viewpoint_camera) -> Image.Image:
    return Image.fromarray(camera_gt_image(viewpoint_camera))


def diff_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Difference map HxW uint8 (L1 amplify x3)."""
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    diff = np.abs(pred.astype(np.int32) - gt.astype(np.int32)).mean(axis=2)
    return np.clip(diff * 3, 0, 255).astype(np.uint8)


def diff_pil(pred: np.ndarray, gt: np.ndarray) -> Image.Image:
    return Image.fromarray(diff_map(pred, gt))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke test render_utils.")
    parser.add_argument("--source", required=True,
                        help="LLFF scene dir, e.g. data/nerf_llff_data/fern")
    parser.add_argument("--ply", required=True,
                        help="Path to point_cloud.ply")
    parser.add_argument("--out", default="/tmp/render_smoke.png",
                        help="Where to save the rendered test view")
    parser.add_argument("--test-idx", type=int, default=0,
                        help="Index of the test camera to render")
    args = parser.parse_args()

    print(f"[demo] load scene from {args.source}")
    scene = load_llff_scene(args.source, n_views=3, resolution=8)
    test_cams = scene.getTestCameras()
    if not test_cams:
        raise RuntimeError("No test cameras found in scene")
    if args.test_idx >= len(test_cams):
        raise IndexError(f"test_idx {args.test_idx} out of range "
                         f"(scene has {len(test_cams)} test cams)")
    cam = test_cams[args.test_idx]

    print(f"[demo] load gaussians from {args.ply}")
    gaussians = load_gaussians_from_ply(args.ply)
    print(f"[demo] N_gauss = {gaussians.get_xyz.shape[0]}")

    print(f"[demo] render test camera idx={args.test_idx} "
          f"(uid={cam.uid}, {cam.image_width}x{cam.image_height})")
    pil = render_view_pil(gaussians, cam)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pil.save(args.out)
    print(f"[demo] saved to {args.out}")

    gt_pil = camera_gt_pil(cam)
    gt_out = Path(args.out).with_name(Path(args.out).stem + "_gt.png")
    gt_pil.save(gt_out)
    print(f"[demo] gt saved to {gt_out}")

    diff = diff_pil(np.array(pil), np.array(gt_pil))
    diff_out = Path(args.out).with_name(Path(args.out).stem + "_diff.png")
    diff.save(diff_out)
    print(f"[demo] diff saved to {diff_out}")
