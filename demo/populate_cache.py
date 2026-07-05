"""Sinh cache pre-computed content cho Tab A của Gradio demo.

Với mỗi scene LLFF, script này sẽ:
1. Copy 3 training images (input views) sang `cache/<scene>/`.
2. Render + save từng test view (GT + our render + raw diff factor=1).
3. Sinh orbit GIF 60 frames interpolate qua TẤT CẢ 20 camera của LLFF
   scene (đọc COLMAP raw, không qua Scene filter llffhold=8 + n_views=3).
4. Ghi `metrics.json` chứa N_gauss + training_time.

Chạy 1 lần trong tuần prep. Sau đó Tab A chỉ đọc file từ cache/ để
hiển thị (không compute lúc UI serve).

Usage (chạy trên server, conda env `gradio_demo`):
    python -m demo.populate_cache                # cache tất cả 8 scenes
    python -m demo.populate_cache --scenes fern  # cache 1 scene (test)
    python -m demo.populate_cache --n-frames 30  # GIF ngắn hơn (default 60)

Output: `demo/cache/<scene>/` với các file (Phase 2.5: PNG lossless):
    train_1.png, train_2.png, train_3.png
    test_0_gt.png, test_0_render.png, test_0_diff.png
    test_1_gt.png, ...
    metrics.json
    orbit.gif
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import imageio.v2 as imageio
import numpy as np
from PIL import Image

# Cho phép import module CRSGaussian từ root repo khi chạy `python -m demo.populate_cache`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from demo.render_utils import (
    camera_gt_image,
    diff_map,
    load_gaussians_from_ply,
    load_llff_scene,
    render_view,
)
from demo.trajectory_utils import build_orbit_trajectory, sort_llff_cameras


# ---------------------------------------------------------------------------
# Constants — scene → checkpoint mapping
# ---------------------------------------------------------------------------


CACHE_ROOT = Path("demo/cache")

DATA_ROOT = Path("data/nerf_llff_data")

CHECKPOINT_ROOT = Path("output/p28_crs_boost/tau65")

LOG_ROOT = Path("logs/p28_crs_boost/tau65")

SCENES = ["fern", "flower", "fortress", "horns", "leaves", "orchids",
          "room", "trex"]

RECIPE_INFO = "A3-TRIM 8-module, tau=0.65, seed 42"

# Resolution downscale factor (khớp `-r 8` production).
RESOLUTION = 8


# ---------------------------------------------------------------------------
# Helper — parse training log for N_gauss + training_time
# ---------------------------------------------------------------------------


_RX_TIMING = re.compile(r"Total training:\s*([\d.]+)s")
_RX_TEST_LINE = re.compile(
    r"^\s*10000\s*\|\s*test\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|"
    r"\s*[\d.]+\s*\|\s*(\d+)"
)


def parse_training_log(log_path: Path) -> Dict[str, float]:
    """Đọc training log tau65 để lấy N_gauss và training_time."""
    result = {"training_time_s": 0.0, "n_gauss": 0}
    if not log_path.exists():
        return result

    for line in log_path.read_text(errors="ignore").splitlines():
        m = _RX_TIMING.search(line)
        if m:
            result["training_time_s"] = float(m.group(1))
        m2 = _RX_TEST_LINE.match(line)
        if m2:
            result["n_gauss"] = int(m2.group(1))
    return result


def format_training_time(seconds: float) -> str:
    """`355.5` → `5m 55s`."""
    if seconds <= 0:
        return "unknown"
    m = int(seconds // 60)
    s = int(round(seconds - m * 60))
    return f"{m}m {s:02d}s"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def save_png(pil_or_array, path: Path) -> None:
    """Lưu ảnh PNG lossless (chuẩn cho defense, không nén artifact)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(pil_or_array, np.ndarray):
        pil = Image.fromarray(pil_or_array)
    else:
        pil = pil_or_array
    pil.save(str(path), format="PNG")


def save_train_views(scene, cache_dir: Path) -> List[str]:
    """Copy 3 ảnh training input sang cache. Trả về list filenames."""
    train_cams = list(scene.getTrainCameras())
    filenames: List[str] = []
    for i, cam in enumerate(train_cams, start=1):
        gt = camera_gt_image(cam)
        out_name = f"train_{i}.png"
        save_png(gt, cache_dir / out_name)
        filenames.append(out_name)
    return filenames


def save_test_comparison(gaussians, scene, cache_dir: Path) -> List[Dict[str, str]]:
    """Với mỗi test camera: save GT + render + raw diff (factor=1).

    Trả về list dict để metrics.json ghi filenames + camera info.
    """
    test_cams = list(scene.getTestCameras())
    entries: List[Dict[str, str]] = []
    for i, cam in enumerate(test_cams):
        gt = camera_gt_image(cam)
        rend = render_view(gaussians, cam)
        # Diff raw factor=1 (mean-L1 grayscale, KHÔNG amplify).
        # Slider UI sẽ amplify in-memory theo user pref.
        diff_raw = np.clip(
            np.abs(rend.astype(np.int32) - gt.astype(np.int32)).mean(axis=2),
            0, 255,
        ).astype(np.uint8)

        gt_name = f"test_{i}_gt.png"
        rend_name = f"test_{i}_render.png"
        diff_name = f"test_{i}_diff.png"
        save_png(gt, cache_dir / gt_name)
        save_png(rend, cache_dir / rend_name)
        save_png(diff_raw, cache_dir / diff_name)

        entries.append({
            "index": i,
            "gt": gt_name,
            "render": rend_name,
            "diff": diff_name,
            "image_name": getattr(cam, "image_name", f"test_{i}"),
            "width": int(cam.image_width),
            "height": int(cam.image_height),
        })
    return entries


# ---------------------------------------------------------------------------
# COLMAP raw camera loader — bypass Scene filter to get ALL 20 cams
# ---------------------------------------------------------------------------


def load_all_colmap_cameras(source_path: Path,
                            resolution: int = RESOLUTION) -> List[SimpleNamespace]:
    """Load TẤT CẢ camera từ COLMAP raw (sparse/0/{images,cameras}.bin).

    Scene class trong CRSGaussian filter: llffhold=8 (test) + n_views=3 (train)
    → chỉ giữ 6 camera cho fern (3 train + 3 test), bỏ 14 cam khác.
    Populate cache muốn dùng cả 20 để orbit GIF quét đủ trajectory.

    Trả về list các SimpleNamespace có các attr mà `build_orbit_trajectory`
    cần: R, T, FoVx, FoVy, image_width, image_height, image_name,
    znear, zfar, trans, scale.
    """
    from scene.colmap_loader import (
        read_extrinsics_binary,
        read_intrinsics_binary,
        qvec2rotmat,
    )
    from utils.graphics_utils import focal2fov

    sparse_dir = source_path / "sparse" / "0"
    ext_path = sparse_dir / "images.bin"
    intr_path = sparse_dir / "cameras.bin"
    if not ext_path.exists() or not intr_path.exists():
        raise FileNotFoundError(f"COLMAP files missing in {sparse_dir}")

    extrinsics = read_extrinsics_binary(str(ext_path))
    intrinsics = read_intrinsics_binary(str(intr_path))

    cams: List[SimpleNamespace] = []
    for key in sorted(extrinsics.keys()):
        extr = extrinsics[key]
        intr = intrinsics[extr.camera_id]

        # Convention 3DGS: transpose(qvec2rotmat) — verified dataset_readers.py:212.
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec, dtype=np.float64)

        if intr.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            focal_x = focal_y = intr.params[0]
        elif intr.model == "PINHOLE":
            focal_x = intr.params[0]
            focal_y = intr.params[1]
        else:
            raise ValueError(f"Unsupported camera model: {intr.model}")

        # FoV KHÔNG đổi theo resolution — chỉ pixel count đổi.
        # Tính FoV từ ORIGINAL width/height (chưa downscale).
        FovX = focal2fov(focal_x, intr.width)
        FovY = focal2fov(focal_y, intr.height)

        # Render tại resolution đã downscale (khớp -r 8 production).
        rend_w = intr.width // resolution
        rend_h = intr.height // resolution

        image_name = Path(extr.name).stem

        cams.append(SimpleNamespace(
            R=R, T=T,
            FoVx=FovX, FoVy=FovY,
            image_width=rend_w, image_height=rend_h,
            image_name=image_name,
            znear=0.01, zfar=100.0,
            trans=np.array([0.0, 0.0, 0.0]),
            scale=1.0,
        ))
    return cams


def save_orbit_gif(gaussians, source_path: Path, cache_dir: Path,
                   n_frames: int = 60, fps: int = 30) -> str:
    """Render orbit trajectory + save GIF.

    Phase 2.5 change: dùng TẤT CẢ 20 cam từ COLMAP raw (thay vì 6 cam từ Scene).
    """
    all_cams = load_all_colmap_cameras(source_path, resolution=RESOLUTION)
    all_cams = sort_llff_cameras(all_cams)
    print(f"[cache] orbit uses {len(all_cams)} raw COLMAP cameras")

    traj = build_orbit_trajectory(all_cams, n_frames=n_frames)

    frames: List[np.ndarray] = []
    for cam in traj:
        frame = render_view(gaussians, cam)
        frames.append(frame)

    gif_path = cache_dir / "orbit.gif"
    imageio.mimsave(str(gif_path), frames, format="GIF",
                    duration=1.0 / fps, loop=0)
    return gif_path.name


# ---------------------------------------------------------------------------
# Main populate flow per scene
# ---------------------------------------------------------------------------


def populate_scene(scene_name: str, n_frames: int = 60) -> None:
    """Populate cache cho 1 scene. Idempotent — chạy lại sẽ overwrite."""
    ply_path = (CHECKPOINT_ROOT / f"A3_seed42_{scene_name}"
                / "point_cloud" / "iteration_10000" / "point_cloud.ply")
    if not ply_path.exists():
        raise FileNotFoundError(f"Checkpoint missing: {ply_path}")

    source_path = DATA_ROOT / scene_name
    if not source_path.exists():
        raise FileNotFoundError(f"LLFF scene missing: {source_path}")

    log_path = LOG_ROOT / f"A3_seed42_{scene_name}.log"

    cache_dir = CACHE_ROOT / scene_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== populate cache for scene: {scene_name} ===")
    t0 = time.time()

    print(f"[cache] load scene from {source_path}")
    scene = load_llff_scene(str(source_path), n_views=3, resolution=8)

    print(f"[cache] load gaussians from {ply_path}")
    gaussians = load_gaussians_from_ply(str(ply_path))
    n_gauss_loaded = int(gaussians.get_xyz.shape[0])
    print(f"[cache] N_gauss = {n_gauss_loaded}")

    print("[cache] save 3 training views")
    train_files = save_train_views(scene, cache_dir)

    print("[cache] render + save test view comparisons")
    test_entries = save_test_comparison(gaussians, scene, cache_dir)

    print(f"[cache] render orbit ({n_frames} frames)")
    orbit_file = save_orbit_gif(gaussians, source_path, cache_dir,
                                n_frames=n_frames)

    # Metrics + info
    log_stats = parse_training_log(log_path)
    metrics = {
        "scene": scene_name,
        "n_gauss": n_gauss_loaded,
        "training_time_s": log_stats["training_time_s"],
        "training_time_str": format_training_time(log_stats["training_time_s"]),
        "recipe": RECIPE_INFO,
        "train_views": train_files,
        "test_views": test_entries,
        "orbit": orbit_file,
        "resolution": [test_entries[0]["width"], test_entries[0]["height"]]
        if test_entries else None,
    }
    (cache_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    elapsed = time.time() - t0
    print(f"[cache] scene {scene_name} done in {elapsed:.1f}s "
          f"→ {cache_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate demo cache.")
    parser.add_argument("--scenes", nargs="*", default=None,
                        help="Scene names to populate (default: all 8 LLFF).")
    parser.add_argument("--n-frames", type=int, default=60,
                        help="Number of orbit GIF frames.")
    args = parser.parse_args()

    scenes = args.scenes if args.scenes else SCENES

    t_start = time.time()
    for sc in scenes:
        try:
            populate_scene(sc, n_frames=args.n_frames)
        except Exception as e:
            print(f"[cache] FAILED scene={sc}: {e}")
            raise
    print(f"\n[cache] TOTAL {len(scenes)} scenes done in "
          f"{time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
