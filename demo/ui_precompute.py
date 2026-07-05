"""Tab 1 (Pre-computed viewer) cho Gradio demo.

Đọc content pre-generated từ `demo/cache/<scene>/` (do
`populate_cache.py` sinh). Không render live, chỉ serve file từ disk +
amplify diff map trong memory theo slider user.

Layout:
- Dropdown chọn scene (mặc định fortress).
- 3 training input images.
- Test view grid: index | GT | Render | Diff (amplified).
- Slider "Difference amplification" 1..10, default 1.
- Orbit GIF autoplay loop.
- Info: N_gauss, training time, recipe.

Không có tier ranking, không hiển thị PSNR/SSIM/LPIPS number
(theo yêu cầu user).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import gradio as gr
import numpy as np
from PIL import Image


CACHE_ROOT_DEFAULT = Path("demo/cache")

DEFAULT_SCENE = "fortress"

SCENES = ["fern", "flower", "fortress", "horns", "leaves", "orchids",
          "room", "trex"]

# Phase 2.5b: preload TẤT CẢ scene vào memory lúc startup để chuyển scene
# instant (~10ms). Chi phí: ~45MB memory (8 × 12 PNG), ~2s startup thêm.
# Dict: scene_name → preloaded data dict.
_PRELOAD_CACHE: dict = {}


# ---------------------------------------------------------------------------
# Cache reading helpers
# ---------------------------------------------------------------------------


def _cache_dir(scene: str, cache_root: Path) -> Path:
    return cache_root / scene


def _load_metrics(scene: str, cache_root: Path) -> dict:
    p = _cache_dir(scene, cache_root) / "metrics.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _load_diff_raw(scene: str, cache_root: Path, idx: int) -> Image.Image | None:
    p = _cache_dir(scene, cache_root) / f"test_{idx}_diff.png"
    if not p.exists():
        return None
    return Image.open(p).convert("L")  # grayscale


def _amplify_diff(diff_raw: Image.Image, factor: float) -> Image.Image:
    """Nhân diff map bằng factor rồi clip. Trả về PIL grayscale."""
    arr = np.asarray(diff_raw, dtype=np.int32)
    amplified = np.clip(arr * factor, 0, 255).astype(np.uint8)
    return Image.fromarray(amplified, mode="L")


def _existing_scenes(cache_root: Path) -> List[str]:
    """Trả về danh sách scene mà cache đã sẵn sàng."""
    if not cache_root.exists():
        return []
    return [s for s in SCENES if (cache_root / s / "metrics.json").exists()]


# ---------------------------------------------------------------------------
# Preload — nạp toàn bộ file PNG/GIF vào memory 1 lần lúc startup
# ---------------------------------------------------------------------------


def _open_and_load(path: Path, mode: str | None = None) -> Image.Image | None:
    """Open PIL image + force decode vào memory (không giữ file handle)."""
    if not path.exists():
        return None
    img = Image.open(path)
    if mode is not None:
        img = img.convert(mode)
    img.load()   # force decode ngay, PIL không giữ file handle sau khi return
    return img


def _preload_scene(scene: str, cache_root: Path) -> dict | None:
    """Nạp toàn bộ cache của 1 scene vào memory.

    Trả về dict với keys: trains (3 PIL), gts (3 PIL), renders (3 PIL),
    diffs_raw (3 PIL L-mode), orbit (str path), info_md (str).
    """
    metrics = _load_metrics(scene, cache_root)
    if not metrics:
        return None
    cache = _cache_dir(scene, cache_root)

    trains: List[Image.Image | None] = []
    for f in metrics.get("train_views", [])[:3]:
        trains.append(_open_and_load(cache / f))
    while len(trains) < 3:
        trains.append(None)

    gts: List[Image.Image | None] = [None, None, None]
    renders: List[Image.Image | None] = [None, None, None]
    diffs_raw: List[Image.Image | None] = [None, None, None]
    for i, entry in enumerate(metrics.get("test_views", [])[:3]):
        gts[i] = _open_and_load(cache / entry["gt"])
        renders[i] = _open_and_load(cache / entry["render"])
        # Diff giữ mode L (grayscale) để amplify in-memory nhân matrix.
        diffs_raw[i] = _open_and_load(cache / entry["diff"], mode="L")

    orbit_p = cache / metrics.get("orbit", "orbit.gif")
    orbit_str = str(orbit_p) if orbit_p.exists() else None

    # Info markdown — Phase 2.5: bỏ dòng Recipe.
    n_gauss = metrics.get("n_gauss", 0)
    training_time = metrics.get("training_time_str", "unknown")
    info_md = (
        f"**Gaussians**: {n_gauss:,}  \n"
        f"**Training time**: {training_time}"
    )

    return {
        "trains": trains,
        "gts": gts,
        "renders": renders,
        "diffs_raw": diffs_raw,
        "orbit": orbit_str,
        "info_md": info_md,
    }


def _preload_all(scenes_available: List[str], cache_root: Path) -> None:
    """Nạp preload cache cho tất cả scene sẵn có. Idempotent."""
    print(f"[ui] preloading {len(scenes_available)} scenes into memory...")
    for scene in scenes_available:
        if scene not in _PRELOAD_CACHE:
            data = _preload_scene(scene, cache_root)
            if data is not None:
                _PRELOAD_CACHE[scene] = data
    print(f"[ui] preloaded {len(_PRELOAD_CACHE)} scenes ready")


# ---------------------------------------------------------------------------
# Callback — cập nhật UI khi đổi scene hoặc slider (đọc từ preload dict)
# ---------------------------------------------------------------------------


def build_scene_callback(cache_root: Path):
    """Factory sinh callback để Gradio gọi khi dropdown scene đổi.

    Phase 2.5b: callback đọc từ `_PRELOAD_CACHE` dict thay vì disk →
    chuyển scene instant, không chớp. Diff amplify tính in-memory.

    Returns: (scene: str, amplify: float) -> tuple(14 widget values)
    """
    def cb(scene: str, amplify: float):
        data = _PRELOAD_CACHE.get(scene)
        if data is None:
            # Fallback: nếu preload miss (cache thêm sau startup), nạp on-demand.
            data = _preload_scene(scene, cache_root)
            if data is not None:
                _PRELOAD_CACHE[scene] = data
        if data is None:
            return (None,) * 14

        # Diff amplify trong memory theo slider factor.
        diffs = [
            _amplify_diff(d, amplify) if d is not None else None
            for d in data["diffs_raw"]
        ]

        return (
            data["trains"][0], data["trains"][1], data["trains"][2],
            data["gts"][0], data["renders"][0], diffs[0],
            data["gts"][1], data["renders"][1], diffs[1],
            data["gts"][2], data["renders"][2], diffs[2],
            data["orbit"],
            data["info_md"],
        )

    return cb


# ---------------------------------------------------------------------------
# Build Tab
# ---------------------------------------------------------------------------


def build_precompute_tab(cache_root: Path = CACHE_ROOT_DEFAULT) -> None:
    """Build layout Tab 1 vào current gr.Blocks context."""
    scenes_available = _existing_scenes(cache_root)
    if not scenes_available:
        gr.Markdown(
            f"⚠️ No cache found in `{cache_root}`. "
            f"Run `python -m demo.populate_cache` first to generate content."
        )
        return

    # Phase 2.5b: preload TẤT CẢ scene vào memory lúc startup (~2s cho 8 scene).
    # Sau đó chuyển scene chỉ đọc từ dict → instant, không chớp.
    _preload_all(scenes_available, cache_root)

    default_scene = DEFAULT_SCENE if DEFAULT_SCENE in scenes_available \
        else scenes_available[0]

    # Compute initial content NGAY để bake vào `value=` lúc construct widget.
    # Gradio 4: gán widget.value = X sau khi tạo sẽ KHÔNG sync frontend.
    callback = build_scene_callback(cache_root)
    initial = callback(default_scene, 1.0)
    (i_train1, i_train2, i_train3,
     i_gt0, i_rend0, i_diff0,
     i_gt1, i_rend1, i_diff1,
     i_gt2, i_rend2, i_diff2,
     i_orbit, i_info) = initial

    with gr.Row():
        scene_dd = gr.Dropdown(
            choices=scenes_available,
            value=default_scene,
            label="Scene",
            interactive=True,
        )
        amplify_slider = gr.Slider(
            minimum=1.0, maximum=10.0, value=1.0, step=0.5,
            label="Difference amplification",
            interactive=True,
        )

    gr.Markdown("### Training views (input)")
    with gr.Row():
        train1 = gr.Image(value=i_train1, label="View 1",
                          interactive=False, height=250)
        train2 = gr.Image(value=i_train2, label="View 2",
                          interactive=False, height=250)
        train3 = gr.Image(value=i_train3, label="View 3",
                          interactive=False, height=250)

    gr.Markdown("### Test view comparison")
    with gr.Row():
        gt0 = gr.Image(value=i_gt0, label="Test 0 — GT",
                       interactive=False, height=250)
        rend0 = gr.Image(value=i_rend0, label="Test 0 — Render",
                         interactive=False, height=250)
        diff0 = gr.Image(value=i_diff0, label="Test 0 — Difference",
                         interactive=False, height=250)
    with gr.Row():
        gt1 = gr.Image(value=i_gt1, label="Test 1 — GT",
                       interactive=False, height=250)
        rend1 = gr.Image(value=i_rend1, label="Test 1 — Render",
                         interactive=False, height=250)
        diff1 = gr.Image(value=i_diff1, label="Test 1 — Difference",
                         interactive=False, height=250)
    with gr.Row():
        gt2 = gr.Image(value=i_gt2, label="Test 2 — GT",
                       interactive=False, height=250)
        rend2 = gr.Image(value=i_rend2, label="Test 2 — Render",
                         interactive=False, height=250)
        diff2 = gr.Image(value=i_diff2, label="Test 2 — Difference",
                         interactive=False, height=250)

    gr.Markdown("### Novel view orbit")
    orbit_img = gr.Image(value=i_orbit,
                         label="Interpolated orbit through all cameras",
                         interactive=False, height=400)

    info_md = gr.Markdown(i_info)

    outputs = [
        train1, train2, train3,
        gt0, rend0, diff0,
        gt1, rend1, diff1,
        gt2, rend2, diff2,
        orbit_img,
        info_md,
    ]

    # Trigger callback khi dropdown/slider thay đổi.
    # KHÔNG có nút "Chạy" — auto-refresh trên change event.
    scene_dd.change(callback, inputs=[scene_dd, amplify_slider], outputs=outputs)
    amplify_slider.change(callback, inputs=[scene_dd, amplify_slider],
                          outputs=outputs)
