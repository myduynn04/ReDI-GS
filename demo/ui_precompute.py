"""Tab 1 (Pre-computed viewer) cho Gradio demo.

Đọc content pre-generated từ `demo/cache/<scene>/` (do
`populate_cache.py` sinh). Không render live, chỉ serve file từ disk +
amplify diff map trong memory theo slider user.

Layout (Phase 2.6 — card design):
- Header: dropdown scene (trái) + info N_gauss / training time (phải).
- Card "Training views": 1 strip = 3 train view compose ngang.
- Card "Held-out test views": slider amplification + 3 row, mỗi row =
  1 ảnh compose (GT | Render | Diff amplified).
- Card "Novel view orbit": GIF autoplay.

Không có tier ranking, không hiển thị PSNR/SSIM/LPIPS number
(theo yêu cầu user).

Phase 2.6 fix (2026-07-05) — ATOMIC BLOCK UPDATE + CARD LAYOUT:
    Vấn đề: đổi scene thì ảnh mới thay ảnh cũ NHỎ GIỌT từng ô một,
    UI trộn lẫn scene cũ/mới trong ~0.5-1s.
    Root cause: preload đã làm data instant, nhưng 14 output riêng lẻ →
    Gradio serialize MỖI gr.Image thành 1 temp file + 1 HTTP fetch riêng;
    browser hoàn tất 13 fetch tại các thời điểm khác nhau (GIF chậm
    nhất). Slider `.change` còn bắn lại CẢ 14 output (gồm GIF) dù chỉ
    diff thay đổi, và bắn liên tục trong lúc kéo.
    Fix:
      1. Compose theo KHỐI: 3 train view → 1 strip; mỗi test view →
         1 row (GT|Render|Diff). 13 ảnh → 5 payload; mỗi khối swap
         nguyên khối, hết cảnh so le trong khối.
      2. Tách callback theo phạm vi ảnh hưởng: đổi scene → update 6
         output; slider → chỉ 3 row (không đụng strip/GIF), dùng
         `.release` để không re-render khi đang kéo.
      3. Card layout: tách hẳn khối Training / Test / Orbit thành 3
         card riêng + CSS inject (không cần sửa app.py).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import gradio as gr
import numpy as np
from PIL import Image


CACHE_ROOT_DEFAULT = Path("demo/cache")

DEFAULT_SCENE = "fortress"

SCENES = ["fern", "flower", "fortress", "horns", "leaves", "orchids",
          "room", "trex"]

# Phase 2.5b: preload TẤT CẢ scene vào memory lúc startup để chuyển scene
# instant (~10ms). Chi phí: ~45MB memory, ~2s startup thêm.
# Dict: scene_name → preloaded data dict.
_PRELOAD_CACHE: dict = {}

# Khe giữa các ảnh trong composed strip/row. Alpha = 0 → lộ nền theme
# (đẹp cả light lẫn dark mode).
_GAP_PX = 10


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
# Compose — ghép nhiều ảnh thành 1 khối để frontend swap nguyên tử
# ---------------------------------------------------------------------------


def _as_rgba(img: Image.Image | None, size: tuple[int, int]) -> np.ndarray:
    """PIL bất kỳ → ndarray RGBA (H,W,4). None → placeholder xám đục."""
    if img is None:
        w, h = size
        arr = np.zeros((h, w, 4), dtype=np.uint8)
        arr[..., :3] = 45
        arr[..., 3] = 255
        return arr
    arr = np.asarray(img.convert("RGBA"), dtype=np.uint8)
    if (arr.shape[1], arr.shape[0]) != size:
        arr = np.asarray(
            Image.fromarray(arr).resize(size, Image.LANCZOS), dtype=np.uint8)
    return arr


def _hstack_images(imgs: List[Image.Image | None]) -> Image.Image | None:
    """Ghép ngang các ảnh cùng cỡ thành 1 ảnh RGBA, khe trong suốt.

    Lý do compose: 1 khối = 1 payload = 1 HTTP fetch → cả khối xuất hiện
    cùng lúc, triệt tiêu hiện tượng ảnh mới thay ảnh cũ so le từng ô.
    Chi phí: vài np.concatenate trên ảnh preloaded ≈ sub-ms.
    """
    ref = next((im for im in imgs if im is not None), None)
    if ref is None:
        return None
    size = ref.size                                   # (W, H)
    gap = np.zeros((size[1], _GAP_PX, 4), dtype=np.uint8)   # alpha = 0
    parts: List[np.ndarray] = []
    for k, im in enumerate(imgs):
        if k:
            parts.append(gap)
        parts.append(_as_rgba(im, size))
    return Image.fromarray(np.concatenate(parts, axis=1))


def _compose_test_rows(data: dict, amplify: float) -> List[Image.Image | None]:
    """3 row test view, mỗi row = compose(GT | Render | Diff × amplify)."""
    rows: List[Image.Image | None] = []
    for gt, rend, diff in zip(data["gts"], data["renders"],
                              data["diffs_raw"]):
        if gt is None and rend is None and diff is None:
            rows.append(None)
            continue
        d_amp = _amplify_diff(diff, amplify) if diff is not None else None
        rows.append(_hstack_images([gt, rend, d_amp]))
    return rows


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

    Trả về dict với keys: train_strip (1 PIL compose), gts (3 PIL),
    renders (3 PIL), diffs_raw (3 PIL L-mode), orbit (str path),
    info_md (str).
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

    n_gauss = metrics.get("n_gauss", 0)
    training_time = metrics.get("training_time_str", "unknown")
    info_md = (
        f"**Gaussians**: {n_gauss:,} · "
        f"**Training time**: {training_time}"
    )

    return {
        # Train strip compose sẵn 1 lần lúc preload (static theo scene).
        "train_strip": _hstack_images(trains),
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


def _get_scene(scene: str, cache_root: Path) -> dict | None:
    """Đọc scene từ preload dict; fallback nạp on-demand nếu miss."""
    data = _PRELOAD_CACHE.get(scene)
    if data is None:
        data = _preload_scene(scene, cache_root)
        if data is not None:
            _PRELOAD_CACHE[scene] = data
    return data


# ---------------------------------------------------------------------------
# Callbacks — tách theo phạm vi ảnh hưởng
# ---------------------------------------------------------------------------


def build_scene_callbacks(cache_root: Path):
    """Trả về (cb_scene, cb_amplify).

    cb_scene   — đổi scene:  (strip, row0, row1, row2, orbit, info_md).
    cb_amplify — thả slider: (row0, row1, row2) — KHÔNG đụng strip/GIF,
                 nên kéo slider không bao giờ refetch payload nặng.
    """

    def cb_scene(scene: str, amplify: float):
        data = _get_scene(scene, cache_root)
        if data is None:
            return (None, None, None, None, None, "")
        rows = _compose_test_rows(data, amplify)
        return (data["train_strip"], rows[0], rows[1], rows[2],
                data["orbit"], data["info_md"])

    def cb_amplify(scene: str, amplify: float):
        data = _get_scene(scene, cache_root)
        if data is None:
            return (None, None, None)
        rows = _compose_test_rows(data, amplify)
        return (rows[0], rows[1], rows[2])

    return cb_scene, cb_amplify


# ---------------------------------------------------------------------------
# Build Tab
# ---------------------------------------------------------------------------

# CSS inject qua gr.HTML → tự chứa trong tab, không cần sửa gr.Blocks(css=)
# ở app.py. Dùng CSS var của Gradio theme để tương thích light/dark.
_CSS = """
<style>
.demo-card {border: 1px solid var(--border-color-primary) !important;
            border-radius: 12px !important;
            padding: 14px 16px !important;
            background: var(--background-fill-secondary) !important;
            margin-bottom: 4px;}
.demo-card .demo-img img {border-radius: 8px;}
.demo-card h4 {margin: 0 0 6px 0;}
.col-headers {display: flex; margin: 2px 0 8px;
              color: var(--body-text-color-subdued);
              font-size: 13px; font-weight: 600; text-align: center;}
.col-headers span {flex: 1;}
.scene-info {text-align: right;}
.scene-info p {margin: 6px 0 0 0;}
</style>
"""


def build_precompute_tab(cache_root: Path = CACHE_ROOT_DEFAULT) -> None:
    """Build layout Tab 1 vào current gr.Blocks context."""
    scenes_available = _existing_scenes(cache_root)
    if not scenes_available:
        gr.Markdown(
            f"⚠️ No cache found in `{cache_root}`. "
            f"Run `python -m demo.populate_cache` first to generate content."
        )
        return

    # Preload TẤT CẢ scene vào memory lúc startup (~2s cho 8 scene).
    _preload_all(scenes_available, cache_root)

    default_scene = DEFAULT_SCENE if DEFAULT_SCENE in scenes_available \
        else scenes_available[0]

    cb_scene, cb_amplify = build_scene_callbacks(cache_root)

    # Compute initial content NGAY để bake vào `value=` lúc construct widget.
    # Gradio 4: gán widget.value = X sau khi tạo sẽ KHÔNG sync frontend.
    (i_strip, i_row0, i_row1, i_row2, i_orbit, i_info) = cb_scene(
        default_scene, 1.0)

    gr.HTML(_CSS)

    with gr.Row():
        with gr.Column(scale=1, min_width=220):
            scene_dd = gr.Dropdown(
                choices=scenes_available,
                value=default_scene,
                label="Scene",
                interactive=True,
            )
        with gr.Column(scale=2, min_width=220):
            info_md = gr.Markdown(i_info, elem_classes=["scene-info"])

    # --- Card 1: Training input --------------------------------------
    with gr.Group(elem_classes=["demo-card"]):
        gr.Markdown("#### Training views — 3-view input")
        gr.HTML('<div class="col-headers">'
                '<span>View 1</span><span>View 2</span><span>View 3</span>'
                '</div>')
        train_strip = gr.Image(
            value=i_strip, show_label=False, container=False,
            interactive=False, elem_classes=["demo-img"],
        )

    # --- Card 2: Held-out test views ----------------------------------
    with gr.Group(elem_classes=["demo-card"]):
        with gr.Row():
            with gr.Column(scale=2, min_width=200):
                gr.Markdown("#### Held-out test views")
            with gr.Column(scale=1, min_width=260):
                amplify_slider = gr.Slider(
                    minimum=1.0, maximum=10.0, value=1.0, step=0.5,
                    label="Difference amplification",
                    interactive=True,
                )
        gr.HTML('<div class="col-headers">'
                '<span>Ground truth</span><span>Render</span>'
                '<span>Difference (amplified)</span></div>')
        row0 = gr.Image(value=i_row0, show_label=False, container=False,
                        interactive=False, elem_classes=["demo-img"])
        row1 = gr.Image(value=i_row1, show_label=False, container=False,
                        interactive=False, elem_classes=["demo-img"])
        row2 = gr.Image(value=i_row2, show_label=False, container=False,
                        interactive=False, elem_classes=["demo-img"])

    # --- Card 3: Novel view orbit --------------------------------------
    with gr.Group(elem_classes=["demo-card"]):
        gr.Markdown("#### Novel view orbit — seamless ellipse loop")
        orbit_img = gr.Image(
            value=i_orbit, show_label=False, container=False,
            interactive=False, height=400, elem_classes=["demo-img"],
        )

    # --- Events ---------------------------------------------------------
    # Đổi scene: update cả 6 output (5 payload ảnh + 1 markdown inline).
    scene_dd.change(
        cb_scene,
        inputs=[scene_dd, amplify_slider],
        outputs=[train_strip, row0, row1, row2, orbit_img, info_md],
    )
    # Slider: `.release` chỉ bắn khi THẢ tay (không re-render lúc kéo),
    # và chỉ update 3 row — strip/GIF không bị refetch.
    amplify_slider.release(
        cb_amplify,
        inputs=[scene_dd, amplify_slider],
        outputs=[row0, row1, row2],
    )