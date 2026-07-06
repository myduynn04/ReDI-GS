"""Tab B (Live Render from .ply) cho Gradio demo.

Layout (Phase 3.3):
- Header: dropdown scene + info (N_gauss + view count).
- Block 1 "Novel view rendering":
  * Training views reference (static markdown, để so sánh với novel).
  * 2 button: "Render ALL held-out" và "Render one" (toggle group).
  * Toggle group: dropdown chọn view + button "Render this view".
  * Output = gr.Gallery scrollable với composed row (GT | Render | Diff)
    cho mỗi rendered view. Batch mode tích lũy tất cả 40+ views.
- Block 2 "Free exploration":
  * Slider [0..59] dọc theo ellipse trajectory (60 pose = Tab A GIF).
  * Live render on slider drag.

Design decisions (Phase 3.3, 2026-07-06):
- **PRELOAD tất cả 8 scenes** (metadata + .ply) lúc Gradio startup.
  Startup ~24s một lần, sau đó mọi thứ INSTANT (đổi scene, render).
  VRAM cost: ~300-500MB tổng. GT vẫn lazy per view (không thể tăng tốc).
- **gr.Gallery scrollable** cho Block 1 output:
  * Height 600px với scroll → user cuộn xem đủ views.
  * Progressive yield trong batch mode: tích lũy từng view.
- Explicit render buttons cho Block 1. Block 2 giữ live drag.
- Format dropdown label:
    'View 00 · pos=(0.32, -0.15, 0.87) · yaw 5° pitch 12°'
- Training views reference hiển thị pose 3 train cam để so sánh.

Server-mode required. Local mode ẩn tab kèm markdown.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, List, Tuple

import gradio as gr
import numpy as np
from PIL import Image

from demo.render_utils import (
    load_gaussians_from_ply,
    load_llff_scene,
    render_view,
)
from demo.trajectory_utils import (
    _build_minicam_from_pose,
    build_orbit_trajectory,
)
from demo.populate_cache import (
    CACHE_ROOT,
    CHECKPOINT_ROOT,
    DATA_ROOT,
    RESOLUTION,
    SCENES,
    load_all_colmap_cameras,
)


DEFAULT_SCENE = "fortress"

# 60 pose = MATCH Tab A GIF n_frames → user có thể verify slider position
# X = frame X của GIF, prove GIF không phải video pre-recorded.
N_ELLIPSE_POSES = 60


# ---------------------------------------------------------------------------
# 3-tier caches — lazy load
# ---------------------------------------------------------------------------


# scene → {train_infos, held_out_views_meta, trajectory, n_gauss}. Fast load.
_METADATA_CACHE: dict = {}

# scene → GaussianModel. Slow load (~2-3s). Lazy on first render click.
_GAUSSIANS_CACHE: dict = {}


def _existing_scenes() -> List[str]:
    """Chỉ liệt kê scene có cả .ply checkpoint + LLFF data."""
    ok = []
    for s in SCENES:
        ply = (CHECKPOINT_ROOT / f"A3_seed42_{s}"
               / "point_cloud" / "iteration_10000" / "point_cloud.ply")
        src = DATA_ROOT / s
        if ply.exists() and src.exists():
            ok.append(s)
    return ok


# ---------------------------------------------------------------------------
# Pose helpers — camera world position + yaw/pitch từ C2W rotation
# ---------------------------------------------------------------------------


def _cam_world_center(cam_ns) -> np.ndarray:
    """Vị trí camera world frame (3,) — pattern từ getNerfppNorm."""
    from utils.graphics_utils import getWorld2View2

    R = np.asarray(cam_ns.R, dtype=np.float64)
    T = np.asarray(cam_ns.T, dtype=np.float64).reshape(3)
    W2C = getWorld2View2(R, T)
    C2W = np.linalg.inv(W2C)
    return C2W[:3, 3]


def _yaw_pitch_deg(R: np.ndarray) -> Tuple[float, float]:
    """Yaw + pitch (degrees) từ C2W rotation matrix.

    Convention COLMAP: X right, Y down, Z forward.
    forward_world = R[:, 2] (Z column vì camera Z-forward, R là C2W).
      yaw   = arctan2(forward.x, forward.z)  — âm=trái, dương=phải.
      pitch = arcsin(-forward.y)              — âm=xuống, dương=lên.
    """
    forward = np.asarray(R, dtype=np.float64)[:, 2]
    forward = forward / (np.linalg.norm(forward) + 1e-12)
    yaw = float(np.degrees(np.arctan2(forward[0], forward[2])))
    pitch = float(np.degrees(np.arcsin(-forward[1])))
    return yaw, pitch


# ---------------------------------------------------------------------------
# Held-out view metadata — WITHOUT GT (lazy load GT per view)
# ---------------------------------------------------------------------------


def _load_held_out_metadata(source_path: Path,
                            train_image_names: set,
                            resolution: int = RESOLUTION) -> List[dict]:
    """Load metadata (pose + rgb path) cho held-out views. KHÔNG load GT PIL.

    GT được load lazily bởi `_get_gt_lazy` khi user actually render view.
    Giảm scene metadata load từ ~23s xuống ~100ms.
    """
    from scene.colmap_loader import (
        qvec2rotmat,
        read_extrinsics_binary,
        read_intrinsics_binary,
    )
    from utils.graphics_utils import focal2fov

    ext_path = source_path / "sparse" / "0" / "images.bin"
    intr_path = source_path / "sparse" / "0" / "cameras.bin"
    extrinsics = read_extrinsics_binary(str(ext_path))
    intrinsics = read_intrinsics_binary(str(intr_path))

    images_dir = source_path / "images"
    rgb_map = {p.stem: p for p in images_dir.iterdir()
               if p.suffix.lower() in (".jpg", ".jpeg", ".png")}

    views: List[dict] = []
    for key in sorted(extrinsics.keys()):
        extr = extrinsics[key]
        intr = intrinsics[extr.camera_id]
        image_name = Path(extr.name).stem
        if image_name in train_image_names:
            continue

        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec, dtype=np.float64)

        if intr.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            fx = fy = intr.params[0]
        elif intr.model == "PINHOLE":
            fx, fy = intr.params[0], intr.params[1]
        else:
            continue

        rend_w = intr.width // resolution
        rend_h = intr.height // resolution
        FovX = focal2fov(fx, intr.width)
        FovY = focal2fov(fy, intr.height)

        rgb_path = rgb_map.get(image_name)  # có thể None nếu missing

        cam_ns = SimpleNamespace(
            R=R, T=T,
            FoVx=FovX, FoVy=FovY,
            image_width=rend_w, image_height=rend_h,
            image_name=image_name,
            znear=0.01, zfar=100.0,
            trans=np.array([0.0, 0.0, 0.0]),
            scale=1.0,
        )
        pos = _cam_world_center(cam_ns)
        yaw, pitch = _yaw_pitch_deg(R)

        views.append({
            "cam": cam_ns,
            "rgb_path": rgb_path,           # để lazy load GT sau
            "gt_pil": None,                  # populate khi cần
            "image_name": image_name,
            "image_width": rend_w,
            "image_height": rend_h,
            "position": pos,
            "yaw": yaw,
            "pitch": pitch,
        })

    views.sort(key=lambda v: v["image_name"])
    return views


def _get_gt_lazy(view: dict) -> Image.Image | None:
    """Load GT PIL cho 1 view (lazy). Cache vào view dict sau khi load."""
    if view["gt_pil"] is not None:
        return view["gt_pil"]
    rgb_path = view["rgb_path"]
    if rgb_path is None or not rgb_path.exists():
        return None
    img = Image.open(rgb_path).convert("RGB")
    # BILINEAR (thay LANCZOS): ~10× nhanh hơn, chất lượng vẫn OK cho demo.
    img = img.resize((view["image_width"], view["image_height"]),
                     Image.BILINEAR)
    view["gt_pil"] = img
    return img


# ---------------------------------------------------------------------------
# Scene metadata loader — cameras + trajectory + train ref info
# ---------------------------------------------------------------------------


def _get_metadata(scene: str) -> dict:
    """Lazy load metadata: cameras + train info + trajectory + held-out.

    ~100-500ms tuỳ scene (fern nhanh, horns chậm hơn vì 62 cam).
    KHÔNG load .ply — .ply loaded riêng bởi `_get_gaussians`.
    """
    if scene in _METADATA_CACHE:
        return _METADATA_CACHE[scene]

    print(f"[live] loading metadata for {scene}...")
    t0 = time.time()

    source_path = DATA_ROOT / scene

    # Load Scene chỉ để lấy 3 train cam (n_views=3 + llffhold=8).
    scene_obj = load_llff_scene(str(source_path))
    train_cams = list(scene_obj.getTrainCameras())
    train_names = {c.image_name for c in train_cams}

    # Info pose của 3 train cams (reference cho user so sánh).
    train_infos: List[dict] = []
    for c in train_cams:
        cam_ns = SimpleNamespace(
            R=np.asarray(c.R, dtype=np.float64),
            T=np.asarray(c.T, dtype=np.float64),
        )
        pos = _cam_world_center(cam_ns)
        yaw, pitch = _yaw_pitch_deg(cam_ns.R)
        train_infos.append({
            "image_name": c.image_name,
            "position": pos,
            "yaw": yaw,
            "pitch": pitch,
        })

    # Ellipse trajectory + held-out views metadata (no GT).
    cameras_raw = load_all_colmap_cameras(source_path, resolution=RESOLUTION)
    trajectory = build_orbit_trajectory(cameras_raw, n_frames=N_ELLIPSE_POSES)
    held_out = _load_held_out_metadata(source_path, train_names, RESOLUTION)

    # N_gauss lấy từ Tab A cache (metrics.json) — không cần load .ply.
    n_gauss = 0
    metrics_path = CACHE_ROOT / scene / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
            n_gauss = int(metrics.get("n_gauss", 0))
        except Exception:
            pass

    entry = {
        "source_path": source_path,
        "train_infos": train_infos,
        "held_out": held_out,
        "trajectory": trajectory,
        "n_gauss": n_gauss,
    }
    _METADATA_CACHE[scene] = entry
    print(f"[live] metadata {scene} loaded in {time.time() - t0:.2f}s "
          f"({len(held_out)} held-out views)")
    return entry


def _get_gaussians(scene: str):
    """Lazy load .ply gaussians (~2-3s first time, cache)."""
    if scene in _GAUSSIANS_CACHE:
        return _GAUSSIANS_CACHE[scene]

    print(f"[live] loading .ply for {scene}...")
    t0 = time.time()

    ply_path = (CHECKPOINT_ROOT / f"A3_seed42_{scene}"
                / "point_cloud" / "iteration_10000" / "point_cloud.ply")
    gaussians = load_gaussians_from_ply(str(ply_path))
    _GAUSSIANS_CACHE[scene] = gaussians
    print(f"[live] .ply {scene} loaded in {time.time() - t0:.1f}s")
    return gaussians


def _preload_all_scenes() -> None:
    """Preload metadata + .ply cho TẤT CẢ 8 scenes lúc Gradio startup.

    Phase 3.3: chấp nhận ~24s startup (8 × ~3s .ply) đổi lấy trải nghiệm
    switching scene INSTANT sau đó (không delay). GT vẫn lazy per view.
    VRAM cost tổng ~300-500MB — chấp nhận được với RTX 3090 24GB.
    """
    scenes = _existing_scenes()
    print(f"[live] preloading {len(scenes)} scenes (metadata + .ply)...")
    t0 = time.time()
    for i, scene in enumerate(scenes, 1):
        print(f"[live]   [{i}/{len(scenes)}] {scene}...")
        _get_metadata(scene)
        _get_gaussians(scene)
    print(f"[live] all {len(scenes)} scenes ready in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Compose helper (giống ui_precompute — 1 payload atomic swap)
# ---------------------------------------------------------------------------


_GAP_PX = 10


def _as_rgba(img: Image.Image | None, size: tuple[int, int]) -> np.ndarray:
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
    ref = next((im for im in imgs if im is not None), None)
    if ref is None:
        return None
    size = ref.size
    gap = np.zeros((size[1], _GAP_PX, 4), dtype=np.uint8)
    parts: List[np.ndarray] = []
    for k, im in enumerate(imgs):
        if k:
            parts.append(gap)
        parts.append(_as_rgba(im, size))
    return Image.fromarray(np.concatenate(parts, axis=1))


def _diff_amplify_3(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """L1 mean grayscale diff, amplify ×3."""
    diff = np.abs(pred.astype(np.int32) - gt.astype(np.int32)).mean(axis=2)
    return np.clip(diff * 3, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Format helpers — dropdown labels, train reference markdown
# ---------------------------------------------------------------------------


def _pose_summary(pos: np.ndarray, yaw: float, pitch: float) -> str:
    """Format compact: 'pos=(0.32, -0.15, 0.87) · yaw 5° pitch 12°'."""
    return (f"pos=({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f}) · "
            f"yaw {yaw:+.0f}° pitch {pitch:+.0f}°")


def _format_train_reference(train_infos: List[dict]) -> str:
    """Static markdown hiển thị 3 train cams pose để so sánh với novel."""
    lines = ["**📸 Training views (reference — để so sánh với novel views):**"]
    for info in train_infos:
        lines.append(f"- `{info['image_name']}` · "
                     f"{_pose_summary(info['position'], info['yaw'], info['pitch'])}")
    return "  \n".join(lines)


def _view_dropdown_label(view: dict, idx: int) -> str:
    """Label: 'View 00 · pos=(0.32, -0.15, 0.87) · yaw 5° pitch 12°'."""
    return (f"View {idx:2d} · "
            f"{_pose_summary(view['position'], view['yaw'], view['pitch'])}")


def _view_idx_from_label(label: str) -> int:
    """Extract idx từ 'View XX · pos...'"""
    try:
        return int(label.split("·")[0].split()[1])
    except (IndexError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Render helpers — gọi model live
# ---------------------------------------------------------------------------


def _render_view_composed(gaussians, view: dict) -> Tuple[Image.Image, float]:
    """Render 1 view + compose GT|Render|Diff. Trả về (composed_row, dt_ms).

    Caller quyết định format caption / info từ view metadata + dt_ms.
    """
    cam_ns = view["cam"]
    minicam = _build_minicam_from_pose(cam_ns.R, cam_ns.T, cam_ns)

    t0 = time.time()
    rend = render_view(gaussians, minicam)         # HxWx3 uint8
    dt_ms = (time.time() - t0) * 1000

    gt_pil = _get_gt_lazy(view)                    # lazy load GT
    if gt_pil is None:
        row = Image.fromarray(rend)
    else:
        gt_arr = np.asarray(gt_pil, dtype=np.uint8)
        diff = _diff_amplify_3(rend, gt_arr)
        row = _hstack_images([
            gt_pil,
            Image.fromarray(rend),
            Image.fromarray(diff),
        ])
    return row, dt_ms


def _view_caption(view: dict, dt_ms: float, prefix: str = "") -> str:
    """Caption ngắn cho Gallery item."""
    fps = 1000.0 / dt_ms if dt_ms > 0 else 0.0
    return (f"{prefix}{view['image_name']} · "
            f"yaw {view['yaw']:+.0f}° pitch {view['pitch']:+.0f}° · "
            f"{dt_ms:.0f}ms ({fps:.0f} FPS)")


def _view_info_md(view: dict, dt_ms: float) -> str:
    """Full markdown info hiển thị dưới Gallery."""
    fps = 1000.0 / dt_ms if dt_ms > 0 else 0.0
    return (
        f"**View**: `{view['image_name']}` · "
        f"**pos**: ({view['position'][0]:+.2f}, "
        f"{view['position'][1]:+.2f}, {view['position'][2]:+.2f}) · "
        f"**yaw** {view['yaw']:+.1f}° · **pitch** {view['pitch']:+.1f}° · "
        f"**Render**: {dt_ms:.0f}ms ({fps:.0f} FPS)"
    )


# ---------------------------------------------------------------------------
# Callbacks — Gradio events
# ---------------------------------------------------------------------------


def cb_scene_change(scene: str) -> Tuple:
    """Đổi scene → load metadata (fast) + update UI. KHÔNG load .ply, KHÔNG render.

    Returns:
        (header_info_md, train_reference_md, view_dd_update,
         select_group_visibility, block1_output_clear, block1_info_clear,
         block2_output_clear, block2_info_clear)
    """
    meta = _get_metadata(scene)

    header = (f"**N_gauss**: {meta['n_gauss']:,} · "
              f"**Held-out views**: {len(meta['held_out'])} · "
              f"**Ellipse poses**: {N_ELLIPSE_POSES}")
    train_ref = _format_train_reference(meta["train_infos"])
    view_choices = [_view_dropdown_label(v, i)
                    for i, v in enumerate(meta["held_out"])]
    default_view = view_choices[0] if view_choices else ""
    view_dd_update = gr.update(choices=view_choices, value=default_view)

    # Reset — ẩn select group, clear outputs.
    select_hide = gr.update(visible=False)
    return (header, train_ref, view_dd_update, select_hide,
            [], "", None, "")


def cb_show_select_group():
    """Click 'Render one' → hiện dropdown + button 'Render this view'.

    KHÔNG render. Chỉ toggle visibility.
    """
    return gr.update(visible=True)


def cb_render_this_view(scene: str, view_label: str
                        ) -> Tuple[List, str]:
    """Render 1 view do user chọn. Trả về Gallery list-with-1-item + info."""
    meta = _get_metadata(scene)
    idx = _view_idx_from_label(view_label)
    if idx >= len(meta["held_out"]):
        return [], "N/A"
    gaussians = _get_gaussians(scene)   # instant nếu đã preload
    view = meta["held_out"][idx]

    row, dt_ms = _render_view_composed(gaussians, view)
    gallery_item = (row, _view_caption(view, dt_ms))
    return [gallery_item], _view_info_md(view, dt_ms)


def cb_render_all(scene: str) -> Iterator[Tuple]:
    """Cycle qua TẤT CẢ held-out views, tích lũy vào Gallery progressive.

    Mỗi yield gửi FULL list rendered so far → Gallery grows từng item →
    user scroll xem toàn bộ khi xong. Info markdown hiện progress N/M.
    """
    meta = _get_metadata(scene)
    views = meta["held_out"]
    n = len(views)
    gaussians = _get_gaussians(scene)

    gallery: List[Tuple[Image.Image, str]] = []
    for i, view in enumerate(views):
        row, dt_ms = _render_view_composed(gaussians, view)
        caption = _view_caption(view, dt_ms, prefix=f"{i + 1:02d}/{n} · ")
        gallery.append((row, caption))
        progress = (f"**Rendering...** {i + 1}/{n} · "
                    f"`{view['image_name']}`")
        yield gallery, progress

    yield gallery, (f"**Done** · Rendered {n} held-out views. "
                    f"Scroll gallery ↕ để xem tất cả.")


def cb_render_free(scene: str, pos_idx: float
                   ) -> Tuple[Image.Image, str]:
    """Render 1 pose trên ellipse trajectory. Load .ply nếu chưa."""
    meta = _get_metadata(scene)
    gaussians = _get_gaussians(scene)   # ~2-3s first time (Block 2 first drag)
    idx = int(pos_idx)
    idx = max(0, min(idx, len(meta["trajectory"]) - 1))
    minicam = meta["trajectory"][idx]

    t0 = time.time()
    rend = render_view(gaussians, minicam)
    dt_ms = (time.time() - t0) * 1000
    fps = 1000.0 / dt_ms if dt_ms > 0 else 0.0

    theta_deg = (idx / max(N_ELLIPSE_POSES, 1)) * 360.0
    wvt = minicam.world_view_transform.cpu().numpy()
    c2w = np.linalg.inv(wvt.T)
    cam_pos = c2w[:3, 3]

    info = (
        f"**Position on ellipse**: {idx}/{N_ELLIPSE_POSES - 1} · "
        f"**θ ≈ {theta_deg:.0f}°** · "
        f"**Camera world**: ({cam_pos[0]:+.2f}, {cam_pos[1]:+.2f}, "
        f"{cam_pos[2]:+.2f}) · "
        f"**Render**: {dt_ms:.0f}ms ({fps:.0f} FPS)"
    )
    return Image.fromarray(rend), info


# ---------------------------------------------------------------------------
# Build Tab
# ---------------------------------------------------------------------------


_CSS = """
<style>
.live-card {border: 1px solid var(--border-color-primary) !important;
            border-radius: 12px !important;
            padding: 14px 16px !important;
            background: var(--background-fill-secondary) !important;
            margin-bottom: 8px;}
.live-card h4 {margin: 0 0 8px 0;}
.live-note {color: var(--body-text-color-subdued) !important;
            font-size: 13px !important;
            font-style: italic;
            margin: 4px 0 8px 0 !important;}
.live-headers {display: flex; margin: 8px 0 6px;
               color: var(--body-text-color-subdued);
               font-size: 13px; font-weight: 600; text-align: center;}
.live-headers span {flex: 1;}
.train-ref {padding: 10px 14px !important;
            background: var(--background-fill-primary) !important;
            border-radius: 8px !important;
            margin: 6px 0 !important;
            font-family: var(--font-mono) !important;
            font-size: 13px !important;}
.scene-info-live {text-align: right;}
.scene-info-live p {margin: 6px 0 0 0;}
.select-group {padding: 12px !important;
               background: var(--background-fill-primary) !important;
               border-radius: 8px !important;
               margin: 8px 0 !important;}

/* Gallery cho composed row (aspect ~4:1, wide+thấp) — Phase 3.3b fix.
   Bỏ padding TRÊN/DƯỚI mặc định của gr.Gallery (object-fit contain
   centered gây feeling "lệch dưới"). Ép item container match aspect
   của composed row: 1532×378 ≈ 4:1 → aspect-ratio: 1532/378. */
.render-gallery .grid-wrap {padding: 4px !important;}
.render-gallery .thumbnail-item {
    aspect-ratio: 1532 / 378 !important;
    height: auto !important;
    min-height: 0 !important;
    padding: 0 !important;
    margin: 4px 0 !important;
}
.render-gallery .thumbnail-item img {
    object-fit: contain !important;
    width: 100% !important;
    height: 100% !important;
}
</style>
"""


def build_liverender_tab(local_mode: bool = False) -> None:
    """Build layout Tab B."""
    if local_mode:
        gr.Markdown(
            "### ⚠️ Live Render — server GPU required\n\n"
            "This tab is unavailable in local backup mode. Only the "
            "**Pre-computed** tab works offline (reads cached PNG/GIF).\n\n"
            "To use live rendering, run the demo on the server:\n"
            "```bash\n"
            "conda activate gradio_demo\n"
            "bash demo/run_server.sh\n"
            "```"
        )
        return

    scenes = _existing_scenes()
    if not scenes:
        gr.Markdown(
            "⚠️ No `.ply` checkpoints found in "
            "`output/p28_crs_boost/tau65/`. Cannot start Live Render."
        )
        return

    default_scene = DEFAULT_SCENE if DEFAULT_SCENE in scenes else scenes[0]

    # Phase 3.3: PRELOAD tất cả 8 scenes (metadata + .ply) upfront.
    # Startup ~24s, sau đó đổi scene INSTANT + render INSTANT.
    _preload_all_scenes()

    init_meta = _get_metadata(default_scene)
    init_header = (f"**N_gauss**: {init_meta['n_gauss']:,} · "
                   f"**Held-out views**: {len(init_meta['held_out'])} · "
                   f"**Ellipse poses**: {N_ELLIPSE_POSES}")
    init_train_ref = _format_train_reference(init_meta["train_infos"])
    init_view_choices = [_view_dropdown_label(v, i)
                         for i, v in enumerate(init_meta["held_out"])]
    init_default_view = init_view_choices[0] if init_view_choices else ""

    gr.HTML(_CSS)

    # --- Header ---------------------------------------------------------
    with gr.Row():
        with gr.Column(scale=1, min_width=220):
            scene_dd = gr.Dropdown(
                choices=scenes, value=default_scene,
                label="Scene", interactive=True,
            )
        with gr.Column(scale=2, min_width=220):
            scene_info = gr.Markdown(
                init_header, elem_classes=["scene-info-live"],
            )

    # --- Block 1: Novel view rendering ----------------------------------
    with gr.Group(elem_classes=["live-card"]):
        gr.Markdown("#### 🎬 Novel view rendering")
        gr.Markdown(
            "Render tại camera pose **KHÔNG dùng để train**. LLFF có ~20-62 "
            "cam, trừ 3 dùng làm training input → held-out views (bao gồm "
            "test set + view khác). Rendering là **explicit** — chỉ chạy "
            "khi click button.",
            elem_classes=["live-note"],
        )

        # Training views reference — static markdown để so sánh pose.
        train_ref_md = gr.Markdown(
            init_train_ref, elem_classes=["train-ref"],
        )

        with gr.Row():
            btn_render_all = gr.Button(
                "🎬 Render ALL held-out views", variant="primary",
            )
            btn_render_one = gr.Button(
                "🎬 Render one (chọn view)", variant="secondary",
            )

        # Select group: ẨN ban đầu, chỉ hiện khi user click "Render one".
        with gr.Group(elem_classes=["select-group"],
                      visible=False) as select_group:
            view_dd = gr.Dropdown(
                choices=init_view_choices,
                value=init_default_view,
                label="Choose held-out view",
                interactive=True,
            )
            btn_render_this = gr.Button(
                "🎬 Render this view", variant="primary",
            )

        # Column headers cho output.
        gr.HTML(
            '<div class="live-headers">'
            '<span>Ground truth</span>'
            '<span>Render (live)</span>'
            '<span>Difference ×3</span>'
            '</div>'
        )

        # Output — Gallery scrollable, TRỐNG ban đầu (không auto-render).
        # Phase 3.3: Gallery thay gr.Image → tích lũy tất cả rendered views,
        # user cuộn ↕ xem toàn bộ khi batch "Render ALL" xong.
        # Phase 3.3b: bỏ `height=600` fixed và `object_fit="contain"` gây
        # padding trên/dưới ("lệch dưới"). Aspect ratio ép qua CSS
        # `.render-gallery` để item match 1532×378 composed image.
        view_gallery = gr.Gallery(
            value=[],
            show_label=False,
            columns=1,
            object_fit="scale-down",
            preview=True,
            allow_preview=True,
            elem_classes=["render-gallery"],
        )
        view_info = gr.Markdown("", elem_classes=["live-note"])

    # --- Block 2: Free exploration --------------------------------------
    with gr.Group(elem_classes=["live-card"]):
        gr.Markdown("#### 🎨 Free exploration (novel view)")
        gr.Markdown(
            "Kéo slider để camera đi dọc quỹ đạo ellipse. Novel view — "
            "KHÔNG có ảnh thật để so sánh (góc chưa chụp). Slider vị trí "
            "X = frame X của GIF Tab A → verify GIF là render thật. "
            "**Lần đầu kéo slider sẽ chờ ~2-3s** load .ply, các lần sau "
            "instant (~50ms/tick).",
            elem_classes=["live-note"],
        )

        pos_slider = gr.Slider(
            minimum=0, maximum=N_ELLIPSE_POSES - 1,
            value=0, step=1,
            label=f"Ellipse position (0 .. {N_ELLIPSE_POSES - 1})",
            interactive=True,
        )
        # Output — TRỐNG ban đầu (không auto-render, chờ user kéo slider).
        free_img = gr.Image(
            show_label=False, container=False, interactive=False,
            height=400,
        )
        free_info = gr.Markdown("", elem_classes=["live-note"])

    # --- Events ---------------------------------------------------------
    # Scene change: load metadata + update UI. NO render.
    scene_dd.change(
        cb_scene_change,
        inputs=[scene_dd],
        outputs=[scene_info, train_ref_md, view_dd, select_group,
                 view_gallery, view_info, free_img, free_info],
    )

    # "Render one" click: chỉ toggle visibility của select group.
    btn_render_one.click(
        cb_show_select_group,
        outputs=[select_group],
    )

    # "Render this view" click: render 1 view đã chọn → Gallery 1 item.
    btn_render_this.click(
        cb_render_this_view,
        inputs=[scene_dd, view_dd],
        outputs=[view_gallery, view_info],
    )

    # "Render ALL" click: cycle progressive tích lũy Gallery.
    btn_render_all.click(
        cb_render_all,
        inputs=[scene_dd],
        outputs=[view_gallery, view_info],
    )

    # Slider drag: live render (Block 2 KHÔNG explicit button per user).
    pos_slider.change(
        cb_render_free,
        inputs=[scene_dd, pos_slider],
        outputs=[free_img, free_info],
    )
