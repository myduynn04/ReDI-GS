"""Tab C (Live Training) cho Gradio demo.

User click "Start Training" → subprocess train.py chạy từ 0 iter tới 10000,
Tab C poll checkpoint folders mỗi 5s và render latest .ply lên UI.

Design (Phase 4):
- Recipe: A3-TRIM 8-module + tau=0.65 (Phase 28 tau65 cell) + seed 42.
- Poll interval: 5s (via gr.Timer).
- Save/test iterations: 1000, 2000, ..., 10000 (10 checkpoints).
- Live render: latest checkpoint (thay cũ mỗi poll).
- End state: timelapse GIF từ 10 checkpoint renders + final metrics.
- Auto-cleanup: xóa output/live_demos/{scene}_{ts} sau khi complete
  hoặc user click "Cleanup".
- Chỉ 1 training tại 1 thời điểm (global session state).

Server-mode required. Local mode ẩn tab.
"""

from __future__ import annotations

import atexit
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple

import gradio as gr
import imageio.v2 as imageio
import numpy as np
from PIL import Image

from demo.render_utils import (
    camera_gt_image,
    load_gaussians_from_ply,
    load_llff_scene,
    render_view,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


DEFAULT_SCENE = "fortress"

SCENES = ["fern", "flower", "fortress", "horns", "leaves", "orchids",
          "room", "trex"]

# Path REPO_ROOT — subprocess phải chạy từ đây.
REPO_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_ROOT = REPO_ROOT / "output" / "live_demos"
LOG_ROOT = REPO_ROOT / "logs" / "live_demos"

# Poll interval (seconds).
# Phase 4c: giảm 5s → 1s cho progress bar UI smooth. Callback cb_poll rất
# nhẹ khi không có checkpoint mới (chỉ format status text), safe 5×/s.
# Rendering checkpoint chỉ xảy ra mỗi ~25s (khi có checkpoint mới) — không
# bị load thêm.
POLL_INTERVAL_S = 1

# Iterations save/test — 10 checkpoints, 1000..10000.
CHECKPOINT_ITERS = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000,
                    9000, 10000]

# Reference wall-clock từ HANDOFF doc — hiển thị cho user.
REFERENCE_TIMES = {
    "fortress": "4m 43s (25.55 dB)",
    "room": "3m 55s (23.15 dB)",
    "trex": "5m 05s (23.70 dB)",
    "horns": "5m 23s (21.05 dB)",
    "orchids": "5m 30s (17.48 dB)",
    "fern": "5m 55s (23.88 dB)",
    "flower": "7m 06s (21.55 dB)",
    "leaves": "5m 45s (19.35 dB)",
}


# ---------------------------------------------------------------------------
# Global training session state (chỉ 1 tại 1 thời điểm)
# ---------------------------------------------------------------------------


_ACTIVE_SESSION: Optional["TrainingSession"] = None


# ---------------------------------------------------------------------------
# Signal handlers — dọn subprocess khi Gradio bị Ctrl+C hoặc kill
# ---------------------------------------------------------------------------


_HANDLERS_REGISTERED = False


def _cleanup_on_exit() -> None:
    """Kill active training subprocess trước khi Gradio thoát.

    Vì subprocess.Popen dùng `preexec_fn=os.setsid` → train.py trong
    process group riêng, Ctrl+C tới Gradio KHÔNG lan sang train.py.
    Handler này bridge signal Gradio → SIGTERM train subprocess group.
    """
    global _ACTIVE_SESSION
    if _ACTIVE_SESSION is not None:
        session = _ACTIVE_SESSION
        _ACTIVE_SESSION = None
        try:
            print(f"\n[train] cleanup: killing active session "
                  f"{session.scene}...")
            session.stop()
            session.cleanup()   # xóa output/live_demos/{scene}_{ts}/
        except Exception as e:
            print(f"[train] cleanup error: {e}")


def _signal_handler(sig, frame):
    """SIGINT (Ctrl+C) / SIGTERM (kill) → cleanup + exit."""
    _cleanup_on_exit()
    # Re-raise default behavior để Gradio exit đúng.
    signal.signal(sig, signal.SIG_DFL)
    os.kill(os.getpid(), sig)


def _register_handlers() -> None:
    """Register cleanup handlers 1 lần lúc first tab build."""
    global _HANDLERS_REGISTERED
    if _HANDLERS_REGISTERED:
        return
    atexit.register(_cleanup_on_exit)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    _HANDLERS_REGISTERED = True
    print("[train] signal handlers registered — Ctrl+C sẽ cleanup train subprocess")


# ---------------------------------------------------------------------------
# Train command builder — recipe tau65 từ scripts/p28_crs_boost.sh
# ---------------------------------------------------------------------------


def _build_train_command(scene: str, output_dir: Path,
                         seed: int = 42, gpu: int = 0) -> List[str]:
    """Recipe tau65 (Phase 28 CRS boost, cell tau65). MATCH training thật."""
    save_iters_str = [str(i) for i in CHECKPOINT_ITERS]

    return [
        "python", "train.py",
        "-s", f"data/nerf_llff_data/{scene}",
        "-m", str(output_dir),
        # Protocol Phase 22
        "--eval",
        "-r", "8",
        "--n_views", "3",
        "--random_background",
        "--iterations", "10000",
        "--densify_until_iter", "5000",
        "--densify_grad_threshold", "0.0005",
        "--gaussiansN", "1",
        "--sample_pseudo_interval", "1",
        "--start_sample_pseudo", "500",
        # Test + save checkpoints tại 10 iter → poll render được
        "--test_iterations", *save_iters_str,
        "--save_iterations", *save_iters_str,
        # Depth prior config
        "--use_depth_prior",
        "--dav2_path", "../Depth-Anything-V2",
        "--crs_ema_decay", "0.3",
        "--crs_update_interval", "100",
        # D_cycle
        "--use_d_cycle",
        "--d_cycle_warmup", "1000",
        "--d_cycle_sigma", "5.0",
        "--d_cycle_update_freq", "100",
        # SH freeze — tau=0.65 (Phase 28 tau65)
        "--use_crs_modulated_sh_freeze",
        "--crs_freeze_start", "1000",
        "--crs_freeze_tau", "0.65",
        # SH reliability
        "--use_sh_reliability",
        "--sh_stability_warmup", "1000",
        "--sh_stability_ema_beta", "0.95",
        "--crs_w_s", "0.33",
        # DropAnSH
        "--use_dropansh",
        "--dropansh_pa", "0.02",
        "--dropansh_psh", "0.2",
        # Opacity decay
        "--use_opacity_decay",
        "--opacity_decay_factor", "0.999",
        # LFCF + AbsGS (Phase 13 A3)
        "--use_lfcf",
        "--lfcf_init_scaling_max", "1.5",
        "--lfcf_init_scaling_min", "1.0",
        "--lfcf_last_scaling_max", "1.0",
        "--lfcf_pow", "1.0",
        "--lfcf_splitting_ub", "1.0",
        "--lfcf_interval_times", "2",
        "--lfcf_tolerance", "1e-5",
        "--lfcf_diffscale", "True",
        "--absdensify",
        # Seed
        "--seed", str(seed),
    ]


# ---------------------------------------------------------------------------
# TrainingSession — manages 1 training run
# ---------------------------------------------------------------------------


class TrainingSession:
    """1 training subprocess + polling state."""

    def __init__(self, scene: str, gpu: int = 0):
        self.scene = scene
        self.gpu = gpu
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.output_dir = OUTPUT_ROOT / f"{scene}_{self.timestamp}"
        self.log_path = LOG_ROOT / f"{scene}_{self.timestamp}.log"

        self.process: Optional[subprocess.Popen] = None
        self.log_file = None
        self._reader_thread: Optional[threading.Thread] = None
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None   # freeze elapsed khi done
        self.completed = False
        self.failed = False
        self.exit_code: Optional[int] = None
        # Cờ báo đã sent tất cả output cuối cùng (timelapse + orbit GIF)
        # sau khi complete → poll tiếp theo skip để không flicker.
        self.outputs_finalized: bool = False

        # Fine-grained iteration counter — parse từ tqdm output train.py
        # (mỗi ~10 iter). Progress bar dùng cái này để cập nhật smooth.
        # Khác với rendered_iters (chỉ tăng khi có checkpoint = mỗi 1000).
        self.current_iter: int = 0

        # Checkpoint state
        self.rendered_iters: set = set()
        self.checkpoint_renders: List[Tuple[int, Image.Image]] = []
        # (iter, composed_row_PIL)
        self.psnr_history: List[Tuple[int, float]] = []  # (iter, psnr)

        # Test camera + GT (loaded once when first checkpoint arrives)
        self.test_cam = None
        self.gt_pil: Optional[Image.Image] = None

    def start(self) -> None:
        """Launch train.py subprocess non-blocking.

        stdout đi qua PIPE, background thread đọc line-by-line rồi TEE tới
        cả log file (để parse PSNR) và terminal (để user monitor live).
        """
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        cmd = _build_train_command(self.scene, self.output_dir,
                                   gpu=self.gpu)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu)

        self.log_file = open(self.log_path, "w", buffering=1)
        self.log_file.write(f"# Live Training session\n")
        self.log_file.write(f"# Scene: {self.scene}\n")
        self.log_file.write(f"# Output: {self.output_dir}\n")
        self.log_file.write(f"# CMD: {' '.join(cmd)}\n\n")
        self.log_file.flush()

        self.process = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,            # decode stdout thành str line-by-line
            bufsize=1,            # line-buffered
            env=env,
            preexec_fn=os.setsid,
        )
        self.started_at = time.time()

        # Reader thread: TEE stdout → log file + terminal (Gradio process).
        prefix = f"[train:{self.scene}]"
        self._reader_thread = threading.Thread(
            target=self._forward_output,
            args=(prefix,),
            daemon=True,
        )
        self._reader_thread.start()

        print(f"[train] started PID={self.process.pid} for {self.scene}")
        print(f"[train] tee log to terminal + {self.log_path}")

    def _forward_output(self, prefix: str) -> None:
        """Background thread: đọc subprocess stdout → log + terminal.

        Đồng thời parse tqdm progress bar `X/10000 [` để update
        `current_iter` cho progress bar UI.
        """
        # Regex match tqdm output: "1500/10000 [00:36<..." (X/10000 + open bracket
        # + time). Safe hơn `X/10000` đơn giản để tránh false match nếu train.py
        # in `10000/10000` ở dòng khác (final metric summary chẳng hạn).
        progress_rx = re.compile(r"(\d+)/10000\s*\[\d+:")

        try:
            for line in self.process.stdout:
                # Write to log file (persistent, for PSNR parsing).
                if self.log_file:
                    try:
                        self.log_file.write(line)
                        self.log_file.flush()
                    except Exception:
                        pass
                # Print to terminal chạy Gradio (user monitor).
                try:
                    sys.stdout.write(f"{prefix} {line}")
                    sys.stdout.flush()
                except Exception:
                    pass
                # Parse current iteration (fine-grained progress).
                m = progress_rx.search(line)
                if m:
                    try:
                        it = int(m.group(1))
                        if 0 <= it <= 10000:
                            self.current_iter = it
                    except ValueError:
                        pass
        except Exception as e:
            print(f"[train] reader thread error: {e}")

    def stop(self) -> None:
        """SIGTERM subprocess + cleanup file handles."""
        if self.process and self.process.poll() is None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if self.log_file:
            try:
                self.log_file.close()
            except Exception:
                pass
        print(f"[train] session {self.scene} stopped")

    def cleanup(self) -> None:
        """Xóa output_dir. KHÔNG xóa log (giữ để debug)."""
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir, ignore_errors=True)
            print(f"[train] cleaned up {self.output_dir}")

    def _ensure_test_cam_gt(self) -> None:
        """Load test camera + GT lần đầu khi có checkpoint."""
        if self.test_cam is not None:
            return
        source_path = REPO_ROOT / "data" / "nerf_llff_data" / self.scene
        scene_obj = load_llff_scene(str(source_path))
        test_cams = list(scene_obj.getTestCameras())
        if not test_cams:
            raise RuntimeError(f"No test cameras for {self.scene}")
        # Dùng test 0 làm view chuẩn (thấy tiến triển rõ nhất).
        self.test_cam = test_cams[0]
        gt_np = camera_gt_image(self.test_cam)   # HxWx3 uint8
        self.gt_pil = Image.fromarray(gt_np)

    def _render_checkpoint(self, ply_path: Path) -> Optional[Image.Image]:
        """Load .ply + render tại test cam + compose GT|Render|Diff row."""
        try:
            gaussians = load_gaussians_from_ply(str(ply_path))
        except Exception as e:
            print(f"[train] failed to load {ply_path}: {e}")
            return None

        self._ensure_test_cam_gt()

        rend = render_view(gaussians, self.test_cam)
        gt_np = np.asarray(self.gt_pil, dtype=np.uint8)

        diff = np.abs(rend.astype(np.int32) - gt_np.astype(np.int32)) \
            .mean(axis=2)
        diff = np.clip(diff * 3, 0, 255).astype(np.uint8)

        # Compose ngang (giống Tab B).
        from demo.ui_liverender import _hstack_images
        row = _hstack_images([
            self.gt_pil,
            Image.fromarray(rend),
            Image.fromarray(diff),
        ])

        # Free VRAM ngay sau render.
        del gaussians
        import torch
        torch.cuda.empty_cache()

        return row

    def _parse_psnr_from_log(self, current_iter: int) -> Optional[float]:
        """Parse PSNR ở iter cụ thể từ log file.

        Pattern: '  1000 | test | 22.5 | 0.8 | 0.15 | 5000 | 30.0'
        """
        if not self.log_path.exists():
            return None
        try:
            content = self.log_path.read_text(errors="ignore")
        except Exception:
            return None
        pattern = rf"\s*{current_iter}\s*\|\s*test\s*\|\s*([\d.]+)\s*\|"
        matches = re.findall(pattern, content)
        if matches:
            return float(matches[-1])
        return None

    def poll(self) -> dict:
        """Check status + render new checkpoints. Return dict status."""
        if self.process is None:
            return {"status": "idle"}

        # Check process alive
        exit_code = self.process.poll()
        if exit_code is not None and not self.completed:
            self.completed = True
            self.completed_at = time.time()   # freeze elapsed tại thời điểm này
            self.exit_code = exit_code
            self.failed = (exit_code != 0)
            print(f"[train] {self.scene} exited code={exit_code}")

        # Scan for new checkpoints
        cp_dir = self.output_dir / "point_cloud"
        new_renders = []
        if cp_dir.exists():
            for d in sorted(cp_dir.iterdir()):
                if not d.is_dir() or not d.name.startswith("iteration_"):
                    continue
                try:
                    it = int(d.name.split("_")[1])
                except ValueError:
                    continue
                if it in self.rendered_iters:
                    continue
                ply = d / "point_cloud.ply"
                if not ply.exists() or ply.stat().st_size == 0:
                    continue
                # Small delay để chắc chắn ply write xong
                if time.time() - ply.stat().st_mtime < 2.0:
                    continue

                row = self._render_checkpoint(ply)
                if row is not None:
                    self.checkpoint_renders.append((it, row))
                    self.rendered_iters.add(it)
                    psnr = self._parse_psnr_from_log(it)
                    if psnr is not None:
                        self.psnr_history.append((it, psnr))
                    new_renders.append(it)

        latest_iter = max(self.rendered_iters) if self.rendered_iters else 0
        elapsed = time.time() - self.started_at if self.started_at else 0

        return {
            "status": ("completed" if self.completed
                       else ("failed" if self.failed else "running")),
            "latest_iter": latest_iter,
            "elapsed_s": elapsed,
            "new_renders": new_renders,
        }

    def latest_render(self) -> Optional[Image.Image]:
        """Ảnh compose row của checkpoint mới nhất."""
        if not self.checkpoint_renders:
            return None
        return self.checkpoint_renders[-1][1]

    def build_timelapse_gif(self, out_path: Path, fps: int = 2) -> None:
        """Ghép 10 checkpoint renders thành GIF timelapse.

        fps=2 → mỗi frame 500ms → 10 frame = 5s loop.
        """
        if not self.checkpoint_renders:
            return
        frames_np = []
        for it, pil in self.checkpoint_renders:
            frames_np.append(np.asarray(pil.convert("RGB"), dtype=np.uint8))
        imageio.mimsave(str(out_path), frames_np, format="GIF",
                        duration=1.0 / fps, loop=0)
        print(f"[train] timelapse saved to {out_path}")

    def build_orbit_gif(self, out_path: Path,
                        n_frames: int = 60, fps: int = 30) -> None:
        """Sinh orbit GIF ellipse trajectory tại final checkpoint.

        Giống Tab A cache GIF nhưng render live từ .ply model vừa train xong.
        Cần ~3s cho 60 frames render (~50ms/frame).
        """
        # Chọn checkpoint cuối cùng (iter 10000) hoặc mới nhất available.
        final_iter = CHECKPOINT_ITERS[-1]
        ply_path = (self.output_dir / "point_cloud"
                    / f"iteration_{final_iter}" / "point_cloud.ply")
        if not ply_path.exists():
            if not self.rendered_iters:
                return
            latest = max(self.rendered_iters)
            ply_path = (self.output_dir / "point_cloud"
                        / f"iteration_{latest}" / "point_cloud.ply")
            if not ply_path.exists():
                return

        try:
            gaussians = load_gaussians_from_ply(str(ply_path))
        except Exception as e:
            print(f"[train] failed to load final ply for orbit: {e}")
            return

        # Build ellipse trajectory từ COLMAP raw cameras (giống Tab A).
        from demo.populate_cache import (
            RESOLUTION,
            load_all_colmap_cameras,
        )
        from demo.trajectory_utils import build_orbit_trajectory

        source_path = REPO_ROOT / "data" / "nerf_llff_data" / self.scene
        try:
            cameras_raw = load_all_colmap_cameras(
                source_path, resolution=RESOLUTION)
            trajectory = build_orbit_trajectory(
                cameras_raw, n_frames=n_frames)
        except Exception as e:
            print(f"[train] failed to build orbit trajectory: {e}")
            del gaussians
            import torch
            torch.cuda.empty_cache()
            return

        print(f"[train] rendering orbit GIF ({n_frames} frames)...")
        t0 = time.time()
        frames_np = []
        for cam in trajectory:
            frame = render_view(gaussians, cam)   # HxWx3 uint8
            frames_np.append(frame)

        imageio.mimsave(str(out_path), frames_np, format="GIF",
                        duration=1.0 / fps, loop=0)
        print(f"[train] orbit GIF saved to {out_path} "
              f"({time.time() - t0:.1f}s)")

        # Free VRAM.
        del gaussians
        import torch
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Helper — format status text
# ---------------------------------------------------------------------------


def _format_time(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s"
    m = int(s // 60)
    sec = int(s % 60)
    return f"{m}m {sec:02d}s"


def _progress_bar_html(pct: float, color: str, label: str) -> str:
    """Progress bar HTML với inline style (theme-agnostic).

    pct: 0-100 percentage.
    color: gradient color (start hex).
    label: text overlay (iter + elapsed + ETA).
    """
    pct = max(0, min(100, pct))
    return (
        f'<div class="train-progress-container">'
        f'  <div class="train-progress-fill" '
        f'       style="width: {pct:.1f}%; background: {color};"></div>'
        f'  <div class="train-progress-label">{label}</div>'
        f'</div>'
    )


def _format_status(session: Optional[TrainingSession]) -> str:
    """Return HTML string với progress bar + status text."""
    if session is None:
        return (
            _progress_bar_html(0, "linear-gradient(90deg, #6b7280, #4b5563)",
                               "Idle — click Start Training to begin")
            + '<div class="train-status-text">⚪ <b>Idle</b> · '
              'no training active</div>'
        )

    if session.failed:
        label = f"❌ FAILED · exit={session.exit_code}"
        return (
            _progress_bar_html(100, "linear-gradient(90deg, #ef4444, #b91c1c)",
                               label)
            + f'<div class="train-status-text">❌ <b>Failed</b> — '
              f'scene <code>{session.scene}</code><br>'
              f'Check log <code>{session.log_path}</code></div>'
        )

    # Elapsed: nếu completed → FREEZE tại completed_at. Nếu đang running →
    # tính realtime từ started_at đến now.
    if session.completed and session.completed_at is not None:
        elapsed = session.completed_at - (session.started_at or 0)
    elif session.started_at is not None:
        elapsed = time.time() - session.started_at
    else:
        elapsed = 0

    max_iter = CHECKPOINT_ITERS[-1]
    # Fine-grained iter từ stdout parse (mỗi ~10 iter update).
    current = session.current_iter
    # Coarse-grained iter từ checkpoint saved (mỗi 1000 iter).
    ckpt = max(session.rendered_iters) if session.rendered_iters else 0
    # Progress bar dùng current (smooth). Fallback checkpoint nếu không parse
    # được stdout (an toàn).
    progress_iter = max(current, ckpt)
    pct = (progress_iter / max_iter) * 100

    if session.completed:
        label = f"✅ Complete · {_format_time(elapsed)} total"
        return (
            _progress_bar_html(100,
                               "linear-gradient(90deg, #10b981, #059669)",
                               label)
            + f'<div class="train-status-text">✅ <b>Completed</b> — '
              f'scene <code>{session.scene}</code> · '
              f'elapsed <b>{_format_time(elapsed)}</b></div>'
        )

    # Running
    if progress_iter > 0:
        eta = (elapsed / progress_iter) * (max_iter - progress_iter)
        eta_str = f" · ETA <b>{_format_time(eta)}</b>"
    else:
        eta_str = " · <i>warming up...</i>"
    label = (f"{progress_iter}/{max_iter} iter · "
             f"{_format_time(elapsed)} elapsed"
             if progress_iter > 0
             else f"warming up · {_format_time(elapsed)}")
    return (
        _progress_bar_html(pct,
                           "linear-gradient(90deg, #3b82f6, #1d4ed8)",
                           label)
        + f'<div class="train-status-text">🟢 <b>Running</b> — '
          f'scene <code>{session.scene}</code> · '
          f'iter <b>{progress_iter}/{max_iter}</b> ({pct:.1f}%) · '
          f'ckpt <b>{ckpt}</b> · '
          f'elapsed <b>{_format_time(elapsed)}</b>{eta_str}</div>'
    )


def _format_psnr_history(session: Optional[TrainingSession]) -> str:
    if session is None or not session.psnr_history:
        return "*(chưa có checkpoint)*"
    lines = ["**PSNR progression on test view 0:**"]
    for it, psnr in session.psnr_history:
        lines.append(f"- Iter {it:>5d}: {psnr:.2f} dB")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def cb_start_training(scene: str) -> Tuple:
    """Start training subprocess.

    Returns (status_html, psnr_md, render_img, timelapse_gif, orbit_gif).
    """
    global _ACTIVE_SESSION

    if _ACTIVE_SESSION is not None and not _ACTIVE_SESSION.completed \
            and not _ACTIVE_SESSION.failed:
        return (f"❌ Training đang chạy cho `{_ACTIVE_SESSION.scene}`. "
                f"Click **Stop** trước.",
                "", None, None, None)

    # Cleanup old session if any.
    if _ACTIVE_SESSION is not None:
        _ACTIVE_SESSION.cleanup()

    session = TrainingSession(scene=scene)
    try:
        session.start()
    except Exception as e:
        return f"❌ Failed to start: {e}", "", None, None, None

    _ACTIVE_SESSION = session
    return (_format_status(session), "*(chưa có checkpoint)*",
            None, None, None)


def cb_stop_training() -> Tuple:
    """Stop subprocess (SIGTERM)."""
    global _ACTIVE_SESSION
    if _ACTIVE_SESSION is None:
        return "⚪ No training to stop", "", None, None, None
    _ACTIVE_SESSION.stop()
    _ACTIVE_SESSION.completed = True
    if _ACTIVE_SESSION.completed_at is None:
        _ACTIVE_SESSION.completed_at = time.time()  # freeze elapsed
    _ACTIVE_SESSION.outputs_finalized = True   # tránh poll flicker
    return (_format_status(_ACTIVE_SESSION),
            _format_psnr_history(_ACTIVE_SESSION),
            _ACTIVE_SESSION.latest_render(),
            None, None)


def cb_poll() -> Tuple:
    """Poll session state; called every 1s bởi gr.Timer.

    Returns:
        (status_html, psnr_history_md, latest_render,
         timelapse_gif_path, orbit_gif_path)

    Sau khi complete + build xong 2 GIF cuối cùng → set flag
    `outputs_finalized=True`. Các poll tiếp theo return `gr.skip()` cho
    5 output → Gradio không re-render → không flicker.
    """
    session = _ACTIVE_SESSION
    if session is None:
        return ("⚪ **Idle** — no training active",
                "*(chưa có checkpoint)*", None, None, None)

    # SKIP: đã complete + đã sent tất cả outputs cuối. Không cần poll nữa.
    if session.outputs_finalized:
        return (gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip())

    session.poll()
    status = _format_status(session)
    history = _format_psnr_history(session)
    render = session.latest_render()

    # Nếu vừa complete: build timelapse GIF + orbit GIF (chạy 1 lần).
    timelapse_path = None
    orbit_path_str = None
    if session.completed and not session.failed \
            and session.checkpoint_renders:
        # 1. Timelapse (progression tại test view 0)
        tl_path = (LOG_ROOT / f"{session.scene}_{session.timestamp}"
                   f"_timelapse.gif")
        if not tl_path.exists():
            try:
                session.build_timelapse_gif(tl_path, fps=2)
            except Exception as e:
                print(f"[train] timelapse build failed: {e}")
        if tl_path.exists():
            timelapse_path = str(tl_path)

        # 2. Orbit GIF (ellipse trajectory tại final checkpoint)
        orbit_path = (LOG_ROOT / f"{session.scene}_{session.timestamp}"
                      f"_orbit.gif")
        if not orbit_path.exists():
            try:
                session.build_orbit_gif(orbit_path, n_frames=60, fps=30)
            except Exception as e:
                print(f"[train] orbit build failed: {e}")
        if orbit_path.exists():
            orbit_path_str = str(orbit_path)

        # Nếu cả 2 GIF đã build xong → đánh dấu finalized để poll tiếp
        # theo skip. Nếu build fail (path == None) → vẫn skip vì retry
        # cũng không giúp gì.
        session.outputs_finalized = True
        print(f"[train] outputs finalized for {session.scene} — "
              f"further polls will skip UI updates")

    # Failed: cũng finalize để không spam UI update.
    if session.failed:
        session.outputs_finalized = True

    return status, history, render, timelapse_path, orbit_path_str


def cb_cleanup() -> Tuple:
    """User request cleanup output/live_demos/{scene}_{ts}/."""
    global _ACTIVE_SESSION
    if _ACTIVE_SESSION is None:
        return "⚪ No session to cleanup", "", None, None, None
    _ACTIVE_SESSION.cleanup()
    _ACTIVE_SESSION = None
    return "🧹 **Cleaned up** — session removed", "", None, None, None


# ---------------------------------------------------------------------------
# Build tab
# ---------------------------------------------------------------------------


_CSS = """
<style>
.train-card {border: 1px solid var(--border-color-primary) !important;
             border-radius: 12px !important;
             padding: 14px 16px !important;
             background: var(--background-fill-secondary) !important;
             margin-bottom: 8px;}
.train-recipe {padding: 10px 14px !important;
               background: var(--background-fill-primary) !important;
               border-radius: 8px !important;
               font-family: var(--font-mono) !important;
               font-size: 13px !important;
               margin: 6px 0 !important;}
.train-status {padding: 12px !important;
               background: var(--background-fill-primary) !important;
               border-radius: 8px !important;
               font-size: 14px !important;
               margin: 8px 0 !important;}
.train-note {color: var(--body-text-color-subdued) !important;
             font-size: 13px !important; font-style: italic;
             margin: 4px 0 8px 0 !important;}

/* Phase 4b: progress bar visual */
.train-progress-container {
    position: relative;
    height: 28px;
    background: var(--background-fill-primary);
    border: 1px solid var(--border-color-primary);
    border-radius: 14px;
    overflow: hidden;
    margin: 8px 0;
    box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.06);
}
.train-progress-fill {
    height: 100%;
    /* Phase 4c: transition dài 1s LINEAR match poll interval 1s → bar
       animation smooth liên tục, cảm giác continuous flow như tqdm. */
    transition: width 1s linear, background 0.4s ease;
    border-radius: 14px;
}
.train-progress-label {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 13px;
    font-weight: 700;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.4);
    letter-spacing: 0.3px;
}
.train-status-text {
    padding: 10px 14px;
    background: var(--background-fill-primary);
    border-radius: 8px;
    font-size: 14px;
    margin: 4px 0;
    line-height: 1.6;
}
.train-status-text code {
    background: var(--background-fill-secondary);
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 12px;
}
</style>
"""


def _format_reference_times() -> str:
    lines = ["**⏱️ Reference wall-clock (Phase 28 tau65, PSNR final):**"]
    for scene in SCENES:
        t = REFERENCE_TIMES.get(scene, "?")
        lines.append(f"- `{scene}` — {t}")
    return "  \n".join(lines)


def build_livetraining_tab(local_mode: bool = False) -> None:
    """Build layout Tab C."""
    if local_mode:
        gr.Markdown(
            "### ⚠️ Live Training — server GPU required\n\n"
            "This tab is unavailable in local backup mode. Only the "
            "**Pre-computed** tab works offline.\n\n"
            "To run live training, use the server:\n"
            "```bash\n"
            "conda activate gradio_demo\n"
            "bash demo/run_server.sh\n"
            "```"
        )
        return

    # Register signal handlers 1 lần: Ctrl+C trên Gradio → cleanup train.
    _register_handlers()

    gr.HTML(_CSS)

    with gr.Row():
        with gr.Column(scale=1, min_width=220):
            scene_dd = gr.Dropdown(
                choices=SCENES, value=DEFAULT_SCENE,
                label="Scene to train",
                interactive=True,
            )
        with gr.Column(scale=2, min_width=220):
            gr.Markdown(
                "**Recipe**: A3-TRIM 8-module · tau=0.65 · seed 42  \n"
                "**Iterations**: 10,000 · **Poll interval**: 5s  \n"
                "**Checkpoints**: 10 (every 1000 iter)",
                elem_classes=["train-recipe"],
            )

    with gr.Group(elem_classes=["train-card"]):
        gr.Markdown("#### 🚀 Live Training")
        gr.Markdown(
            "Click **Start Training** để chạy train.py từ 0 iter → 10000 "
            "trực tiếp trên server. Tab tự poll checkpoint mỗi 5s và "
            "render test view 0 tại checkpoint mới nhất. Khi xong sẽ có "
            "timelapse GIF từ 10 checkpoints. **Auto-cleanup** output "
            "khi complete.",
            elem_classes=["train-note"],
        )

        gr.Markdown(
            _format_reference_times(),
            elem_classes=["train-recipe"],
        )

        with gr.Row():
            btn_start = gr.Button("🚀 Start Training", variant="primary")
            btn_stop = gr.Button("⏹ Stop", variant="stop")
            btn_cleanup = gr.Button("🧹 Cleanup output")

        # HTML thay Markdown để render progress bar + text.
        status_html = gr.HTML(_format_status(None))

    with gr.Group(elem_classes=["train-card"]):
        gr.Markdown("#### 📸 Latest checkpoint render (test view 0)")
        gr.Markdown(
            "Composed row: **Ground truth** | **Render at latest iter** "
            "| **Difference ×3**. Cập nhật mỗi lần có checkpoint mới "
            "(~mỗi 1000 iter).",
            elem_classes=["train-note"],
        )
        render_img = gr.Image(
            show_label=False, container=False, interactive=False,
        )

    with gr.Group(elem_classes=["train-card"]):
        gr.Markdown("#### 📈 PSNR progression")
        psnr_md = gr.Markdown(
            "*(chưa có checkpoint)*",
            elem_classes=["train-status"],
        )

    with gr.Group(elem_classes=["train-card"]):
        gr.Markdown("#### 🎞️ Training progression timelapse (hiện sau complete)")
        gr.Markdown(
            "10 frame ghép từ 10 checkpoint renders tại test view 0 — "
            "1000, 2000, ..., 10000 iter. fps=2 → 5s loop. "
            "Chứng minh **model học tốt theo iter**.",
            elem_classes=["train-note"],
        )
        timelapse_gif = gr.Image(
            show_label=False, container=False, interactive=False,
            height=400,
        )

    with gr.Group(elem_classes=["train-card"]):
        gr.Markdown("#### 🎨 Novel view orbit — final checkpoint (Tab A style)")
        gr.Markdown(
            "60 frame ellipse trajectory quanh scene, render live tại "
            "checkpoint cuối (iter 10000). Giống Tab A GIF nhưng sinh "
            "**live sau khi training vừa xong** — chứng minh **full "
            "pipeline** train → render → output.",
            elem_classes=["train-note"],
        )
        orbit_gif = gr.Image(
            show_label=False, container=False, interactive=False,
            height=400,
        )

    # ── Events ────────────────────────────────────────────────────────
    btn_start.click(
        cb_start_training,
        inputs=[scene_dd],
        outputs=[status_html, psnr_md, render_img,
                 timelapse_gif, orbit_gif],
    )
    btn_stop.click(
        cb_stop_training,
        outputs=[status_html, psnr_md, render_img,
                 timelapse_gif, orbit_gif],
    )
    btn_cleanup.click(
        cb_cleanup,
        outputs=[status_html, psnr_md, render_img,
                 timelapse_gif, orbit_gif],
    )

    # Auto-poll every 5s via gr.Timer (Gradio 4.13+).
    timer = gr.Timer(POLL_INTERVAL_S)
    timer.tick(
        cb_poll,
        outputs=[status_html, psnr_md, render_img,
                 timelapse_gif, orbit_gif],
    )
