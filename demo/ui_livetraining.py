"""Tab C (Live Training) cho Gradio demo.

User click "Start Training" → subprocess train.py chạy từ 0 iter tới 10000,
Tab C poll checkpoint folders và render latest .ply lên UI.

Design (Phase 4):
- Recipe: A3-TRIM 8-module + tau=0.65 (Phase 28 tau65 cell) + seed 42.
- Save/test iterations: 1000, 2000, ..., 10000 (10 checkpoints).
- Live render: latest checkpoint (thay cũ mỗi khi có checkpoint mới).
- End state: timelapse GIF từ 10 checkpoint renders + orbit GIF.
- Auto-cleanup: xóa output/live_demos/{scene}_{ts} sau khi complete
  hoặc user click "Cleanup".
- Chỉ 1 training tại 1 thời điểm (global session state).

Server-mode required. Local mode ẩn tab.

Phase 4d fix (2026-07-06) — HẾT "LOAD MÃI + GIẬT GIẬT" SAU COMPLETE:
    Triệu chứng: training xong (elapsed đã freeze) nhưng tab browser vẫn
    quay loading mỗi giây, 5 output chớp chớp; PSNR progression luôn
    "(chưa có checkpoint)".
    Root cause chuỗi:
      1. `outputs_finalized` CHỈ được set khi `checkpoint_renders` khác
         rỗng → nếu render checkpoint fail (exception trong
         _ensure_test_cam_gt/render KHÔNG được catch, hoặc load ply fail
         im lặng) thì finalize không bao giờ xảy ra → gr.Timer tiếp tục
         bắn FULL update 5 output mỗi 1s vĩnh viễn → favicon loading +
         component chớp.
      2. PSNR rỗng vì (a) chuỗi trên, và (b) regex parse log dạng bảng
         pipe `1000 | test | 22.5 |` không khớp format chuẩn của train.py
         gốc 3DGS/CoR-GS: `[ITER 1000] Evaluating test: L1 x PSNR y`.
      3. Kể cả happy path vẫn giật: timer active từ lúc load trang (poll
         cả khi Idle), event tick mặc định show_progress="full" phủ
         loading overlay lên cả 5 output mỗi giây, và latest_render (PIL)
         bị re-serialize + refetch mỗi tick dù không đổi.
    Fix:
      1. Timer lifecycle: tick tự TẮT timer (gr.Timer(active=False)) khi
         idle/finalized/failed; Start bật lại. Page refresh giữa chừng
         vẫn resume được (timer khởi tạo active, tick đầu tự tắt nếu
         không có việc).
      2. `show_progress="hidden"` cho tick → không còn overlay chớp.
      3. Granular skip: render/timelapse/orbit chỉ update khi NỘI DUNG
         mới; mỗi tick chỉ status + PSNR text thay đổi.
      4. Finalize TÁCH KHỎI render thành công: hết grace window (5 poll
         sau complete, chờ ply cuối) là finalize + tắt timer, kể cả khi
         0 checkpoint render được. Render checkpoint được try/except
         toàn bộ + in traceback, retry tối đa 3 lần rồi blacklist.
      5. PSNR tính TRỰC TIẾP từ render vs GT tại test view 0 (đúng nhãn
         UI, không phụ thuộc format log). Parse log chỉ còn là fallback
         đa pattern lúc kết thúc.
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
import traceback
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
# Phase 4c: 1s cho progress bar smooth. Phase 4d: tick nhẹ (chỉ text) vì
# ảnh/GIF đã granular-skip; timer TỰ TẮT khi không còn việc.
POLL_INTERVAL_S = 1

# Iterations save/test — 10 checkpoints, 1000..10000.
CHECKPOINT_ITERS = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000,
                    9000, 10000]

# Số poll grace sau khi process exit — chờ ply cuối ghi xong rồi mới
# finalize (ply 10000 thường xuất hiện ngay trước/sau khi exit).
GRACE_POLLS_AFTER_COMPLETE = 5

# Retry tối đa cho 1 checkpoint render fail trước khi blacklist.
MAX_RENDER_ATTEMPTS = 3

# ---------------------------------------------------------------------------
# Global training session state (chỉ 1 tại 1 thời điểm)
# ---------------------------------------------------------------------------


_ACTIVE_SESSION: Optional["TrainingSession"] = None


def _skip():
    """gr.skip() nếu có (Gradio ≥4.36), fallback gr.update() (no-op)."""
    return gr.skip() if hasattr(gr, "skip") else gr.update()


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
        # sau khi complete → poll tiếp theo skip + tắt timer.
        self.outputs_finalized: bool = False
        # Phase 4d: đếm số poll sau khi process exit (grace window chờ
        # ply cuối cùng ghi xong trước khi finalize).
        self.polls_after_complete: int = 0

        # Fine-grained iteration counter — parse từ tqdm output train.py
        # (mỗi ~10 iter). Progress bar dùng cái này để cập nhật smooth.
        # Khác với rendered_iters (chỉ tăng khi có checkpoint = mỗi 1000).
        self.current_iter: int = 0

        # Checkpoint state
        self.rendered_iters: set = set()
        # Phase 4d: retry đếm theo iter; hết MAX_RENDER_ATTEMPTS →
        # blacklist (coi như "đã xử lý" để không retry vĩnh viễn).
        self.render_attempts: dict = {}
        self.render_failed: set = set()
        self.checkpoint_renders: List[Tuple[int, Image.Image]] = []
        # (iter, composed_row_PIL)
        self.psnr_history: List[Tuple[int, float]] = []  # (iter, psnr)

        # Test camera + GT (loaded once when first checkpoint arrives)
        self.test_cam = None
        self.gt_pil: Optional[Image.Image] = None

    def start(self) -> None:
        """Launch train.py subprocess non-blocking.

        stdout đi qua PIPE, background thread đọc line-by-line rồi TEE tới
        cả log file (để debug) và terminal (để user monitor live).
        """
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        cmd = _build_train_command(self.scene, self.output_dir,
                                   gpu=self.gpu)
        env = os.environ.copy()
        # Kế thừa CUDA_VISIBLE_DEVICES từ Gradio process nếu user đã export
        # (vd: `export CUDA_VISIBLE_DEVICES=1 && bash demo/run_server.sh`).
        # Nếu chưa export → default self.gpu (0).
        if "CUDA_VISIBLE_DEVICES" not in env:
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
                # Write to log file (persistent, for debugging).
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
        """Load test camera + GT lần đầu khi có checkpoint.

        Phase 4d: truyền TƯỜNG MINH n_views=3, resolution=8 khớp protocol
        train (`-r 8 --n_views 3`) — trước đây dùng default của
        load_llff_scene, nếu default là full-res thì render chậm/lệch GT.
        """
        if self.test_cam is not None:
            return
        source_path = REPO_ROOT / "data" / "nerf_llff_data" / self.scene
        scene_obj = load_llff_scene(str(source_path), n_views=3, resolution=8)
        test_cams = list(scene_obj.getTestCameras())
        if not test_cams:
            raise RuntimeError(f"No test cameras for {self.scene}")
        # Dùng test 0 làm view chuẩn (thấy tiến triển rõ nhất).
        self.test_cam = test_cams[0]
        gt_np = camera_gt_image(self.test_cam)   # HxWx3 uint8
        self.gt_pil = Image.fromarray(gt_np)

    def _render_checkpoint(
            self, ply_path: Path
    ) -> Optional[Image.Image]:
        """Load .ply → render test view 0 → composed row.

        Phase 4e: KHÔNG return PSNR — PSNR đúng phải là mean toàn test
        set (compute bởi train.py và log `[ITER X] Evaluating test:
        PSNR Y`). Render 1 view chỉ để preview visual + timelapse GIF.
        Populate psnr_history CHỈ ở finalize via `rescan_psnr_from_log`.
        """
        gaussians = None
        try:
            gaussians = load_gaussians_from_ply(str(ply_path))
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
            return row
        except Exception:
            print(f"[train] render checkpoint {ply_path} FAILED:")
            traceback.print_exc()
            return None
        finally:
            # Free VRAM ngay sau render (kể cả khi fail giữa chừng).
            if gaussians is not None:
                del gaussians
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _parse_metrics_from_log(self, current_iter: int
                                ) -> Optional[dict]:
        """Parse PSNR / SSIM / LPIPS ở iter cụ thể từ log file.

        Log format 3DGS/CoR-GS:
          `[ITER 10000] Evaluating test: L1 0.039 PSNR 22.67 SSIM 0.88 LPIPS 0.13`

        Trả về dict {psnr, ssim, lpips} nếu parse được PSNR (SSIM/LPIPS
        có thể None nếu format khác).
        """
        if not self.log_path.exists():
            return None
        try:
            content = self.log_path.read_text(errors="ignore")
        except Exception:
            return None
        it = current_iter
        # Line pattern: [ITER 10000] Evaluating test: ... PSNR X SSIM Y LPIPS Z
        line_pattern = (rf"\[ITER\s+{it}\]\s*Evaluating\s+test\s*:"
                        rf"([^\n]+)")
        m = re.search(line_pattern, content)
        if not m:
            return None
        line = m.group(1)

        def _grab(name: str) -> Optional[float]:
            mm = re.search(rf"\b{name}\s*[:=]?\s*([\d.]+)", line)
            if mm:
                try:
                    return float(mm.group(1))
                except ValueError:
                    return None
            return None

        psnr = _grab("PSNR")
        if psnr is None:
            return None
        return {
            "psnr": psnr,
            "ssim": _grab("SSIM"),
            "lpips": _grab("LPIPS"),
        }

    def rescan_psnr_from_log(self) -> None:
        """Parse metrics từ log train.py cho tất cả CHECKPOINT_ITERS.

        Phase 4e: PSNR/SSIM/LPIPS = mean toàn test set (compute bởi
        train.py). Store vào `psnr_history` dạng (iter, dict) —
        dict chứa keys psnr, ssim, lpips.
        """
        have = {it for it, _ in self.psnr_history}
        added = 0
        for it in CHECKPOINT_ITERS:
            if it in have:
                continue
            metrics = self._parse_metrics_from_log(it)
            if metrics is not None:
                self.psnr_history.append((it, metrics))
                added += 1
        if added:
            self.psnr_history.sort(key=lambda t: t[0])
            print(f"[train] parse log: {added} metric entries "
                  f"(PSNR/SSIM/LPIPS mean test set)")

    def accounted_iters(self) -> set:
        """Các checkpoint đã xử lý xong (render OK hoặc blacklist)."""
        return self.rendered_iters | self.render_failed

    def poll(self) -> dict:
        """Check status + render new checkpoints. Return dict status."""
        if self.process is None:
            return {"status": "idle", "new_renders": []}

        # Check process alive
        exit_code = self.process.poll()
        if exit_code is not None and not self.completed:
            self.completed = True
            self.completed_at = time.time()   # freeze elapsed tại thời điểm này
            self.exit_code = exit_code
            self.failed = (exit_code != 0)
            print(f"[train] {self.scene} exited code={exit_code}")
        if self.completed:
            self.polls_after_complete += 1

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
                if it in self.rendered_iters or it in self.render_failed:
                    continue
                ply = d / "point_cloud.ply"
                if not ply.exists() or ply.stat().st_size == 0:
                    continue
                # Small delay để chắc chắn ply write xong. Sau khi process
                # exit thì không còn ai ghi nữa → đọc ngay (tránh miss ply
                # cuối trong grace window).
                if not self.completed \
                        and time.time() - ply.stat().st_mtime < 2.0:
                    continue

                row = self._render_checkpoint(ply)
                if row is not None:
                    self.checkpoint_renders.append((it, row))
                    self.rendered_iters.add(it)
                    new_renders.append(it)
                    # PSNR sẽ được parse từ log lúc finalize (mean toàn
                    # test set, KHÔNG phải single view). Ở đây chỉ log
                    # tiến trình render.
                    print(f"[train] ckpt {it}: rendered preview")
                else:
                    n = self.render_attempts.get(it, 0) + 1
                    self.render_attempts[it] = n
                    if n >= MAX_RENDER_ATTEMPTS:
                        self.render_failed.add(it)
                        print(f"[train] ckpt {it}: blacklist sau {n} lần fail")

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
    """Chỉ hiển thị metrics tại iter 10000 (final) — PSNR/SSIM/LPIPS."""
    if session is None:
        return "*(chưa có training active)*"
    if not session.psnr_history:
        if session.outputs_finalized:
            return "*(không parse được metrics từ log)*"
        if session.completed:
            return "*(đang tính metrics sau complete...)*"
        return ("*(PSNR/SSIM/LPIPS mean toàn test set sẽ hiển thị sau "
                "khi training complete)*")

    # Lấy entry iter 10000 (checkpoint cuối cùng).
    final_iter = CHECKPOINT_ITERS[-1]
    final_metrics = None
    for it, m in session.psnr_history:
        if it == final_iter:
            final_metrics = m
            break

    if final_metrics is None:
        return f"*(chưa parse được metrics tại iter {final_iter})*"

    psnr = final_metrics.get("psnr")
    ssim = final_metrics.get("ssim")
    lpips = final_metrics.get("lpips")

    lines = [f"**Final metrics on test set at iter {final_iter} "
             f"(mean across all test views):**"]
    if psnr is not None:
        lines.append(f"- **PSNR**: {psnr:.2f} dB")
    if ssim is not None:
        lines.append(f"- **SSIM**: {ssim:.4f}")
    if lpips is not None:
        lines.append(f"- **LPIPS**: {lpips:.4f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
# Mọi callback trả 6 outputs:
#   (status_html, psnr_md, render_img, timelapse_gif, orbit_gif, timer)
# Timer là output cuối để bật/tắt polling từ chính callback.


def cb_start_training(scene: str) -> Tuple:
    """Start training subprocess + BẬT timer poll."""
    global _ACTIVE_SESSION

    if _ACTIVE_SESSION is not None and not _ACTIVE_SESSION.completed \
            and not _ACTIVE_SESSION.failed:
        return (f"❌ Training đang chạy cho `{_ACTIVE_SESSION.scene}`. "
                f"Click **Stop** trước.",
                _skip(), _skip(), _skip(), _skip(), _skip())

    # Cleanup old session if any.
    if _ACTIVE_SESSION is not None:
        _ACTIVE_SESSION.cleanup()

    session = TrainingSession(scene=scene)
    try:
        session.start()
    except Exception as e:
        return (f"❌ Failed to start: {e}", "", None, None, None,
                gr.Timer(active=False))

    _ACTIVE_SESSION = session
    return (_format_status(session), "*(chưa có checkpoint)*",
            None, None, None, gr.Timer(active=True))


def cb_stop_training() -> Tuple:
    """Stop subprocess (SIGTERM) + tắt timer."""
    global _ACTIVE_SESSION
    if _ACTIVE_SESSION is None:
        return ("⚪ No training to stop", "", None, None, None,
                gr.Timer(active=False))
    _ACTIVE_SESSION.stop()
    _ACTIVE_SESSION.completed = True
    if _ACTIVE_SESSION.completed_at is None:
        _ACTIVE_SESSION.completed_at = time.time()  # freeze elapsed
    _ACTIVE_SESSION.outputs_finalized = True   # tránh poll flicker
    return (_format_status(_ACTIVE_SESSION),
            _format_psnr_history(_ACTIVE_SESSION),
            _ACTIVE_SESSION.latest_render(),
            None, None, gr.Timer(active=False))


def cb_poll() -> Tuple:
    """Poll session state; gọi bởi gr.Timer.

    Phase 4d — nguyên tắc:
    - Idle / finalized → TẮT timer (gr.Timer(active=False)); mọi output
      khác skip. Không còn request lặp vô hạn → hết favicon loading.
    - Đang chạy → chỉ status + PSNR text update mỗi tick; render_img chỉ
      update khi CÓ checkpoint mới; GIF chỉ update đúng 1 lần lúc
      finalize. Không re-serialize ảnh mỗi giây → hết giật.
    - Finalize TÁCH KHỎI render thành công: sau grace window
      (GRACE_POLLS_AFTER_COMPLETE poll từ lúc process exit, chờ ply cuối)
      hoặc khi đã xử lý đủ 10 checkpoint → finalize + tắt timer, KỂ CẢ
      khi 0 checkpoint render được (trước đây kẹt vĩnh viễn ở đây).
    """
    session = _ACTIVE_SESSION
    if session is None:
        # Idle: tắt luôn timer (tick đầu sau page load sẽ rơi vào đây).
        return (_skip(), _skip(), _skip(), _skip(), _skip(),
                gr.Timer(active=False))

    if session.outputs_finalized:
        return (_skip(), _skip(), _skip(), _skip(), _skip(),
                gr.Timer(active=False))

    info = session.poll()
    status = _format_status(session)
    history = _format_psnr_history(session)

    # Render chỉ update khi có checkpoint MỚI trong tick này.
    render_out = session.latest_render() if info["new_renders"] else _skip()
    timelapse_out = _skip()
    orbit_out = _skip()
    timer_out = _skip()

    if session.completed or session.failed:
        all_accounted = all(it in session.accounted_iters()
                            for it in CHECKPOINT_ITERS)
        grace_over = (session.polls_after_complete
                      >= GRACE_POLLS_AFTER_COMPLETE)
        if session.failed or all_accounted or grace_over:
            # --- FINALIZE (chạy đúng 1 lần) ---
            if not session.failed and session.checkpoint_renders:
                # 1. Timelapse (progression tại test view 0)
                tl_path = (LOG_ROOT / f"{session.scene}_{session.timestamp}"
                           f"_timelapse.gif")
                if not tl_path.exists():
                    try:
                        session.build_timelapse_gif(tl_path, fps=2)
                    except Exception as e:
                        print(f"[train] timelapse build failed: {e}")
                if tl_path.exists():
                    timelapse_out = str(tl_path)

                # 2. Orbit GIF (ellipse trajectory tại final checkpoint)
                orbit_path = (LOG_ROOT
                              / f"{session.scene}_{session.timestamp}"
                              f"_orbit.gif")
                if not orbit_path.exists():
                    try:
                        session.build_orbit_gif(orbit_path,
                                                n_frames=60, fps=30)
                    except Exception as e:
                        print(f"[train] orbit build failed: {e}")
                if orbit_path.exists():
                    orbit_out = str(orbit_path)

                # Render cuối cùng (phòng khi ply 10000 vừa render tick này).
                render_out = session.latest_render()

            # PSNR fallback từ log cho iter còn thiếu.
            try:
                session.rescan_psnr_from_log()
                history = _format_psnr_history(session)
            except Exception as e:
                print(f"[train] psnr rescan failed: {e}")

            session.outputs_finalized = True
            timer_out = gr.Timer(active=False)
            print(f"[train] outputs finalized for {session.scene} — "
                  f"timer off, no more UI polls")

    return status, history, render_out, timelapse_out, orbit_out, timer_out


def cb_cleanup() -> Tuple:
    """User request cleanup output/live_demos/{scene}_{ts}/."""
    global _ACTIVE_SESSION
    if _ACTIVE_SESSION is None:
        return ("⚪ No session to cleanup", "", None, None, None,
                gr.Timer(active=False))
    _ACTIVE_SESSION.cleanup()
    _ACTIVE_SESSION = None
    return ("🧹 **Cleaned up** — session removed", "", None, None, None,
            gr.Timer(active=False))


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
                "**Iterations**: 10,000 · **Poll interval**: 1s  \n"
                "**Checkpoints**: 10 (every 1000 iter)",
                elem_classes=["train-recipe"],
            )

    with gr.Group(elem_classes=["train-card"]):
        gr.Markdown("#### 🚀 Live Training")
        gr.Markdown(
            "Click **Start Training** để chạy train.py từ 0 iter → 10000 "
            "trực tiếp trên server. Tab tự poll checkpoint và render test "
            "view 0 tại checkpoint mới nhất. Khi xong sẽ có timelapse GIF "
            "từ 10 checkpoints. **Auto-cleanup** output khi complete.",
            elem_classes=["train-note"],
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
        gr.Markdown("#### 📈 Final test metrics (iter 10000, mean over all test views)")
        psnr_md = gr.Markdown(
            "*(PSNR / SSIM / LPIPS sẽ hiển thị sau khi training complete)*",
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
    # Timer khởi tạo ACTIVE: nếu user refresh trang giữa lúc training,
    # tick đầu tiên sẽ resume polling; còn nếu không có việc (idle /
    # finalized) tick đầu tự TẮT timer luôn → không poll vô hạn.
    timer = gr.Timer(POLL_INTERVAL_S, active=True)

    outputs_all = [status_html, psnr_md, render_img,
                   timelapse_gif, orbit_gif, timer]

    btn_start.click(cb_start_training, inputs=[scene_dd],
                    outputs=outputs_all)
    btn_stop.click(cb_stop_training, outputs=outputs_all)
    btn_cleanup.click(cb_cleanup, outputs=outputs_all)

    # show_progress="hidden": tick KHÔNG phủ loading overlay lên 5 output
    # mỗi giây — nguồn "giật giật" chính khi UI đứng yên.
    timer.tick(cb_poll, outputs=outputs_all, show_progress="hidden")
