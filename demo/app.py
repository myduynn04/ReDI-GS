"""ReDI-GS Gradio demo — entry point.

Sinh 1 web UI với 2 tab:
- Tab 1 (Pre-computed): duyệt kết quả 8 scene LLFF từ checkpoint tau=0.65.
- Tab 2 (Live Training): chạy training live 1 scene, hiển thị progression.

Chạy trên server:
    conda activate corgs
    python demo/app.py --port 7860

Chạy local backup (cache-only, không cần GPU):
    python demo/app.py --local-mode --port 7860
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Cho phép chạy `python demo/app.py` (không có -m). Khi chạy như script,
# sys.path[0] là `demo/`, không thấy được package `demo` → cần thêm repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Auto-pick free GPU trước khi import torch/gradio.
# Server shared → GPU 0 thường bị chiếm bởi user khác. Tự chọn GPU rảnh
# nhất (min mem used + util). User có thể override qua env var
# CUDA_VISIBLE_DEVICES=X trước khi chạy.
# ---------------------------------------------------------------------------


def _auto_pick_gpu() -> None:
    """Query nvidia-smi, set CUDA_VISIBLE_DEVICES tới GPU rảnh nhất.

    Skip nếu user đã export sẵn CUDA_VISIBLE_DEVICES.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        print(f"[gpu-pick] CUDA_VISIBLE_DEVICES đã set = "
              f"{os.environ['CUDA_VISIBLE_DEVICES']} (respect user)")
        return
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            print("[gpu-pick] nvidia-smi failed, GPU 0 default")
            return

        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    idx = int(parts[0])
                    mem_used = int(parts[1])   # MiB
                    util = int(parts[2])       # %
                    # Score: prioritize low memory used, weight util heavier
                    # (util = active compute, mem = just allocated).
                    score = mem_used + util * 100
                    gpus.append((idx, score, mem_used, util))
                except ValueError:
                    continue

        if not gpus:
            print("[gpu-pick] no GPU found, default")
            return

        gpus.sort(key=lambda g: g[1])   # ascending by score
        chosen_idx = gpus[0][0]

        print("[gpu-pick] GPU status (auto-pick least busy):")
        for idx, _, mem, util in gpus:
            marker = "  ← SELECTED" if idx == chosen_idx else ""
            print(f"  GPU {idx}: mem {mem:>6} MiB · util {util:>3}%{marker}")

        os.environ["CUDA_VISIBLE_DEVICES"] = str(chosen_idx)
    except FileNotFoundError:
        print("[gpu-pick] nvidia-smi không tìm thấy, GPU 0 default")
    except Exception as e:
        print(f"[gpu-pick] error: {e}, GPU 0 default")


_auto_pick_gpu()


import gradio as gr


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ReDI-GS Gradio demo.")
    parser.add_argument("--port", type=int, default=7860,
                        help="Gradio server port")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Interface to bind (0.0.0.0 for network share)")
    parser.add_argument("--server-mode", action="store_true", default=True,
                        help="Full compute, render live (default)")
    parser.add_argument("--local-mode", action="store_true", default=False,
                        help="Cache-only backup, no GPU, no live training")
    parser.add_argument("--cache-dir", default="demo/cache",
                        help="Where the pre-generated content lives")
    parser.add_argument("--share", action="store_true", default=False,
                        help="Create a public Gradio share URL")
    args = parser.parse_args()

    if args.local_mode:
        args.server_mode = False
    return args


# ---------------------------------------------------------------------------
# Tab 1 — Pre-computed viewer (Phase 2 full)
# ---------------------------------------------------------------------------


def build_precompute_tab_wrapper(cache_dir: Path, local_mode: bool) -> None:
    """Layout Tab 1 dùng demo/ui_precompute.py."""
    from demo.ui_precompute import build_precompute_tab
    build_precompute_tab(cache_dir)


# ---------------------------------------------------------------------------
# Tab B — Live Render from .ply (Phase 3, MỚI)
# ---------------------------------------------------------------------------


def build_liverender_tab_wrapper(local_mode: bool) -> None:
    """Layout Tab B dùng demo/ui_liverender.py.

    Local mode: KHÔNG import demo.ui_liverender (module đó cần torch cho
    CUDA rasterizer). Chỉ hiện markdown giải thích.
    """
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
    from demo.ui_liverender import build_liverender_tab
    build_liverender_tab(local_mode)


# ---------------------------------------------------------------------------
# Tab C — Live Training (Phase 4)
# ---------------------------------------------------------------------------


def build_livetraining_tab_wrapper(local_mode: bool) -> None:
    """Layout Tab C dùng demo/ui_livetraining.py.

    Local mode: KHÔNG import demo.ui_livetraining (cần torch cho render
    checkpoint + subprocess train.py). Chỉ hiện markdown giải thích.
    """
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
    from demo.ui_livetraining import build_livetraining_tab
    build_livetraining_tab(local_mode)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_app(args: argparse.Namespace) -> gr.Blocks:
    """Compose UI 2 tab."""
    cache_dir = Path(args.cache_dir)
    if args.local_mode and not cache_dir.exists():
        raise FileNotFoundError(
            f"Local mode yêu cầu cache đã sync về, không thấy {cache_dir}. "
            "Chạy `bash demo/populate_cache.sh` trên server rồi rsync về."
        )

    with gr.Blocks(title="ReDI-GS Demo",
                   theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# ReDI-GS — Reliable Dense Initialization for Sparse-view GS\n"
            "Novel view synthesis with 3D Gaussian Splatting from sparse images."
        )
        with gr.Tabs():
            with gr.TabItem("Pre-computed"):
                build_precompute_tab_wrapper(cache_dir, args.local_mode)
            with gr.TabItem("Live Render"):
                build_liverender_tab_wrapper(args.local_mode)
            with gr.TabItem("Live Training"):
                build_livetraining_tab_wrapper(args.local_mode)
    return app


def main() -> None:
    args = parse_args()
    app = build_app(args)
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
