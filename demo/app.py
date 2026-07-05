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
import sys
from pathlib import Path

# Cho phép chạy `python demo/app.py` (không có -m). Khi chạy như script,
# sys.path[0] là `demo/`, không thấy được package `demo` → cần thêm repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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
# Tab 2 — Live training viewer (placeholder Phase 1, đầy đủ ở Phase 3)
# ---------------------------------------------------------------------------


def build_live_training_tab(local_mode: bool) -> gr.Blocks:
    """Layout Tab 2. Ở Phase 1 chỉ hiện label placeholder."""
    with gr.Blocks() as tab:
        gr.Markdown("## Tab 2 — Live training viewer")
        if local_mode:
            gr.Markdown(
                "**Local mode**: sẽ play video MP4 pre-recorded thay vì "
                "chạy train live (server không sẵn)."
            )
        else:
            gr.Markdown(
                "**Server mode**: sẽ chạy `train.py` subprocess và poll "
                "iteration folders mỗi 30 giây để update test view + PSNR."
            )
        gr.Markdown(
            "🚧 Phase 1 scaffold. Subprocess launcher và iteration poller "
            "sẽ được implement ở Phase 3."
        )
    return tab


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
            with gr.TabItem("Live Training"):
                build_live_training_tab(args.local_mode)
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
