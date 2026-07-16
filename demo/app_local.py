"""ReDI-GS Gradio LOCAL backup — entry point riêng cho laptop offline.

Chỉ Tab A Pre-computed. KHÔNG import torch / ui_liverender / ui_livetraining.
Chạy được trên Windows/Mac laptop không cần CUDA.

Usage:
    conda activate gradio_local
    python demo/app_local.py --port 7861

Layout:
- Tab "Pre-computed" — full work (đọc PNG/GIF từ demo/cache/).
- Tab "Live Render"    — placeholder markdown "server required".
- Tab "Live Training"  — placeholder markdown "server required".

Khác với `demo/app.py`:
- KHÔNG import demo/ui_liverender.py hoặc demo/ui_livetraining.py
  → tránh module-level `import torch` fail trên laptop không có CUDA.
- KHÔNG có auto-pick GPU (không dùng GPU).
- Default port 7861 (thay 7860) để không clash với SSH tunnel.
- `show_api=False` để tránh gradio_client schema bug.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Cho phép chạy `python demo/app_local.py` (không có -m). Khi chạy như
# script, sys.path[0] là `demo/`, không thấy được package `demo`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Bypass proxy cho localhost — Windows thường có HTTP_PROXY set khiến
# Gradio self-check (request tới 127.0.0.1) đi qua proxy fail.
# Set NO_PROXY BEFORE import gradio để urllib/requests respect.
# ---------------------------------------------------------------------------


os.environ["NO_PROXY"] = "127.0.0.1,localhost,0.0.0.0"
os.environ["no_proxy"] = "127.0.0.1,localhost,0.0.0.0"


# ---------------------------------------------------------------------------
# Monkey-patch gradio_client bug — TRƯỚC khi import gradio.
#
# gradio_client 1.3.0 (đi cùng gradio 4.44.1) crash trên schema=bool (True/
# False): `if "const" in schema` fail với "argument of type 'bool' is not
# iterable". Gradio startup gọi /gradio_api → parse schema → crash → tưởng
# localhost inaccessible → raise ValueError.
#
# Patch `gradio_client.utils.get_type` để handle bool trước khi in check.
# ---------------------------------------------------------------------------


def _patch_gradio_client() -> None:
    """Patch 3 layer — deep-fix cho bug schema=bool.

    Layer 1: `get_type(bool)` → return "bool".
    Layer 2: `_json_schema_to_python_type(bool, ...)` → return "bool".
    Layer 3 (safety net): override `Blocks.get_api_info` → return empty
             dict. Local backup KHÔNG cần API info (chỉ dùng browser UI).
    """
    try:
        import gradio_client.utils as gcu

        # Layer 1
        _orig_get_type = gcu.get_type

        def _safe_get_type(schema):
            if isinstance(schema, bool):
                return "bool"
            if schema is None:
                return "None"
            try:
                return _orig_get_type(schema)
            except (TypeError, Exception):
                return "Any"

        gcu.get_type = _safe_get_type

        # Layer 2: patch _json_schema_to_python_type recursion
        _orig_json = gcu._json_schema_to_python_type

        def _safe_json(schema, defs=None):
            if isinstance(schema, bool):
                return "bool"
            if schema is None:
                return "None"
            try:
                return _orig_json(schema, defs)
            except Exception:
                return "Any"

        gcu._json_schema_to_python_type = _safe_json

        # Layer 2b: outer wrapper
        _orig_outer = gcu.json_schema_to_python_type

        def _safe_outer(schema):
            try:
                return _orig_outer(schema)
            except Exception:
                return "Any"

        gcu.json_schema_to_python_type = _safe_outer

        print("[local] patched gradio_client.utils (3 layers, bool-schema)")
    except Exception as e:
        print(f"[local] gradio_client patch skipped: {e}")


_patch_gradio_client()


import gradio as gr


def _patch_gradio_blocks() -> None:
    """Layer 3: override Blocks.get_api_info → return empty.

    Local backup KHÔNG dùng API info. Bypass hoàn toàn schema generation.
    """
    try:
        _orig = gr.Blocks.get_api_info

        def _empty_api_info(self):
            return {"named_endpoints": {}, "unnamed_endpoints": {}}

        gr.Blocks.get_api_info = _empty_api_info
        print("[local] patched gr.Blocks.get_api_info (bypass schema gen)")
    except Exception as e:
        print(f"[local] Blocks.get_api_info patch skipped: {e}")


_patch_gradio_blocks()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ReDI-GS Local backup demo (cache-only, no GPU).")
    parser.add_argument("--port", type=int, default=7861,
                        help="Gradio server port (default 7861 để không "
                             "clash SSH tunnel 7860)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Interface to bind (127.0.0.1 = localhost only)")
    parser.add_argument("--cache-dir", default="demo/cache",
                        help="Where the pre-generated content lives")
    parser.add_argument("--share", action="store_true", default=False,
                        help="Create a public Gradio share URL")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Placeholder tab cho Tab B/C (không import module thật)
# ---------------------------------------------------------------------------


def _placeholder_tab(name: str) -> None:
    """Markdown giải thích tab này chỉ work trên server."""
    gr.Markdown(
        f"### ⚠️ {name} — server GPU required\n\n"
        f"This tab is unavailable in **local backup mode**. Only the "
        f"**Pre-computed** tab works offline (reads cached PNG/GIF from "
        f"disk).\n\n"
        f"To use {name.lower()}, run the demo on the server:\n"
        f"```bash\n"
        f"conda activate gradio_demo\n"
        f"bash demo/run_server.sh\n"
        f"```"
    )


# ---------------------------------------------------------------------------
# Build app
# ---------------------------------------------------------------------------


def build_app(args: argparse.Namespace) -> gr.Blocks:
    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        raise FileNotFoundError(
            f"Local mode yêu cầu cache đã sync về, không thấy {cache_dir}. "
            f"Chạy: scp -r aidev@<server>:~/workspace/representation-3d/"
            f"duyen/CoR-GS/demo/cache/* {cache_dir}/"
        )

    with gr.Blocks(
        title="ReDI-GS Demo (Local Backup)",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            "# ReDI-GS — Reliable Dense Initialization for Sparse-view GS\n"
            "**Local backup mode** · Novel view synthesis with 3D Gaussian "
            "Splatting from sparse images."
        )
        with gr.Tabs():
            with gr.TabItem("Pre-computed"):
                # Import Tab A chỉ khi cần → không rác cho local (Tab A
                # không cần torch, chỉ đọc PIL từ cache).
                from demo.ui_precompute import build_precompute_tab
                build_precompute_tab(cache_dir)
            with gr.TabItem("Live Render"):
                _placeholder_tab("Live Render")
            with gr.TabItem("Live Training"):
                _placeholder_tab("Live Training")
    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    print(f"[local] launching Gradio on http://{args.host}:{args.port}")
    print(f"[local] cache dir: {args.cache_dir}")

    app = build_app(args)

    # Try 1: bind localhost bình thường.
    try:
        app.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            show_error=True,
            show_api=False,     # tránh gradio_client schema bug
        )
    except ValueError as e:
        if "localhost is not accessible" not in str(e):
            raise
        # Windows firewall/proxy block self-check → thử 2 fallback:
        print(f"\n[local] ⚠️  localhost check failed: {e}")
        print("[local] retry với server_name=0.0.0.0 (bind all interfaces)...")
        try:
            app.launch(
                server_name="0.0.0.0",
                server_port=args.port,
                share=args.share,
                show_error=True,
                show_api=False,
            )
        except ValueError as e2:
            if "localhost is not accessible" not in str(e2):
                raise
            print(f"\n[local] ⚠️  0.0.0.0 vẫn fail: {e2}")
            print("[local] final fallback: share=True (tạo public URL)...")
            app.launch(
                server_name="0.0.0.0",
                server_port=args.port,
                share=True,        # tạo public share URL, bypass check
                show_error=True,
                show_api=False,
            )


if __name__ == "__main__":
    main()
