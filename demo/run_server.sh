#!/usr/bin/env bash
# ============================================================
# Launch Gradio demo trên server.
# ============================================================
# Prerequisites:
#   conda activate gradio_demo
#   cd ~/workspace/representation-3d/duyen/CoR-GS
#   pip install -r demo/requirements.txt   (chạy 1 lần)
#
# Usage:
#   bash demo/run_server.sh
#   bash demo/run_server.sh --port 7861   # đổi port
# ============================================================
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Env sanity: chắc chắn đúng conda env
if [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
    echo "[demo] WARNING: no conda env active; expected 'gradio_demo'"
elif [ "${CONDA_DEFAULT_ENV}" != "gradio_demo" ]; then
    echo "[demo] WARNING: current env = ${CONDA_DEFAULT_ENV}, expected 'gradio_demo'"
fi

PORT="${PORT:-7860}"
HOST="${HOST:-127.0.0.1}"
EXTRA_ARGS="${*:-}"

echo "[demo] launching Gradio on http://${HOST}:${PORT}"
python demo/app.py --host "${HOST}" --port "${PORT}" ${EXTRA_ARGS}
