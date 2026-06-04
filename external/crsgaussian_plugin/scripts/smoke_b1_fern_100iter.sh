#!/bin/bash
# ============================================================
# [CRSGaussian Path A B1] Smoke test — fern 100 iter init verify
# File: crsgaussian_plugin/scripts/smoke_b1_fern_100iter.sh
#
# Mục đích: Verify method `crsgaussian` register OK + init 23420 points từ
#           fused.ply.romav1 + 100 iter không crash.
#
# B1 expectations (KHÔNG expect PSNR cao):
#   - Init Gaussians = 23420 (RoMa v1 fern)
#   - 100 iter complete (Trainer loop không crash)
#   - TensorBoard ghi train_loss + psnr metric
#   - PSNR có thể thấp vì B1 chưa hook CoR-GS optimizer.step (loss không giảm)
#
# Run từ working dir Nerfstudio:
#   cd ~/workspace/representation-3d/duyen/nerfstudio
#   bash crsgaussian_plugin/scripts/smoke_b1_fern_100iter.sh
# ============================================================

set -e

# ── ENV setup ──
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

# CRITICAL: PYTHONPATH = nerfstudio workdir (cwd shadow safe)
# CORGS_SOURCE_PATH = CoR-GS clone trên server
export PYTHONPATH=/home/aidev/workspace/representation-3d/duyen/nerfstudio:$PYTHONPATH
export CORGS_SOURCE_PATH=/home/aidev/workspace/representation-3d/duyen/CoR-GS

# Register method qua env var
export NERFSTUDIO_METHOD_CONFIGS="crsgaussian=crsgaussian_plugin:crsgaussian_method_spec"

# ── Paths ──
DATA_ROOT=/home/aidev/workspace/representation-3d/duyen/CoR-GS/data/nerf_llff_data
SCENE=fern
OUT_ROOT=/home/aidev/workspace/representation-3d/duyen/nerfstudio/outputs_smoke_b1
LOG_DIR=/home/aidev/workspace/representation-3d/duyen/nerfstudio/logs/smoke_b1
mkdir -p $OUT_ROOT $LOG_DIR

LOG=$LOG_DIR/smoke_b1_${SCENE}_100iter.log

# ── Pre-check: PLY exist ──
PLY=$DATA_ROOT/$SCENE/3_views/dense/fused.ply.romav1
if [ ! -f "$PLY" ]; then
    echo "❌ MISSING PLY: $PLY"
    echo "   Run RoMa v1 preprocess for $SCENE first."
    exit 1
fi

# ── Pre-check: gsplat OK (no JIT compile, cwd-safe) ──
cd /tmp
python -c "import gsplat; print('gsplat OK', gsplat.__file__)" || {
    echo "❌ gsplat import fail"
    exit 1
}

# ── Pre-check: CoR-GS import ──
python -c "
import sys
sys.path.insert(0, '$CORGS_SOURCE_PATH')
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
print('CoR-GS import OK')
" || {
    echo "❌ CoR-GS import fail"
    exit 1
}

# ── Pre-check: method registered ──
cd /home/aidev/workspace/representation-3d/duyen/nerfstudio
python -c "
from crsgaussian_plugin import crsgaussian_method_spec
print('Method spec OK:', crsgaussian_method_spec.config.method_name)
" || {
    echo "❌ Method spec build fail"
    exit 1
}

echo "[$(date +%H:%M:%S)] All pre-checks PASS → start smoke train"

# ── Smoke train ──
cd /home/aidev/workspace/representation-3d/duyen/nerfstudio

CUDA_VISIBLE_DEVICES=1 ns-train crsgaussian \
    --data $DATA_ROOT/$SCENE/3_views/ \
    --output-dir $OUT_ROOT \
    --experiment-name $SCENE \
    --max-num-iterations 100 \
    --steps-per-eval-all-images 100 \
    --vis tensorboard \
    2>&1 | tee $LOG

# ── Post-check: B1 verify gates ──
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  B1 VERIFY GATES"
echo "═══════════════════════════════════════════════════════════"

# Gate B1.1: Method registered
if grep -q "crsgaussian" $LOG; then
    echo "✅ B1.1 Method registered (log có 'crsgaussian')"
else
    echo "❌ B1.1 Method NOT registered"
fi

# Gate B1.2: Init 23420 points
if grep -q "Loaded.*from dense/fused.ply.romav1\|Init CoR-GS GaussianModel from 23420" $LOG; then
    echo "✅ B1.2 Init RoMa PLY OK"
else
    echo "❌ B1.2 Init RoMa PLY FAIL (check log)"
fi

# Gate B1.3: Train complete (rich progress format: "Training Finished" or "99 (99.00%)")
if grep -qE "Training Finished|99 \(99\.00%\)|100 \(100\.00%\)" $LOG; then
    echo "✅ B1.3 Train complete"
else
    echo "❌ B1.3 Train did NOT complete"
fi

# Gate B1.4: TensorBoard event file exists
RUN_DIR=$(ls -td $OUT_ROOT/$SCENE/crsgaussian/* 2>/dev/null | head -1)
if [ -n "$RUN_DIR" ] && ls $RUN_DIR/events.out.tfevents.* 1>/dev/null 2>&1; then
    echo "✅ B1.4 TensorBoard event file exists: $RUN_DIR"
else
    echo "❌ B1.4 No TensorBoard event file"
fi

echo "═══════════════════════════════════════════════════════════"
echo "  Log full: $LOG"
echo "  Run dir:  $RUN_DIR"
echo "═══════════════════════════════════════════════════════════"
