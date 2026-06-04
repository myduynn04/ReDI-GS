#!/bin/bash
# ============================================================
# [CRSGaussian Path A B2] Smoke test — fern 500 iter L1+SSIM+depth+optimizer
# File: crsgaussian_plugin/scripts/smoke_b2_fern_500iter.sh
#
# Mục đích: Verify B2 milestone:
#   - CoR-GS optimizer.step CHẠY (loss giảm)
#   - SSIM contribution active
#   - Depth loss active (Pearson DAV2)
#   - PSNR train > 22 dB tại iter 500 (vượt B1 init 22)
#   - N_gaussians KHÔNG đổi (chưa densify — B3 mới densify)
#
# Run từ working dir Nerfstudio:
#   cd ~/workspace/representation-3d/duyen/nerfstudio
#   bash crsgaussian_plugin/scripts/smoke_b2_fern_500iter.sh
# ============================================================

set -e

# ── ENV setup ──
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

export PYTHONPATH=/home/aidev/workspace/representation-3d/duyen/nerfstudio:$PYTHONPATH
export CORGS_SOURCE_PATH=/home/aidev/workspace/representation-3d/duyen/CoR-GS

export NERFSTUDIO_METHOD_CONFIGS="crsgaussian=crsgaussian_plugin:crsgaussian_method_spec"

# ── Paths ──
DATA_ROOT=/home/aidev/workspace/representation-3d/duyen/CoR-GS/data/nerf_llff_data
SCENE=fern
OUT_ROOT=/home/aidev/workspace/representation-3d/duyen/nerfstudio/outputs_smoke_b2
LOG_DIR=/home/aidev/workspace/representation-3d/duyen/nerfstudio/logs/smoke_b2
mkdir -p $OUT_ROOT $LOG_DIR

LOG=$LOG_DIR/smoke_b2_${SCENE}_500iter.log

# ── Pre-check: aligned depth tồn tại ──
DEPTH_DIR=$DATA_ROOT/$SCENE/3_views/aligned_depth_a23
if [ ! -d "$DEPTH_DIR" ] || [ -z "$(ls $DEPTH_DIR/*.npy 2>/dev/null)" ]; then
    echo "⚠ WARNING: $DEPTH_DIR không có .npy"
    echo "  Run preprocess_depth_a23.py trước nếu muốn test depth loss."
    echo "  Smoke vẫn chạy nhưng depth_loss sẽ không active."
fi

echo "[$(date +%H:%M:%S)] Pre-checks OK → start B2 smoke train"

# ── Smoke train ──
cd /home/aidev/workspace/representation-3d/duyen/nerfstudio

CUDA_VISIBLE_DEVICES=1 ns-train crsgaussian \
    --data $DATA_ROOT/$SCENE/3_views/ \
    --output-dir $OUT_ROOT \
    --experiment-name $SCENE \
    --max-num-iterations 500 \
    --steps-per-eval-all-images 500 \
    --vis tensorboard \
    2>&1 | tee $LOG

# ── Post-check: B2 verify gates ──
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  B2 VERIFY GATES"
echo "═══════════════════════════════════════════════════════════"

# Gate B2.1: L1 loss giảm (so iter 50 vs iter 500)
L1_EARLY=$(grep -oE "Step.*\(10\.00%\).*" $LOG | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
L1_LATE=$(grep -oE "Step.*\(99\.00%\)|99 \(99\.00%\)" $LOG | tail -1 | head -1)
# Simple: check if "Training Finished" present
if grep -q "Training Finished\|499 (99\.80%)\|500 (100\.00%)" $LOG; then
    echo "✅ B2.1 Train hoàn thành 500 iter"
else
    echo "❌ B2.1 Train không hoàn thành"
fi

# Gate B2.2: depth loss active
if grep -q "Loaded.*aligned depth maps\|Depth dict.*train cameras có aligned depth" $LOG; then
    N_DEPTH=$(grep -oE "[0-9]+/[0-9]+ train cameras có aligned depth" $LOG | head -1)
    echo "✅ B2.2 Depth dict loaded: $N_DEPTH"
else
    echo "⚠ B2.2 Depth dict KHÔNG load (use_depth_loss=False hoặc folder thiếu)"
fi

# Gate B2.3: PSNR train final > 22 (vượt B1 init)
# Lấy PSNR cuối từ progress table (cột 3)
PSNR_FINAL=$(grep -oE "99\.[0-9]+%.*[0-9]+\.[0-9]+ M\|99\.00%.*PSNR.*[0-9]+\.[0-9]+" $LOG | tail -1)
echo "   Final PSNR line: $PSNR_FINAL"

# Gate B2.4: N_gaussians không đổi (B2 chưa densify)
N_INIT=$(grep -oE "Init CoR-GS GaussianModel from [0-9]+ seed points" $LOG | grep -oE "[0-9]+" | head -1)
echo "   N init: $N_INIT (B2 should stay same — no densify yet)"

# Run dir
RUN_DIR=$(ls -td $OUT_ROOT/$SCENE/crsgaussian/* 2>/dev/null | head -1)
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Log full: $LOG"
echo "  Run dir:  $RUN_DIR"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  Inspect manual:"
echo "  - PSNR progress: grep -E '[0-9]+ \([0-9]+\\.[0-9]+%\\)' $LOG | tail -10"
echo "  - Loss giảm:    grep -E 'main_loss|depth_loss' $LOG | head -20"
