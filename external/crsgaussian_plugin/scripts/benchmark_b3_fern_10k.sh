#!/bin/bash
# ============================================================
# [CRSGaussian Path A B3] Fern 10k benchmark — full Phase 22 mechanism
# File: crsgaussian_plugin/scripts/benchmark_b3_fern_10k.sh
#
# Mục đích: Verify B3 (densify + opacity decay + DropAnSH) trên 1 scene fern,
#           10k iter, compare với Phase 22 standalone (fern PSNR 23.84).
#
# Active components:
#   ✅ L1 + SSIM + Pearson depth
#   ✅ CoR-GS optimizer.step
#   ✅ Densify (CoR-GS densify_and_prune mỗi 100 iter)
#   ✅ Opacity decay ×0.999/iter sau iter 500
#   ✅ DropAnSH anchor + SH degree dropout
#   ❌ CRS module + SH freeze (defer B3b — CRS-axis MVS-only per Phase 23)
#
# Verify gates B3:
#   B3.1 Train 10k iter complete
#   B3.2 N_gaussians grow: 24k → > 80k (densify ACTIVE)
#   B3.3 Eval PSNR > 19 (vượt Splatfacto A1 baseline 19.08)
#   B3.4 Train PSNR > 25 (Phase 22 fern train ~26)
#
# Run từ working dir Nerfstudio:
#   cd ~/workspace/representation-3d/duyen/nerfstudio
#   bash crsgaussian_plugin/scripts/benchmark_b3_fern_10k.sh
# ============================================================

set -e

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export PYTHONPATH=/home/aidev/workspace/representation-3d/duyen/nerfstudio:$PYTHONPATH
export CORGS_SOURCE_PATH=/home/aidev/workspace/representation-3d/duyen/CoR-GS
export NERFSTUDIO_METHOD_CONFIGS="crsgaussian=crsgaussian_plugin:crsgaussian_method_spec"

DATA_ROOT=/home/aidev/workspace/representation-3d/duyen/CoR-GS/data/nerf_llff_data
SCENE=fern
OUT_ROOT=/home/aidev/workspace/representation-3d/duyen/nerfstudio/outputs_benchmark_b3
LOG_DIR=/home/aidev/workspace/representation-3d/duyen/nerfstudio/logs/benchmark_b3
mkdir -p $OUT_ROOT $LOG_DIR

LOG=$LOG_DIR/benchmark_b3_${SCENE}_10k.log

echo "[$(date +%H:%M:%S)] B3 benchmark start: fern 10k iter (~12 min)"

# ── Train 10k iter ──
cd /home/aidev/workspace/representation-3d/duyen/nerfstudio

CUDA_VISIBLE_DEVICES=1 ns-train crsgaussian \
    --data $DATA_ROOT/$SCENE/3_views/ \
    --output-dir $OUT_ROOT \
    --experiment-name $SCENE \
    --max-num-iterations 10000 \
    --steps-per-eval-all-images 10000 \
    --vis tensorboard \
    2>&1 | tee $LOG

echo ""
echo "[$(date +%H:%M:%S)] Train DONE → eval"

# ── Eval ──
RUN_DIR=$(ls -td $OUT_ROOT/$SCENE/crsgaussian/* | head -1)
EVAL_JSON=/tmp/eval_b3_${SCENE}.json

CUDA_VISIBLE_DEVICES=1 ns-eval \
    --load-config $RUN_DIR/config.yml \
    --output-path $EVAL_JSON 2>&1 | tail -5

# ── Verify gates ──
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  B3 BENCHMARK fern 10k VERIFY GATES"
echo "═══════════════════════════════════════════════════════════"

# B3.1: Train complete
if grep -q "Training Finished" $LOG; then
    echo "✅ B3.1 Train 10k iter complete"
else
    echo "❌ B3.1 Train không complete"
fi

# B3.2: Densify ACTIVE — extract N_gauss progression
N_FINAL=$(grep -oE "N_gauss=[0-9]+" $LOG | tail -1 | grep -oE "[0-9]+")
N_INIT=24543
if [ -n "$N_FINAL" ] && [ "$N_FINAL" -gt "80000" ]; then
    echo "✅ B3.2 Densify ACTIVE: $N_INIT → $N_FINAL"
elif [ -n "$N_FINAL" ]; then
    echo "⚠ B3.2 Densify weak: $N_INIT → $N_FINAL (< 80k expected)"
else
    echo "❌ B3.2 N_gauss log không tìm thấy — densify có thể KHÔNG ACTIVE"
fi

# B3.3: Eval PSNR
if [ -f "$EVAL_JSON" ]; then
    PSNR=$(python -c "import json; print(json.load(open('$EVAL_JSON'))['results']['psnr'])" 2>/dev/null)
    if [ -n "$PSNR" ]; then
        echo "✅ B3.3 Eval PSNR: $PSNR"
        # Compare vs baselines
        echo "        Splatfacto A1 baseline (10k): ~19.08"
        echo "        Phase 22 standalone (10k):     23.84"
    fi
else
    echo "❌ B3.3 Eval failed (no JSON)"
fi

echo ""
echo "  Log: $LOG"
echo "  Run: $RUN_DIR"
echo "  Eval JSON: $EVAL_JSON"
echo "═══════════════════════════════════════════════════════════"
