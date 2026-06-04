#!/bin/bash
# ============================================================
# [CRSGaussian Path A B3 PHASE22] Fern 10k benchmark — full Phase 22 recipe
# File: crsgaussian_plugin/scripts/benchmark_phase22_fern_10k.sh
#
# 7 modules ON: depth + D_cycle + SH_freeze + S_stability +
#               DropAnSH + opacity_decay + LFCF/AbsGS + reset_opacity
#
# Target: PSNR ≥ 19.0 (vượt Splatfacto baseline 19.08)
# Stretch: PSNR ≥ 22.0 (95% Phase 22 standalone fern 23.84)
# ============================================================

set -e

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export PYTHONPATH=/home/aidev/workspace/representation-3d/duyen/nerfstudio:$PYTHONPATH
export CORGS_SOURCE_PATH=/home/aidev/workspace/representation-3d/duyen/CoR-GS
export NERFSTUDIO_METHOD_CONFIGS="crsgaussian=crsgaussian_plugin:crsgaussian_method_spec"

DATA_ROOT=/home/aidev/workspace/representation-3d/duyen/CoR-GS/data/nerf_llff_data
SCENE=fern
OUT_ROOT=/home/aidev/workspace/representation-3d/duyen/nerfstudio/outputs_phase22_b3
LOG_DIR=/home/aidev/workspace/representation-3d/duyen/nerfstudio/logs/phase22_b3
mkdir -p $OUT_ROOT $LOG_DIR

LOG=$LOG_DIR/phase22_${SCENE}_10k.log

echo "[$(date +%H:%M:%S)] Phase 22 plug-in benchmark fern 10k (~12-15 min)"

cd /home/aidev/workspace/representation-3d/duyen/nerfstudio

CUDA_VISIBLE_DEVICES=1 ns-train crsgaussian \
    --data $DATA_ROOT/$SCENE/3_views/ \
    --output-dir $OUT_ROOT \
    --experiment-name $SCENE \
    --max-num-iterations 10000 \
    --steps-per-eval-all-images 10000 \
    --vis tensorboard \
    2>&1 | tee $LOG

echo "[$(date +%H:%M:%S)] Train DONE → eval"

RUN_DIR=$(ls -td $OUT_ROOT/$SCENE/crsgaussian/* | head -1)
EVAL_JSON=/tmp/eval_phase22_${SCENE}.json

CUDA_VISIBLE_DEVICES=1 ns-eval \
    --load-config $RUN_DIR/config.yml \
    --output-path $EVAL_JSON 2>&1 | tail -5

# ── Verify gates ──
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Phase 22 PLUG-IN fern 10k VERIFY"
echo "═══════════════════════════════════════════════════════════"

# Gate 1: Train complete
if grep -q "Training Finished" $LOG; then
    echo "✅ Train 10k complete"
else
    echo "❌ Train DID NOT complete — check $LOG"
fi

# Gate 2: Reset opacity calls (expect 4: iter 501, 3501, 6501, 9501)
N_RESETS=$(grep -c "reset_opacity at iter" $LOG || echo 0)
echo "  reset_opacity calls: $N_RESETS (expect 4)"

# Gate 3: Densify N_gauss progression
N_FINAL=$(grep -oE "N_gauss=[0-9]+" $LOG | tail -1 | grep -oE "[0-9]+")
echo "  N_gauss final: ${N_FINAL:-NOT_LOGGED}"

# Gate 4: PSNR
if [ -f "$EVAL_JSON" ]; then
    PSNR=$(python -c "import json; print(f\"{json.load(open('$EVAL_JSON'))['results']['psnr']:.2f}\")" 2>/dev/null)
    echo ""
    echo "  ┌─────────────────────────────────────────┐"
    echo "  │  Eval PSNR:   $PSNR dB"
    echo "  │  Splatfacto:  19.08 (baseline)"
    echo "  │  Phase 22:    23.84 (standalone fern)"
    echo "  └─────────────────────────────────────────┘"
else
    echo "❌ Eval JSON not found"
fi

echo ""
echo "  Log: $LOG"
echo "  Run: $RUN_DIR"
echo "═══════════════════════════════════════════════════════════"
