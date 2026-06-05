#!/bin/bash
# ============================================================
# [CRSGaussian Path A B4] 4-scene multi-scene benchmark — Phase 22 plug-in
# File: crsgaussian_plugin/scripts/benchmark_phase22_4scenes.sh
#
# Scenes: fern + horns + fortress + flower (mix easy/medium/hard)
# Compute: 2 GPU parallel, single seed first pass, ~30 phút wall-clock.
#
# Phase 22 standalone reference (3-seed N=24 mean):
#   fern     23.84   horns    21.08
#   fortress 25.57   flower   21.41
#   AVG = 22.97
#
# Acceptance:
#   Gate 1: All 4 scenes train complete
#   Gate 2: AVG PSNR > 22.5
#   Gate 3: 3/4 scenes within ±1 dB of Phase 22 ref
# ============================================================

set -e

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export PYTHONPATH=/home/aidev/workspace/representation-3d/duyen/nerfstudio:$PYTHONPATH
export CORGS_SOURCE_PATH=/home/aidev/workspace/representation-3d/duyen/CoR-GS
export NERFSTUDIO_METHOD_CONFIGS="crsgaussian=crsgaussian_plugin:crsgaussian_method_spec"

DATA_ROOT=/home/aidev/workspace/representation-3d/duyen/CoR-GS/data/nerf_llff_data
OUT_ROOT=/home/aidev/workspace/representation-3d/duyen/nerfstudio/outputs/phase22_b4
LOG_DIR=/home/aidev/workspace/representation-3d/duyen/nerfstudio/logs/phase22_b4
mkdir -p $OUT_ROOT $LOG_DIR

cd /home/aidev/workspace/representation-3d/duyen/nerfstudio

# ── Phase 22 standalone reference (3-seed N=24 mean) ──
declare -A PHASE22_REF=(
    [fern]=23.84
    [horns]=21.08
    [fortress]=25.57
    [flower]=21.41
)

# ── Train function ──
run_scene() {
    local SCENE=$1
    local GPU=$2
    local LOG=$LOG_DIR/${SCENE}_seed0.log

    echo "[$(date +%H:%M:%S)] [GPU $GPU] Starting $SCENE 10k train..."

    CUDA_VISIBLE_DEVICES=$GPU ns-train crsgaussian \
        --data $DATA_ROOT/$SCENE/ \
        --output-dir $OUT_ROOT \
        --experiment-name $SCENE \
        --max-num-iterations 10000 \
        --steps-per-eval-all-images 10000 \
        --vis tensorboard \
        > $LOG 2>&1

    echo "[$(date +%H:%M:%S)] [GPU $GPU] $SCENE train DONE → eval"

    local RUN_DIR=$(ls -td $OUT_ROOT/$SCENE/crsgaussian/* | head -1)
    local EVAL_JSON=/tmp/eval_phase22_b4_${SCENE}.json

    CUDA_VISIBLE_DEVICES=$GPU ns-eval \
        --load-config $RUN_DIR/config.yml \
        --output-path $EVAL_JSON >> $LOG 2>&1

    echo "[$(date +%H:%M:%S)] [GPU $GPU] $SCENE eval DONE"
}

# ════════════════════════════════════════════════════════
# Run 4 scenes parallel: GPU 0 (fern + fortress), GPU 1 (horns + flower)
# Each GPU runs 2 scenes sequentially (12-15 min each = 25-30 min total)
# ════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════════════════"
echo "  Phase 22 PLUG-IN B4 — 4-scene benchmark"
echo "  GPU 0: fern → fortress"
echo "  GPU 1: horns → flower"
echo "  Expected: ~30 phút wall-clock"
echo "═══════════════════════════════════════════════════════════"

(run_scene fern 0 && run_scene fortress 0) &
GPU0_PID=$!
(run_scene horns 1 && run_scene flower 1) &
GPU1_PID=$!

wait $GPU0_PID
wait $GPU1_PID

echo ""
echo "[$(date +%H:%M:%S)] ALL 4 SCENES DONE"
echo ""

# ════════════════════════════════════════════════════════
# Summary table + acceptance gates
# ════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════════════════"
echo "  Phase 22 PLUG-IN B4 RESULTS"
echo "═══════════════════════════════════════════════════════════"
echo ""
printf "  %-10s | %-8s | %-8s | %-10s | %-7s\n" "Scene" "Plug-in" "Phase 22" "Δ vs ref" "Verdict"
echo "  -------------------------------------------------------------"

SUM_PLUGIN=0
SUM_REF=0
COUNT=0
N_PASS=0  # scenes within ±1 dB of ref

for SCENE in fern horns fortress flower; do
    EVAL_JSON=/tmp/eval_phase22_b4_${SCENE}.json
    if [ -f "$EVAL_JSON" ]; then
        PSNR=$(python -c "import json; print(f\"{json.load(open('$EVAL_JSON'))['results']['psnr']:.2f}\")" 2>/dev/null)
        REF=${PHASE22_REF[$SCENE]}
        DELTA=$(python -c "print(f\"{$PSNR - $REF:+.2f}\")" 2>/dev/null)
        # Verdict: within ±1 dB
        VERDICT=$(python -c "print('✅ PASS' if abs($PSNR - $REF) < 1.0 or $PSNR >= $REF else '⚠️ DROP')" 2>/dev/null)
        printf "  %-10s | %-8s | %-8s | %-10s | %-7s\n" "$SCENE" "$PSNR" "$REF" "$DELTA" "$VERDICT"
        SUM_PLUGIN=$(python -c "print($SUM_PLUGIN + $PSNR)")
        SUM_REF=$(python -c "print($SUM_REF + $REF)")
        COUNT=$((COUNT + 1))
        if [ "$VERDICT" = "✅ PASS" ]; then
            N_PASS=$((N_PASS + 1))
        fi
    else
        printf "  %-10s | %-8s | %-8s | %-10s | %-7s\n" "$SCENE" "NO_JSON" "${PHASE22_REF[$SCENE]}" "—" "❌ FAIL"
    fi
done

echo "  -------------------------------------------------------------"
if [ $COUNT -gt 0 ]; then
    AVG_PLUGIN=$(python -c "print(f\"{$SUM_PLUGIN / $COUNT:.2f}\")")
    AVG_REF=$(python -c "print(f\"{$SUM_REF / $COUNT:.2f}\")")
    DELTA_AVG=$(python -c "print(f\"{$SUM_PLUGIN / $COUNT - $SUM_REF / $COUNT:+.2f}\")")
    printf "  %-10s | %-8s | %-8s | %-10s |\n" "AVG" "$AVG_PLUGIN" "$AVG_REF" "$DELTA_AVG"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Acceptance gates"
echo "═══════════════════════════════════════════════════════════"
echo "  Gate 1: All 4 scenes complete:  $COUNT/4"
echo "  Gate 2: AVG PSNR > 22.5:         $AVG_PLUGIN (target > 22.5)"
echo "  Gate 3: 3/4 scenes within ref:   $N_PASS/4"
echo ""
echo "  Run dir:  $OUT_ROOT"
echo "  Logs:     $LOG_DIR"
echo "═══════════════════════════════════════════════════════════"
