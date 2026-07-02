#!/bin/bash
# ============================================================
# [Verify script 2026-07-02] Verify plug-in không hỏng sau clean/refactor.
# File: crsgaussian_plugin/scripts/verify_after_clean.sh
#
# 2 modes:
#   ./verify_after_clean.sh              → single fern (default)
#   ./verify_after_clean.sh fortress     → single scene by name
#   ./verify_after_clean.sh --full       → all 8 scenes parallel 2-GPU
#
# Flow mỗi scene:
#   1. Train 10k iter (~13 phút single GPU)
#   2. Compute PSNR + SSIM + LPIPS (eval_full_metrics.py)
#   3. Compare vs B6 baseline (2026-06-05 single-seed) trong tolerance ±1.3
#   4. Print PASS/FAIL
#
# Output isolated tại outputs/verify_after_clean/ — KHÔNG đụng phase22_b4 cũ.
#
# Wall-time:
#   single (fern default) → ~14 phút
#   --full 8-scene 2-GPU parallel → ~55 phút
# ============================================================

set -e

# ── Env vars ──
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export PYTHONPATH=/home/aidev/workspace/representation-3d/duyen/nerfstudio:$PYTHONPATH
export CORGS_SOURCE_PATH=/home/aidev/workspace/representation-3d/duyen/CoR-GS
export NERFSTUDIO_METHOD_CONFIGS="crsgaussian=crsgaussian_plugin:crsgaussian_method_spec"

# ── Paths ──
WORKDIR=/home/aidev/workspace/representation-3d/duyen/nerfstudio
DATA_ROOT=/home/aidev/workspace/representation-3d/duyen/CoR-GS/data/nerf_llff_data
OUT_ROOT=$WORKDIR/outputs/verify_after_clean
LOG_DIR=$WORKDIR/logs/verify_after_clean

# ── B6 baselines from doc 09 (single-seed 2026-06-05) ──
declare -A BASELINE_PSNR=(
    [fern]=25.77
    [horns]=21.41
    [fortress]=25.44
    [flower]=21.80
    [leaves]=20.27
    [orchids]=17.98
    [room]=22.76
    [trex]=24.20
)
BASELINE_AVG=22.45
TOLERANCE=1.3

# ── Parse args ──
if [ "$1" == "--full" ] || [ "$1" == "-f" ]; then
    SCENES=(fern horns fortress flower leaves orchids room trex)
    MODE="FULL 8-SCENE (2-GPU parallel)"
    PARALLEL=1
elif [ -n "$1" ]; then
    if [ -z "${BASELINE_PSNR[$1]}" ]; then
        echo "❌ Unknown scene: $1"
        echo "   Valid scenes: fern horns fortress flower leaves orchids room trex"
        exit 1
    fi
    SCENES=("$1")
    MODE="SINGLE SCENE: $1"
    PARALLEL=0
else
    SCENES=(fern)
    MODE="SINGLE SCENE: fern (default)"
    PARALLEL=0
fi

echo "═══════════════════════════════════════════════════════════"
echo "  Verify Plug-in After Clean — $MODE"
echo "  Started at $(date +%H:%M:%S)"
echo "  Output → $OUT_ROOT"
echo "  Baseline reference: B6 2026-06-05 single-seed (doc 09)"
echo "═══════════════════════════════════════════════════════════"

cd $WORKDIR
mkdir -p $OUT_ROOT $LOG_DIR

# ── Function chạy 1 scene ──
run_scene() {
    local SCENE=$1
    local GPU=$2
    local EXP_NAME="${SCENE}_verify"
    local LOG=$LOG_DIR/${SCENE}.log
    local EVAL_JSON=$OUT_ROOT/eval_${SCENE}_full_metrics.json

    echo "[$(date +%H:%M:%S)] [GPU$GPU] $SCENE — train 10k..."

    # Train 10k iter
    CUDA_VISIBLE_DEVICES=$GPU ns-train crsgaussian \
        --data $DATA_ROOT/$SCENE/ \
        --output-dir $OUT_ROOT \
        --experiment-name $EXP_NAME \
        --max-num-iterations 10000 \
        --steps-per-eval-all-images 10000 \
        --vis tensorboard \
        > $LOG 2>&1

    # Locate run dir
    local RUN_DIR=$(ls -td $OUT_ROOT/$EXP_NAME/crsgaussian/* | head -1)

    echo "[$(date +%H:%M:%S)] [GPU$GPU] $SCENE — eval PSNR+SSIM+LPIPS..."

    # Compute full metrics
    CUDA_VISIBLE_DEVICES=$GPU python crsgaussian_plugin/scripts/eval_full_metrics.py \
        --load-config $RUN_DIR/config.yml \
        --output-path $EVAL_JSON >> $LOG 2>&1

    echo "[$(date +%H:%M:%S)] [GPU$GPU] $SCENE — DONE"
}

# ── Chạy scene(s) ──
if [ $PARALLEL -eq 1 ]; then
    # FULL 8-scene 2-GPU parallel — split 4/4
    for i in "${!SCENES[@]}"; do
        SCENE="${SCENES[$i]}"
        GPU=$((i % 2))
        run_scene "$SCENE" "$GPU" &
        # Sau 2 scene (1 pair GPU0+GPU1) → wait để tránh over-schedule
        if [ $((i % 2)) -eq 1 ]; then
            wait
        fi
    done
    wait  # cuối cùng đảm bảo tất cả done
else
    # Single scene → GPU 1 (default trong bench script gốc)
    run_scene "${SCENES[0]}" 1
fi

# ── Final report ──
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  FINAL REPORT"
echo "═══════════════════════════════════════════════════════════"
printf "  %-10s %-8s %-8s %-8s %-10s %-8s %-9s\n" \
    "Scene" "PSNR" "SSIM" "LPIPS" "Baseline" "Δ_PSNR" "Verdict"
printf "  %-10s %-8s %-8s %-8s %-10s %-8s %-9s\n" \
    "─────" "────" "────" "─────" "────────" "──────" "───────"

TOTAL_PSNR=0
TOTAL_SSIM=0
TOTAL_LPIPS=0
N_PASS=0

for SCENE in "${SCENES[@]}"; do
    EVAL_JSON=$OUT_ROOT/eval_${SCENE}_full_metrics.json
    if [ ! -f "$EVAL_JSON" ]; then
        printf "  %-10s %-8s %-8s %-8s %-10s %-8s %-9s\n" "$SCENE" "—" "—" "—" "—" "—" "❌ NO_EVAL"
        continue
    fi

    PSNR=$(python -c "import json; print(f\"{json.load(open('$EVAL_JSON'))['psnr']:.2f}\")")
    SSIM=$(python -c "import json; print(f\"{json.load(open('$EVAL_JSON'))['ssim']:.4f}\")")
    LPIPS=$(python -c "import json; print(f\"{json.load(open('$EVAL_JSON'))['lpips']:.4f}\")")
    BASELINE=${BASELINE_PSNR[$SCENE]}
    DIFF=$(python -c "print(f'{$PSNR - $BASELINE:+.2f}')")
    ABS_DIFF=$(python -c "print(f'{abs($PSNR - $BASELINE):.2f}')")
    WITHIN_TOL=$(python -c "print('yes' if $ABS_DIFF < $TOLERANCE else 'no')")

    if [ "$WITHIN_TOL" == "yes" ]; then
        VERDICT="✅ PASS"
        N_PASS=$((N_PASS + 1))
    else
        VERDICT="❌ FAIL"
    fi

    printf "  %-10s %-8s %-8s %-8s %-10s %-8s %-9s\n" \
        "$SCENE" "$PSNR" "$SSIM" "$LPIPS" "$BASELINE" "$DIFF" "$VERDICT"

    TOTAL_PSNR=$(python -c "print($TOTAL_PSNR + $PSNR)")
    TOTAL_SSIM=$(python -c "print($TOTAL_SSIM + $SSIM)")
    TOTAL_LPIPS=$(python -c "print($TOTAL_LPIPS + $LPIPS)")
done

N=${#SCENES[@]}
if [ $N -gt 1 ]; then
    AVG_PSNR=$(python -c "print(f'{$TOTAL_PSNR/$N:.2f}')")
    AVG_SSIM=$(python -c "print(f'{$TOTAL_SSIM/$N:.4f}')")
    AVG_LPIPS=$(python -c "print(f'{$TOTAL_LPIPS/$N:.4f}')")
    DIFF_AVG=$(python -c "print(f'{$AVG_PSNR - $BASELINE_AVG:+.2f}')")

    echo ""
    printf "  %-10s %-8s %-8s %-8s %-10s %-8s\n" \
        "─────" "────" "────" "─────" "────────" "──────"
    printf "  %-10s %-8s %-8s %-8s %-10s %-8s\n" \
        "AVG" "$AVG_PSNR" "$AVG_SSIM" "$AVG_LPIPS" "$BASELINE_AVG" "$DIFF_AVG"
fi

echo ""
echo "  Passed $N_PASS/$N scenes within ±$TOLERANCE dB tolerance"
echo "  Completed at $(date +%H:%M:%S)"
echo "  Logs: $LOG_DIR/"
echo "  Eval JSON: $OUT_ROOT/eval_*_full_metrics.json"
echo "═══════════════════════════════════════════════════════════"

if [ $N_PASS -eq $N ]; then
    echo "  ✅ VERIFY SUCCESS — plug-in không hỏng bởi clean"
    exit 0
else
    echo "  ❌ VERIFY PARTIAL — $((N - N_PASS)) scene ngoài tolerance → check regression"
    exit 1
fi
