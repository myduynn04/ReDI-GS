#!/bin/bash
# ============================================================
# [CRSGaussian Plug-in A2.1] Run 4 scene splatfacto-roma-opdecay 10k iter
# Mirror run_4scene_roma.sh — chỉ đổi method sang splatfacto-roma-opdecay
# Output: outputs_4scene_roma_opdecay/
# ============================================================

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export PYTHONPATH=/home/aidev/workspace/representation-3d/duyen/nerfstudio:$PYTHONPATH

DATA_ROOT=/home/aidev/workspace/representation-3d/duyen/CoR-GS/data/nerf_llff_data
OUT_ROOT=/home/aidev/workspace/representation-3d/duyen/nerfstudio/outputs_4scene_roma_opdecay
LOG_DIR=/home/aidev/workspace/representation-3d/duyen/nerfstudio/logs/4scene_roma_opdecay
mkdir -p $OUT_ROOT $LOG_DIR

GPU0_SCENES=(fern horns)
GPU1_SCENES=(fortress flower)

run_scene() {
    local GPU=$1
    local SC=$2
    local LOG=$LOG_DIR/$SC.log

    if [ -f "$LOG" ] && grep -q "Training Finished" "$LOG" 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] SKIP $SC (already done)"
        return 0
    fi

    PLY=$DATA_ROOT/$SC/3_views/dense/fused.ply.romav1
    if [ ! -f "$PLY" ]; then
        echo "[$(date +%H:%M:%S)] SKIP $SC — missing $PLY"
        return 1
    fi

    echo "[$(date +%H:%M:%S)] START $SC on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
    ns-train splatfacto-roma-opdecay \
        --data $DATA_ROOT/$SC/3_views/ \
        --output-dir $OUT_ROOT \
        --experiment-name $SC \
        --max-num-iterations 10000 \
        --steps-per-eval-all-images 10000 \
        --vis tensorboard \
        2>&1 | tee $LOG
    echo "[$(date +%H:%M:%S)] DONE $SC"
}

(
    for SC in "${GPU0_SCENES[@]}"; do
        run_scene 0 $SC
    done
) &
PID0=$!

(
    for SC in "${GPU1_SCENES[@]}"; do
        run_scene 1 $SC
    done
) &
PID1=$!

wait $PID0 $PID1
echo "[$(date +%H:%M:%S)] All 4 scenes done"
echo ""
echo "==== SUMMARY PSNR splatfacto-roma-opdecay (A2.1) ===="
for SC in "${GPU0_SCENES[@]}" "${GPU1_SCENES[@]}"; do
    RUN=$(ls -td $OUT_ROOT/$SC/splatfacto-roma-opdecay/* 2>/dev/null | head -1)
    if [ -n "$RUN" ]; then
        ns-eval --load-config $RUN/config.yml --output-path $RUN/eval.json 2>&1 | tail -3
        PSNR=$(python3 -c "import json; print(json.load(open('$RUN/eval.json'))['results']['psnr'])" 2>/dev/null)
        echo "$SC: PSNR=$PSNR"
    fi
done

echo ""
echo "==== CASCADE COMPARE: vanilla → A1 (roma) → A2.1 (roma+opdecay) ===="
for SC in "${GPU0_SCENES[@]}" "${GPU1_SCENES[@]}"; do
    RUN_V=$(ls -td /home/aidev/workspace/representation-3d/duyen/nerfstudio/outputs_4scene_10k/$SC/splatfacto/* 2>/dev/null | head -1)
    RUN_A1=$(ls -td /home/aidev/workspace/representation-3d/duyen/nerfstudio/outputs_4scene_roma/$SC/splatfacto-roma/* 2>/dev/null | head -1)
    RUN_A2=$(ls -td $OUT_ROOT/$SC/splatfacto-roma-opdecay/* 2>/dev/null | head -1)
    if [ -f "$RUN_V/eval.json" ] && [ -f "$RUN_A1/eval.json" ] && [ -f "$RUN_A2/eval.json" ]; then
        P_V=$(python3 -c "import json; print(f\"{json.load(open('$RUN_V/eval.json'))['results']['psnr']:.3f}\")")
        P_A1=$(python3 -c "import json; print(f\"{json.load(open('$RUN_A1/eval.json'))['results']['psnr']:.3f}\")")
        P_A2=$(python3 -c "import json; print(f\"{json.load(open('$RUN_A2/eval.json'))['results']['psnr']:.3f}\")")
        D_A1=$(python3 -c "print(f\"{float('$P_A1')-float('$P_V'):+.3f}\")")
        D_A2=$(python3 -c "print(f\"{float('$P_A2')-float('$P_A1'):+.3f}\")")
        D_TOT=$(python3 -c "print(f\"{float('$P_A2')-float('$P_V'):+.3f}\")")
        echo "$SC: $P_V → $P_A1 (Δ_A1=$D_A1) → $P_A2 (Δ_A2.1=$D_A2)  |  Δ_total=$D_TOT"
    fi
done
