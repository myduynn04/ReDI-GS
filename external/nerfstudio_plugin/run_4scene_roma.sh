#!/bin/bash
# ============================================================
# [CRSGaussian Plug-in A1] Run 4 scene splatfacto-roma 10k iter
# File: crsgaussian_plugin/run_4scene_roma.sh
#
# Mirror cấu trúc run_4scene_10k.sh (vanilla) — đổi method sang splatfacto-roma
# Output: outputs_4scene_roma/ (parallel với outputs_4scene_10k/)
#
# Pre-requisite:
#   - Phase 22 RoMa preprocess cho 4 scene đã chạy:
#       data/nerf_llff_data/{fern,horns,fortress,flower}/3_views/dense/fused.ply.romav1
#   - Pre-generate images_4/, images_2/ cho 4 scene (đã làm cho vanilla)
#   - Plugin verified PASS 10/10 tier
# ============================================================

# ── Critical: CUDA env cho gsplat JIT ──
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

# ── Plugin env (PYTHONPATH + method config) ──
# Persist trong $CONDA_PREFIX/etc/conda/activate.d/crsgaussian_plugin.sh — nhưng export lại để chắc
export PYTHONPATH=/home/aidev/workspace/representation-3d/duyen/nerfstudio:$PYTHONPATH
export NERFSTUDIO_METHOD_CONFIGS="splatfacto-roma=crsgaussian_plugin:roma_method_spec"

DATA_ROOT=/home/aidev/workspace/representation-3d/duyen/CoR-GS/data/nerf_llff_data
OUT_ROOT=/home/aidev/workspace/representation-3d/duyen/nerfstudio/outputs_4scene_roma
LOG_DIR=/home/aidev/workspace/representation-3d/duyen/nerfstudio/logs/4scene_roma
mkdir -p $OUT_ROOT $LOG_DIR

GPU0_SCENES=(fern horns)
GPU1_SCENES=(fortress flower)

run_scene() {
    local GPU=$1
    local SC=$2
    local LOG=$LOG_DIR/$SC.log

    # Skip if already done
    if [ -f "$LOG" ] && grep -q "Training Finished" "$LOG" 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] SKIP $SC (already done)"
        return 0
    fi

    # Verify PLY tồn tại trước khi train
    PLY=$DATA_ROOT/$SC/3_views/dense/fused.ply.romav1
    if [ ! -f "$PLY" ]; then
        echo "[$(date +%H:%M:%S)] SKIP $SC — missing $PLY"
        echo "  Run Phase 22 preprocess first: SCENE=$SC python scripts/p22_romav1_preprocess.py"
        return 1
    fi

    echo "[$(date +%H:%M:%S)] START $SC on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
    ns-train splatfacto-roma \
        --data $DATA_ROOT/$SC/3_views/ \
        --output-dir $OUT_ROOT \
        --experiment-name $SC \
        --max-num-iterations 10000 \
        --steps-per-eval-all-images 10000 \
        --vis tensorboard \
        2>&1 | tee $LOG
    echo "[$(date +%H:%M:%S)] DONE $SC"
}

# Parallel 2 GPUs
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
echo "==== SUMMARY PSNR ===="
for SC in "${GPU0_SCENES[@]}" "${GPU1_SCENES[@]}"; do
    RUN=$(ls -td $OUT_ROOT/$SC/splatfacto-roma/* 2>/dev/null | head -1)
    if [ -n "$RUN" ]; then
        ns-eval --load-config $RUN/config.yml --output-path $RUN/eval.json 2>&1 | tail -3
        PSNR=$(python3 -c "import json; print(json.load(open('$RUN/eval.json'))['results']['psnr'])" 2>/dev/null)
        echo "$SC: PSNR=$PSNR"
    fi
done

echo ""
echo "==== COMPARE vs splatfacto vanilla (outputs_4scene_10k) ===="
for SC in "${GPU0_SCENES[@]}" "${GPU1_SCENES[@]}"; do
    # Vanilla
    RUN_V=$(ls -td /home/aidev/workspace/representation-3d/duyen/nerfstudio/outputs_4scene_10k/$SC/splatfacto/* 2>/dev/null | head -1)
    # Roma
    RUN_R=$(ls -td $OUT_ROOT/$SC/splatfacto-roma/* 2>/dev/null | head -1)
    if [ -f "$RUN_V/eval.json" ] && [ -f "$RUN_R/eval.json" ]; then
        PSNR_V=$(python3 -c "import json; print(f\"{json.load(open('$RUN_V/eval.json'))['results']['psnr']:.3f}\")")
        PSNR_R=$(python3 -c "import json; print(f\"{json.load(open('$RUN_R/eval.json'))['results']['psnr']:.3f}\")")
        DELTA=$(python3 -c "print(f\"{float('$PSNR_R')-float('$PSNR_V'):+.3f}\")")
        echo "$SC: vanilla=$PSNR_V → roma=$PSNR_R  (Δ=$DELTA dB)"
    fi
done
