#!/bin/bash
# ============================================================
# [CRSGaussian Plug-in A2.3] Run 4 scene splatfacto-roma-depth 10k iter
# CLEAN ABLATION — skip opdecay + dropansh, chỉ test depth loss alone effect.
# ============================================================

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export PYTHONPATH=/home/aidev/workspace/representation-3d/duyen/nerfstudio:$PYTHONPATH

DATA_ROOT=/home/aidev/workspace/representation-3d/duyen/CoR-GS/data/nerf_llff_data
OUT_ROOT=/home/aidev/workspace/representation-3d/duyen/nerfstudio/outputs_4scene_roma_depth
LOG_DIR=/home/aidev/workspace/representation-3d/duyen/nerfstudio/logs/4scene_roma_depth
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
        echo "[$(date +%H:%M:%S)] SKIP $SC — missing RoMa PLY $PLY"
        return 1
    fi

    DEPTH_DIR=$DATA_ROOT/$SC/3_views/aligned_depth_a23
    if [ ! -d "$DEPTH_DIR" ] || [ -z "$(ls $DEPTH_DIR/*.npy 2>/dev/null)" ]; then
        echo "[$(date +%H:%M:%S)] SKIP $SC — missing aligned depth $DEPTH_DIR"
        echo "  → conda activate corgs && SCENE=$SC python crsgaussian_plugin/preprocess_depth_a23.py"
        return 1
    fi

    echo "[$(date +%H:%M:%S)] START $SC on GPU $GPU"
    # Set CRSG_A23_DATA_ROOT để model biết path aligned_depth
    CUDA_VISIBLE_DEVICES=$GPU CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
    CRSG_A23_DATA_ROOT=$DATA_ROOT/$SC/3_views \
    ns-train splatfacto-roma-depth \
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
echo "==== SUMMARY PSNR splatfacto-roma-depth (A2.3) ===="
for SC in "${GPU0_SCENES[@]}" "${GPU1_SCENES[@]}"; do
    RUN=$(ls -td $OUT_ROOT/$SC/splatfacto-roma-depth/* 2>/dev/null | head -1)
    if [ -n "$RUN" ]; then
        ns-eval --load-config $RUN/config.yml --output-path $RUN/eval.json 2>&1 | tail -3
        PSNR=$(python3 -c "import json; print(json.load(open('$RUN/eval.json'))['results']['psnr'])" 2>/dev/null)
        echo "$SC: PSNR=$PSNR"
    fi
done

echo ""
echo "==== COMPARE: vanilla → A1 → A2.3 (skip opdecay/dropansh) ===="
NS_OUT=/home/aidev/workspace/representation-3d/duyen/nerfstudio
for SC in "${GPU0_SCENES[@]}" "${GPU1_SCENES[@]}"; do
    RUN_V=$(ls -td $NS_OUT/outputs_4scene_10k/$SC/splatfacto/* 2>/dev/null | head -1)
    RUN_A1=$(ls -td $NS_OUT/outputs_4scene_roma/$SC/splatfacto-roma/* 2>/dev/null | head -1)
    RUN_A23=$(ls -td $OUT_ROOT/$SC/splatfacto-roma-depth/* 2>/dev/null | head -1)

    get_psnr() { python3 -c "import json; print(f\"{json.load(open('$1/eval.json'))['results']['psnr']:.3f}\")" 2>/dev/null || echo "—"; }
    delta() {
        if [ "$1" != "—" ] && [ "$2" != "—" ]; then
            python3 -c "print(f\"{float('$1') - float('$2'):+.3f}\")"
        else
            echo "—"
        fi
    }

    P_V=$(get_psnr "$RUN_V")
    P_A1=$(get_psnr "$RUN_A1")
    P_A23=$(get_psnr "$RUN_A23")
    D_A1=$(delta "$P_A1" "$P_V")
    D_A23=$(delta "$P_A23" "$P_A1")
    D_TOT=$(delta "$P_A23" "$P_V")

    echo "$SC: V=$P_V → A1=$P_A1 (Δ=$D_A1) → A2.3=$P_A23 (Δ_depth=$D_A23)  |  Δ_total=$D_TOT"
done
