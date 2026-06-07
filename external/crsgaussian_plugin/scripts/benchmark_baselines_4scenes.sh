#!/bin/bash
# ============================================================
# [B5 Defense Compare v2 — 2026-06-05] 2 Splatfacto baselines × 4 scene
# File: crsgaussian_plugin/scripts/benchmark_baselines_4scenes.sh
#
# 2 baselines × 4 scenes = 8 runs total trên 2 GPU parallel.
# GPU 0: splatfacto-17  (17-cam ns default + COLMAP init) × 4 scene
# GPU 1: splatfacto-sparse  (3-cam Phase 22 split + COLMAP init, NO RoMa) × 4 scene
#
# ⚠️ v2 FIX (2026-06-05): splat-sparse pass --use-roma-init False
# Vì RoMa init = anh's Phase 22 contribution, baseline phải dùng COLMAP sparse.
#
# Wall-clock: ~30-40 phút.
# Output: outputs/splatfacto_17view/<scene>/ + outputs/splatfacto_3view/<scene>/
# ============================================================

set -e

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export PYTHONPATH=/home/aidev/workspace/representation-3d/duyen/nerfstudio:$PYTHONPATH
export CORGS_SOURCE_PATH=/home/aidev/workspace/representation-3d/duyen/CoR-GS
export NERFSTUDIO_METHOD_CONFIGS="crsgaussian=crsgaussian_plugin:crsgaussian_method_spec,splatfacto-sparse=crsgaussian_plugin:splatfacto_sparse_method_spec,splatfacto-17=crsgaussian_plugin:splatfacto_17_method_spec"

DATA_ROOT=/home/aidev/workspace/representation-3d/duyen/CoR-GS/data/nerf_llff_data
OUT_17=/home/aidev/workspace/representation-3d/duyen/nerfstudio/outputs/splatfacto_17view
OUT_3=/home/aidev/workspace/representation-3d/duyen/nerfstudio/outputs/splatfacto_3view
LOG_DIR=/home/aidev/workspace/representation-3d/duyen/nerfstudio/logs/baselines_b5_v2

# Backup old (if exists) — v1 had splat-sparse với RoMa unfair
[ -d "$OUT_17" ] && [ "$(ls -A $OUT_17 2>/dev/null)" ] && mv $OUT_17 ${OUT_17}_v1_old 2>/dev/null
[ -d "$OUT_3" ] && [ "$(ls -A $OUT_3 2>/dev/null)" ] && mv $OUT_3 ${OUT_3}_v1_roma_unfair 2>/dev/null

mkdir -p $OUT_17 $OUT_3 $LOG_DIR

cd /home/aidev/workspace/representation-3d/duyen/nerfstudio

SCENES="fern horns fortress flower"

# ────────────────────────────────────────────────────────
# GPU 0 sequential: splatfacto 17-cam × 4 scene
# ────────────────────────────────────────────────────────
run_splatfacto_17() {
    for SCENE in $SCENES; do
        LOG=$LOG_DIR/splatfacto17_${SCENE}.log
        EVAL=/tmp/eval_splatfacto17_${SCENE}.json
        echo "[$(date +%H:%M:%S)] [GPU 0] splatfacto $SCENE 10k..."

        CUDA_VISIBLE_DEVICES=0 ns-train splatfacto-17 \
            --data $DATA_ROOT/$SCENE/ \
            --output-dir $OUT_17 \
            --experiment-name $SCENE \
            --max-num-iterations 10000 \
            --steps-per-eval-all-images 10000 \
            --vis tensorboard \
            > $LOG 2>&1

        local RUN=$(ls -td $OUT_17/$SCENE/splatfacto-17/* | head -1)
        CUDA_VISIBLE_DEVICES=0 ns-eval --load-config $RUN/config.yml --output-path $EVAL >> $LOG 2>&1
        echo "[$(date +%H:%M:%S)] [GPU 0] splatfacto $SCENE DONE"
    done
}

# ────────────────────────────────────────────────────────
# GPU 1 sequential: splatfacto-sparse 3-cam COLMAP init (NO RoMa) × 4 scene
# ⚠️ FIX v2 (2026-06-05): --use-roma-init False
# Vì RoMa = anh's contribution; baseline phải dùng COLMAP sparse init thật.
# ────────────────────────────────────────────────────────
run_splatfacto_3() {
    for SCENE in $SCENES; do
        LOG=$LOG_DIR/splatfacto3_${SCENE}.log
        EVAL=/tmp/eval_splatfacto3_${SCENE}.json
        echo "[$(date +%H:%M:%S)] [GPU 1] splatfacto-sparse $SCENE 10k (COLMAP init)..."

        # [v2 FIX] use_roma_init=False NÀY GẮN DEFAULT trong __init__.py method spec
        # (KHÔNG dùng CLI flag vì plug-in DataParser KHÔNG register thành tyro subcommand)
        CUDA_VISIBLE_DEVICES=1 ns-train splatfacto-sparse \
            --data $DATA_ROOT/$SCENE/ \
            --output-dir $OUT_3 \
            --experiment-name $SCENE \
            --max-num-iterations 10000 \
            --steps-per-eval-all-images 10000 \
            --vis tensorboard \
            > $LOG 2>&1

        local RUN=$(ls -td $OUT_3/$SCENE/splatfacto-sparse/* | head -1)
        CUDA_VISIBLE_DEVICES=1 ns-eval --load-config $RUN/config.yml --output-path $EVAL >> $LOG 2>&1
        echo "[$(date +%H:%M:%S)] [GPU 1] splatfacto-sparse $SCENE DONE"
    done
}

# Launch parallel
echo "═══════════════════════════════════════════════════════════"
echo "  B5 v2 Defense Compare — 2 baselines × 4 scenes"
echo "  GPU 0: splatfacto-17 (17-cam + COLMAP init)"
echo "  GPU 1: splatfacto-sparse (3-cam + COLMAP init, NO RoMa)"
echo "  Old (v1 RoMa unfair): backed up tới *_v1_roma_unfair/"
echo "  Wall-clock: ~30-40 phút"
echo "═══════════════════════════════════════════════════════════"

run_splatfacto_17 &
GPU0_PID=$!
run_splatfacto_3 &
GPU1_PID=$!

wait $GPU0_PID
wait $GPU1_PID

echo ""
echo "[$(date +%H:%M:%S)] ALL 8 RUNS DONE"
echo ""

# ════════════════════════════════════════════════════════
# 3-way comparison table: crsgaussian vs splatfacto-sparse vs splatfacto
# ════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════════════════"
echo "  B5 3-WAY COMPARISON RESULTS"
echo "═══════════════════════════════════════════════════════════"
echo ""
printf "  %-10s | %-10s | %-13s | %-12s | %-10s\n" \
    "Scene" "crsgauss" "splat-sparse" "splat-17" "Δ anh→17"
echo "  ────────────────────────────────────────────────────────────────"

SUM_PLUGIN=0; SUM_SPARSE=0; SUM_17=0; COUNT=0
for SCENE in $SCENES; do
    # crsgaussian from B4
    EVAL_P=/tmp/eval_phase22_b4_${SCENE}.json
    EVAL_S=/tmp/eval_splatfacto3_${SCENE}.json
    EVAL_17=/tmp/eval_splatfacto17_${SCENE}.json

    PSNR_P="?"; PSNR_S="?"; PSNR_17="?"
    [ -f "$EVAL_P" ] && PSNR_P=$(python -c "import json; print(f\"{json.load(open('$EVAL_P'))['results']['psnr']:.2f}\")")
    [ -f "$EVAL_S" ] && PSNR_S=$(python -c "import json; print(f\"{json.load(open('$EVAL_S'))['results']['psnr']:.2f}\")")
    [ -f "$EVAL_17" ] && PSNR_17=$(python -c "import json; print(f\"{json.load(open('$EVAL_17'))['results']['psnr']:.2f}\")")

    DELTA_17="?"
    if [ "$PSNR_P" != "?" ] && [ "$PSNR_17" != "?" ]; then
        DELTA_17=$(python -c "print(f\"{$PSNR_P - $PSNR_17:+.2f}\")")
        SUM_PLUGIN=$(python -c "print($SUM_PLUGIN + $PSNR_P)")
        SUM_17=$(python -c "print($SUM_17 + $PSNR_17)")
        COUNT=$((COUNT + 1))
    fi
    [ "$PSNR_S" != "?" ] && SUM_SPARSE=$(python -c "print($SUM_SPARSE + $PSNR_S)")

    printf "  %-10s | %-10s | %-13s | %-12s | %-10s\n" \
        "$SCENE" "$PSNR_P" "$PSNR_S" "$PSNR_17" "$DELTA_17"
done

echo "  ────────────────────────────────────────────────────────────────"
if [ $COUNT -gt 0 ]; then
    AVG_P=$(python -c "print(f\"{$SUM_PLUGIN/$COUNT:.2f}\")")
    AVG_S=$(python -c "print(f\"{$SUM_SPARSE/$COUNT:.2f}\")")
    AVG_17=$(python -c "print(f\"{$SUM_17/$COUNT:.2f}\")")
    DELTA_AVG=$(python -c "print(f\"{$SUM_PLUGIN/$COUNT - $SUM_17/$COUNT:+.2f}\")")
    printf "  %-10s | %-10s | %-13s | %-12s | %-10s\n" \
        "AVG" "$AVG_P" "$AVG_S" "$AVG_17" "$DELTA_AVG"
fi

echo ""
echo "  Legend (all baselines TRULY vanilla, no anh's contribution):"
echo "    crsgauss     = anh full (3-cam + RoMa init + 8 Phase 22 modules)"
echo "    splat-sparse = Splatfacto vanilla, 3-cam + COLMAP init (NO RoMa)"
echo "    splat-17     = Splatfacto vanilla, 17-cam + COLMAP init"
echo "    Δ anh→17     = anh 3-cam vs baseline 17-cam (5.6× more data)"
echo ""
echo "  Outputs: $OUT_17 + $OUT_3"
echo "  Logs:    $LOG_DIR"
echo "═══════════════════════════════════════════════════════════"
