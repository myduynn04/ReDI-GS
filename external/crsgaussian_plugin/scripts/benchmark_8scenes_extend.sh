#!/bin/bash
# ============================================================
# [B6 Defense Final 2026-06-05] Extend 8-scene + AbsGS ablation
# File: crsgaussian_plugin/scripts/benchmark_8scenes_extend.sh
#
# 3 methods × 8 scenes (LLFF full) = 24 runs total
#   crsgaussian             — anh full Phase 22 (8 modules + RoMa)
#   splatfacto-sparse       — vanilla ns (AbsGS=ON default)
#   splatfacto-sparse-noabs — vanilla ns NO AbsGS (pure 3DGS-style)
#
# Phase 1 — Extend B4 + B5 sang 4 scene mới (leaves, orchids, room, trex)
#   GPU 0: crsgaussian × 4 scene  → outputs/phase22_b4/
#   GPU 1: splatfacto-sparse × 4 scene → outputs/splatfacto_3view/
#   Time: ~30 phút
#
# Phase 2 — splatfacto-sparse-noabs trên FULL 8 scene (chưa có)
#   GPU 0: 4 scene (fern, horns, fortress, flower)
#   GPU 1: 4 scene (leaves, orchids, room, trex)
#   Output: outputs/splatfacto_3view_noabs/
#   Time: ~60 phút
#
# Total wall-clock: ~90 phút
# ============================================================

set -e

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export PYTHONPATH=/home/aidev/workspace/representation-3d/duyen/nerfstudio:$PYTHONPATH
export CORGS_SOURCE_PATH=/home/aidev/workspace/representation-3d/duyen/CoR-GS
export NERFSTUDIO_METHOD_CONFIGS="crsgaussian=crsgaussian_plugin:crsgaussian_method_spec,splatfacto-sparse=crsgaussian_plugin:splatfacto_sparse_method_spec,splatfacto-17=crsgaussian_plugin:splatfacto_17_method_spec,splatfacto-sparse-noabs=crsgaussian_plugin:splatfacto_sparse_noabs_method_spec"

DATA_ROOT=/home/aidev/workspace/representation-3d/duyen/CoR-GS/data/nerf_llff_data
OUT_CRS=/home/aidev/workspace/representation-3d/duyen/nerfstudio/outputs/phase22_b4
OUT_SPARSE=/home/aidev/workspace/representation-3d/duyen/nerfstudio/outputs/splatfacto_3view
OUT_NOABS=/home/aidev/workspace/representation-3d/duyen/nerfstudio/outputs/splatfacto_3view_noabs
LOG_DIR=/home/aidev/workspace/representation-3d/duyen/nerfstudio/logs/b6_8scene

mkdir -p $OUT_CRS $OUT_SPARSE $OUT_NOABS $LOG_DIR
cd /home/aidev/workspace/representation-3d/duyen/nerfstudio

SCENES_NEW="leaves orchids room trex"
SCENES_ALL="fern horns fortress flower leaves orchids room trex"

# ────────────────────────────────────────────────────────
# PHASE 1 — Extend crsgaussian + splatfacto-sparse sang 4 scene mới
# ────────────────────────────────────────────────────────

run_crs_extend() {
    for SCENE in $SCENES_NEW; do
        LOG=$LOG_DIR/p1_crs_${SCENE}.log
        EVAL=/tmp/eval_phase22_b4_${SCENE}.json
        echo "[$(date +%H:%M:%S)] [GPU 0 P1] crsgaussian $SCENE..."
        CUDA_VISIBLE_DEVICES=0 ns-train crsgaussian \
            --data $DATA_ROOT/$SCENE/ \
            --output-dir $OUT_CRS \
            --experiment-name $SCENE \
            --max-num-iterations 10000 \
            --steps-per-eval-all-images 10000 \
            --vis tensorboard \
            > $LOG 2>&1
        local RUN=$(ls -td $OUT_CRS/$SCENE/crsgaussian/* | head -1)
        CUDA_VISIBLE_DEVICES=0 ns-eval --load-config $RUN/config.yml --output-path $EVAL >> $LOG 2>&1
        echo "[$(date +%H:%M:%S)] [GPU 0 P1] crsgaussian $SCENE DONE"
    done
}

run_sparse_extend() {
    for SCENE in $SCENES_NEW; do
        LOG=$LOG_DIR/p1_sparse_${SCENE}.log
        EVAL=/tmp/eval_splatfacto3_${SCENE}.json
        echo "[$(date +%H:%M:%S)] [GPU 1 P1] splatfacto-sparse $SCENE..."
        CUDA_VISIBLE_DEVICES=1 ns-train splatfacto-sparse \
            --data $DATA_ROOT/$SCENE/ \
            --output-dir $OUT_SPARSE \
            --experiment-name $SCENE \
            --max-num-iterations 10000 \
            --steps-per-eval-all-images 10000 \
            --vis tensorboard \
            > $LOG 2>&1
        local RUN=$(ls -td $OUT_SPARSE/$SCENE/splatfacto-sparse/* | head -1)
        CUDA_VISIBLE_DEVICES=1 ns-eval --load-config $RUN/config.yml --output-path $EVAL >> $LOG 2>&1
        echo "[$(date +%H:%M:%S)] [GPU 1 P1] splatfacto-sparse $SCENE DONE"
    done
}

# ────────────────────────────────────────────────────────
# PHASE 2 — splatfacto-sparse-noabs trên 8 scene (split 4-4 trên 2 GPU)
# ────────────────────────────────────────────────────────

SCENES_GPU0="fern horns fortress flower"
SCENES_GPU1="leaves orchids room trex"

run_noabs_gpu() {
    local SCENES_LIST="$1"
    local GPU=$2
    for SCENE in $SCENES_LIST; do
        LOG=$LOG_DIR/p2_noabs_${SCENE}.log
        EVAL=/tmp/eval_splatfacto3_noabs_${SCENE}.json
        echo "[$(date +%H:%M:%S)] [GPU $GPU P2] splatfacto-sparse-noabs $SCENE..."
        CUDA_VISIBLE_DEVICES=$GPU ns-train splatfacto-sparse-noabs \
            --data $DATA_ROOT/$SCENE/ \
            --output-dir $OUT_NOABS \
            --experiment-name $SCENE \
            --max-num-iterations 10000 \
            --steps-per-eval-all-images 10000 \
            --vis tensorboard \
            > $LOG 2>&1
        local RUN=$(ls -td $OUT_NOABS/$SCENE/splatfacto-sparse-noabs/* | head -1)
        CUDA_VISIBLE_DEVICES=$GPU ns-eval --load-config $RUN/config.yml --output-path $EVAL >> $LOG 2>&1
        echo "[$(date +%H:%M:%S)] [GPU $GPU P2] splatfacto-sparse-noabs $SCENE DONE"
    done
}

# ════════════════════════════════════════════════════════
# Run
# ════════════════════════════════════════════════════════

echo "═══════════════════════════════════════════════════════════"
echo "  B6 — Extend 8 scene + AbsGS ablation"
echo "  Phase 1 (~30 phút): crs + sparse trên 4 scene mới"
echo "  Phase 2 (~60 phút): sparse-noabs trên 8 scene"
echo "  Total: ~90 phút wall-clock"
echo "═══════════════════════════════════════════════════════════"

# Phase 1 parallel
echo ""
echo "═══ PHASE 1 START ═══"
run_crs_extend &
P0=$!
run_sparse_extend &
P1=$!
wait $P0 $P1
echo "[$(date +%H:%M:%S)] PHASE 1 DONE"

# Phase 2 parallel
echo ""
echo "═══ PHASE 2 START ═══"
run_noabs_gpu "$SCENES_GPU0" 0 &
P0=$!
run_noabs_gpu "$SCENES_GPU1" 1 &
P1=$!
wait $P0 $P1
echo "[$(date +%H:%M:%S)] PHASE 2 DONE"

# ════════════════════════════════════════════════════════
# 4-way comparison table 8 scenes
# ════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  B6 FINAL — 4-way 8-scene comparison"
echo "═══════════════════════════════════════════════════════════"
echo ""
printf "  %-10s | %-9s | %-13s | %-15s | %-9s | %-12s\n" \
    "Scene" "crsgauss" "splat-sparse" "splat-noabs" "Δ vs sparse" "Δ vs noabs"
echo "  ────────────────────────────────────────────────────────────────────────"

SUM_P=0; SUM_S=0; SUM_NA=0; COUNT=0
declare -A DIFF_SPARSE
for SCENE in $SCENES_ALL; do
    EVAL_P=/tmp/eval_phase22_b4_${SCENE}.json
    EVAL_S=/tmp/eval_splatfacto3_${SCENE}.json
    EVAL_NA=/tmp/eval_splatfacto3_noabs_${SCENE}.json
    PSNR_P=$(python -c "import json; print(f\"{json.load(open('$EVAL_P'))['results']['psnr']:.2f}\")" 2>/dev/null || echo "—")
    PSNR_S=$(python -c "import json; print(f\"{json.load(open('$EVAL_S'))['results']['psnr']:.2f}\")" 2>/dev/null || echo "—")
    PSNR_NA=$(python -c "import json; print(f\"{json.load(open('$EVAL_NA'))['results']['psnr']:.2f}\")" 2>/dev/null || echo "—")
    D_S=$(python -c "print(f\"{$PSNR_P - $PSNR_S:+.2f}\")" 2>/dev/null || echo "—")
    D_NA=$(python -c "print(f\"{$PSNR_P - $PSNR_NA:+.2f}\")" 2>/dev/null || echo "—")
    DIFF_SPARSE[$SCENE]=$D_S
    printf "  %-10s | %-9s | %-13s | %-15s | %-9s | %-12s\n" \
        "$SCENE" "$PSNR_P" "$PSNR_S" "$PSNR_NA" "$D_S" "$D_NA"
    if [ "$PSNR_P" != "—" ]; then SUM_P=$(python -c "print($SUM_P + $PSNR_P)"); COUNT=$((COUNT+1)); fi
    [ "$PSNR_S" != "—" ] && SUM_S=$(python -c "print($SUM_S + $PSNR_S)")
    [ "$PSNR_NA" != "—" ] && SUM_NA=$(python -c "print($SUM_NA + $PSNR_NA)")
done

echo "  ────────────────────────────────────────────────────────────────────────"
if [ $COUNT -gt 0 ]; then
    AP=$(python -c "print(f\"{$SUM_P/$COUNT:.2f}\")")
    AS=$(python -c "print(f\"{$SUM_S/$COUNT:.2f}\")")
    ANA=$(python -c "print(f\"{$SUM_NA/$COUNT:.2f}\")")
    DA_S=$(python -c "print(f\"{$SUM_P/$COUNT - $SUM_S/$COUNT:+.2f}\")")
    DA_NA=$(python -c "print(f\"{$SUM_P/$COUNT - $SUM_NA/$COUNT:+.2f}\")")
    printf "  %-10s | %-9s | %-13s | %-15s | %-9s | %-12s\n" "AVG" "$AP" "$AS" "$ANA" "$DA_S" "$DA_NA"
fi

# Best scene for demo
echo ""
echo "═══ TOP scene gap (crsgauss vs splat-sparse) — anchor cho demo ═══"
for SCENE in $SCENES_ALL; do
    echo "  $SCENE: ${DIFF_SPARSE[$SCENE]}"
done | sort -k2 -t: -nr -V | head -5

echo ""
echo "  crsgauss     = anh (RoMa + 8 modules)"
echo "  splat-sparse = ns Splatfacto vanilla AbsGS=ON (default — strong baseline)"
echo "  splat-noabs  = ns Splatfacto vanilla AbsGS=OFF (pure 3DGS-style)"
echo "═══════════════════════════════════════════════════════════"
