#!/bin/bash
# ============================================================
# [CRSGaussian] T_warmup Ablation
#
# USAGE:
#   bash scripts/ablation_warmup.sh [HALF] [GPU]
#
# EXAMPLES:
#   bash scripts/ablation_warmup.sh A 0     # Half A, GPU 0
#   bash scripts/ablation_warmup.sh B 1     # Half B, GPU 1
#   bash scripts/ablation_warmup.sh ALL 0
#   bash scripts/ablation_warmup.sh summary
#
# SONG SONG:
#   bash scripts/ablation_warmup.sh A 0 &
#   bash scripts/ablation_warmup.sh B 1 &
#   wait
#
# BỐI CẢNH:
#   T_warmup = iter bắt đầu CRS pruning + CRS update.
#   Trước T_warmup: CRS chỉ có informed init (nếu bật), không update.
#   Câu hỏi: với informed CRS₀, có thể bắt đầu sớm hơn 1000?
#
# CONFIGS (fix ema=0.3, interval=100, informed_crs_init=True):
#   T_warmup ∈ {0, 250, 500, 750, 1000, 1500, 2000}
#
# OUTPUT:
#   logs/ablation_warmup/warmup<X>_<scene>.log
# ============================================================

set -e

HALF=${1:-ALL}
GPU=${2:-0}

SCENES_A="fern flower fortress horns"
SCENES_B="leaves orchids room trex"
ALL_SCENES="${SCENES_A} ${SCENES_B}"

case ${HALF} in
    A|a) SCENES="${SCENES_A}" ;;
    B|b) SCENES="${SCENES_B}" ;;
    summary|SUMMARY) SCENES="" ;;
    *) SCENES="${ALL_SCENES}" ;;
esac

WARMUP_VALUES="0 250 500 750 1000 1500 2000"

# ── Best EMA from phase 1 ──
EMA=0.3
INTERVAL=100

DATA_ROOT="data/nerf_llff_data"
DAV2_PATH="../Depth-Anything-V2"
ITERATIONS=10000
N_VIEWS=3
RESOLUTION=8
DENSIFY_UNTIL=10000
DENSIFY_GRAD=0.0005
TEST_ITERS="1000 2000 3000 5000 7000 10000"

LOGDIR="logs/ablation_warmup"
mkdir -p ${LOGDIR}

run_one() {
    local warmup=$1
    local scene=$2
    local tag="warmup${warmup}_${scene}"
    local log="${LOGDIR}/${tag}.log"
    local out="output/ablation_warmup/${tag}"

    if [ -f "$log" ] && grep -q "Best test PSNR" "$log" 2>/dev/null; then
        echo "[SKIP] ${tag} — already done"
        return
    fi

    echo ""
    echo "========================================"
    echo " T_warmup=${warmup} | ${scene} | GPU=${GPU}"
    echo "========================================"

    CUDA_VISIBLE_DEVICES=${GPU} python train.py \
        --source_path ${DATA_ROOT}/${scene} \
        -m ${out} \
        --eval -r ${RESOLUTION} --n_views ${N_VIEWS} \
        --random_background \
        --iterations ${ITERATIONS} \
        --densify_until_iter ${DENSIFY_UNTIL} \
        --densify_grad_threshold ${DENSIFY_GRAD} \
        --gaussiansN 1 \
        --use_depth_prior --dav2_path ${DAV2_PATH} \
        --sample_pseudo_interval 1 \
        --start_sample_pseudo 500 \
        --informed_crs_init \
        --use_crs_pruning \
        --crs_ema_decay ${EMA} \
        --crs_update_interval ${INTERVAL} \
        --T_warmup ${warmup} \
        --test_iterations ${TEST_ITERS} \
        2>&1 | tee ${log}

    echo "[DONE] ${tag}"
}

# ── Main loop ──
if [ -n "${SCENES}" ]; then
    total=0
    for w in ${WARMUP_VALUES}; do
        for scene in ${SCENES}; do
            total=$((total + 1))
        done
    done
    echo "============================================"
    echo " T_warmup Ablation: ${total} runs"
    echo " Warmup: ${WARMUP_VALUES}"
    echo " EMA=${EMA}, interval=${INTERVAL}"
    echo " Scenes: ${SCENES}"
    echo " GPU: ${GPU}"
    echo "============================================"

    count=0
    for warmup in ${WARMUP_VALUES}; do
        for scene in ${SCENES}; do
            count=$((count + 1))
            echo "[${count}/${total}]"
            run_one ${warmup} ${scene}
        done
    done
fi

# ── Summary ──
echo ""
echo "=================================================================="
echo "  SUMMARY — T_warmup Ablation (ema=${EMA}, int=${INTERVAL})"
echo "=================================================================="
echo ""
printf "  %-10s" "Warmup"
for scene in ${ALL_SCENES}; do
    printf " | %7s" "${scene}"
done
echo ""
echo "  -----------$(printf -- '----------%.0s' ${ALL_SCENES})"

for warmup in ${WARMUP_VALUES}; do
    printf "  %-10s" "T=${warmup}"
    for scene in ${ALL_SCENES}; do
        log="${LOGDIR}/warmup${warmup}_${scene}.log"
        if [ -f "$log" ]; then
            val=$(grep "Evaluating test.*PSNR" "$log" 2>/dev/null | tail -1 | sed 's/.*PSNR \([0-9.]*\).*/\1/' | cut -c1-7)
            printf " | %7s" "${val:-—}"
        else
            printf " | %7s" "—"
        fi
    done
    echo ""
done
echo "=================================================================="
