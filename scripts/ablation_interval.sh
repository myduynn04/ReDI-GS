#!/bin/bash
# ============================================================
# [CRSGaussian] CRS Update Interval Ablation
#
# USAGE:
#   bash scripts/ablation_interval.sh [HALF] [GPU]
#
# EXAMPLES:
#   bash scripts/ablation_interval.sh A 0     # Half A (fern→horns), GPU 0
#   bash scripts/ablation_interval.sh B 1     # Half B (leaves→trex), GPU 1
#   bash scripts/ablation_interval.sh ALL 0   # Tất cả scenes
#   bash scripts/ablation_interval.sh summary # Chỉ in bảng tổng hợp
#
# CHẠY SONG SONG:
#   bash scripts/ablation_interval.sh A 0 &
#   bash scripts/ablation_interval.sh B 1 &
#   wait
#
# CONFIGS (9 = 3 ema × 3 interval):
#   ema=0.3, interval={25, 50, 75}
#   ema=0.5, interval={25, 50, 75}
#   ema=0.7, interval={25, 50, 75}
#
# OUTPUT:
#   logs/ablation_interval/ema<X>_int<Y>_<scene>.log
# ============================================================

set -e

# ── ARGS ──
HALF=${1:-ALL}
GPU=${2:-0}

# ── SCENES ──
SCENES_A="fern flower fortress horns"
SCENES_B="leaves orchids room trex"
ALL_SCENES="${SCENES_A} ${SCENES_B}"

case ${HALF} in
    A|a) SCENES="${SCENES_A}"; echo ">> Half A: ${SCENES}" ;;
    B|b) SCENES="${SCENES_B}"; echo ">> Half B: ${SCENES}" ;;
    summary|SUMMARY) SCENES=""; echo ">> Summary only" ;;
    *)   SCENES="${ALL_SCENES}"; echo ">> All scenes: ${SCENES}" ;;
esac

# ── SWEEP VALUES ──
EMA_VALUES="0.3 0.5 0.7"
INTERVAL_VALUES="25 50 75"

# ── SHARED PARAMS ──
DATA_ROOT="data/nerf_llff_data"
DAV2_PATH="../Depth-Anything-V2"
ITERATIONS=10000
N_VIEWS=3
RESOLUTION=8
DENSIFY_UNTIL=10000
DENSIFY_GRAD=0.0005
TEST_ITERS="1000 2000 3000 5000 7000 10000"

LOGDIR="logs/ablation_interval"
mkdir -p ${LOGDIR}

# ── Run function ──
run_one() {
    local ema=$1
    local interval=$2
    local scene=$3
    local tag="ema${ema}_int${interval}_${scene}"
    local log="${LOGDIR}/${tag}.log"
    local out="output/ablation_interval/${tag}"

    # Skip nếu log đã có kết quả
    if [ -f "$log" ] && grep -q "Best test PSNR" "$log" 2>/dev/null; then
        echo "[SKIP] ${tag} — already done"
        return
    fi

    echo ""
    echo "========================================"
    echo " ema=${ema} | int=${interval} | ${scene} | GPU=${GPU}"
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
        --crs_ema_decay ${ema} \
        --crs_update_interval ${interval} \
        --test_iterations ${TEST_ITERS} \
        2>&1 | tee ${log}

    echo "[DONE] ${tag}"
}

# ── Main loop ──
if [ -n "${SCENES}" ]; then
    total=0
    for ema in ${EMA_VALUES}; do
        for interval in ${INTERVAL_VALUES}; do
            for scene in ${SCENES}; do
                total=$((total + 1))
            done
        done
    done
    echo "============================================"
    echo " Interval Ablation: ${total} runs"
    echo " EMA: ${EMA_VALUES}"
    echo " Intervals: ${INTERVAL_VALUES}"
    echo " Scenes: ${SCENES}"
    echo " GPU: ${GPU}"
    echo "============================================"

    count=0
    for ema in ${EMA_VALUES}; do
        for interval in ${INTERVAL_VALUES}; do
            for scene in ${SCENES}; do
                count=$((count + 1))
                echo ""
                echo "[${count}/${total}]"
                run_one ${ema} ${interval} ${scene}
            done
        done
    done
fi

# ── Summary table ──
echo ""
echo "=================================================================="
echo "  SUMMARY — Interval Ablation — Test PSNR @10k"
echo "=================================================================="
echo ""

for ema in ${EMA_VALUES}; do
    echo "--- ema=${ema} ---"
    printf "  %-8s" "int"
    for scene in ${ALL_SCENES}; do
        printf " | %7s" "${scene}"
    done
    printf " | %7s\n" "AVG"
    echo "  ---------$(printf -- '----------%.0s' ${ALL_SCENES})----------"

    for interval in ${INTERVAL_VALUES} 100; do
        printf "  %-8s" "int=${interval}"
        sum=0
        count=0
        for scene in ${ALL_SCENES}; do
            # Tìm log trong cả ablation_interval và ablation_ema (int=100)
            log="${LOGDIR}/ema${ema}_int${interval}_${scene}.log"
            if [ ! -f "$log" ]; then
                log="logs/ablation_ema/ema${ema}_int${interval}_${scene}.log"
            fi
            if [ -f "$log" ]; then
                val=$(grep "Evaluating test.*PSNR" "$log" 2>/dev/null | tail -1 | sed 's/.*PSNR \([0-9.]*\).*/\1/')
                if [ -n "$val" ]; then
                    printf " | %7s" "${val:0:7}"
                    # Can't compute avg without bc, just print
                else
                    printf " | %7s" "—"
                fi
            else
                printf " | %7s" "—"
            fi
        done
        printf " |\n"
    done
    echo ""
done
echo "=================================================================="
