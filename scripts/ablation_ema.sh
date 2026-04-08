#!/bin/bash
# ============================================================
# [CRSGaussian] EMA Hyperparameter Ablation — Full Sweep
#
# USAGE:
#   bash scripts/ablation_ema.sh [GPU] [PHASE]
#
# EXAMPLES:
#   bash scripts/ablation_ema.sh              # GPU=0, phase 1
#   bash scripts/ablation_ema.sh 1            # GPU=1, phase 1
#   bash scripts/ablation_ema.sh 0 2          # Phase 2 (sweep interval)
#   bash scripts/ablation_ema.sh 0 summary    # Chỉ in bảng tổng hợp
#
# PHASE 1: Sweep EMA decay × ALL scenes (fix interval=100)
#   ema ∈ {0.9, 0.7, 0.5, 0.4, 0.35, 0.3, 0.2, 0.1, 0.0}
#
# PHASE 2: Sweep interval × ALL scenes (fix ema=BEST)
#   interval ∈ {25, 50, 100}
#
# OUTPUT:
#   Mỗi (ema, scene) → 1 log file:
#     logs/ablation_ema/ema<X>_int<Y>_<scene>.log
#   Bảng tổng hợp cuối: PSNR @10k cho tất cả scenes × configs
# ============================================================

set -e

# ── CONFIG ──
GPU=${1:-0}
PHASE=${2:-1}

# ── SAU PHASE 1: sửa giá trị này ──
BEST_EMA=0.3

# ── SCENES ──
SCENES="fern flower fortress horns leaves orchids room trex"

# ── EMA VALUES ──
EMA_VALUES="0.9 0.7 0.5 0.4 "
# EMA_VALUES="0.35 0.3 0.2 0.1 0"

# ── INTERVAL VALUES (phase 2) ──
INTERVAL_VALUES="25 50 100"

# ── SHARED PARAMS ──
DATA_ROOT="data/nerf_llff_data"
DAV2_PATH="../Depth-Anything-V2"
ITERATIONS=10000
N_VIEWS=3
RESOLUTION=8
DENSIFY_UNTIL=10000
DENSIFY_GRAD=0.0005
TEST_ITERS="1000 2000 3000 5000 7000 10000"

LOGDIR="logs/ablation_ema"
mkdir -p ${LOGDIR}

# ── Run function ──
run_one() {
    local ema=$1
    local interval=$2
    local scene=$3
    local tag="ema${ema}_int${interval}_${scene}"
    local log="${LOGDIR}/${tag}.log"
    local out="output/ablation_ema/${tag}"

    # Skip nếu log đã tồn tại và có kết quả
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

# ── Phase 1: Sweep EMA × ALL scenes ──
if [ "$PHASE" = "1" ]; then
    echo "============================================"
    echo " PHASE 1 — Sweep EMA × ALL scenes"
    echo " EMA: ${EMA_VALUES}"
    echo " Scenes: ${SCENES}"
    echo "============================================"

    for ema in ${EMA_VALUES}; do
        for scene in ${SCENES}; do
            run_one ${ema} 100 ${scene}
        done
    done
fi

# ── Phase 2: Sweep interval × ALL scenes ──
if [ "$PHASE" = "2" ]; then
    echo "============================================"
    echo " PHASE 2 — Sweep interval (ema=${BEST_EMA})"
    echo " Intervals: ${INTERVAL_VALUES}"
    echo " Scenes: ${SCENES}"
    echo "============================================"

    for interval in ${INTERVAL_VALUES}; do
        for scene in ${SCENES}; do
            run_one ${BEST_EMA} ${interval} ${scene}
        done
    done
fi

# ── Summary table ──
print_summary() {
    local pattern=$1   # e.g. "ema*_int100" or "ema0.3_int*"
    local label=$2

    echo ""
    echo "=================================================================="
    echo "  ${label}"
    echo "=================================================================="

    # Header
    printf "  %-12s" "Config"
    for scene in ${SCENES}; do
        printf " | %7s" "${scene}"
    done
    printf " | %7s\n" "AVG"
    echo "  ------------$(printf -- '----------%.0s' ${SCENES})----------"

    # Collect configs
    local configs=""
    for log in ${LOGDIR}/${pattern}_fern.log; do
        [ -f "$log" ] || continue
        local name=$(basename "$log" _fern.log)
        configs="${configs} ${name}"
    done

    # Rows
    for cfg in ${configs}; do
        printf "  %-12s" "${cfg}"
        local sum=0
        local count=0
        for scene in ${SCENES}; do
            local log="${LOGDIR}/${cfg}_${scene}.log"
            if [ -f "$log" ]; then
                local val=$(grep "Evaluating test.*PSNR" "$log" 2>/dev/null | tail -1 | sed 's/.*PSNR \([0-9.]*\).*/\1/')
                if [ -n "$val" ]; then
                    printf " | %7s" "${val:0:7}"
                    sum=$(echo "$sum + $val" | bc)
                    count=$((count + 1))
                else
                    printf " | %7s" "—"
                fi
            else
                printf " | %7s" "—"
            fi
        done
        # AVG
        if [ $count -gt 0 ]; then
            local avg=$(echo "scale=2; $sum / $count" | bc)
            printf " | %7s" "${avg}"
        else
            printf " | %7s" "—"
        fi
        echo ""
    done
    echo "=================================================================="
}

# Always print summary at end (or when PHASE=summary)
if [ "$PHASE" = "summary" ] || [ "$PHASE" = "1" ] || [ "$PHASE" = "2" ]; then
    print_summary "ema*_int100" "PHASE 1 — EMA sweep (interval=100) — Test PSNR @10k"

    # Check if phase 2 logs exist
    if ls ${LOGDIR}/ema${BEST_EMA}_int25_*.log 1>/dev/null 2>&1; then
        print_summary "ema${BEST_EMA}_int*" "PHASE 2 — Interval sweep (ema=${BEST_EMA}) — Test PSNR @10k"
    fi
fi
