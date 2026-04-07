#!/bin/bash
# ============================================================
# [CRSGaussian] EMA Hyperparameter Ablation
# File: scripts/ablation_ema.sh
#
# USAGE:
#   bash scripts/ablation_ema.sh [GPU] [SCENE] [PHASE]
#
# EXAMPLES:
#   bash scripts/ablation_ema.sh              # GPU=0, fern, phase 1+2
#   bash scripts/ablation_ema.sh 1            # GPU=1, fern, phase 1+2
#   bash scripts/ablation_ema.sh 0 flower     # GPU=0, flower
#   bash scripts/ablation_ema.sh 0 fern 1     # Chỉ phase 1 (sweep ema)
#   bash scripts/ablation_ema.sh 0 fern 2     # Chỉ phase 2 (sweep interval)
#                                              # → sửa BEST_EMA trước khi chạy!
#
# STRATEGY:
#   Phase 1 — Sweep EMA decay (fix interval=100):
#     Config 1: ema=0.9  (baseline C7)
#     Config 2: ema=0.7
#     Config 3: ema=0.5
#   → Chọn ema tốt nhất (best test PSNR)
#
#   Phase 2 — Sweep interval (fix ema=BEST từ phase 1):
#     Config 4: interval=50
#     Config 5: interval=25
#     (interval=100 đã có từ phase 1)
#
# OUTPUT:
#   logs/ablation_ema/<scene>_ema<X>_int<Y>.log
#
# SAU KHI CHẠY PHASE 1:
#   1. Đọc PSNR summary cuối mỗi log
#   2. Sửa BEST_EMA bên dưới
#   3. Chạy phase 2
# ============================================================

set -e

# ── ARGS ──
GPU=${1:-0}
SCENE=${2:-fern}
PHASE=${3:-0}       # 0=cả hai, 1=chỉ phase 1, 2=chỉ phase 2

# ── SAU PHASE 1: sửa giá trị này ──
BEST_EMA=0.9        # ← ĐỔI SAU KHI CÓ KẾT QUẢ PHASE 1

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
run_config() {
    local ema=$1
    local interval=$2
    local tag="${SCENE}_ema${ema}_int${interval}"
    local log="${LOGDIR}/${tag}.log"
    local out="output/ablation_ema/${tag}"

    echo ""
    echo "========================================"
    echo " CONFIG: ema=${ema}, interval=${interval}"
    echo " Scene:  ${SCENE} | GPU: ${GPU}"
    echo " Log:    ${log}"
    echo "========================================"

    CUDA_VISIBLE_DEVICES=${GPU} python train.py \
        --source_path ${DATA_ROOT}/${SCENE} \
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

    echo ""
    echo "[DONE] ${tag}"
    echo ""
}

# ── Phase 1: Sweep EMA decay ──
if [ "$PHASE" = "0" ] || [ "$PHASE" = "1" ]; then
    echo "============================================"
    echo " PHASE 1 — Sweep EMA (interval=100)"
    echo "============================================"

    run_config 0.9 100    # Config 1 — baseline
    run_config 0.7 100    # Config 2
    run_config 0.5 100    # Config 3
fi

# ── Phase 2: Sweep interval ──
if [ "$PHASE" = "0" ] || [ "$PHASE" = "2" ]; then
    echo "============================================"
    echo " PHASE 2 — Sweep interval (ema=${BEST_EMA})"
    echo "============================================"

    run_config ${BEST_EMA} 50     # Config 4
    run_config ${BEST_EMA} 25     # Config 5
fi

# ── Summary: extract best test PSNR từ mỗi log ──
echo ""
echo "============================================"
echo " SUMMARY — All configs for ${SCENE}"
echo "============================================"
echo ""
printf "  %-35s | %s\n" "Config" "Best test PSNR"
echo "  ------------------------------------------------"
for log in ${LOGDIR}/${SCENE}_*.log; do
    if [ -f "$log" ]; then
        name=$(basename "$log" .log)
        best=$(grep "Best test PSNR" "$log" 2>/dev/null | tail -1)
        if [ -n "$best" ]; then
            printf "  %-35s | %s\n" "$name" "$best"
        else
            # Fallback: grep last test PSNR
            last=$(grep "Evaluating test.*PSNR" "$log" 2>/dev/null | tail -1 | grep -oP 'PSNR \K[0-9.]+')
            printf "  %-35s | last test PSNR: %s\n" "$name" "$last"
        fi
    fi
done
echo "  ================================================"
echo ""
echo " Next steps:"
echo "   1. Chọn ema tốt nhất từ Phase 1"
echo "   2. Sửa BEST_EMA trong script"
echo "   3. Chạy: bash scripts/ablation_ema.sh ${GPU} ${SCENE} 2"
echo ""
