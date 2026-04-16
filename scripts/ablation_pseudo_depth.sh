#!/bin/bash
# ============================================================
# [CRSGaussian] Pseudo-view Depth Loss Ablation
#
# USAGE:
#   bash scripts/ablation_pseudo_depth.sh [SCENE_OR_HALF] [GPU]
#
#   Single scene:
#     bash scripts/ablation_pseudo_depth.sh fern 0
#
#   Full 8 scenes one GPU:
#     bash scripts/ablation_pseudo_depth.sh ALL 0
#
#   Parallel 2 GPUs (recommended):
#     bash scripts/ablation_pseudo_depth.sh A 0 &
#     bash scripts/ablation_pseudo_depth.sh B 1 &
#     wait
#       A: fern flower fortress horns   (GPU 0)
#       B: leaves orchids room trex     (GPU 1)
#
#   Summary only:
#     bash scripts/ablation_pseudo_depth.sh summary
#
# CONFIGS — mỗi config trả lời 1 câu hỏi nghiên cứu cụ thể:
#   B0    — Baseline best-config (KHÔNG pseudo loss)
#   PM    — pseudo λ=0.05 start=2000 until=10000   (default candidate)
#   PL    — pseudo λ=0.02 start=2000 until=10000   (Q2: weight thấp)
#   PH    — pseudo λ=0.10 start=2000 until=10000   (Q2: weight cao)
#   PU5   — pseudo λ=0.05 start=2000 until=5000    (Q3: combine densify sớm)
#   PE    — pseudo λ=0.05 start=1000 until=10000   (Q4: bắt đầu sớm)
#   PT    — pseudo λ=0.05 start=3000 until=10000   (Q4: bắt đầu muộn)
#
# Câu hỏi nghiên cứu:
#   Q1  Pseudo loss có giúp gì?            B0  vs PM
#   Q2  λ nào tối ưu?                      PL  vs PM  vs PH
#   Q3  Tương tác với densify_until?       PM  vs PU5
#   Q4  Khi nào bắt đầu là tốt nhất?       PE  vs PM  vs PT
#
# OUTPUT:
#   logs/ablation_pseudo/<config>_<scene>.log
#   output/ablation_pseudo/<config>_<scene>/
# ============================================================

set -e

SCENE=${1:-fern}
GPU=${2:-0}

SCENES_A="fern flower fortress horns"
SCENES_B="leaves orchids room trex"
ALL_SCENES="${SCENES_A} ${SCENES_B}"

case ${SCENE} in
    ALL|all)         SCENES="${ALL_SCENES}" ;;
    A|a)             SCENES="${SCENES_A}" ;;
    B|b)             SCENES="${SCENES_B}" ;;
    summary|SUMMARY) SCENES="" ;;
    *)               SCENES="${SCENE}" ;;
esac

# ── Common settings (best config từ ablation trước) ──
DATA_ROOT="data/nerf_llff_data"
DAV2_PATH="../Depth-Anything-V2"
ITERATIONS=10000
N_VIEWS=3
RESOLUTION=8
DENSIFY_GRAD=0.0005
TEST_ITERS="1000 2000 3000 5000 7000 10000"

LOGDIR="logs/ablation_pseudo"
mkdir -p ${LOGDIR}

# ── Config: name|use_pseudo|lambda|start_iter|densify_until ──
CONFIGS="
B0|False|0.00|2000|10000
PM|True|0.05|2000|10000
PL|True|0.02|2000|10000
PH|True|0.10|2000|10000
PU5|True|0.05|2000|5000
PE|True|0.05|1000|10000
PT|True|0.05|3000|10000
"

run_one() {
    local cfg_name=$1
    local use_pseudo=$2
    local lam=$3
    local start_iter=$4
    local dens_until=$5
    local scene=$6

    local tag="${cfg_name}_${scene}"
    local log="${LOGDIR}/${tag}.log"
    local out="output/ablation_pseudo/${tag}"

    if [ -f "$log" ] && grep -q "Best test PSNR\|\\[ITER 10000\\] Evaluating" "$log" 2>/dev/null; then
        echo "[SKIP] ${tag} — already done"
        return
    fi

    echo ""
    echo "========================================"
    echo " ${cfg_name} | scene=${scene} | use_pseudo=${use_pseudo}"
    echo " lambda=${lam} | start=${start_iter} | densify_until=${dens_until} | GPU=${GPU}"
    echo "========================================"

    local pseudo_flag=""
    if [ "$use_pseudo" = "True" ]; then
        pseudo_flag="--use_pseudo_depth_loss --lambda_pseudo_depth ${lam} --pseudo_depth_start_iter ${start_iter}"
    fi

    CUDA_VISIBLE_DEVICES=${GPU} python train.py \
        --source_path ${DATA_ROOT}/${scene} \
        -m ${out} \
        --eval -r ${RESOLUTION} --n_views ${N_VIEWS} \
        --random_background \
        --iterations ${ITERATIONS} \
        --densify_until_iter ${dens_until} \
        --densify_grad_threshold ${DENSIFY_GRAD} \
        --gaussiansN 1 \
        --use_depth_prior --dav2_path ${DAV2_PATH} \
        --sample_pseudo_interval 1 \
        --start_sample_pseudo 500 \
        --informed_crs_init \
        --use_crs_pruning \
        --crs_ema_decay 0.3 \
        --crs_update_interval 100 \
        --crs_init_use_view False \
        --crs_init_w_reproj 0.4 \
        --crs_init_w_depth 0.6 \
        --crs_init_w_view 0.0 \
        ${pseudo_flag} \
        --test_iterations ${TEST_ITERS} \
        2>&1 | tee ${log}

    echo "[DONE] ${tag}"
}

# ── Main loop ──
if [ -n "${SCENES}" ]; then
    total=0
    for c_line in ${CONFIGS}; do
        [ -z "$c_line" ] && continue
        for s in ${SCENES}; do
            total=$((total + 1))
        done
    done

    echo "============================================"
    echo " Pseudo Depth Ablation: ${total} runs"
    echo " GPU: ${GPU}, Scenes: ${SCENES}"
    echo "============================================"

    count=0
    for c_line in ${CONFIGS}; do
        [ -z "$c_line" ] && continue
        IFS='|' read -r c_name c_use c_lam c_start c_dens <<< "$c_line"

        for s in ${SCENES}; do
            count=$((count + 1))
            echo "[${count}/${total}]"
            run_one "$c_name" "$c_use" "$c_lam" "$c_start" "$c_dens" "$s"
        done
    done
fi

# ── Summary ──
echo ""
echo "=================================================================="
echo "  SUMMARY — Pseudo Depth Ablation — Test PSNR @10k"
echo "=================================================================="

printf "  %-5s" "Cfg"
for s in ${ALL_SCENES}; do
    printf " | %8s" "${s}"
done
echo ""
echo "  ------$(printf -- '-----------%.0s' ${ALL_SCENES})"

for c_line in ${CONFIGS}; do
    [ -z "$c_line" ] && continue
    IFS='|' read -r c_name c_use c_lam c_start c_dens <<< "$c_line"

    printf "  %-5s" "${c_name}"
    for s in ${ALL_SCENES}; do
        log="${LOGDIR}/${c_name}_${s}.log"
        if [ -f "$log" ]; then
            val=$(grep "\\[ITER 10000\\] Evaluating test" "$log" 2>/dev/null | tail -1 | sed 's/.*PSNR \([0-9.]*\).*/\1/' | cut -c1-7)
            printf " | %8s" "${val:-—}"
        else
            printf " | %8s" "—"
        fi
    done
    echo ""
done

echo ""
echo "=================================================================="
echo "  Legend:"
echo "    B0  — Baseline (no pseudo loss)"
echo "    PM  — pseudo λ=0.05 start=2000 until=10k (default candidate)"
echo "    PL  — pseudo λ=0.02 start=2000 until=10k (lower weight)"
echo "    PH  — pseudo λ=0.10 start=2000 until=10k (higher weight)"
echo "    PU5 — pseudo λ=0.05 start=2000 until=5k  (combine densify sớm)"
echo "    PE  — pseudo λ=0.05 start=1000 until=10k (sớm hơn)"
echo "    PT  — pseudo λ=0.05 start=3000 until=10k (muộn hơn)"
echo ""
echo "  Câu hỏi:"
echo "    Q1 — Pseudo loss có giúp gì?    B0 vs PM"
echo "    Q2 — λ tối ưu?                  PL vs PM vs PH"
echo "    Q3 — Compound với densify_until? PM vs PU5"
echo "    Q4 — Khi nào start tối ưu?      PE vs PM vs PT"
echo ""
