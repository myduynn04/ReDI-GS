#!/bin/bash
# ============================================================
# [CRSGaussian] Pseudo-view Photometric Consistency Ablation
#
# Approach 2 — warp GT image từ training cam → pseudo cam (3° away)
# Detect floater qua parallax photometric mismatch
#
# USAGE:
#   bash scripts/ablation_pseudo_photo.sh [SCENE_OR_HALF] [GPU]
#
#   Smoke test fern (5 configs × 1 scene = 5 runs):
#     bash scripts/ablation_pseudo_photo.sh fern 0
#
#   Full 8 scenes parallel:
#     bash scripts/ablation_pseudo_photo.sh A 0 &
#     bash scripts/ablation_pseudo_photo.sh B 1 &
#     wait
#
#   Summary:
#     bash scripts/ablation_pseudo_photo.sh summary
#
# CONFIGS:
#   B0       — Baseline best-config (KHÔNG pseudo photo)
#   AP2_005  — pseudo photo λ=0.005   (rất thấp, an toàn)
#   AP2_01   — pseudo photo λ=0.01
#   AP2_02   — pseudo photo λ=0.02
#   AP2_05   — pseudo photo λ=0.05    (mạnh)
#
# Lambda thấp vì warped GT có artifacts ở boundaries
# (forward warp scatter holes) → tránh ép Gaussians fit artifacts.
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

DATA_ROOT="data/nerf_llff_data"
DAV2_PATH="../Depth-Anything-V2"
ITERATIONS=10000
N_VIEWS=3
RESOLUTION=8
DENSIFY_GRAD=0.0005
DENSIFY_UNTIL=10000
TEST_ITERS="1000 2000 3000 5000 7000 10000"

LOGDIR="logs/ablation_pseudo_photo"
mkdir -p ${LOGDIR}

# ── Config: name|use_photo|lambda ──
CONFIGS="
B0|False|0.00
AP2_005|True|0.005
AP2_01|True|0.01
AP2_02|True|0.02
AP2_05|True|0.05
"

run_one() {
    local cfg_name=$1
    local use_photo=$2
    local lam=$3
    local scene=$4

    local tag="${cfg_name}_${scene}"
    local log="${LOGDIR}/${tag}.log"
    local out="output/ablation_pseudo_photo/${tag}"

    if [ -f "$log" ] && grep -q "\\[ITER 10000\\] Evaluating" "$log" 2>/dev/null; then
        echo "[SKIP] ${tag} — already done"
        return
    fi

    echo ""
    echo "========================================"
    echo " ${cfg_name} | scene=${scene} | use_photo=${use_photo} | lambda=${lam}"
    echo " GPU=${GPU}"
    echo "========================================"

    local photo_flag=""
    if [ "$use_photo" = "True" ]; then
        photo_flag="--use_pseudo_photo_loss --lambda_pseudo_photo ${lam}"
    fi

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
        --crs_ema_decay 0.3 \
        --crs_update_interval 100 \
        --crs_init_use_view False \
        --crs_init_w_reproj 0.4 \
        --crs_init_w_depth 0.6 \
        --crs_init_w_view 0.0 \
        ${photo_flag} \
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
    echo " Pseudo Photo Ablation: ${total} runs"
    echo " GPU: ${GPU}, Scenes: ${SCENES}"
    echo "============================================"

    count=0
    for c_line in ${CONFIGS}; do
        [ -z "$c_line" ] && continue
        IFS='|' read -r c_name c_use c_lam <<< "$c_line"

        for s in ${SCENES}; do
            count=$((count + 1))
            echo "[${count}/${total}]"
            run_one "$c_name" "$c_use" "$c_lam" "$s"
        done
    done
fi

# ── Summary ──
echo ""
echo "=================================================================="
echo "  SUMMARY — Pseudo Photo Ablation — Test PSNR @10k"
echo "=================================================================="

printf "  %-8s" "Cfg"
for s in ${ALL_SCENES}; do
    printf " | %8s" "${s}"
done
echo ""
echo "  ---------$(printf -- '-----------%.0s' ${ALL_SCENES})"

for c_line in ${CONFIGS}; do
    [ -z "$c_line" ] && continue
    IFS='|' read -r c_name c_use c_lam <<< "$c_line"

    printf "  %-8s" "${c_name}"
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
echo "    B0       — Baseline (no pseudo photo)"
echo "    AP2_005  — pseudo photo λ=0.005"
echo "    AP2_01   — pseudo photo λ=0.01"
echo "    AP2_02   — pseudo photo λ=0.02"
echo "    AP2_05   — pseudo photo λ=0.05"
echo ""
