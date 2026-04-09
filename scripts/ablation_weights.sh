#!/bin/bash
# ============================================================
# [CRSGaussian] CRS₀ Weight Ablation
#
# USAGE:
#   bash scripts/ablation_weights.sh [HALF] [GPU]
#
# EXAMPLES:
#   bash scripts/ablation_weights.sh A 0
#   bash scripts/ablation_weights.sh B 1
#   bash scripts/ablation_weights.sh ALL 0
#   bash scripts/ablation_weights.sh summary
#
# SONG SONG:
#   bash scripts/ablation_weights.sh A 0 &
#   bash scripts/ablation_weights.sh B 1 &
#   wait
#
# CONFIGS:
#   6 bộ weights × 2 EMA settings × 8 scenes = 96 runs
#
#   Weights (w_reproj / w_depth / w_view):
#     WA: 0.3  / 0.6  / 0.1    (depth heavy)
#     WB: 0.6  / 0.1  / 0.3    (reproj heavy)
#     WC: 0.33 / 0.33 / 0.34   (equal = C7)
#     WD: 0.5  / 0.3  / 0.2    (reproj+depth)
#     WE: 0.3  / 0.5  / 0.2    (depth+reproj)
#     WF: 0.5  / 0.5  / 0      (no view = W4)
#
#   EMA settings:
#     S1: ema=0.3, interval=100
#     S2: ema=0.5, interval=75
#
# OUTPUT:
#   logs/ablation_weights/<setting>_<weight>_<scene>.log
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

# ── Weight configs: name|w_reproj|w_depth|w_view|use_view ──
WEIGHT_CONFIGS="
WA|0.3|0.6|0.1|True
WB|0.6|0.1|0.3|True
WC|0.33|0.33|0.34|True
WD|0.5|0.3|0.2|True
WE|0.3|0.5|0.2|True
WF|0.5|0.5|0.0|False
"

# ── EMA settings: name|ema|interval ──
EMA_SETTINGS="
S1|0.3|100
S2|0.5|75
"

DATA_ROOT="data/nerf_llff_data"
DAV2_PATH="../Depth-Anything-V2"
ITERATIONS=10000
N_VIEWS=3
RESOLUTION=8
DENSIFY_UNTIL=10000
DENSIFY_GRAD=0.0005
TEST_ITERS="1000 2000 3000 5000 7000 10000"

LOGDIR="logs/ablation_weights"
mkdir -p ${LOGDIR}

run_one() {
    local setting_name=$1
    local ema=$2
    local interval=$3
    local weight_name=$4
    local w_reproj=$5
    local w_depth=$6
    local w_view=$7
    local use_view=$8
    local scene=$9

    local tag="${setting_name}_${weight_name}_${scene}"
    local log="${LOGDIR}/${tag}.log"
    local out="output/ablation_weights/${tag}"

    if [ -f "$log" ] && grep -q "Best test PSNR" "$log" 2>/dev/null; then
        echo "[SKIP] ${tag} — already done"
        return
    fi

    echo ""
    echo "========================================"
    echo " ${setting_name} | ${weight_name} (${w_reproj}/${w_depth}/${w_view}) | ${scene}"
    echo " ema=${ema} int=${interval} | GPU=${GPU}"
    echo "========================================"

    local view_flag=""
    if [ "$use_view" = "False" ]; then
        view_flag="--crs_init_use_view False"
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
        --crs_ema_decay ${ema} \
        --crs_update_interval ${interval} \
        --crs_init_w_reproj ${w_reproj} \
        --crs_init_w_depth ${w_depth} \
        --crs_init_w_view ${w_view} \
        ${view_flag} \
        --test_iterations ${TEST_ITERS} \
        2>&1 | tee ${log}

    echo "[DONE] ${tag}"
}

# ── Main loop ──
if [ -n "${SCENES}" ]; then
    # Count total
    total=0
    for s_line in ${EMA_SETTINGS}; do
        [ -z "$s_line" ] && continue
        for w_line in ${WEIGHT_CONFIGS}; do
            [ -z "$w_line" ] && continue
            for scene in ${SCENES}; do
                total=$((total + 1))
            done
        done
    done

    echo "============================================"
    echo " Weight Ablation: ${total} runs"
    echo " GPU: ${GPU}"
    echo " Scenes: ${SCENES}"
    echo "============================================"

    count=0
    for s_line in ${EMA_SETTINGS}; do
        [ -z "$s_line" ] && continue
        IFS='|' read -r s_name s_ema s_int <<< "$s_line"

        for w_line in ${WEIGHT_CONFIGS}; do
            [ -z "$w_line" ] && continue
            IFS='|' read -r w_name w_r w_d w_v w_use <<< "$w_line"

            for scene in ${SCENES}; do
                count=$((count + 1))
                echo "[${count}/${total}]"
                run_one "$s_name" "$s_ema" "$s_int" "$w_name" "$w_r" "$w_d" "$w_v" "$w_use" "$scene"
            done
        done
    done
fi

# ── Summary ──
echo ""
echo "=================================================================="
echo "  SUMMARY — Weight Ablation — Test PSNR @10k"
echo "=================================================================="

for s_line in ${EMA_SETTINGS}; do
    [ -z "$s_line" ] && continue
    IFS='|' read -r s_name s_ema s_int <<< "$s_line"

    echo ""
    echo "--- ${s_name} (ema=${s_ema}, int=${s_int}) ---"
    printf "  %-5s" "Wt"
    for scene in ${ALL_SCENES}; do
        printf " | %7s" "${scene}"
    done
    echo ""
    echo "  ------$(printf -- '----------%.0s' ${ALL_SCENES})"

    for w_line in ${WEIGHT_CONFIGS}; do
        [ -z "$w_line" ] && continue
        IFS='|' read -r w_name w_r w_d w_v w_use <<< "$w_line"

        printf "  %-5s" "${w_name}"
        for scene in ${ALL_SCENES}; do
            log="${LOGDIR}/${s_name}_${w_name}_${scene}.log"
            if [ -f "$log" ]; then
                val=$(grep "Evaluating test.*PSNR" "$log" 2>/dev/null | tail -1 | sed 's/.*PSNR \([0-9.]*\).*/\1/' | cut -c1-7)
                printf " | %7s" "${val:-—}"
            else
                printf " | %7s" "—"
            fi
        done
        echo ""
    done
done
echo "=================================================================="
echo ""
echo "  Legend:"
echo "    WA: 0.3/0.6/0.1 (depth heavy)"
echo "    WB: 0.6/0.1/0.3 (reproj heavy)"
echo "    WC: 0.33/0.33/0.34 (equal = C7)"
echo "    WD: 0.5/0.3/0.2 (reproj+depth)"
echo "    WE: 0.3/0.5/0.2 (depth+reproj)"
echo "    WF: 0.5/0.5/0 (no view = W4)"
echo "    S1: ema=0.3, int=100"
echo "    S2: ema=0.5, int=75"
echo ""
