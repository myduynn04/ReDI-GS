#!/bin/bash
# ============================================================
# [CRSGaussian] Pseudo-view Photometric — FULL 8 LLFF scenes
#
# Approach 2 — λ sweep với best λ từ fern smoke test
# Mục đích: verify gain +0.21 dB AP2_05 fern có generalize sang
#           7 scenes còn lại + sweep λ cao hơn (0.10, 0.20)
#
# USAGE:
#   bash scripts/ablation_pseudo_photo_full.sh [SCENE_OR_HALF] [GPU]
#
#   Single scene:
#     bash scripts/ablation_pseudo_photo_full.sh fern 0
#
#   Parallel 2 GPUs (recommended):
#     bash scripts/ablation_pseudo_photo_full.sh A 0 &
#     bash scripts/ablation_pseudo_photo_full.sh B 1 &
#     wait
#       A: fern flower fortress horns   (GPU 0)
#       B: leaves orchids room trex     (GPU 1)
#
#   Summary:
#     bash scripts/ablation_pseudo_photo_full.sh summary
#
# CONFIGS (4 configs × 8 scenes = 32 runs):
#   B0      — Baseline best-config (KHÔNG pseudo photo)
#   AP2_05  — pseudo photo λ=0.05  (fern: +0.21 dB)
#   AP2_10  — pseudo photo λ=0.10  (sweep cao hơn)
#   AP2_20  — pseudo photo λ=0.20  (sweep cao nhất)
#
# SKIP LOGIC: dùng cùng LOGDIR với smoke test (logs/ablation_pseudo_photo/)
# → fern B0 và fern AP2_05 đã có → skip tự động.
# ============================================================

# [CRSGaussian] Bỏ `set -e` — không cho 1 run fail kill toàn bộ batch.
# Skip logic ở đầu run_one() đảm bảo có thể re-run an toàn để pick up
# những run đã miss/fail.

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
DENSIFY_UNTIL=5000
TEST_ITERS="1000 2000 3000 5000 7000 10000"

# ── Reuse smoke test LOGDIR cho skip logic ──
LOGDIR="logs/ablation_pseudo_photo"
mkdir -p ${LOGDIR}

# ── Config: name|use_photo|lambda ──
CONFIGS="
B0|False|0.00
AP2_05|True|0.05
AP2_10|True|0.10
AP2_20|True|0.20
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

    # [CRSGaussian] Wrap python call: never propagate failure to outer loop.
    # Nếu 1 run crash, log file vẫn được tạo (rỗng/partial), skip logic
    # ở đầu run_one() sẽ bỏ qua nếu có ITER 10000, hoặc retry nếu không.
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
        2>&1 | tee ${log} || echo "[ERROR] ${tag} — train.py crashed, continuing"

    # Verify run completion — nếu thiếu ITER 10000, in cảnh báo
    if grep -q "\\[ITER 10000\\] Evaluating" "$log" 2>/dev/null; then
        echo "[DONE] ${tag}"
    else
        echo "[FAIL] ${tag} — no ITER 10000 in log, will retry on next batch"
    fi
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
    echo " Pseudo Photo Full Ablation: ${total} runs"
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
echo "=================================================================================="
echo "  SUMMARY — Pseudo Photo Full Ablation — Test PSNR @10k"
echo "=================================================================================="

# Header
printf "  %-7s" "Cfg"
for s in ${ALL_SCENES}; do
    printf " | %7s" "${s}"
done
printf " | %7s\n" "AVG"
printf "  -------"
for s in ${ALL_SCENES}; do
    printf -- "----------"
done
printf -- "----------\n"

# Lưu B0 averages cho delta calc
declare -A B0_VALS
B0_AVG=""

for c_line in ${CONFIGS}; do
    [ -z "$c_line" ] && continue
    IFS='|' read -r c_name c_use c_lam <<< "$c_line"

    printf "  %-7s" "${c_name}"

    sum="0"
    n="0"
    for s in ${ALL_SCENES}; do
        log="${LOGDIR}/${c_name}_${s}.log"
        if [ -f "$log" ]; then
            val=$(grep "\\[ITER 10000\\] Evaluating test" "$log" 2>/dev/null | tail -1 | sed 's/.*PSNR \([0-9.]*\).*/\1/' | cut -c1-7)
            if [ -n "$val" ]; then
                printf " | %7s" "${val}"
                sum=$(awk -v s="$sum" -v v="$val" 'BEGIN {printf "%.6f", s + v}')
                n=$((n + 1))
                # Lưu B0 value cho scene này
                if [ "$c_name" = "B0" ]; then
                    B0_VALS[$s]=$val
                fi
            else
                printf " | %7s" "—"
            fi
        else
            printf " | %7s" "—"
        fi
    done

    if [ "$n" -gt 0 ]; then
        avg=$(awk -v s="$sum" -v n="$n" 'BEGIN {printf "%.3f", s / n}')
        printf " | %7s\n" "${avg}"
        if [ "$c_name" = "B0" ]; then
            B0_AVG=$avg
        fi
    else
        printf " | %7s\n" "—"
    fi
done

# Delta row
echo ""
echo "  Delta vs B0 (AP2 - B0):"
for c_line in ${CONFIGS}; do
    [ -z "$c_line" ] && continue
    IFS='|' read -r c_name c_use c_lam <<< "$c_line"
    [ "$c_name" = "B0" ] && continue

    printf "  %-7s" "Δ${c_name}"

    sum_d="0"
    n_d="0"
    for s in ${ALL_SCENES}; do
        log="${LOGDIR}/${c_name}_${s}.log"
        b0_val="${B0_VALS[$s]}"
        if [ -f "$log" ] && [ -n "$b0_val" ]; then
            val=$(grep "\\[ITER 10000\\] Evaluating test" "$log" 2>/dev/null | tail -1 | sed 's/.*PSNR \([0-9.]*\).*/\1/' | cut -c1-7)
            if [ -n "$val" ]; then
                delta=$(awk -v v="$val" -v b="$b0_val" 'BEGIN {printf "%+.3f", v - b}')
                printf " | %7s" "${delta}"
                sum_d=$(awk -v s="$sum_d" -v v="$val" -v b="$b0_val" 'BEGIN {printf "%.6f", s + (v - b)}')
                n_d=$((n_d + 1))
            else
                printf " | %7s" "—"
            fi
        else
            printf " | %7s" "—"
        fi
    done

    if [ "$n_d" -gt 0 ]; then
        avg_d=$(awk -v s="$sum_d" -v n="$n_d" 'BEGIN {printf "%+.3f", s / n}')
        printf " | %7s\n" "${avg_d}"
    else
        printf " | %7s\n" "—"
    fi
done

echo ""
echo "=================================================================================="
echo "  Legend:"
echo "    B0      — Baseline (no pseudo photo loss)"
echo "    AP2_05  — pseudo photo λ=0.05"
echo "    AP2_10  — pseudo photo λ=0.10"
echo "    AP2_20  — pseudo photo λ=0.20"
echo ""
echo "  Hyperparams: ema=0.3, int=100, w=0.4/0.6/0, densify_until=5000"
echo "=================================================================================="
