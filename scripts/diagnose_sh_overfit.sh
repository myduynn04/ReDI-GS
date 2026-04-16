#!/bin/bash
# ============================================================
# [CRSGaussian DIAG E1] SH Overfit Test — Freeze f_dc + f_rest
# File: scripts/diagnose_sh_overfit.sh
#
# Mục đích: Test hypothesis H4 — "SH overfit memorize training views
#   là nguyên nhân chính của train-test gap ~16dB".
#
# Cơ chế: Sau iter N, set lr=0 cho f_dc + f_rest → SH đóng băng.
#   Chỉ xyz/opacity/scaling/rotation tiếp tục update 5000 iters còn lại.
#
# Configs (3 × 1 scene = 3 runs, fern only):
#   E1_baseline     — freeze_sh_after=0        (KHÔNG freeze, control)
#   E1_freeze3k     — freeze_sh_after=3000     (freeze sớm: 7k iter còn lại)
#   E1_freeze5k     — freeze_sh_after=5000     (freeze giữa: 5k iter còn lại)
#   E1_freeze7k     — freeze_sh_after=7000     (freeze trễ: 3k iter còn lại)
#
# Diễn giải kết quả:
#   - Nếu E1_freeze5k > baseline test PSNR → SH overfit CONFIRMED
#     → hướng giải: freeze SH sau iter X, hoặc regularize SH
#   - Nếu E1_freeze5k ≈ baseline → SH không phải root cause
#     → H4 bác bỏ, cần tìm nguyên nhân khác (geometry, densification)
#   - Nếu E1_freeze5k < baseline → SH vẫn đang học useful info
#
# USAGE:
#   bash scripts/diagnose_sh_overfit.sh [SCENE] [GPU]
#
# EXAMPLES:
#   bash scripts/diagnose_sh_overfit.sh fern 0
#   bash scripts/diagnose_sh_overfit.sh flower 1
# ============================================================

SCENE=${1:-fern}
GPU=${2:-0}

DATA_ROOT="data/nerf_llff_data"
DAV2_PATH="../Depth-Anything-V2"
ITERATIONS=10000
N_VIEWS=3
RESOLUTION=8
DENSIFY_GRAD=0.0005
DENSIFY_UNTIL=5000
TEST_ITERS="1000 2000 3000 5000 7000 10000"

LOGDIR="logs/diag_e1_sh"
mkdir -p ${LOGDIR}

# ── Config: name|freeze_after ──
CONFIGS="
E1_baseline|0
E1_freeze3k|3000
E1_freeze5k|5000
E1_freeze7k|7000
"

run_one() {
    local cfg_name=$1
    local freeze_after=$2
    local scene=$3

    local tag="${cfg_name}_${scene}"
    local log="${LOGDIR}/${tag}.log"
    local out="output/diag_e1_sh/${tag}"

    if [ -f "$log" ] && grep -q "\\[ITER 10000\\] Evaluating" "$log" 2>/dev/null; then
        echo "[SKIP] ${tag} — already done"
        return
    fi

    echo ""
    echo "========================================"
    echo " ${cfg_name} | scene=${scene} | freeze_sh_after=${freeze_after}"
    echo " GPU=${GPU}"
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
        --crs_ema_decay 0.3 \
        --crs_update_interval 100 \
        --crs_init_use_view False \
        --crs_init_w_reproj 0.4 \
        --crs_init_w_depth 0.6 \
        --crs_init_w_view 0.0 \
        --freeze_sh_after ${freeze_after} \
        --test_iterations ${TEST_ITERS} \
        2>&1 | tee ${log} || echo "[ERROR] ${tag} — train.py crashed, continuing"

    if grep -q "\\[ITER 10000\\] Evaluating" "$log" 2>/dev/null; then
        echo "[DONE] ${tag}"
    else
        echo "[FAIL] ${tag} — no ITER 10000 in log"
    fi
}

# ── Main loop ──
total=0
for c_line in ${CONFIGS}; do
    [ -z "$c_line" ] && continue
    total=$((total + 1))
done

echo "============================================"
echo " DIAG E1 — SH Overfit Test"
echo " Scene: ${SCENE}, GPU: ${GPU}, ${total} configs"
echo "============================================"

count=0
for c_line in ${CONFIGS}; do
    [ -z "$c_line" ] && continue
    IFS='|' read -r c_name c_freeze <<< "$c_line"
    count=$((count + 1))
    echo "[${count}/${total}]"
    run_one "$c_name" "$c_freeze" "${SCENE}"
done

# ── Summary ──
echo ""
echo "=================================================================="
echo "  SUMMARY — DIAG E1 — Test PSNR @10k — scene=${SCENE}"
echo "=================================================================="

printf "  %-15s | %10s | %10s | %10s\n" "Config" "Test PSNR" "Train PSNR" "Gap"
printf "  %-15s-+-%10s-+-%10s-+-%10s\n" "---------------" "----------" "----------" "----------"

for c_line in ${CONFIGS}; do
    [ -z "$c_line" ] && continue
    IFS='|' read -r c_name c_freeze <<< "$c_line"

    log="${LOGDIR}/${c_name}_${SCENE}.log"
    if [ -f "$log" ]; then
        test_psnr=$(grep "\\[ITER 10000\\] Evaluating test" "$log" 2>/dev/null | tail -1 | sed 's/.*PSNR \([0-9.]*\).*/\1/' | cut -c1-7)
        train_psnr=$(grep "\\[ITER 10000\\] Evaluating train" "$log" 2>/dev/null | tail -1 | sed 's/.*PSNR \([0-9.]*\).*/\1/' | cut -c1-7)
        if [ -n "$test_psnr" ] && [ -n "$train_psnr" ]; then
            gap=$(awk -v tr="$train_psnr" -v te="$test_psnr" 'BEGIN {printf "%.2f", tr - te}')
            printf "  %-15s | %10s | %10s | %10s\n" "${c_name}" "${test_psnr}" "${train_psnr}" "${gap}"
        else
            printf "  %-15s | %10s | %10s | %10s\n" "${c_name}" "—" "—" "—"
        fi
    else
        printf "  %-15s | %10s | %10s | %10s\n" "${c_name}" "MISS" "MISS" "—"
    fi
done

echo ""
echo "=================================================================="
echo "  Diễn giải:"
echo "    Gap giảm + Test PSNR tăng → SH overfit CONFIRMED (H4 đúng)"
echo "    Gap không đổi              → H4 bác bỏ, tìm nguyên nhân khác"
echo "=================================================================="
