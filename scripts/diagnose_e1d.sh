#!/bin/bash
# ============================================================
# [CRSGaussian DIAG E1d] CRS Pruning Diagnosis
# File: scripts/diagnose_e1d.sh
#
# Mục đích: Test hypothesis "CRS pruning từ iter 1000 gây drop
#   test PSNR (over-prune Gaussians hữu ích cho novel views)".
#
# Configs (2 × fern = 2 runs):
#   E1d_delay_crs   — T_warmup=3000 (CRS active sau iter 3000)
#   E1d_no_crs      — use_crs_pruning=False (tắt hoàn toàn CRS pruning)
#
# Reference: E1_baseline_fern (T_warmup=1000 default):
#   Test@1k=22.65 Test@3k=22.16 Test@10k=21.99 Train=37.78 Gap=15.79
#
# Diễn giải:
#   E1d_no_crs ≥ 22.65 xuyên suốt → CRS over-prune là culprit
#   E1d_delay peak tại iter 3000 rồi drop → CRS timing gây drop
#   Cả 2 ≈ baseline → CRS không phải thủ phạm, gap do SH
#
# USAGE (GPU 1):
#   CUDA_VISIBLE_DEVICES=1 bash scripts/diagnose_e1d.sh
# ============================================================

GPU=${CUDA_VISIBLE_DEVICES:-0}
SCENE="fern"

DATA_ROOT="data/nerf_llff_data"
DAV2_PATH="../Depth-Anything-V2"
ITERATIONS=10000
N_VIEWS=3
RESOLUTION=8
DENSIFY_GRAD=0.0005
DENSIFY_UNTIL=5000
TEST_ITERS="1000 2000 3000 5000 7000 10000"

LOGDIR="logs/diag_e1d"
mkdir -p ${LOGDIR}

# ── Configs: array để tránh IFS splitting issues ──
# Format: "cfg_name|extra_flags"
CONFIG_NAMES=(
    "E1d_delay_crs"
    "E1d_no_crs"
)
CONFIG_FLAGS=(
    "--use_crs_pruning --T_warmup 3000"
    ""
)
# Ghi chú: E1d_no_crs KHÔNG pass --use_crs_pruning (default=False trong
# OptimizationParams → CRS pruning tắt hoàn toàn). Flag là action="store_true",
# nếu pass "False" sẽ bị argparse báo "unrecognized arguments: False".

run_one() {
    local cfg_name=$1
    local extra_flags=$2

    local tag="${cfg_name}_${SCENE}"
    local log="${LOGDIR}/${tag}.log"
    local out="output/diag_e1d/${tag}"

    if [ -f "$log" ] && grep -q "\\[ITER 10000\\] Evaluating" "$log" 2>/dev/null; then
        echo "[SKIP] ${tag} — already done"
        return
    fi

    echo ""
    echo "========================================"
    echo " ${cfg_name} | scene=${SCENE}"
    echo " extra: ${extra_flags}"
    echo " GPU=${GPU}"
    echo "========================================"

    python -u train.py \
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
        --crs_ema_decay 0.3 \
        --crs_update_interval 100 \
        --crs_init_use_view False \
        --crs_init_w_reproj 0.4 \
        --crs_init_w_depth 0.6 \
        --crs_init_w_view 0.0 \
        ${extra_flags} \
        --test_iterations ${TEST_ITERS} \
        2>&1 | tee ${log} || echo "[ERROR] ${tag} — train.py crashed, continuing"

    if grep -q "\\[ITER 10000\\] Evaluating" "$log" 2>/dev/null; then
        echo "[DONE] ${tag}"
    else
        echo "[FAIL] ${tag} — no ITER 10000 in log"
    fi
}

# ── Parse config với separator @@ ──
extract_psnr() {
    local log=$1
    local iter=$2
    local split=$3
    if [ ! -f "$log" ]; then echo "—"; return; fi
    local val
    val=$(grep "\\[ITER ${iter}\\] Evaluating ${split}" "$log" 2>/dev/null | tail -1 | sed 's/.*PSNR \([0-9.]*\).*/\1/' | cut -c1-7)
    if [ -z "$val" ]; then echo "—"; else echo "$val"; fi
}

# ── Main loop (iterate array index) ──
echo "============================================"
echo " DIAG E1d — CRS Pruning Diagnosis"
echo " Scene: ${SCENE}, GPU: ${GPU}"
echo "============================================"

total=${#CONFIG_NAMES[@]}
for i in "${!CONFIG_NAMES[@]}"; do
    count=$((i + 1))
    echo "[${count}/${total}]"
    run_one "${CONFIG_NAMES[$i]}" "${CONFIG_FLAGS[$i]}"
done

# ── Summary ──
echo ""
echo "========================================================================="
echo "  SUMMARY — DIAG E1d — scene=${SCENE}"
echo "========================================================================="
printf "  %-15s | %7s | %7s | %7s | %7s | %7s | %7s\n" \
    "Config" "T@1k" "T@3k" "T@5k" "T@10k" "Tr@10k" "Gap"
printf "  %-15s-+-%7s-+-%7s-+-%7s-+-%7s-+-%7s-+-%7s\n" \
    "---------------" "-------" "-------" "-------" "-------" "-------" "-------"

printf "  %-15s | %7s | %7s | %7s | %7s | %7s | %7s\n" \
    "baseline (E1)" "22.65" "22.16" "22.06" "21.99" "37.78" "15.79"

for i in "${!CONFIG_NAMES[@]}"; do
    c_name="${CONFIG_NAMES[$i]}"
    log="${LOGDIR}/${c_name}_${SCENE}.log"

    t1=$(extract_psnr "$log" 1000 test)
    t3=$(extract_psnr "$log" 3000 test)
    t5=$(extract_psnr "$log" 5000 test)
    t10=$(extract_psnr "$log" 10000 test)
    tr10=$(extract_psnr "$log" 10000 train)
    if [ "$t10" != "—" ] && [ "$tr10" != "—" ]; then
        gap=$(awk -v tr="$tr10" -v te="$t10" 'BEGIN {printf "%.2f", tr - te}')
    else
        gap="—"
    fi
    printf "  %-15s | %7s | %7s | %7s | %7s | %7s | %7s\n" \
        "${c_name}" "${t1}" "${t3}" "${t5}" "${t10}" "${tr10}" "${gap}"
done

echo ""
echo "========================================================================="
echo "  Diễn giải E1d:"
echo "    no_crs ≥ 22.65 xuyên suốt     → CRS over-prune là culprit"
echo "    delay_crs peak tại iter 3000  → CRS timing gây drop"
echo "    Cả 2 ≈ baseline               → CRS không phải thủ phạm"
echo "========================================================================="
