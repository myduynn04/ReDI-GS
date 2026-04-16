#!/bin/bash
# ============================================================
# [CRSGaussian DIAG E1b] SH Freeze Optimal Point
# File: scripts/diagnose_e1b.sh
#
# Mục đích: Tìm optimal freeze_sh_after — có vượt được Test peak
#   22.65 @iter 1000 của baseline không?
#
# Configs (3 × fern = 3 runs):
#   E1b_freeze1k    — freeze_sh_after=1000 (đúng peak)
#   E1b_freeze1.5k  — freeze_sh_after=1500
#   E1b_freeze2k    — freeze_sh_after=2000
#
# Reference: E1_baseline_fern (freeze=0):
#   Test@1k=22.65 Test@3k=22.16 Test@10k=21.99 Train=37.78 Gap=15.79
#   E1_freeze3k (best E1):
#   Test@10k=22.40 Train=35.60 Gap=13.20
#
# USAGE (GPU 0):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/diagnose_e1b.sh
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

LOGDIR="logs/diag_e1b"
mkdir -p ${LOGDIR}

# ── Config: name|freeze_after ──
CONFIGS="
E1b_freeze1k|1000
E1b_freeze1.5k|1500
E1b_freeze2k|2000
"

run_one() {
    local cfg_name=$1
    local freeze_after=$2

    local tag="${cfg_name}_${SCENE}"
    local log="${LOGDIR}/${tag}.log"
    local out="output/diag_e1b/${tag}"

    if [ -f "$log" ] && grep -q "\\[ITER 10000\\] Evaluating" "$log" 2>/dev/null; then
        echo "[SKIP] ${tag} — already done"
        return
    fi

    echo ""
    echo "========================================"
    echo " ${cfg_name} | scene=${SCENE} | freeze_sh_after=${freeze_after}"
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
echo " DIAG E1b — SH Freeze Optimal Point"
echo " Scene: ${SCENE}, GPU: ${GPU}, ${total} configs"
echo "============================================"

count=0
for c_line in ${CONFIGS}; do
    [ -z "$c_line" ] && continue
    IFS='|' read -r c_name c_freeze <<< "$c_line"
    count=$((count + 1))
    echo "[${count}/${total}]"
    run_one "$c_name" "$c_freeze"
done

# ── Extract metric from log at specified iter ──
extract_psnr() {
    local log=$1
    local iter=$2
    local split=$3   # test | train
    if [ ! -f "$log" ]; then
        echo "—"
        return
    fi
    local val
    val=$(grep "\\[ITER ${iter}\\] Evaluating ${split}" "$log" 2>/dev/null | tail -1 | sed 's/.*PSNR \([0-9.]*\).*/\1/' | cut -c1-7)
    if [ -z "$val" ]; then echo "—"; else echo "$val"; fi
}

# ── Summary ──
echo ""
echo "========================================================================="
echo "  SUMMARY — DIAG E1b — scene=${SCENE}"
echo "========================================================================="
printf "  %-15s | %7s | %7s | %7s | %7s | %7s | %7s\n" \
    "Config" "T@1k" "T@3k" "T@5k" "T@10k" "Tr@10k" "Gap"
printf "  %-15s-+-%7s-+-%7s-+-%7s-+-%7s-+-%7s-+-%7s\n" \
    "---------------" "-------" "-------" "-------" "-------" "-------" "-------"

# Baseline reference row (từ E1)
printf "  %-15s | %7s | %7s | %7s | %7s | %7s | %7s\n" \
    "baseline (E1)" "22.65" "22.16" "22.06" "21.99" "37.78" "15.79"
printf "  %-15s | %7s | %7s | %7s | %7s | %7s | %7s\n" \
    "E1_freeze3k"   "22.63" "22.29" "22.31" "22.40" "35.60" "13.20"

for c_line in ${CONFIGS}; do
    [ -z "$c_line" ] && continue
    IFS='|' read -r c_name c_freeze <<< "$c_line"
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
echo "  Diễn giải E1b:"
echo "    Có config > 22.65 @10k → SH freeze SỚM là bottleneck chính"
echo "    Peak ở freeze1k nhưng < 22.65 → cần fix khác (densification/CRS)"
echo "========================================================================="
