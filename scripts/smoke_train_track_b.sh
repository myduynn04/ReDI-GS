#!/bin/bash
# ============================================================
# [CRSGaussian Track B] Smoke test 3 dropout modes trên fern
# File: scripts/smoke_train_track_b.sh
#
# Mục đích: Verify 3 modes (uniform/sh_norm/hybrid) chạy được, không crash,
#   drop rate in log đúng kỳ vọng. KHÔNG phải full ablation.
#
# Chạy 3000 iter thay vì 10k để nhanh (~4 min/config × 3 = ~12 min).
# Eval tại 1000, 2000, 3000 → đủ để thấy dropout active + PSNR evolve.
#
# USAGE (1 GPU sequential):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/smoke_train_track_b.sh
#
# Parallel 2 GPUs (B1+B3 trên GPU0, B4 trên GPU1):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/smoke_train_track_b.sh B1 &
#   CUDA_VISIBLE_DEVICES=1 bash scripts/smoke_train_track_b.sh B3 &
#   wait
#   CUDA_VISIBLE_DEVICES=0 bash scripts/smoke_train_track_b.sh B4
# ============================================================

MODE=${1:-ALL}

DATA_ROOT="data/nerf_llff_data"
DAV2_PATH="../Depth-Anything-V2"
SCENE="fern"
ITERATIONS=3000
N_VIEWS=3
RESOLUTION=8
TEST_ITERS="1000 2000 3000"

LOGDIR="logs/smoke_track_b"
mkdir -p ${LOGDIR}

# ── Base flags (A1 config, match verify_off) ──
BASE_FLAGS=(
    --source_path ${DATA_ROOT}/${SCENE}
    --eval -r ${RESOLUTION} --n_views ${N_VIEWS}
    --random_background
    --iterations ${ITERATIONS}
    --densify_until_iter 5000
    --densify_grad_threshold 0.0005
    --gaussiansN 1
    --sh_degree 1
    --freeze_sh_after 1000
    --use_depth_prior --dav2_path ${DAV2_PATH}
    --sample_pseudo_interval 1
    --start_sample_pseudo 500
    --informed_crs_init
    --use_crs_pruning
    --crs_ema_decay 0.3
    --crs_update_interval 100
    --crs_init_use_view False
    --crs_init_w_reproj 0.4
    --crs_init_w_depth 0.6
    --crs_init_w_view 0.0
    --test_iterations ${TEST_ITERS}
)

run_smoke() {
    local tag=$1
    shift
    local extra_flags=("$@")
    local log="${LOGDIR}/${tag}_fern.log"
    local out="output/smoke_track_b/${tag}_fern"

    echo ""
    echo "================================================================"
    echo " SMOKE ${tag} | fern | 3k iter | extra=${extra_flags[*]}"
    echo "================================================================"

    python -u train.py \
        "${BASE_FLAGS[@]}" \
        -m ${out} \
        "${extra_flags[@]}" \
        2>&1 | tee ${log} || echo "[ERROR] ${tag} crashed"

    echo ""
    echo "── ${tag} drop rate log ──"
    grep "DIAG Track B" "$log" | tail -5
    echo "── ${tag} test PSNR ──"
    grep "\[ITER [0-9]\+\] Evaluating test" "$log"
}

# ── 3 smoke configs ──
case ${MODE} in
    B1|b1)
        run_smoke "B1_uniform" \
            --use_dropout --dropout_mode uniform --dropout_base 0.2
        ;;
    B3|b3)
        run_smoke "B3_sh_norm" \
            --use_dropout --dropout_mode sh_norm --dropout_base 0.1 --dropout_w_sh 0.5
        ;;
    B4|b4)
        run_smoke "B4_hybrid" \
            --use_dropout --dropout_mode hybrid --dropout_base 0.1 --dropout_w_crs 0.3 --dropout_w_sh 0.3
        ;;
    ALL|all)
        run_smoke "B1_uniform" \
            --use_dropout --dropout_mode uniform --dropout_base 0.2
        run_smoke "B3_sh_norm" \
            --use_dropout --dropout_mode sh_norm --dropout_base 0.1 --dropout_w_sh 0.5
        run_smoke "B4_hybrid" \
            --use_dropout --dropout_mode hybrid --dropout_base 0.1 --dropout_w_crs 0.3 --dropout_w_sh 0.3
        ;;
    *)
        echo "Unknown MODE: ${MODE}"; exit 1 ;;
esac

# ── Summary ──
echo ""
echo "======================================================================"
echo "  SMOKE SUMMARY — drop rate expectations"
echo "======================================================================"
printf "  %-12s | %-15s | %-18s | %-15s\n" \
    "Config" "expected drop" "actual drop (avg)" "test@3k PSNR"
printf "  %-12s-+-%-15s-+-%-18s-+-%-15s\n" \
    "------------" "---------------" "------------------" "---------------"

for cfg in "B1_uniform|~0.20" "B3_sh_norm|varies 0.10-0.60" "B4_hybrid|varies 0.10-0.60"; do
    tag="${cfg%%|*}"
    expect="${cfg#*|}"
    log="${LOGDIR}/${tag}_fern.log"

    if [ -f "$log" ]; then
        actual=$(grep "DIAG Track B" "$log" \
            | sed -n 's/.*drop_rate=\([0-9.]*\).*/\1/p' \
            | awk 'BEGIN{s=0;n=0} {s+=$1; n++} END{if(n>0) printf "%.3f",s/n; else print "—"}')
        psnr=$(grep "\[ITER 3000\] Evaluating test" "$log" 2>/dev/null \
            | tail -1 | sed 's/.*PSNR \([0-9.]*\).*/\1/' | cut -c1-7)
        [ -z "$psnr" ] && psnr="—"
    else
        actual="—"
        psnr="—"
    fi

    printf "  %-12s | %-15s | %-18s | %-15s\n" "${tag}" "${expect}" "${actual}" "${psnr}"
done

echo ""
echo "======================================================================"
echo "  Kiểm tra thủ công:"
echo "    1. B1 actual drop ≈ 0.20 (constant)"
echo "    2. B3 actual drop trong [0.10, 0.60] (varies theo sh_norm)"
echo "    3. B4 actual drop trong [0.10, 0.60] (varies theo CRS+sh_norm)"
echo "    4. Không có [ERROR] hay crash"
echo "    5. Test PSNR @3k ~22 dB (gần baseline)"
echo "======================================================================"
