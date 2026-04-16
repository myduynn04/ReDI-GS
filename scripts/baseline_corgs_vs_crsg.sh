#!/bin/bash
# ============================================================
# [CRSGaussian Baseline] CoR-GS gốc vs CRSGaussian B0 vs SH1k
# File: scripts/baseline_corgs_vs_crsg.sh
#
# Mục đích: Tạo bảng đối chiếu chuẩn 3 configs cùng code state,
#   tránh CUDA noise drift khi so sánh với T0.5 (cũ ~3 tháng).
#
# 3 Configs × 8 scenes = 24 runs:
#   CORGS  — CoR-GS gốc (gaussiansN=2, --coreg, --coprune, NO depth)
#   CRSG_B0 — CRSGaussian best WG (single-field, informed CRS, no SH freeze)
#   CRSG_SH1k — B0 + freeze_sh_after=1000
#
# REUSE LOGIC:
#   - CRSG_B0 + CRSG_SH1k: reuse từ logs/ablation_sh_freeze/ (vừa chạy xong)
#   - CORGS fern: reuse từ logs/fern_corgs_multitest.log (cũ, NẾU MUỐN BỎ
#     reuse này thì xóa log đích → script sẽ rerun)
#   - CORGS 7 scenes khác: chạy mới
#
# Nếu muốn FRESH HOÀN TOÀN: xóa logs/baseline_corgs_vs_crsg/ trước khi chạy
#
# USAGE (parallel 2 GPUs):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/baseline_corgs_vs_crsg.sh A 0 &
#   CUDA_VISIBLE_DEVICES=1 bash scripts/baseline_corgs_vs_crsg.sh B 1 &
#   wait
#   bash scripts/baseline_corgs_vs_crsg.sh summary
#
# Single scene:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/baseline_corgs_vs_crsg.sh fern 0
# ============================================================

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
TEST_ITERS="1000 2000 3000 5000 7000 10000"

LOGDIR="logs/baseline_corgs_vs_crsg"
mkdir -p ${LOGDIR}

# ── Reuse logs từ batch ablation_sh_freeze (CRSG_B0 + CRSG_SH1k đã chạy) ──
reuse_existing_logs() {
    local src dst
    for s in ${ALL_SCENES}; do
        # CRSG_B0 ← ablation_sh_freeze/B0_*
        src="logs/ablation_sh_freeze/B0_${s}.log"
        dst="${LOGDIR}/CRSG_B0_${s}.log"
        if [ -f "$src" ] && [ ! -f "$dst" ]; then
            cp "$src" "$dst"
            echo "[REUSE] CRSG_B0_${s} ← ablation_sh_freeze/B0_${s}"
        fi
        # CRSG_SH1k ← ablation_sh_freeze/SH1k_*
        src="logs/ablation_sh_freeze/SH1k_${s}.log"
        dst="${LOGDIR}/CRSG_SH1k_${s}.log"
        if [ -f "$src" ] && [ ! -f "$dst" ]; then
            cp "$src" "$dst"
            echo "[REUSE] CRSG_SH1k_${s} ← ablation_sh_freeze/SH1k_${s}"
        fi
    done
    # CORGS fern multitest log nếu có
    src="logs/fern_corgs_multitest.log"
    dst="${LOGDIR}/CORGS_fern.log"
    if [ -f "$src" ] && [ ! -f "$dst" ]; then
        cp "$src" "$dst"
        echo "[REUSE] CORGS_fern ← fern_corgs_multitest.log"
    fi
}

run_corgs() {
    local scene=$1
    local tag="CORGS_${scene}"
    local log="${LOGDIR}/${tag}.log"
    local out="output/baseline_corgs_vs_crsg/${tag}"

    if [ -f "$log" ] && grep -q "\\[ITER 10000\\] Evaluating" "$log" 2>/dev/null; then
        echo "[SKIP] ${tag} — already done"
        return
    fi

    echo ""
    echo "========================================"
    echo " CORGS gốc | scene=${scene} | gaussiansN=2 + coreg + coprune | GPU=${GPU}"
    echo "========================================"

    python -u train.py \
        --source_path ${DATA_ROOT}/${scene} \
        -m ${out} \
        --eval -r ${RESOLUTION} --n_views ${N_VIEWS} \
        --random_background \
        --iterations ${ITERATIONS} \
        --gaussiansN 2 \
        --coreg --coprune \
        --sample_pseudo_interval 1 \
        --start_sample_pseudo 500 \
        --test_iterations ${TEST_ITERS} \
        2>&1 | tee ${log} || echo "[ERROR] ${tag} — train.py crashed"

    if grep -q "\\[ITER 10000\\] Evaluating" "$log" 2>/dev/null; then
        echo "[DONE] ${tag}"
    else
        echo "[FAIL] ${tag} — no ITER 10000 in log"
    fi
}

run_crsg() {
    local cfg=$1            # "B0" or "SH1k"
    local freeze_after=$2   # 0 or 1000
    local scene=$3
    local tag="CRSG_${cfg}_${scene}"
    local log="${LOGDIR}/${tag}.log"
    local out="output/baseline_corgs_vs_crsg/${tag}"

    if [ -f "$log" ] && grep -q "\\[ITER 10000\\] Evaluating" "$log" 2>/dev/null; then
        echo "[SKIP] ${tag} — already done"
        return
    fi

    echo ""
    echo "========================================"
    echo " CRSG_${cfg} | scene=${scene} | freeze_sh_after=${freeze_after} | GPU=${GPU}"
    echo "========================================"

    python -u train.py \
        --source_path ${DATA_ROOT}/${scene} \
        -m ${out} \
        --eval -r ${RESOLUTION} --n_views ${N_VIEWS} \
        --random_background \
        --iterations ${ITERATIONS} \
        --densify_until_iter 5000 \
        --densify_grad_threshold 0.0005 \
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
        2>&1 | tee ${log} || echo "[ERROR] ${tag} — train.py crashed"

    if grep -q "\\[ITER 10000\\] Evaluating" "$log" 2>/dev/null; then
        echo "[DONE] ${tag}"
    else
        echo "[FAIL] ${tag} — no ITER 10000 in log"
    fi
}

extract_psnr() {
    local log=$1
    local iter=$2
    local split=$3
    if [ ! -f "$log" ]; then echo ""; return; fi
    local val
    val=$(grep "\\[ITER ${iter}\\] Evaluating ${split}" "$log" 2>/dev/null | tail -1 | sed 's/.*PSNR \([0-9.]*\).*/\1/' | cut -c1-7)
    echo "$val"
}

# ── Main ──
reuse_existing_logs

if [ -n "${SCENES}" ]; then
    n_scenes=$(echo ${SCENES} | wc -w)
    total=$((n_scenes * 3))
    echo "============================================"
    echo " Baseline CoR-GS vs CRSGaussian: ${total} runs"
    echo " GPU: ${GPU}, Scenes: ${SCENES}"
    echo "============================================"

    count=0
    # Order: chạy CORGS trước (chậm hơn 2x), rồi CRSG_B0, rồi CRSG_SH1k
    for s in ${SCENES}; do
        count=$((count + 1)); echo "[${count}/${total}]"
        run_corgs "$s"
        count=$((count + 1)); echo "[${count}/${total}]"
        run_crsg "B0" 0 "$s"
        count=$((count + 1)); echo "[${count}/${total}]"
        run_crsg "SH1k" 1000 "$s"
    done
fi

# ── Summary ──
echo ""
echo "==================================================================================================="
echo "  SUMMARY — Baseline comparison — Test PSNR @10k"
echo "==================================================================================================="

# Header
printf "  %-10s" "Cfg"
for s in ${ALL_SCENES}; do
    printf " | %7s" "${s}"
done
printf " | %7s\n" "AVG"
printf "  %-10s" "----------"
for s in ${ALL_SCENES}; do printf -- "----------"; done
printf -- "----------\n"

declare -A CORGS_VALS

for cfg_label in "CORGS" "CRSG_B0" "CRSG_SH1k"; do
    printf "  %-10s" "${cfg_label}"
    sum="0"
    n=0
    for s in ${ALL_SCENES}; do
        log="${LOGDIR}/${cfg_label}_${s}.log"
        val=$(extract_psnr "$log" 10000 test)
        if [ -n "$val" ]; then
            printf " | %7s" "${val}"
            sum=$(awk -v s="$sum" -v v="$val" 'BEGIN {printf "%.6f", s + v}')
            n=$((n + 1))
            if [ "$cfg_label" = "CORGS" ]; then CORGS_VALS[$s]="$val"; fi
        else
            printf " | %7s" "—"
        fi
    done
    if [ "$n" -gt 0 ]; then
        avg=$(awk -v s="$sum" -v n="$n" 'BEGIN {printf "%.3f", s / n}')
        printf " | %7s\n" "${avg}"
    else
        printf " | %7s\n" "—"
    fi
done

# Delta vs CORGS
echo ""
for cfg_label in "CRSG_B0" "CRSG_SH1k"; do
    printf "  %-10s" "Δ ${cfg_label#CRSG_}"
    sum_d="0"
    n_d=0
    for s in ${ALL_SCENES}; do
        log="${LOGDIR}/${cfg_label}_${s}.log"
        v=$(extract_psnr "$log" 10000 test)
        b="${CORGS_VALS[$s]}"
        if [ -n "$v" ] && [ -n "$b" ]; then
            d=$(awk -v a="$v" -v b="$b" 'BEGIN {printf "%+.3f", a - b}')
            printf " | %7s" "${d}"
            sum_d=$(awk -v s="$sum_d" -v v="$v" -v b="$b" 'BEGIN {printf "%.6f", s + (v - b)}')
            n_d=$((n_d + 1))
        else
            printf " | %7s" "—"
        fi
    done
    if [ "$n_d" -gt 0 ]; then
        avg_d=$(awk -v s="$sum_d" -v n="$n_d" 'BEGIN {printf "%+.3f", s / n_d}')
        printf " | %7s\n" "${avg_d}"
    else
        printf " | %7s\n" "—"
    fi
done

echo ""
echo "==================================================================================================="
echo "  Legend:"
echo "    CORGS      — CoR-GS gốc (gaussiansN=2, coreg, coprune, NO depth/CRS/SH-freeze)"
echo "    CRSG_B0    — CRSGaussian best WG (single-field, informed CRS, depth loss, no SH freeze)"
echo "    CRSG_SH1k  — CRSG_B0 + freeze_sh_after=1000"
echo "    Δ          — CRSG_* minus CORGS (positive = CRSG tốt hơn)"
echo "==================================================================================================="
