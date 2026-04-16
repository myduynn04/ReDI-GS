#!/bin/bash
# ============================================================
# [CRSGaussian Ablation E3] sh_degree {1, 2, 3} — full 8 LLFF scenes
# File: scripts/ablation_sh_degree.sh
#
# Mục đích: Test hypothesis SH capacity là culprit gây overfit.
#   E1b confirm freeze SH từ iter 1000 → +0.60 dB.
#   Câu hỏi: giảm sh_degree (capacity nhỏ hơn) có giúp tương tự không?
#
# 3 Configs × 8 scenes = 24 runs (REUSE SH3 = 16 runs mới):
#   SH1 — --sh_degree 1   (12 params/Gaussian)
#   SH2 — --sh_degree 2   (27 params/Gaussian)
#   SH3 — --sh_degree 3   (48 params/Gaussian, baseline) ← REUSE từ
#         logs/baseline_corgs_vs_crsg/CRSG_B0_<scene>.log
#
# USAGE (parallel 2 GPUs, ~1h):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/ablation_sh_degree.sh A 0 \
#       2>&1 | tee logs/ablation_sh_degree/_wrapperA.log &
#   CUDA_VISIBLE_DEVICES=1 bash scripts/ablation_sh_degree.sh B 1 \
#       2>&1 | tee logs/ablation_sh_degree/_wrapperB.log &
#   wait
#   bash scripts/ablation_sh_degree.sh summary
#
# Single scene:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/ablation_sh_degree.sh fern 0
# ============================================================

SCENE=${1:-ALL}
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

LOGDIR="logs/ablation_sh_degree"
mkdir -p ${LOGDIR}

# ── Reuse SH3 từ baseline_corgs_vs_crsg/CRSG_B0_<scene> (cùng config y hệt) ──
reuse_sh3() {
    local src dst
    for s in ${ALL_SCENES}; do
        src="logs/baseline_corgs_vs_crsg/CRSG_B0_${s}.log"
        dst="${LOGDIR}/SH3_${s}.log"
        if [ -f "$src" ] && [ ! -f "$dst" ]; then
            cp "$src" "$dst"
            echo "[REUSE] SH3_${s} ← baseline_corgs_vs_crsg/CRSG_B0_${s}.log"
        fi
    done
}

run_sh() {
    local degree=$1
    local scene=$2
    local tag="SH${degree}_${scene}"
    local log="${LOGDIR}/${tag}.log"
    local out="output/ablation_sh_degree/${tag}"

    if [ -f "$log" ] && grep -q "\\[ITER 10000\\] Evaluating" "$log" 2>/dev/null; then
        echo "[SKIP] ${tag} — already done"
        return
    fi

    echo ""
    echo "========================================"
    echo " ${tag} | --sh_degree ${degree} | scene=${scene} | GPU=${GPU}"
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
        --sh_degree ${degree} \
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
        --test_iterations ${TEST_ITERS} \
        2>&1 | tee ${log} || echo "[ERROR] ${tag} — train.py crashed"

    if grep -q "\\[ITER 10000\\] Evaluating" "$log" 2>/dev/null; then
        echo "[DONE] ${tag}"
    else
        echo "[FAIL] ${tag} — no ITER 10000 in log"
    fi
}

extract_psnr() {
    local log=$1; local iter=$2; local split=$3
    if [ ! -f "$log" ]; then echo ""; return; fi
    grep "\\[ITER ${iter}\\] Evaluating ${split}" "$log" 2>/dev/null \
      | tail -1 | sed 's/.*PSNR \([0-9.]*\).*/\1/' | cut -c1-7
}

# ── Main ──
reuse_sh3

if [ -n "${SCENES}" ]; then
    n_scenes=$(echo ${SCENES} | wc -w)
    total=$((n_scenes * 2))   # chỉ SH1, SH2 mới chạy (SH3 reuse)
    echo "============================================"
    echo " Ablation sh_degree: ${total} new runs (SH1 + SH2 × ${n_scenes} scenes)"
    echo " GPU: ${GPU}, Scenes: ${SCENES}"
    echo "============================================"

    count=0
    for s in ${SCENES}; do
        count=$((count + 1)); echo "[${count}/${total}]"
        run_sh 1 "$s"
        count=$((count + 1)); echo "[${count}/${total}]"
        run_sh 2 "$s"
    done
fi

# ── Summary ──
echo ""
echo "==================================================================================================="
echo "  SUMMARY — Ablation sh_degree — Test PSNR @10k"
echo "==================================================================================================="

# Header
printf "  %-6s" "Cfg"
for s in ${ALL_SCENES}; do printf " | %7s" "${s}"; done
printf " | %7s\n" "AVG"
printf "  %-6s" "------"
for s in ${ALL_SCENES}; do printf -- "----------"; done
printf -- "----------\n"

declare -A SH3_VALS

for cfg in SH1 SH2 SH3; do
    printf "  %-6s" "${cfg}"
    sum="0"; n=0
    for s in ${ALL_SCENES}; do
        log="${LOGDIR}/${cfg}_${s}.log"
        val=$(extract_psnr "$log" 10000 test)
        if [ -n "$val" ]; then
            printf " | %7s" "${val}"
            sum=$(awk -v s="$sum" -v v="$val" 'BEGIN {printf "%.6f", s + v}')
            n=$((n + 1))
            if [ "$cfg" = "SH3" ]; then SH3_VALS[$s]="$val"; fi
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

# Delta vs SH3
echo ""
for cfg in SH1 SH2; do
    printf "  Δ %-4s" "${cfg}"
    sum_d="0"; n_d=0
    for s in ${ALL_SCENES}; do
        log="${LOGDIR}/${cfg}_${s}.log"
        v=$(extract_psnr "$log" 10000 test)
        b="${SH3_VALS[$s]}"
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

# Train + Gap @10k cho insight overfit
echo ""
echo "  Train PSNR @10k:"
for cfg in SH1 SH2 SH3; do
    printf "  %-6s" "${cfg}"
    sum="0"; n=0
    for s in ${ALL_SCENES}; do
        log="${LOGDIR}/${cfg}_${s}.log"
        val=$(extract_psnr "$log" 10000 train)
        if [ -n "$val" ]; then
            printf " | %7s" "${val}"
            sum=$(awk -v s="$sum" -v v="$val" 'BEGIN {printf "%.6f", s + v}')
            n=$((n + 1))
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

echo ""
echo "==================================================================================================="
echo "  Reference: freeze SH 1k AVG = +0.32~+0.47 dB so với baseline."
echo "  Interpret:"
echo "    SH1 AVG > SH3 AVG đáng kể (>0.3 dB) → SH capacity culprit → CRS dropout"
echo "    SH1 AVG ≈ SH3 AVG (<0.1 dB)        → degree không quan trọng → DC drift mới là culprit"
echo "==================================================================================================="
