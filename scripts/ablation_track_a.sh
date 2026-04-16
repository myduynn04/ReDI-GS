#!/bin/bash
# ============================================================
# [CRSGaussian Track A] Diagnose DC vs rest contribution in SH overfit
# File: scripts/ablation_track_a.sh
#
# Hypothesis cần test:
#   SH overfit có thể do (a) DC drift, (b) higher-order SH capacity,
#   hoặc (c) cả hai. Cần isolate từng component.
#
# Referencing từ batch trước (AVG 8 scenes):
#   SH1 (sh_degree=1)     → +0.20 dB vs SH3
#   Freeze1k (cả DC+rest) → +0.32 dB vs B0
#
# 2 configs mới × 3 scenes (fern, orchids, room):
#   A1 — sh_degree=1 + freeze_sh_after=1000
#        (reduce capacity + hard freeze all SH)
#        Kỳ vọng: effects cộng → > +0.32 hay saturate?
#   A2 — freeze_dc_only (chỉ freeze f_dc, f_rest tự do)
#        (isolate DC drift từ rest)
#        Kỳ vọng:
#          A2 ≈ Freeze1k → DC là culprit chính
#          A2 << Freeze1k → rest cũng contribute mạnh
#
# Full 8 LLFF scenes × 2 configs = 16 runs mới.
# fern/orchids/room đã có từ lần trước → SKIP tự động, chỉ cần 10 runs mới.
#
# USAGE (parallel 2 GPUs, ~50-60 min wall):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/ablation_track_a.sh A1 0 \
#       2>&1 | tee logs/ablation_track_a/_wrapperA1.log &
#   CUDA_VISIBLE_DEVICES=1 bash scripts/ablation_track_a.sh A2 1 \
#       2>&1 | tee logs/ablation_track_a/_wrapperA2.log &
#   wait
#   bash scripts/ablation_track_a.sh summary
#
# Chia 4 GPUs nếu có (A1A/A1B/A2A/A2B, ~25 min):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/ablation_track_a.sh A1A 0 &
#   CUDA_VISIBLE_DEVICES=1 bash scripts/ablation_track_a.sh A1B 1 &
#   CUDA_VISIBLE_DEVICES=2 bash scripts/ablation_track_a.sh A2A 2 &
#   CUDA_VISIBLE_DEVICES=3 bash scripts/ablation_track_a.sh A2B 3 &
#   wait
#   bash scripts/ablation_track_a.sh summary
# ============================================================

CFG=${1:-ALL}
GPU=${2:-0}

ALL_SCENES="fern flower fortress horns leaves orchids room trex"
SCENES_A="fern flower fortress horns"
SCENES_B="leaves orchids room trex"

# Xác định SCENES theo CFG:
#   A1A / A1B / A1 → chạy A1 trên subset A / B / full
#   A2A / A2B / A2 → chạy A2 trên subset A / B / full
#   ALL            → cả A1+A2 × full 8 scenes (1 GPU sequential)
#   summary        → in bảng
case ${CFG} in
    A1A|a1a) SCENES="${SCENES_A}"; RUN_A1=1; RUN_A2=0 ;;
    A1B|a1b) SCENES="${SCENES_B}"; RUN_A1=1; RUN_A2=0 ;;
    A1|a1)   SCENES="${ALL_SCENES}"; RUN_A1=1; RUN_A2=0 ;;
    A2A|a2a) SCENES="${SCENES_A}"; RUN_A1=0; RUN_A2=1 ;;
    A2B|a2b) SCENES="${SCENES_B}"; RUN_A1=0; RUN_A2=1 ;;
    A2|a2)   SCENES="${ALL_SCENES}"; RUN_A1=0; RUN_A2=1 ;;
    ALL|all) SCENES="${ALL_SCENES}"; RUN_A1=1; RUN_A2=1 ;;
    summary|SUMMARY) SCENES=""; RUN_A1=0; RUN_A2=0 ;;
    *) echo "Unknown CFG: ${CFG}"; exit 1 ;;
esac

DATA_ROOT="data/nerf_llff_data"
DAV2_PATH="../Depth-Anything-V2"
ITERATIONS=10000
N_VIEWS=3
RESOLUTION=8
TEST_ITERS="1000 2000 3000 5000 7000 10000"

LOGDIR="logs/ablation_track_a"
mkdir -p ${LOGDIR}

# ── Common flags (best WG base) ──
BASE_FLAGS=(
    --eval -r ${RESOLUTION} --n_views ${N_VIEWS}
    --random_background
    --iterations ${ITERATIONS}
    --densify_until_iter 5000
    --densify_grad_threshold 0.0005
    --gaussiansN 1
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

run_a1() {
    local scene=$1
    local tag="A1_${scene}"
    local log="${LOGDIR}/${tag}.log"
    local out="output/ablation_track_a/${tag}"

    if [ -f "$log" ] && grep -q "\\[ITER 10000\\] Evaluating" "$log" 2>/dev/null; then
        echo "[SKIP] ${tag} — already done"
        return
    fi

    echo ""
    echo "========================================"
    echo " ${tag} | sh_degree=1 + freeze_sh_after=1000 | GPU=${GPU}"
    echo "========================================"

    python -u train.py \
        --source_path ${DATA_ROOT}/${scene} \
        -m ${out} \
        "${BASE_FLAGS[@]}" \
        --sh_degree 1 \
        --freeze_sh_after 1000 \
        2>&1 | tee ${log} || echo "[ERROR] ${tag} crashed"

    if grep -q "\\[ITER 10000\\] Evaluating" "$log" 2>/dev/null; then
        echo "[DONE] ${tag}"
    else
        echo "[FAIL] ${tag} — no ITER 10000 in log"
    fi
}

run_a2() {
    local scene=$1
    local tag="A2_${scene}"
    local log="${LOGDIR}/${tag}.log"
    local out="output/ablation_track_a/${tag}"

    if [ -f "$log" ] && grep -q "\\[ITER 10000\\] Evaluating" "$log" 2>/dev/null; then
        echo "[SKIP] ${tag} — already done"
        return
    fi

    echo ""
    echo "========================================"
    echo " ${tag} | freeze_dc_only @ iter=1000 | GPU=${GPU}"
    echo "========================================"

    python -u train.py \
        --source_path ${DATA_ROOT}/${scene} \
        -m ${out} \
        "${BASE_FLAGS[@]}" \
        --freeze_dc_only \
        --freeze_dc_start_iter 1000 \
        2>&1 | tee ${log} || echo "[ERROR] ${tag} crashed"

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
if [ -n "${SCENES}" ]; then
    for s in ${SCENES}; do
        [ "${RUN_A1}" = "1" ] && run_a1 "$s"
        [ "${RUN_A2}" = "1" ] && run_a2 "$s"
    done
fi

# ── Summary ──
# Force C locale để awk dùng dấu . thay vì , (fix +inf bug do vi_VN locale)
export LC_ALL=C

echo ""
echo "======================================================================================================================"
echo "  SUMMARY — Track A: DC vs rest contribution (full 8 LLFF scenes)"
echo "======================================================================================================================"
printf "  %-14s" "Config"
for s in ${ALL_SCENES}; do printf " | %7s" "${s}"; done
printf " | %7s | %7s\n" "AVG" "Δ vs B0"
printf "  %-14s" "--------------"
for s in ${ALL_SCENES}; do printf -- "----------"; done
printf -- "----------\n"

declare -A B0_VALS

for cfg_spec in \
    "B0|logs/baseline_corgs_vs_crsg/CRSG_B0" \
    "SH1 (ref)|logs/ablation_sh_degree/SH1" \
    "Freeze1k (ref)|logs/baseline_corgs_vs_crsg/CRSG_SH1k" \
    "A1: SH1+freeze|logs/ablation_track_a/A1" \
    "A2: DC-only|logs/ablation_track_a/A2"; do
    label="${cfg_spec%%|*}"
    prefix="${cfg_spec#*|}"

    printf "  %-14s" "${label}"
    sum="0"; n=0
    for s in ${ALL_SCENES}; do
        log="${prefix}_${s}.log"
        val=$(extract_psnr "$log" 10000 test)
        if [ -n "$val" ]; then
            printf " | %7s" "${val}"
            sum=$(awk -v s="$sum" -v v="$val" 'BEGIN {printf "%.6f", s + v}')
            n=$((n + 1))
            if [ "$label" = "B0" ]; then B0_VALS[$s]="$val"; fi
        else
            printf " | %7s" "—"
        fi
    done

    if [ "$n" -gt 0 ]; then
        avg=$(awk -v s="$sum" -v n="$n" 'BEGIN {printf "%.3f", s / n}')
        printf " | %7s" "${avg}"
        if [ "$label" = "B0" ]; then
            printf " | %7s\n" "—"
        else
            sum_d="0"; n_d=0
            for s in ${ALL_SCENES}; do
                log="${prefix}_${s}.log"
                v=$(extract_psnr "$log" 10000 test)
                b="${B0_VALS[$s]}"
                if [ -n "$v" ] && [ -n "$b" ]; then
                    sum_d=$(awk -v s="$sum_d" -v v="$v" -v b="$b" 'BEGIN {printf "%.6f", s + (v - b)}')
                    n_d=$((n_d + 1))
                fi
            done
            if [ "$n_d" -gt 0 ]; then
                avg_d=$(awk -v s="$sum_d" -v n="$n_d" 'BEGIN {printf "%+.3f", s / n_d}')
                printf " | %7s\n" "${avg_d}"
            else
                printf " | %7s\n" "—"
            fi
        fi
    else
        printf " | %7s | %7s\n" "—" "—"
    fi
done

# Gap avg (Train-Test @10k) full 8 scenes
echo ""
echo "  Gap avg (Train@10k − Test@10k, 8 scenes):"
for cfg_spec in \
    "B0|logs/baseline_corgs_vs_crsg/CRSG_B0" \
    "SH1 (ref)|logs/ablation_sh_degree/SH1" \
    "Freeze1k (ref)|logs/baseline_corgs_vs_crsg/CRSG_SH1k" \
    "A1: SH1+freeze|logs/ablation_track_a/A1" \
    "A2: DC-only|logs/ablation_track_a/A2"; do
    label="${cfg_spec%%|*}"
    prefix="${cfg_spec#*|}"
    sum="0"; n=0
    for s in ${ALL_SCENES}; do
        log="${prefix}_${s}.log"
        te=$(extract_psnr "$log" 10000 test)
        tr=$(extract_psnr "$log" 10000 train)
        if [ -n "$te" ] && [ -n "$tr" ]; then
            g=$(awk -v tr="$tr" -v te="$te" 'BEGIN {printf "%.3f", tr - te}')
            sum=$(awk -v s="$sum" -v g="$g" 'BEGIN {printf "%.6f", s + g}')
            n=$((n + 1))
        fi
    done
    if [ "$n" -gt 0 ]; then
        avg=$(awk -v s="$sum" -v n="$n" 'BEGIN {printf "%.2f", s / n}')
        printf "    %-14s %s dB\n" "${label}" "${avg}"
    fi
done

echo ""
echo "==========================================================================="
echo "  Interpret:"
echo "    A2 ≈ Freeze1k      → DC drift là culprit chính"
echo "    A2 << Freeze1k     → higher-order SH cũng contribute"
echo "    A1 > Freeze1k      → reduce+freeze có effect cộng → hybrid hợp lý"
echo "    A1 ≈ Freeze1k      → capacity và freeze saturate ở cùng mức"
echo "==========================================================================="
