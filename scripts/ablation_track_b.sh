#!/bin/bash
# ============================================================
# [CRSGaussian Track B] Full ablation: Dropout regularization
# File: scripts/ablation_track_b.sh
#
# 7 configs × 8 LLFF scenes = 56 runs.
#   B0'  — A1 baseline (no dropout) — REUSE từ logs/ablation_track_a/A1_<s>.log
#   B1α  — uniform dropout, start iter 0     (Co-Adapt reproduce, early)
#   B1β  — uniform dropout, start iter 1000  (Co-Adapt after warmup)
#   B3α  — sh_norm dropout, start iter 0     (target rest overfit, early)
#   B3β  — sh_norm dropout, start iter 1000  (target rest, after warmup)
#   B4α  — hybrid CRS+sh_norm, start iter 0  (dual signal, early)
#   B4β  — hybrid, start iter 1000           (dual signal, after warmup)
#
# REUSE: 8 B0' logs từ ablation_track_a/A1_<scene>.log → 48 runs mới.
# Estimate: 48 × ~7 min = ~5.6h total. Parallel 2 GPUs ~3h wall.
#
# USAGE (parallel 2 GPUs):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/ablation_track_b.sh A 0 \
#       2>&1 | tee logs/ablation_track_b/_wrapperA.log &
#   CUDA_VISIBLE_DEVICES=1 bash scripts/ablation_track_b.sh B 1 \
#       2>&1 | tee logs/ablation_track_b/_wrapperB.log &
#   wait
#   bash scripts/ablation_track_b.sh summary
#
# Single scene test:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/ablation_track_b.sh fern 0
# ============================================================

SCENE_ARG=${1:-ALL}
GPU=${2:-0}

ALL_SCENES="fern flower fortress horns leaves orchids room trex"
SCENES_A="fern flower fortress horns"
SCENES_B="leaves orchids room trex"

case ${SCENE_ARG} in
    ALL|all)         SCENES="${ALL_SCENES}" ;;
    A|a)             SCENES="${SCENES_A}" ;;
    B|b)             SCENES="${SCENES_B}" ;;
    summary|SUMMARY) SCENES="" ;;
    *)               SCENES="${SCENE_ARG}" ;;
esac

DATA_ROOT="data/nerf_llff_data"
DAV2_PATH="../Depth-Anything-V2"
ITERATIONS=10000
N_VIEWS=3
RESOLUTION=8
TEST_ITERS="1000 2000 3000 5000 7000 10000"

LOGDIR="logs/ablation_track_b"
mkdir -p ${LOGDIR}

# ── Base flags (A1 = best config hiện tại) ──
BASE_FLAGS=(
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

# ── Reuse B0' từ ablation_track_a/A1_* (cùng config y hệt) ──
reuse_b0p() {
    local src dst
    for s in ${ALL_SCENES}; do
        src="logs/ablation_track_a/A1_${s}.log"
        dst="${LOGDIR}/B0p_${s}.log"
        if [ -f "$src" ] && [ ! -f "$dst" ]; then
            cp "$src" "$dst"
            echo "[REUSE] B0p_${s} ← ablation_track_a/A1_${s}.log"
        fi
    done
}

# run_cfg <tag> <scene> <extra_flags...>
run_cfg() {
    local tag=$1
    local scene=$2
    shift 2
    local extra_flags=("$@")
    local full_tag="${tag}_${scene}"
    local log="${LOGDIR}/${full_tag}.log"
    local out="output/ablation_track_b/${full_tag}"

    if [ -f "$log" ] && grep -q "\\[ITER 10000\\] Evaluating" "$log" 2>/dev/null; then
        echo "[SKIP] ${full_tag} — already done"
        return
    fi

    echo ""
    echo "=================================================================="
    echo " ${full_tag} | GPU=${GPU} | flags: ${extra_flags[*]}"
    echo "=================================================================="

    python -u train.py \
        --source_path ${DATA_ROOT}/${scene} \
        -m ${out} \
        "${BASE_FLAGS[@]}" \
        "${extra_flags[@]}" \
        2>&1 | tee ${log} || echo "[ERROR] ${full_tag} crashed"

    if grep -q "\\[ITER 10000\\] Evaluating" "$log" 2>/dev/null; then
        echo "[DONE] ${full_tag}"
    else
        echo "[FAIL] ${full_tag} — no ITER 10000 in log"
    fi
}

extract_psnr() {
    local log=$1; local iter=$2; local split=$3
    if [ ! -f "$log" ]; then echo ""; return; fi
    grep "\\[ITER ${iter}\\] Evaluating ${split}" "$log" 2>/dev/null \
      | tail -1 | sed 's/.*PSNR \([0-9.]*\).*/\1/' | cut -c1-7
}

# ── Main ──
reuse_b0p

if [ -n "${SCENES}" ]; then
    n_scenes=$(echo ${SCENES} | wc -w)
    total=$((n_scenes * 6))   # 6 configs mới × scenes (B0' reuse)
    echo "============================================"
    echo " Track B full: ${total} new runs (6 configs × ${n_scenes} scenes)"
    echo " GPU: ${GPU}, Scenes: ${SCENES}"
    echo "============================================"

    count=0
    for s in ${SCENES}; do
        # B1α — uniform from iter 0
        count=$((count + 1)); echo "[${count}/${total}]"
        run_cfg "B1a" "$s" \
            --use_dropout --dropout_mode uniform --dropout_base 0.2 --dropout_start_iter 0

        # B1β — uniform from iter 1000
        count=$((count + 1)); echo "[${count}/${total}]"
        run_cfg "B1b" "$s" \
            --use_dropout --dropout_mode uniform --dropout_base 0.2 --dropout_start_iter 1000

        # B3α — sh_norm from iter 0
        count=$((count + 1)); echo "[${count}/${total}]"
        run_cfg "B3a" "$s" \
            --use_dropout --dropout_mode sh_norm \
            --dropout_base 0.1 --dropout_w_sh 0.5 --dropout_start_iter 0

        # B3β — sh_norm from iter 1000
        count=$((count + 1)); echo "[${count}/${total}]"
        run_cfg "B3b" "$s" \
            --use_dropout --dropout_mode sh_norm \
            --dropout_base 0.1 --dropout_w_sh 0.5 --dropout_start_iter 1000

        # B4α — hybrid from iter 0
        count=$((count + 1)); echo "[${count}/${total}]"
        run_cfg "B4a" "$s" \
            --use_dropout --dropout_mode hybrid \
            --dropout_base 0.1 --dropout_w_crs 0.3 --dropout_w_sh 0.3 --dropout_start_iter 0

        # B4β — hybrid from iter 1000
        count=$((count + 1)); echo "[${count}/${total}]"
        run_cfg "B4b" "$s" \
            --use_dropout --dropout_mode hybrid \
            --dropout_base 0.1 --dropout_w_crs 0.3 --dropout_w_sh 0.3 --dropout_start_iter 1000
    done
fi

# ── Summary ──
export LC_ALL=C

echo ""
echo "============================================================================================================================="
echo "  SUMMARY — Track B: Dropout regularization (full 8 LLFF scenes) — Test PSNR @10k"
echo "============================================================================================================================="
printf "  %-5s" "Cfg"
for s in ${ALL_SCENES}; do printf " | %7s" "${s}"; done
printf " | %7s | %7s\n" "AVG" "Δ B0p"
printf "  %-5s" "-----"
for s in ${ALL_SCENES}; do printf -- "----------"; done
printf -- "----------\n"

declare -A B0P_VALS

for cfg in B0p B1a B1b B3a B3b B4a B4b; do
    printf "  %-5s" "${cfg}"
    sum="0"; n=0
    for s in ${ALL_SCENES}; do
        log="${LOGDIR}/${cfg}_${s}.log"
        val=$(extract_psnr "$log" 10000 test)
        if [ -n "$val" ]; then
            printf " | %7s" "${val}"
            sum=$(awk -v s="$sum" -v v="$val" 'BEGIN {printf "%.6f", s + v}')
            n=$((n + 1))
            if [ "$cfg" = "B0p" ]; then B0P_VALS[$s]="$val"; fi
        else
            printf " | %7s" "—"
        fi
    done

    if [ "$n" -gt 0 ]; then
        avg=$(awk -v s="$sum" -v n="$n" 'BEGIN {printf "%.3f", s / n}')
        printf " | %7s" "${avg}"
        if [ "$cfg" = "B0p" ]; then
            printf " | %7s\n" "—"
        else
            sum_d="0"; n_d=0
            for s in ${ALL_SCENES}; do
                log="${LOGDIR}/${cfg}_${s}.log"
                v=$(extract_psnr "$log" 10000 test)
                b="${B0P_VALS[$s]}"
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

# Gap avg (Train-Test @10k)
echo ""
echo "  Gap avg (Train@10k − Test@10k, 8 scenes):"
for cfg in B0p B1a B1b B3a B3b B4a B4b; do
    sum="0"; n=0
    for s in ${ALL_SCENES}; do
        log="${LOGDIR}/${cfg}_${s}.log"
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
        printf "    %-5s %s dB\n" "${cfg}" "${avg}"
    fi
done

# Wins vs B0p
echo ""
echo "  Wins vs B0p (out of 8 scenes):"
for cfg in B1a B1b B3a B3b B4a B4b; do
    wins=0
    for s in ${ALL_SCENES}; do
        log="${LOGDIR}/${cfg}_${s}.log"
        v=$(extract_psnr "$log" 10000 test)
        b="${B0P_VALS[$s]}"
        if [ -n "$v" ] && [ -n "$b" ]; then
            w=$(awk -v v="$v" -v b="$b" 'BEGIN {print (v > b) ? 1 : 0}')
            wins=$((wins + w))
        fi
    done
    printf "    %-5s %d/8\n" "${cfg}" "${wins}"
done

echo ""
echo "============================================================================================================================="
echo "  Legend:"
echo "    B0p  — A1 baseline (no dropout, sh_degree=1 + freeze_sh_after=1000)"
echo "    B1α  — uniform dropout (Co-Adapt reproduce) from iter 0"
echo "    B1β  — uniform dropout from iter 1000"
echo "    B3α  — SH-norm guided dropout from iter 0 (target rest overfit)"
echo "    B3β  — SH-norm guided dropout from iter 1000"
echo "    B4α  — Hybrid CRS+SH-norm dropout from iter 0"
echo "    B4β  — Hybrid CRS+SH-norm dropout from iter 1000"
echo "============================================================================================================================="
