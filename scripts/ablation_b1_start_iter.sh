#!/bin/bash
# ============================================================
# [CRSGaussian B1-timing] Ablation dropout_start_iter cho B1 uniform
# File: scripts/ablation_b1_start_iter.sh
#
# Mục đích: Fine-tune dropout_start_iter quanh giá trị 1000 (current best).
#   Track B đã confirm start=1000 > start=0 (+0.16 dB AVG). Nhưng chưa
#   sweep dense → có thể có điểm tốt hơn (500, 1500, 2000).
#
# 5 configs × 8 scenes = 40 runs, REUSE 16 → 24 runs mới.
#   B1_s0      — reuse từ ablation_track_b/B1a
#   B1_s500    — NEW (sau warmup densify bắt đầu ~500)
#   B1_s1000   — reuse từ ablation_track_b/B1b
#   B1_s1500   — NEW (sau CRS stable)
#   B1_s2000   — NEW (trễ hơn, sau densify gần ổn)
#
# USAGE:
#   Dry run:
#     bash scripts/ablation_b1_start_iter.sh dry
#
#   Parallel 2 GPUs (~1.5h wall):
#     CUDA_VISIBLE_DEVICES=0 bash scripts/ablation_b1_start_iter.sh A 0 \
#         2>&1 | tee logs/ablation_b1_start/_wrapperA.log &
#     CUDA_VISIBLE_DEVICES=1 bash scripts/ablation_b1_start_iter.sh B 1 \
#         2>&1 | tee logs/ablation_b1_start/_wrapperB.log &
#     wait
#     bash scripts/ablation_b1_start_iter.sh summary
# ============================================================

SCENE_ARG=${1:-ALL}
GPU=${2:-0}

ALL_SCENES="fern flower fortress horns leaves orchids room trex"
SCENES_A="fern flower fortress horns"
SCENES_B="leaves orchids room trex"
NEW_STARTS="500 1500 2000"   # 3 configs mới cần chạy

case ${SCENE_ARG} in
    ALL|all)         SCENES="${ALL_SCENES}" ;;
    A|a)             SCENES="${SCENES_A}" ;;
    B|b)             SCENES="${SCENES_B}" ;;
    summary|SUMMARY) SCENES="" ;;
    dry|DRY)         SCENES="DRYRUN" ;;
    *)               SCENES="${SCENE_ARG}" ;;
esac

DATA_ROOT="data/nerf_llff_data"
DAV2_PATH="../Depth-Anything-V2"
ITERATIONS=10000
N_VIEWS=3
RESOLUTION=8
TEST_ITERS="1000 2000 3000 5000 7000 10000"

LOGDIR="logs/ablation_b1_start"
REUSE_SRC_DIR="logs/ablation_track_b"
mkdir -p ${LOGDIR}

# ── Reuse logic: verify + copy ──
verify_reuse_sources() {
    local missing=0
    for s in ${ALL_SCENES}; do
        for src_tag in B1a B1b; do
            local src="${REUSE_SRC_DIR}/${src_tag}_${s}.log"
            if [ ! -f "$src" ]; then
                echo "[ERROR] Missing reuse source: $src"
                missing=$((missing + 1))
            fi
        done
    done
    if [ "$missing" -gt 0 ]; then
        echo ""
        echo "Cannot continue. ${missing} source files missing."
        echo "Run Track B first (scripts/ablation_track_b.sh) or fix REUSE_SRC_DIR."
        return 1
    fi
    return 0
}

reuse_logs() {
    for s in ${ALL_SCENES}; do
        # B1a → B1_s0
        local src_a="${REUSE_SRC_DIR}/B1a_${s}.log"
        local dst_a="${LOGDIR}/B1_s0_${s}.log"
        if [ -f "$src_a" ] && [ ! -f "$dst_a" ]; then
            cp "$src_a" "$dst_a"
            echo "[REUSE] B1_s0_${s} ← ablation_track_b/B1a_${s}.log"
        fi
        # B1b → B1_s1000
        local src_b="${REUSE_SRC_DIR}/B1b_${s}.log"
        local dst_b="${LOGDIR}/B1_s1000_${s}.log"
        if [ -f "$src_b" ] && [ ! -f "$dst_b" ]; then
            cp "$src_b" "$dst_b"
            echo "[REUSE] B1_s1000_${s} ← ablation_track_b/B1b_${s}.log"
        fi
    done
}

# ── Base flags (match Track B / A1 recipe) ──
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
    --use_dropout
    --dropout_mode uniform
    --dropout_base 0.2
)

# run_one <start_iter> <scene>
run_one() {
    local start=$1
    local scene=$2
    local tag="B1_s${start}_${scene}"
    local log="${LOGDIR}/${tag}.log"
    local out="output/ablation_b1_start/B1_s${start}_${scene}"

    if [ -f "$log" ] && grep -q "\\[ITER 10000\\] Evaluating" "$log" 2>/dev/null; then
        echo "[SKIP] ${tag} — already done"
        return
    fi

    echo ""
    echo "========================================================"
    echo " ${tag} | start_iter=${start} | GPU=${GPU}"
    echo "========================================================"

    python -u train.py \
        --source_path ${DATA_ROOT}/${scene} \
        -m ${out} \
        "${BASE_FLAGS[@]}" \
        --dropout_start_iter ${start} \
        2>&1 | tee ${log} || echo "[ERROR] ${tag} crashed"

    if grep -q "\\[ITER 10000\\] Evaluating" "$log" 2>/dev/null; then
        echo "[DONE] ${tag}"
    else
        echo "[FAIL] ${tag} — no ITER 10000 in log"
    fi
}

extract_psnr() {
    local log=$1; local iter=$2; local split=$3
    [ -f "$log" ] || { echo ""; return; }
    grep "\\[ITER ${iter}\\] Evaluating ${split}" "$log" 2>/dev/null \
      | tail -1 | sed 's/.*PSNR \([0-9.]*\).*/\1/' | cut -c1-7
}

# Tìm peak test PSNR + iter across all eval points.
# Print: "peak_psnr peak_iter"
extract_peak() {
    local log=$1
    [ -f "$log" ] || { echo "0 0"; return; }
    awk '
      /\[ITER [0-9]+\] Evaluating test/ {
        match($0, /\[ITER ([0-9]+)\]/, a); iter = a[1];
        match($0, /PSNR ([0-9.]+)/, b); psnr = b[1] + 0;
        if (psnr > best) { best = psnr; best_iter = iter; }
      }
      END { if (best > 0) printf "%.3f %s", best, best_iter; else print "0 0"; }
    ' "$log"
}

# ── DRY RUN mode ──
if [ "${SCENES}" = "DRYRUN" ]; then
    echo ""
    echo "=============================================================="
    echo "  DRY RUN — Ablation B1 start_iter"
    echo "=============================================================="

    echo ""
    echo "▶ Verifying reuse sources in ${REUSE_SRC_DIR}/ ..."
    if verify_reuse_sources; then
        echo "  [OK] All 16 reuse sources present (B1a/B1b × 8 scenes)"
    else
        echo "  [FAIL] Reuse sources missing → cannot summary B1_s0 / B1_s1000"
        exit 1
    fi

    echo ""
    echo "▶ Configs to run:"
    new_total=0
    for start in ${NEW_STARTS}; do
        for s in ${ALL_SCENES}; do
            log="${LOGDIR}/B1_s${start}_${s}.log"
            if [ -f "$log" ] && grep -q "\\[ITER 10000\\] Evaluating" "$log" 2>/dev/null; then
                status="SKIP (done)"
            else
                status="NEW"
                new_total=$((new_total + 1))
            fi
            printf "    B1_s%-5s %-10s %s\n" "${start}" "${s}" "${status}"
        done
    done

    echo ""
    echo "▶ Summary:"
    echo "    Total runs planned : 24 (3 configs × 8 scenes)"
    echo "    Already done       : $((24 - new_total))"
    echo "    NEW runs needed    : ${new_total}"
    echo "    Est. time per run  : ~7 min (10k iter on A1 config)"
    echo "    Est. wall time     : $(( new_total * 7 / 2 )) min (parallel 2 GPUs)"
    echo "                         $(( new_total * 7 ))     min (1 GPU sequential)"
    echo ""
    echo "▶ Disk estimate:"
    echo "    output/ablation_b1_start/ : ~${new_total}00 MB (checkpoints + renders)"
    echo "    logs/ablation_b1_start/   : ~$((24 * 500 / 1024)) MB"
    echo ""
    echo "Run this to start:"
    echo "  CUDA_VISIBLE_DEVICES=0 bash scripts/ablation_b1_start_iter.sh A 0 &"
    echo "  CUDA_VISIBLE_DEVICES=1 bash scripts/ablation_b1_start_iter.sh B 1 &"
    echo "  wait"
    echo "  bash scripts/ablation_b1_start_iter.sh summary"
    exit 0
fi

# ── Main ──
# Verify reuse trước mọi thứ (fail-fast)
if ! verify_reuse_sources; then
    exit 1
fi
reuse_logs

if [ -n "${SCENES}" ]; then
    n_scenes=$(echo ${SCENES} | wc -w)
    total=$((n_scenes * 3))
    echo "========================================================"
    echo " B1 start_iter sweep: ${total} new runs (3 configs × ${n_scenes} scenes)"
    echo " GPU: ${GPU}, Scenes: ${SCENES}"
    echo "========================================================"

    count=0
    for s in ${SCENES}; do
        for start in ${NEW_STARTS}; do
            count=$((count + 1)); echo "[${count}/${total}]"
            run_one "${start}" "$s"
        done
    done
fi

# ── Summary ──
export LC_ALL=C

ALL_STARTS="0 500 1000 1500 2000"

echo ""
echo "========================================================================================================================================"
echo "  SUMMARY — B1 uniform dropout × start_iter sweep (8 LLFF scenes)"
echo "========================================================================================================================================"
echo ""
echo "── Bảng 1: Test PSNR @10k ──"
printf "  %-10s" "Cfg"
for s in ${ALL_SCENES}; do printf " | %7s" "${s}"; done
printf " | %7s | %8s\n" "AVG" "Δ s1000"
printf "  %-10s" "----------"
for s in ${ALL_SCENES}; do printf -- "----------"; done
printf -- "-----------\n"

declare -A REF_VALS  # B1_s1000 values for delta reference

for start in ${ALL_STARTS}; do
    cfg="B1_s${start}"
    printf "  %-10s" "${cfg}"
    sum="0"; n=0
    for s in ${ALL_SCENES}; do
        log="${LOGDIR}/${cfg}_${s}.log"
        val=$(extract_psnr "$log" 10000 test)
        if [ -n "$val" ]; then
            printf " | %7s" "${val}"
            sum=$(awk -v s="$sum" -v v="$val" 'BEGIN {printf "%.6f", s + v}')
            n=$((n + 1))
            if [ "${start}" = "1000" ]; then REF_VALS[$s]="$val"; fi
        else
            printf " | %7s" "—"
        fi
    done
    if [ "$n" -gt 0 ]; then
        avg=$(awk -v s="$sum" -v n="$n" 'BEGIN {printf "%.3f", s / n}')
        printf " | %7s" "${avg}"
        if [ "${start}" = "1000" ]; then
            printf " | %8s\n" "—"
        else
            sum_d="0"; n_d=0
            for s in ${ALL_SCENES}; do
                log="${LOGDIR}/${cfg}_${s}.log"
                v=$(extract_psnr "$log" 10000 test)
                b="${REF_VALS[$s]}"
                if [ -n "$v" ] && [ -n "$b" ]; then
                    sum_d=$(awk -v s="$sum_d" -v v="$v" -v b="$b" 'BEGIN {printf "%.6f", s + (v - b)}')
                    n_d=$((n_d + 1))
                fi
            done
            if [ "$n_d" -gt 0 ]; then
                avg_d=$(awk -v s="$sum_d" -v n="$n_d" 'BEGIN {printf "%+.3f", s / n_d}')
                printf " | %8s\n" "${avg_d}"
            else
                printf " | %8s\n" "—"
            fi
        fi
    else
        printf " | %7s | %8s\n" "—" "—"
    fi
done

# ── Gap + Wins ──
echo ""
echo "── Bảng 2: Gap + Wins vs B1_s1000 ──"
printf "  %-10s | %-8s | %-8s\n" "Cfg" "Gap avg" "Wins vs s1000"
printf "  %-10s-+-%-8s-+-%-14s\n" "----------" "--------" "--------------"
for start in ${ALL_STARTS}; do
    cfg="B1_s${start}"
    # Gap
    sum_g="0"; n_g=0
    for s in ${ALL_SCENES}; do
        log="${LOGDIR}/${cfg}_${s}.log"
        te=$(extract_psnr "$log" 10000 test)
        tr=$(extract_psnr "$log" 10000 train)
        if [ -n "$te" ] && [ -n "$tr" ]; then
            g=$(awk -v tr="$tr" -v te="$te" 'BEGIN {printf "%.3f", tr - te}')
            sum_g=$(awk -v s="$sum_g" -v g="$g" 'BEGIN {printf "%.6f", s + g}')
            n_g=$((n_g + 1))
        fi
    done
    gap_avg="—"
    if [ "$n_g" -gt 0 ]; then
        gap_avg=$(awk -v s="$sum_g" -v n="$n_g" 'BEGIN {printf "%.2f", s / n}')
    fi
    # Wins vs s1000
    if [ "${start}" = "1000" ]; then
        wins_str="—"
    else
        wins=0
        for s in ${ALL_SCENES}; do
            log="${LOGDIR}/${cfg}_${s}.log"
            v=$(extract_psnr "$log" 10000 test)
            b="${REF_VALS[$s]}"
            if [ -n "$v" ] && [ -n "$b" ]; then
                w=$(awk -v v="$v" -v b="$b" 'BEGIN {print (v > b) ? 1 : 0}')
                wins=$((wins + w))
            fi
        done
        wins_str="${wins}/8"
    fi
    printf "  %-10s | %-8s | %-14s\n" "${cfg}" "${gap_avg}" "${wins_str}"
done

# ── Diagnostic bonus: Peak + curve snapshots + decay ──
echo ""
echo "── Bảng 3: Peak test PSNR + iter (diagnostic — check H1: peak shift theo start_iter?) ──"
printf "  %-10s" "Cfg"
for s in ${ALL_SCENES}; do printf " | %9s" "${s}"; done
printf "\n"
printf "  %-10s" "----------"
for s in ${ALL_SCENES}; do printf -- "------------"; done
printf "\n"
for start in ${ALL_STARTS}; do
    cfg="B1_s${start}"
    printf "  %-10s" "${cfg}"
    for s in ${ALL_SCENES}; do
        log="${LOGDIR}/${cfg}_${s}.log"
        peak=$(extract_peak "$log")
        pk_psnr=$(echo "$peak" | awk '{print $1}')
        pk_iter=$(echo "$peak" | awk '{print $2}')
        if [ "${pk_psnr}" = "0" ]; then
            printf " | %9s" "—"
        else
            printf " | %4s@%-4s" "${pk_psnr}" "${pk_iter}"
        fi
    done
    printf "\n"
done

echo ""
echo "── Bảng 4: Test PSNR curve snapshots (AVG 8 scenes, check H2: decay magnitude) ──"
printf "  %-10s | %-7s | %-7s | %-7s | %-7s | %-7s | %-8s\n" \
    "Cfg" "@1k" "@3k" "@5k" "@7k" "@10k" "Decay"
printf "  %-10s-+-%-7s-+-%-7s-+-%-7s-+-%-7s-+-%-7s-+-%-8s\n" \
    "----------" "-------" "-------" "-------" "-------" "-------" "--------"
for start in ${ALL_STARTS}; do
    cfg="B1_s${start}"
    row="${cfg}"
    peak_sum="0"; peak_n=0
    ten_sum="0"; ten_n=0
    printf "  %-10s" "${cfg}"
    for iter in 1000 3000 5000 7000 10000; do
        sum="0"; n=0
        for s in ${ALL_SCENES}; do
            log="${LOGDIR}/${cfg}_${s}.log"
            v=$(extract_psnr "$log" "${iter}" test)
            if [ -n "$v" ]; then
                sum=$(awk -v s="$sum" -v v="$v" 'BEGIN {printf "%.6f", s + v}')
                n=$((n + 1))
            fi
        done
        if [ "$n" -gt 0 ]; then
            avg=$(awk -v s="$sum" -v n="$n" 'BEGIN {printf "%.3f", s / n}')
            printf " | %7s" "${avg}"
        else
            printf " | %7s" "—"
        fi
    done
    # Decay = peak_AVG - @10k_AVG
    peak_avg="0"; ten_avg="0"; dn=0
    for s in ${ALL_SCENES}; do
        log="${LOGDIR}/${cfg}_${s}.log"
        peak=$(extract_peak "$log")
        pk=$(echo "$peak" | awk '{print $1}')
        ten=$(extract_psnr "$log" 10000 test)
        if [ -n "$ten" ] && [ "${pk}" != "0" ]; then
            peak_avg=$(awk -v s="$peak_avg" -v v="$pk" 'BEGIN {printf "%.6f", s + v}')
            ten_avg=$(awk -v s="$ten_avg" -v v="$ten" 'BEGIN {printf "%.6f", s + v}')
            dn=$((dn + 1))
        fi
    done
    if [ "$dn" -gt 0 ]; then
        decay=$(awk -v p="$peak_avg" -v t="$ten_avg" -v n="$dn" \
            'BEGIN {printf "%+.3f", -(p - t) / n}')
        printf " | %8s\n" "${decay}"
    else
        printf " | %8s\n" "—"
    fi
done

echo ""
echo "========================================================================================================================================"
echo "  Interpret:"
echo "    H1 (peak shift): check Bảng 3 — nếu peak iter tăng khi start_iter tăng → dropout trì hoãn overfit"
echo "    H2 (decay): check Bảng 4 Decay column — negative=decay. Nhỏ hơn (gần 0) = dropout chặn overfit cuối"
echo "    Config tốt: AVG cao + Gap nhỏ + Wins ≥ 5/8 + Decay gần 0"
echo "========================================================================================================================================"
