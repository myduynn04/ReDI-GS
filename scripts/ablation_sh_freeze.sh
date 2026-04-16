#!/bin/bash
# ============================================================
# [CRSGaussian Ablation] SH Freeze — FULL 8 LLFF scenes
# File: scripts/ablation_sh_freeze.sh
#
# Mục đích: Verify freeze_sh_after=1000 (đã +0.60 dB trên fern từ E1b)
#   có generalize sang 7 scenes còn lại không.
#
# Configs (2 × 8 scenes = 16 runs):
#   B0   — baseline (freeze_sh_after=0, best WG config hiện tại)
#   SH1k — freeze_sh_after=1000
#
# USAGE (parallel 2 GPUs):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/ablation_sh_freeze.sh A 0 &
#   CUDA_VISIBLE_DEVICES=1 bash scripts/ablation_sh_freeze.sh B 1 &
#   wait
#
# Single scene:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/ablation_sh_freeze.sh fern 0
#
# Summary only (sau khi tất cả xong):
#   bash scripts/ablation_sh_freeze.sh summary
#
# Fern đã có từ E1 + E1b → script tự reuse log cũ (không re-run).
# ============================================================

# Không dùng `set -e` — 1 run fail không kill toàn batch
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

LOGDIR="logs/ablation_sh_freeze"
mkdir -p ${LOGDIR}

# ── Configs: name|freeze_sh_after ──
CONFIG_NAMES=("B0" "SH1k")
CONFIG_FREEZE=(0 1000)

# ── Reuse fern logs từ E1/E1b (không re-run fern) ──
# Gọi 1 lần ở đầu script — symlink nếu chưa có.
reuse_fern_logs() {
    local src dst
    # B0_fern ← E1_baseline_fern
    src="logs/diag_e1_sh/E1_baseline_fern.log"
    dst="${LOGDIR}/B0_fern.log"
    if [ -f "$src" ] && [ ! -f "$dst" ]; then
        cp "$src" "$dst"
        echo "[REUSE] B0_fern ← ${src}"
    fi
    # SH1k_fern ← E1b_freeze1k_fern
    src="logs/diag_e1b/E1b_freeze1k_fern.log"
    dst="${LOGDIR}/SH1k_fern.log"
    if [ -f "$src" ] && [ ! -f "$dst" ]; then
        cp "$src" "$dst"
        echo "[REUSE] SH1k_fern ← ${src}"
    fi
}

run_one() {
    local cfg_name=$1
    local freeze_after=$2
    local scene=$3

    local tag="${cfg_name}_${scene}"
    local log="${LOGDIR}/${tag}.log"
    local out="output/ablation_sh_freeze/${tag}"

    if [ -f "$log" ] && grep -q "\\[ITER 10000\\] Evaluating" "$log" 2>/dev/null; then
        echo "[SKIP] ${tag} — already done"
        return
    fi

    echo ""
    echo "========================================"
    echo " ${cfg_name} | scene=${scene} | freeze_sh_after=${freeze_after}"
    echo " GPU=${GPU}"
    echo "========================================"

    python -u train.py \
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

extract_psnr() {
    local log=$1
    local iter=$2
    local split=$3
    if [ ! -f "$log" ]; then echo ""; return; fi
    local val
    val=$(grep "\\[ITER ${iter}\\] Evaluating ${split}" "$log" 2>/dev/null | tail -1 | sed 's/.*PSNR \([0-9.]*\).*/\1/' | cut -c1-7)
    echo "$val"
}

# ── Main loop ──
reuse_fern_logs

if [ -n "${SCENES}" ]; then
    total=$(( ${#CONFIG_NAMES[@]} * $(echo ${SCENES} | wc -w) ))
    echo "============================================"
    echo " SH Freeze Ablation: ${total} runs"
    echo " GPU: ${GPU}, Scenes: ${SCENES}"
    echo "============================================"

    count=0
    for i in "${!CONFIG_NAMES[@]}"; do
        c_name="${CONFIG_NAMES[$i]}"
        c_freeze="${CONFIG_FREEZE[$i]}"
        for s in ${SCENES}; do
            count=$((count + 1))
            echo "[${count}/${total}]"
            run_one "$c_name" "$c_freeze" "$s"
        done
    done
fi

# ── Summary ──
echo ""
echo "========================================================================================="
echo "  SUMMARY — SH Freeze Ablation — Test PSNR @10k"
echo "========================================================================================="

# Header
printf "  %-6s" "Cfg"
for s in ${ALL_SCENES}; do
    printf " | %7s" "${s}"
done
printf " | %7s\n" "AVG"
printf "  ------"
for s in ${ALL_SCENES}; do
    printf -- "----------"
done
printf -- "----------\n"

# Store B0 values for delta computation
declare -A B0_VALS

for i in "${!CONFIG_NAMES[@]}"; do
    c_name="${CONFIG_NAMES[$i]}"
    printf "  %-6s" "${c_name}"

    sum="0"
    n=0
    for s in ${ALL_SCENES}; do
        log="${LOGDIR}/${c_name}_${s}.log"
        val=$(extract_psnr "$log" 10000 test)
        if [ -n "$val" ]; then
            printf " | %7s" "${val}"
            sum=$(awk -v s="$sum" -v v="$val" 'BEGIN {printf "%.6f", s + v}')
            n=$((n + 1))
            if [ "$c_name" = "B0" ]; then
                B0_VALS[$s]="$val"
            fi
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

# Delta row
printf "  %-6s" "Δ"
sum_d="0"
n_d=0
for s in ${ALL_SCENES}; do
    log_b0="${LOGDIR}/B0_${s}.log"
    log_sh="${LOGDIR}/SH1k_${s}.log"
    v_b0=$(extract_psnr "$log_b0" 10000 test)
    v_sh=$(extract_psnr "$log_sh" 10000 test)
    if [ -n "$v_b0" ] && [ -n "$v_sh" ]; then
        delta=$(awk -v a="$v_sh" -v b="$v_b0" 'BEGIN {printf "%+.3f", a - b}')
        printf " | %7s" "${delta}"
        sum_d=$(awk -v s="$sum_d" -v v="$v_sh" -v b="$v_b0" 'BEGIN {printf "%.6f", s + (v - b)}')
        n_d=$((n_d + 1))
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

echo ""
echo "========================================================================================="
echo "  Legend:"
echo "    B0   — baseline (freeze_sh_after=0, best WG config)"
echo "    SH1k — freeze_sh_after=1000"
echo "    Δ    — SH1k - B0 (positive = SH freeze có lợi)"
echo ""
echo "  Ref (fern E1): B0=21.99, SH1k=22.59, Δ=+0.60"
echo "========================================================================================="
