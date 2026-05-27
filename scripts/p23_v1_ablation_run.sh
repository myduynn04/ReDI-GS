#!/bin/bash
# ============================================================
# [CRSGaussian Phase 23] Cross-backbone ablation on RoMa v1
# File: scripts/p23_v1_ablation_run.sh  (KEEP LOCAL — server-only)
#
# Mục tiêu: identify cross-backbone-stable contribution trên v1 backbone.
# 13 configs:
#   base + trim_full (reuse Phase 22) + 6 LOO + 6 single-add
#
# Pre-requisite (server):
#   1. conda activate corgs
#   2. cd ~/workspace/representation-3d/duyen/CoR-GS
#   3. fused.ply per scene PHẢI = RoMa v1 (chạy p22_place_romav1_init.py trước)
#   4. chmod +x scripts/p23_v1_ablation_run.sh
#
# Usage (loop tất cả 12 configs mới — bỏ trim_full vì reuse Phase 22):
#   ./scripts/p23_v1_ablation_run.sh
#
# Single config (debug):
#   GPU=0 CONFIG=single_efa SCENES_OVERRIDE='fern' SEEDS_OVERRIDE='42' \
#       LOG_DIR=logs/p23_v1_ablation/single_efa \
#       OUT_DIR=output/p23_v1_ablation/single_efa \
#       bash scripts/p20_ablation_dense_run.sh  # reuse runner
#
# Output:
#   logs/p23_v1_ablation/<config>/A3_seed42_<scene>.log (12 × 8 = 96 files)
#   output/p23_v1_ablation/<config>/A3_seed42_<scene>/
#
# Cost: 96 runs × 5min = ~8h 1-GPU, ~4h 2-GPU split
# ============================================================

set -e

# ── Safety: verify fused.ply = RoMa v1 ──
FERN_PLY="data/nerf_llff_data/fern/3_views/dense/fused.ply"
FERN_V1="data/nerf_llff_data/fern/3_views/dense/fused.ply.romav1"
if [ ! -f "$FERN_V1" ]; then
    echo "❌ fused.ply.romav1 chưa được sinh — chạy p22_run_all_scenes.sh trước"
    exit 1
fi
if ! cmp -s "$FERN_PLY" "$FERN_V1"; then
    echo "❌ fused.ply ≠ fused.ply.romav1 — chạy p22_place_romav1_init.py trước"
    exit 1
fi
echo "✓ verified fused.ply = RoMa v1"

# ── Module flag-strings (exact copy from p20_ablation_dense_run.sh) ──
PROTOCOL="--eval -r 8 --n_views 3 --random_background --iterations 10000 \
--densify_until_iter 5000 --densify_grad_threshold 0.0005 --gaussiansN 1 \
--sample_pseudo_interval 1 --start_sample_pseudo 500 --test_iterations 10000"

M_DEPTHCFG="--use_depth_prior --dav2_path ../Depth-Anything-V2 --crs_ema_decay 0.3 --crs_update_interval 100"
M_DCYCLE="--use_d_cycle --d_cycle_warmup 1000 --d_cycle_sigma 5.0 --d_cycle_update_freq 100"
M_SHFREEZE="--use_crs_modulated_sh_freeze --crs_freeze_start 1000 --crs_freeze_tau 0.5"
M_SHREL="--use_sh_reliability --sh_stability_warmup 1000 --sh_stability_ema_beta 0.95 --crs_w_s 0.33"
M_DROP="--use_dropansh --dropansh_pa 0.02 --dropansh_psh 0.2"
M_OPACITY="--use_opacity_decay --opacity_decay_factor 0.999"
M_EFA="--use_lfcf --lfcf_init_scaling_max 1.5 --lfcf_init_scaling_min 1.0 \
--lfcf_last_scaling_max 1.0 --lfcf_pow 1.0 --lfcf_splitting_ub 1.0 \
--lfcf_interval_times 2 --lfcf_tolerance 1e-5 --lfcf_diffscale True --absdensify"

CRS_TRIM="${M_DEPTHCFG} ${M_DCYCLE} ${M_SHFREEZE} ${M_SHREL}"
INDEP_FULL="${M_DROP} ${M_OPACITY} ${M_EFA}"

# Build flags per config
build_flags() {
    case "$1" in
        # Reference (skip reuse Phase 22)
        base)              echo "" ;;
        # LOO: trim_full minus 1 bucket
        trim_no_efa)       echo "${CRS_TRIM} ${M_DROP} ${M_OPACITY}" ;;
        trim_no_drop)      echo "${CRS_TRIM} ${M_OPACITY} ${M_EFA}" ;;
        trim_no_opacity)   echo "${CRS_TRIM} ${M_DROP} ${M_EFA}" ;;
        trim_no_dcycle)    echo "${M_DEPTHCFG} ${M_SHFREEZE} ${M_SHREL} ${INDEP_FULL}" ;;
        trim_no_shcrs)     echo "${M_DEPTHCFG} ${M_DCYCLE} ${INDEP_FULL}" ;;
        trim_no_depthcrs)  echo "${INDEP_FULL}" ;;
        # Single-add: base + 1 bucket
        single_efa)        echo "${M_EFA}" ;;
        single_drop)       echo "${M_DROP}" ;;
        single_opacity)    echo "${M_OPACITY}" ;;
        single_dcycle)     echo "${M_DEPTHCFG} ${M_DCYCLE}" ;;
        single_shcrs)      echo "${M_DEPTHCFG} ${M_SHFREEZE} ${M_SHREL}" ;;
        single_depthcrs)   echo "${CRS_TRIM}" ;;
        *)                 echo "ERROR_UNKNOWN_CONFIG"; return 1 ;;
    esac
}

# 12 configs (skip trim_full reuse)
CONFIGS_GPU0=("base" "trim_no_efa" "trim_no_drop" "trim_no_opacity" "trim_no_dcycle" "trim_no_shcrs")
CONFIGS_GPU1=("trim_no_depthcrs" "single_efa" "single_drop" "single_opacity" "single_dcycle" "single_shcrs" "single_depthcrs")

SCENES=(fern flower fortress horns leaves orchids room trex)
SEED=42

GPU0_ID=${GPU0_ID:-0}
GPU1_ID=${GPU1_ID:-1}

mkdir -p logs/p23_v1_ablation output/p23_v1_ablation
MASTER_LOG=logs/p23_v1_ablation/_master.log

run_one_config_gpu() {
    local GPU_ID=$1; local CONFIG=$2
    local FLAGS=$(build_flags "$CONFIG")
    local LOG_DIR="logs/p23_v1_ablation/${CONFIG}"
    local OUT_DIR_VAL="output/p23_v1_ablation/${CONFIG}"
    mkdir -p "$LOG_DIR" "$OUT_DIR_VAL"

    for SC in "${SCENES[@]}"; do
        local OUTDIR=${OUT_DIR_VAL}/A3_seed${SEED}_${SC}
        local LOG=${LOG_DIR}/A3_seed${SEED}_${SC}.log

        # SKIP-IF-EXISTS: nếu log đã có "Best test PSNR" → đã chạy xong, skip
        if [ -f "$LOG" ] && grep -q "Best test PSNR" "$LOG" 2>/dev/null; then
            echo "[$(date +%H:%M:%S)] SKIP  [P23 ${CONFIG}] seed${SEED}_${SC} (already done)"
            continue
        fi

        echo "[$(date +%H:%M:%S)] START [P23 ${CONFIG}] seed${SEED}_${SC} (GPU${GPU_ID})"
        CUDA_VISIBLE_DEVICES=${GPU_ID} python -u train.py \
            --source_path data/nerf_llff_data/${SC} \
            -m ${OUTDIR} --seed ${SEED} \
            ${PROTOCOL} ${FLAGS} \
            2>&1 | tee ${LOG} > /dev/null
        echo "[$(date +%H:%M:%S)] DONE  [P23 ${CONFIG}] seed${SEED}_${SC}"
    done
}

echo "[p23 ablation] start $(date '+%H:%M:%S')" | tee "$MASTER_LOG"
echo "  GPU $GPU0_ID: ${CONFIGS_GPU0[*]}" | tee -a "$MASTER_LOG"
echo "  GPU $GPU1_ID: ${CONFIGS_GPU1[*]}" | tee -a "$MASTER_LOG"

# Background loops per GPU
(
    for CFG in "${CONFIGS_GPU0[@]}"; do
        run_one_config_gpu $GPU0_ID "$CFG"
    done
) > logs/p23_v1_ablation/_gpu0.log 2>&1 &
PID0=$!

(
    for CFG in "${CONFIGS_GPU1[@]}"; do
        run_one_config_gpu $GPU1_ID "$CFG"
    done
) > logs/p23_v1_ablation/_gpu1.log 2>&1 &
PID1=$!

echo "  PID GPU$GPU0_ID=$PID0  PID GPU$GPU1_ID=$PID1" | tee -a "$MASTER_LOG"
echo "  monitor: tail -f logs/p23_v1_ablation/_gpu0.log logs/p23_v1_ablation/_gpu1.log"
echo "  count:   watch -n 30 'ls logs/p23_v1_ablation/*/A3_seed42_*.log 2>/dev/null | wc -l'  # target 96"

wait $PID0 $PID1
RC0=$?; RC1=$?

echo "[p23 ablation] done $(date '+%H:%M:%S')  RC GPU$GPU0_ID=$RC0  GPU$GPU1_ID=$RC1" | tee -a "$MASTER_LOG"

N_LOGS=$(ls logs/p23_v1_ablation/*/A3_seed42_*.log 2>/dev/null | wc -l)
echo "  produced $N_LOGS / 96 log files" | tee -a "$MASTER_LOG"

if [ "$N_LOGS" -ne 96 ]; then
    echo "  ⚠ thiếu run — check logs/p23_v1_ablation/_gpu*.log"
fi

echo ""
echo "next: python scripts/p23_v1_ablation_analyze.py"

[ $RC0 -ne 0 ] || [ $RC1 -ne 0 ] && exit 1 || exit 0
