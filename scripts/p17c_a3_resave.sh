#!/bin/bash
# ============================================================
# [CRSGaussian Phase 17c] A3 baseline RE-TRAIN với intermediate saves
# Cần cho Tier 2 trajectory pre-check (∇depth noise per iter).
#
# Lý do: default `--save_iterations=[10000,30000]` (train.py:1085) chỉ lưu
# iter 10000 → không có ckpt intermediate cho trajectory. Re-train với
# --save_iterations 500 2000 5000 7000 10000 để có 5 mốc đo noise.
#
# Mirror P8_CORE+A3_FLAGS EXACT từ p15_shape_pilot.sh:38-64 (= cùng A3
# baseline đã train Phase 13). KHÔNG thêm flag mới, KHÔNG đụng production.
#
# Output: output/p17c_a3_resave/A3_seed42_<scene>/  (NEW dir, KHÔNG ghi đè
# output/p13_lfcf/A3_seed42_<scene>/ — preserve evidence Tier 1 dùng).
#
# Note atomicAdd: re-train iter 10000 PSNR có thể chênh ±1.3 dB vs original
# (single-scene atomicAdd noise floor). Trajectory đo INTRA-run nên
# self-consistent. C1a Δ per-scene từ original logs/p17_c1/ (cross-tab
# informational only, không phải verdict pivot).
#
# Cost: ~30 min/scene × 4 scenes/GPU = ~2h wall-clock 2-GPU. Disk ~0.5-2 GB.
#
# Usage (2 GPU parallel — feedback_use_both_gpus):
#   mkdir -p logs/p17c_resave
#   GPU=0 SCENES_OVERRIDE="fern flower fortress horns" \
#       bash scripts/p17c_a3_resave.sh > logs/p17c_resave/gpu0.log 2>&1 &
#   GPU=1 SCENES_OVERRIDE="leaves orchids room trex" \
#       bash scripts/p17c_a3_resave.sh > logs/p17c_resave/gpu1.log 2>&1 &
#   wait
#
# Sau đó verify saves đã có:
#   for sc in fern flower fortress horns leaves orchids room trex; do
#     echo "$sc:" $(ls -d output/p17c_a3_resave/A3_seed42_$sc/point_cloud/iteration_* 2>/dev/null \
#                    | sed 's/.*iteration_//' | sort -n | tr '\n' ' ')
#   done
#
# Rồi chạy Tier 2:
#   OUTPUT_ROOT=output/p17c_a3_resave bash scripts/p17c_tier2_run.sh
# ============================================================
set -eo pipefail
GPU=${GPU:-0}
LOG_DIR=${LOG_DIR:-logs/p17c_resave}
OUT_DIR_VAL=${OUT_DIR:-output/p17c_a3_resave}
mkdir -p ${LOG_DIR} ${OUT_DIR_VAL}

# P8_CORE + A3_FLAGS = EXACT copy p15_shape_pilot.sh:38-64 (= CFG A3).
P8_CORE=(
    --eval -r 8 --n_views 3 --random_background
    --iterations 10000
    --densify_until_iter 5000 --densify_grad_threshold 0.0005
    --gaussiansN 1
    --use_depth_prior --dav2_path ../Depth-Anything-V2
    --use_crs_pruning --informed_crs_init
    --crs_init_use_view False
    --crs_init_w_reproj 0.4 --crs_init_w_depth 0.6 --crs_init_w_view 0.0
    --crs_ema_decay 0.3 --crs_update_interval 100
    --use_dropansh --dropansh_pa 0.02 --dropansh_psh 0.2
    --use_opacity_decay --opacity_decay_factor 0.999
    --sample_pseudo_interval 1 --start_sample_pseudo 500
    --test_iterations 10000
    --use_r_visible
    --use_d_cycle --d_cycle_warmup 1000 --d_cycle_sigma 5.0 --d_cycle_update_freq 100
    --use_crs_modulated_sh_freeze --crs_freeze_start 1000 --crs_freeze_tau 0.5
    --use_sh_reliability --sh_stability_warmup 1000
    --sh_stability_ema_beta 0.95 --crs_w_s 0.33
)
A3_FLAGS=(
    --use_lfcf
    --lfcf_init_scaling_max 1.5 --lfcf_init_scaling_min 1.0
    --lfcf_last_scaling_max 1.0 --lfcf_pow 1.0 --lfcf_splitting_ub 1.0
    --lfcf_interval_times 2 --lfcf_tolerance 1e-5 --lfcf_diffscale True
    --absdensify
)

# Pre-registered iter mocs (LOCKED with Tier 2 EARLY≤2500/LATE≥6000 cutoffs)
SAVE_ITERS=(500 2000 5000 7000 10000)

run_one() {
    local SC=$1; local SEED=$2
    local OUTDIR=${OUT_DIR_VAL}/A3_seed${SEED}_${SC}
    local LOG=${LOG_DIR}/A3_seed${SEED}_${SC}.log
    echo "[$(date +%H:%M:%S)] START A3-resave seed${SEED}_${SC} (GPU${GPU})"
    CUDA_VISIBLE_DEVICES=${GPU} python -u train.py \
        --source_path data/nerf_llff_data/${SC} \
        -m ${OUTDIR} --seed ${SEED} \
        --save_iterations "${SAVE_ITERS[@]}" \
        "${P8_CORE[@]}" "${A3_FLAGS[@]}" \
        2>&1 | tee ${LOG}
    echo "[$(date +%H:%M:%S)] DONE  A3-resave seed${SEED}_${SC}"
}

SCENES=(${SCENES_OVERRIDE:-fern flower fortress horns leaves orchids room trex})
SEEDS=(${SEEDS_OVERRIDE:-42})

echo "=== Phase 17c A3 RE-SAVE (GPU${GPU}) ==="
echo "Save iters: ${SAVE_ITERS[*]}"
echo "Output dir: ${OUT_DIR_VAL}"
for SEED in "${SEEDS[@]}"; do
    for S in "${SCENES[@]}"; do
        run_one $S $SEED
    done
done
echo "[GPU${GPU}] Phase 17c A3 re-save complete."
