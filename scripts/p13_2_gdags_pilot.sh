#!/bin/bash
# ============================================================
# [CRSGaussian Phase 13.2.5] GDAGS A/B pilot.
#
# A/B 1 biến duy nhất = --use_gdags:
#   Arm A (baseline) = A3 = LFCF + AbsGS-OR  → REUSE logs/p13_lfcf/A3_seed42_*
#                       (KHÔNG re-run, paired vs existing như HF-pilot)
#   Arm B (GDAGS)    = A3 + --use_gdags → standard-path AbsGS-OR thay bằng
#                       GDAGS coherence-weight (LFCF path KHÔNG đụng)
#
# Full 8 scenes, seed 42 = 8 runs ~30-40 phút (2-GPU split 4+4).
# (GDAGS expected modest → 3-scene single-seed rơi trong noise; full-8
#  paired là verdict đầu đúng — feedback_full_8scene_ablation.)
#
# Caveat (decisions_log [2026-05-17]): GDAGS GCR=grads/grads_abs KHÔNG
# orthogonal với AbsGS → policy A/B trên trục proven, KHÔNG +feature.
# Kỳ vọng modest, có thể ≈/< A3. Verdict bằng paired Δ multi-seed nếu
# signal > noise (project_3dgs_variance_floor: ±0.10 floor).
#
# ── VERIFY flag-OFF = A3 byte-identical (BẮT BUỘC trước khi tin pilot) ──
#   Chạy 1 scene với code MỚI nhưng KHÔNG --use_gdags → phải khớp
#   logs/p13_lfcf/A3_seed42_orchids (PSNR identical ± atomicAdd noise):
#     CUDA_VISIBLE_DEVICES=0 SCENES_OVERRIDE="orchids" GDAGS_OFF=1 \
#       bash scripts/p13_2_gdags_pilot.sh
#   So PSNR cuối vs logs/p13_lfcf/A3_seed42_orchids.log
#
# Usage (2 GPU parallel):
#   GPU=0 SCENES_OVERRIDE="trex horns" bash scripts/p13_2_gdags_pilot.sh \
#       > logs/p13_2_gdags/gpu0.log 2>&1 &
#   GPU=1 SCENES_OVERRIDE="orchids"    bash scripts/p13_2_gdags_pilot.sh \
#       > logs/p13_2_gdags/gpu1.log 2>&1 &
#   wait
#
# Output: logs/p13_2_gdags/GDAGS_seed42_<scene>.log
#         (GDAGS_OFF=1 → VERIFYOFF_seed42_<scene>.log)
# ============================================================
set -eo pipefail
GPU=${GPU:-0}
LOG_DIR=${LOG_DIR:-logs/p13_2_gdags}
OUT_DIR_VAL=${OUT_DIR:-output/p13_2_gdags}
mkdir -p ${LOG_DIR} ${OUT_DIR_VAL}

# P8_FULL_BACKBONE — EXACT copy từ scripts/p13_1_sweep.sh (Phase 8 FULL)
P8_FULL_BACKBONE=(
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
# A3 add-ons = LFCF + AbsGS (Phase 13 FULL recipe locked)
A3_FLAGS=(
    --use_lfcf
    --lfcf_init_scaling_min 1.0
    --lfcf_init_scaling_max 1.5
    --lfcf_last_scaling_max 1.0
    --lfcf_pow 1.0
    --lfcf_splitting_ub 1.0
    --lfcf_interval_times 2
    --lfcf_tolerance 1e-5
    --lfcf_diffscale True
    --absdensify
)

GDAGS_OFF=${GDAGS_OFF:-0}   # 1 → verify flag-OFF reproduces A3 (no --use_gdags)
# Full 8-scene seed42 (feedback_full_8scene_ablation: GDAGS expected modest →
# 3-scene single-seed Δ rơi trong noise ±0.10, screening vô nghĩa. Full-8
# paired vs A3 baseline logs/p13_lfcf/A3_seed42_* là verdict đầu đúng).
SCENES=(${SCENES_OVERRIDE:-fern flower fortress horns leaves orchids room trex})
SEEDS=(${SEEDS_OVERRIDE:-42})

run_one() {
    local SC=$1; local SEED=$2
    if [ "$GDAGS_OFF" = "1" ]; then
        local CFG="VERIFYOFF"; local EXTRA=()
    else
        local CFG="GDAGS"; local EXTRA=(--use_gdags)
    fi
    local OUTDIR=${OUT_DIR_VAL}/${CFG}_seed${SEED}_${SC}
    local LOG=${LOG_DIR}/${CFG}_seed${SEED}_${SC}.log
    echo "[$(date +%H:%M:%S)] START ${CFG}_seed${SEED}_${SC} (GPU${GPU})"
    CUDA_VISIBLE_DEVICES=${GPU} python -u train.py \
        --source_path data/nerf_llff_data/${SC} \
        -m ${OUTDIR} \
        --seed ${SEED} \
        "${P8_FULL_BACKBONE[@]}" \
        "${A3_FLAGS[@]}" \
        "${EXTRA[@]}" \
        2>&1 | tee ${LOG}
    echo "[$(date +%H:%M:%S)] DONE  ${CFG}_seed${SEED}_${SC}"
}

echo "=== Phase 13.2.5 GDAGS pilot (GPU${GPU}, GDAGS_OFF=${GDAGS_OFF}) ==="
echo "  Scenes: ${SCENES[@]}  Seeds: ${SEEDS[@]}"
echo ""
for SEED in "${SEEDS[@]}"; do
    for S in "${SCENES[@]}"; do
        run_one $S $SEED
    done
done
echo "[GPU${GPU}] GDAGS pilot (GDAGS_OFF=${GDAGS_OFF}) complete."
