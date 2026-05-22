#!/bin/bash
# ============================================================
# [CRSGaussian Phase 18b] Cyclic-gate triage — A3 với PDCNet+ gated init
# File: scripts/p18b_cyclic_triage.sh  (TẠO MỚI — keep local)
#
# Triage 4 scene × 3 seed = 12 run:
#   horns, trex   = must-rescue (Phase 18 raw thua −0.318 / −0.357)
#   fortress      = control nhẹ (thắng đậm, gate tỉa ~10%)
#   orchids       = control nặng (thắng nhỏ, gate tỉa ~26%)
#
# Init = PDCNet+ cloud GATED theo cyclic τ=1.0 (pre-registered, decisions_log
# 2026-05-22) — đã place qua p18_gate2_place_dense_init.py.
#
# P8_CORE + A3_FLAGS = EXACT copy p18_pilot.sh (= recipe A3). 0 đổi code
# CRSGaussian — chỉ init .ply khác (gated thay raw).
#
# So sánh = REUSE logs/p18_pilot (A3-PDCNet+ raw) — paired theo (seed,scene).
# Verdict triage: scripts/p18b_triage_analyze.py.
#
# Usage (2-GPU split):
#   mkdir -p logs/p18b_triage
#   GPU=0 SCENES_OVERRIDE="horns trex" \
#       nohup bash scripts/p18b_cyclic_triage.sh > logs/p18b_triage/gpu0.log 2>&1 & disown
#   GPU=1 SCENES_OVERRIDE="fortress orchids" \
#       nohup bash scripts/p18b_cyclic_triage.sh > logs/p18b_triage/gpu1.log 2>&1 & disown
# Output: output/p18b_triage/A3_seed<sd>_<scene>/ , log logs/p18b_triage/A3_seed<sd>_<scene>.log
# ============================================================
set -eo pipefail
GPU=${GPU:-0}
LOG_DIR=${LOG_DIR:-logs/p18b_triage}
OUT_DIR_VAL=${OUT_DIR:-output/p18b_triage}
mkdir -p ${LOG_DIR} ${OUT_DIR_VAL}

# P8_CORE + A3_FLAGS = EXACT copy p18_pilot.sh (CFG A3).
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

run_one() {
    local SC=$1; local SEED=$2
    local OUTDIR=${OUT_DIR_VAL}/A3_seed${SEED}_${SC}
    local LOG=${LOG_DIR}/A3_seed${SEED}_${SC}.log
    echo "[$(date +%H:%M:%S)] START A3-gated seed${SEED}_${SC} (GPU${GPU})"
    CUDA_VISIBLE_DEVICES=${GPU} python -u train.py \
        --source_path data/nerf_llff_data/${SC} \
        -m ${OUTDIR} --seed ${SEED} \
        "${P8_CORE[@]}" "${A3_FLAGS[@]}" \
        2>&1 | tee ${LOG}
    echo "[$(date +%H:%M:%S)] DONE  A3-gated seed${SEED}_${SC}"
}

SCENES=(${SCENES_OVERRIDE:-horns trex fortress orchids})
SEEDS=(${SEEDS_OVERRIDE:-42 137 9999})

echo "=== Phase 18b cyclic-gate triage (GPU${GPU}) SCENES=${SCENES[*]} SEEDS=${SEEDS[*]} ==="
for SEED in "${SEEDS[@]}"; do
    for S in "${SCENES[@]}"; do
        run_one $S $SEED
    done
done
echo "[GPU${GPU}] Phase 18b triage complete."
