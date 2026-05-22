#!/bin/bash
# ============================================================
# [CRSGaussian Phase 18] Full pilot N=24 — A3 với PDCNet+ dense init
# File: scripts/p18_pilot.sh  (TẠO MỚI — keep local)
#
# A3 × 8 scene × 3 seed (42/137/9999) = N=24, init = PDCNet+ dense
# (đã place qua p18_gate2_place_dense_init.py SCENES=all-8).
#
# P8_CORE + A3_FLAGS = EXACT copy p17c_a3_resave.sh:38-64 (= recipe A3,
# verified từ p15). 0 đổi code CRSGaussian — chỉ init .ply khác.
# KHÔNG --save_iterations → chỉ lưu ckpt iter 10000 (tiết kiệm disk —
# init dày → ckpt to).
#
# Baseline so sánh = REUSE logs/p13_lfcf (A3-MVS N=24) — KHÔNG train lại.
# Pilot chạy train.py HIỆN TẠI (có hook C1 Phase-17 default OFF =
# byte-identical A3 — verified Phase 17 VOFF) → so p13_lfcf là fair.
#
# Usage (2-GPU split cân theo init-density ~1.6M điểm/GPU):
#   mkdir -p logs/p18_pilot
#   GPU=0 SCENES_OVERRIDE="room leaves horns orchids" \
#       nohup bash scripts/p18_pilot.sh > logs/p18_pilot/gpu0.log 2>&1 & disown
#   GPU=1 SCENES_OVERRIDE="trex fern flower fortress" \
#       nohup bash scripts/p18_pilot.sh > logs/p18_pilot/gpu1.log 2>&1 & disown
# Output: output/p18_pilot/A3_seed<sd>_<scene>/ , log logs/p18_pilot/A3_seed<sd>_<scene>.log
# ============================================================
set -eo pipefail
GPU=${GPU:-0}
LOG_DIR=${LOG_DIR:-logs/p18_pilot}
OUT_DIR_VAL=${OUT_DIR:-output/p18_pilot}
mkdir -p ${LOG_DIR} ${OUT_DIR_VAL}

# P8_CORE + A3_FLAGS = EXACT copy p17c_a3_resave.sh:38-64 (CFG A3).
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
    echo "[$(date +%H:%M:%S)] START A3-pdcnet seed${SEED}_${SC} (GPU${GPU})"
    CUDA_VISIBLE_DEVICES=${GPU} python -u train.py \
        --source_path data/nerf_llff_data/${SC} \
        -m ${OUTDIR} --seed ${SEED} \
        "${P8_CORE[@]}" "${A3_FLAGS[@]}" \
        2>&1 | tee ${LOG}
    echo "[$(date +%H:%M:%S)] DONE  A3-pdcnet seed${SEED}_${SC}"
}

SCENES=(${SCENES_OVERRIDE:-fern flower fortress horns leaves orchids room trex})
SEEDS=(${SEEDS_OVERRIDE:-42 137 9999})

echo "=== Phase 18 pilot (GPU${GPU}) SCENES=${SCENES[*]} SEEDS=${SEEDS[*]} ==="
for SEED in "${SEEDS[@]}"; do
    for S in "${SCENES[@]}"; do
        run_one $S $SEED
    done
done
echo "[GPU${GPU}] Phase 18 pilot complete."
