#!/bin/bash
# ============================================================
# [CRSGaussian Phase 19] SH-degree curriculum pilot — A3 + curriculum
# File: scripts/p19_curriculum_pilot.sh  (TẠO MỚI — keep local)
#
# Train A3 + SH-degree coarse-to-fine curriculum.
#   --use_sh_curriculum --sh_curriculum_interval 1500  (PRE-REGISTERED)
#   → SH degree 0→1→2→3 mở @ iter 1500/3000/4500 (thay vì warmup gốc @1500).
#
# P8_CORE + A3_FLAGS = EXACT copy p18b_cyclic_triage.sh (= recipe A3).
# Chỉ THÊM P19_FLAGS. 0 đổi gì khác → curriculum OFF = A3 byte-identical.
#
# ⚠️ BACKBONE chỉ ĐẶT TÊN output. Init thật (MVS / PDCNet+ dense) do file
# fused.ply quyết định — PHẢI place đúng init qua p18_gate2 TRƯỚC khi chạy:
#   A3-MVS:   SCENES="..." RESTORE=1 python scripts/p18_gate2_place_dense_init.py
#   A3-dense: PDCNET_DIR=p18_dense_init SCENES="..." python scripts/p18_gate2_place_dense_init.py
#
# So sánh (analyzer) = curriculum vs baseline CÙNG backbone, reuse log:
#   BACKBONE=mvs   → baseline logs/p13_lfcf  (A3-MVS 21.330)
#   BACKBONE=dense → baseline logs/p18_pilot (A3-dense 21.599)
# (baseline KHÔNG train lại). Verdict: scripts/p19_curriculum_analyze.py.
#
# Usage (2-GPU split, ví dụ backbone mvs):
#   mkdir -p logs/p19_curriculum_mvs
#   GPU=0 BACKBONE=mvs SCENES_OVERRIDE="horns trex fern fortress" \
#       nohup bash scripts/p19_curriculum_pilot.sh > logs/p19_curriculum_mvs/gpu0.log 2>&1 & disown
#   GPU=1 BACKBONE=mvs SCENES_OVERRIDE="leaves orchids room flower" \
#       nohup bash scripts/p19_curriculum_pilot.sh > logs/p19_curriculum_mvs/gpu1.log 2>&1 & disown
# Output: output/p19_curriculum_<backbone>/A3_seed<sd>_<scene>/
# Log:    logs/p19_curriculum_<backbone>/A3_seed<sd>_<scene>.log
# ============================================================
set -eo pipefail
GPU=${GPU:-0}
BACKBONE=${BACKBONE:-mvs}
LOG_DIR=${LOG_DIR:-logs/p19_curriculum_${BACKBONE}}
OUT_DIR_VAL=${OUT_DIR:-output/p19_curriculum_${BACKBONE}}
mkdir -p ${LOG_DIR} ${OUT_DIR_VAL}

# P8_CORE + A3_FLAGS = EXACT copy p18b_cyclic_triage.sh (CFG A3).
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
# ── [Phase 19] SH-degree curriculum — PRE-REGISTERED interval 1500 ──
P19_FLAGS=(
    --use_sh_curriculum --sh_curriculum_interval 1500
)

run_one() {
    local SC=$1; local SEED=$2
    local OUTDIR=${OUT_DIR_VAL}/A3_seed${SEED}_${SC}
    local LOG=${LOG_DIR}/A3_seed${SEED}_${SC}.log
    echo "[$(date +%H:%M:%S)] START A3-curric(${BACKBONE}) seed${SEED}_${SC} (GPU${GPU})"
    CUDA_VISIBLE_DEVICES=${GPU} python -u train.py \
        --source_path data/nerf_llff_data/${SC} \
        -m ${OUTDIR} --seed ${SEED} \
        "${P8_CORE[@]}" "${A3_FLAGS[@]}" "${P19_FLAGS[@]}" \
        2>&1 | tee ${LOG}
    echo "[$(date +%H:%M:%S)] DONE  A3-curric(${BACKBONE}) seed${SEED}_${SC}"
}

SCENES=(${SCENES_OVERRIDE:-fern flower fortress horns leaves orchids room trex})
SEEDS=(${SEEDS_OVERRIDE:-42 137 9999})

echo "=== Phase 19 curriculum pilot (GPU${GPU}, backbone=${BACKBONE}) "
echo "    SCENES=${SCENES[*]} SEEDS=${SEEDS[*]} ==="
for SEED in "${SEEDS[@]}"; do
    for S in "${SCENES[@]}"; do
        run_one $S $SEED
    done
done
echo "[GPU${GPU}] Phase 19 curriculum pilot (${BACKBONE}) complete."
