#!/bin/bash
# ============================================================
# [CRSGaussian Phase 25_3] 6-view experiment (Phase 22 recipe scaled)
# File: scripts/p25_3_6view_run.sh
#
# Mục đích: Train Phase 22 A3-TRIM 8-module recipe at n_views=6.
# Phase 22 (3-view) đã LOCK = 21.89. 6-view typically easier (more train cams)
# → expected PSNR 23-25 region (per FSGS/CoR-GS baseline trends).
#
# Recipe = identical Phase 22 + chỉ đổi --n_views 3 → 6.
# Init = RoMa v1 từ data/nerf_llff_data/<scene>/6_views/dense/fused.ply
# (preprocess riêng — chạy scripts/p25_3_romav1_6view_preprocess.py trước nếu chưa)
#
# Pre-requisite:
#   conda activate corgs
#   cd ~/workspace/representation-3d/duyen/CoR-GS
#   chmod +x scripts/p25_3_6view_run.sh
#   # 6_views/dense/fused.ply phải = RoMa v1 (verify hash)
#
# Usage:
#   tmux new -s p25_3
#   nohup bash scripts/p25_3_6view_run.sh > /tmp/p25_3_6view.log 2>&1 &
#   disown
#   Ctrl-B D
#
# Cost: 24 runs × ~10-15min / 2 GPU = ~2-3h (6-view slightly slower vs 3-view)
#
# Analyze:
#   python scripts/p25_3_6view_analyze.py
# ============================================================

set -e

# ── Verify init = RoMa v1 cho 6-view ──
FERN_PLY="data/nerf_llff_data/fern/6_views/dense/fused.ply"
FERN_V1="data/nerf_llff_data/fern/6_views/dense/fused.ply.romav1"
if [ ! -f "$FERN_PLY" ]; then
    echo "❌ $FERN_PLY KHÔNG tồn tại — chạy preprocess 6_views trước"
    exit 1
fi
if [ -f "$FERN_V1" ] && ! cmp -s "$FERN_PLY" "$FERN_V1"; then
    echo "❌ fused.ply ≠ fused.ply.romav1 — chạy p25_3_place_6view_init.sh"
    exit 1
fi
echo "✓ verified 6_views/fused.ply = RoMa v1"

# ── Phase 22 protocol — KHÔNG đổi gì ngoài --n_views ──
PROTOCOL="--eval -r 8 --n_views 6 --random_background --iterations 10000 \
--densify_until_iter 5000 --densify_grad_threshold 0.0005 --gaussiansN 1 \
--sample_pseudo_interval 1 --start_sample_pseudo 500 --test_iterations 10000"

# ── A3-TRIM 8-module recipe (identical Phase 22) ──
M_DEPTHCFG="--use_depth_prior --dav2_path ../Depth-Anything-V2 --crs_ema_decay 0.3 --crs_update_interval 100"
M_DCYCLE="--use_d_cycle --d_cycle_warmup 1000 --d_cycle_sigma 5.0 --d_cycle_update_freq 100"
M_SHFREEZE="--use_crs_modulated_sh_freeze --crs_freeze_start 1000 --crs_freeze_tau 0.5"
M_SHREL="--use_sh_reliability --sh_stability_warmup 1000 --sh_stability_ema_beta 0.95 --crs_w_s 0.33"
M_DROP="--use_dropansh --dropansh_pa 0.02 --dropansh_psh 0.2"
M_OPACITY="--use_opacity_decay --opacity_decay_factor 0.999"
M_EFA="--use_lfcf --lfcf_init_scaling_max 1.5 --lfcf_init_scaling_min 1.0 \
--lfcf_last_scaling_max 1.0 --lfcf_pow 1.0 --lfcf_splitting_ub 1.0 \
--lfcf_interval_times 2 --lfcf_tolerance 1e-5 --lfcf_diffscale True --absdensify"

A3_TRIM="${M_DEPTHCFG} ${M_DCYCLE} ${M_SHFREEZE} ${M_SHREL} ${M_DROP} ${M_OPACITY} ${M_EFA}"

SEEDS=(42 137 9999)

GPU0_SCENES=(fern flower fortress horns)
GPU1_SCENES=(leaves orchids room trex)

LOG_ROOT="logs/p25_3_6view"
OUT_ROOT="output/p25_3_6view"
MASTER_LOG="${LOG_ROOT}/_master.log"
mkdir -p "$LOG_ROOT"

run_one() {
    local GPU=$1; local SEED=$2; local SC=$3
    mkdir -p "$LOG_ROOT" "$OUT_ROOT"

    local OUTDIR=${OUT_ROOT}/A3_seed${SEED}_${SC}
    local LOG=${LOG_ROOT}/A3_seed${SEED}_${SC}.log

    if [ -f "$LOG" ] && grep -q "Best test PSNR" "$LOG" 2>/dev/null; then
        return 0
    fi

    echo "[$(date +%H:%M:%S)] START [P25_3 6view seed${SEED}/${SC}] GPU${GPU}" | tee -a "$MASTER_LOG"
    CUDA_VISIBLE_DEVICES=${GPU} python -u train.py \
        --source_path data/nerf_llff_data/${SC} \
        -m ${OUTDIR} --seed ${SEED} \
        ${PROTOCOL} ${A3_TRIM} \
        2>&1 | tee ${LOG} > /dev/null
    echo "[$(date +%H:%M:%S)] DONE  [P25_3 6view seed${SEED}/${SC}]" | tee -a "$MASTER_LOG"
}

run_gpu_loop() {
    local GPU=$1; shift
    local SCENES=("$@")
    for SEED in "${SEEDS[@]}"; do
        for SC in "${SCENES[@]}"; do
            run_one "$GPU" "$SEED" "$SC"
        done
    done
}

echo "[P25_3 6view] start $(date '+%Y-%m-%d %H:%M:%S')" | tee "$MASTER_LOG"
echo "  recipe: Phase 22 A3-TRIM 8-module + n_views=6" | tee -a "$MASTER_LOG"
echo "  seeds: ${SEEDS[*]}" | tee -a "$MASTER_LOG"
echo "  GPU0: ${GPU0_SCENES[*]}" | tee -a "$MASTER_LOG"
echo "  GPU1: ${GPU1_SCENES[*]}" | tee -a "$MASTER_LOG"
echo "  total: 24 runs × ~10-15min / 2 GPU = ~2-3h" | tee -a "$MASTER_LOG"

run_gpu_loop 0 "${GPU0_SCENES[@]}" &
PID0=$!
run_gpu_loop 1 "${GPU1_SCENES[@]}" &
PID1=$!

wait $PID0 $PID1

echo "[P25_3 6view] done $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "next: python scripts/p25_3_6view_analyze.py" | tee -a "$MASTER_LOG"
