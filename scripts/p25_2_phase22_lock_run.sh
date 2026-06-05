#!/bin/bash
# ============================================================
# [CRSGaussian Phase 25_2] Phase 22 LOCK — 4th N=24 measurement
# File: scripts/p25_2_phase22_lock_run.sh
#
# Mục đích: Final confirmation Phase 22 PSNR mean.
# Sau khi có 4 N=24 measurements, lock final value cho paper:
#   mean of 4 N=24 → SEM ÷√4 = SEM/2 (tighter)
#
# Existing 3 measurements:
#   Phase 22 pilot (2026-05-25):     21.918  (pre-cleanup)
#   Phase 24 c0_base (2026-05-29):   21.882  (pre-cleanup, fresh re-run)
#   Phase 24 postcleanup (2026-05-29): 21.869  (post-cleanup)
#   Mean of 3 = 21.890 ± 0.014 SEM
#
# Phase 25_2 (this run) = 4th measurement:
#   Recipe: Phase 22 A3-TRIM 8-module (10k iter, RoMa v1 init)
#   Code state: post-cleanup (3 trim flags removed)
#   N=24 = 3 seeds × 8 scenes
#   Cost: ~1.5h on 2 GPU parallel
#
# After this run:
#   - 4 N=24 measurements → tighter SEM
#   - Mean ≈ 21.88-21.90 expected (consistent với existing 3)
#   - Variance > 0.05 vs mean → flag as concerning
#
# Pre-requisite:
#   conda activate corgs
#   cd ~/workspace/representation-3d/duyen/CoR-GS
#   chmod +x scripts/p25_2_phase22_lock_run.sh
#
# Usage:
#   tmux new -s p25_2
#   nohup bash scripts/p25_2_phase22_lock_run.sh > /tmp/p25_2_lock.log 2>&1 &
#   disown
#   Ctrl-B D
#
# Monitor:
#   tail -f /tmp/p25_2_lock.log
#   ls logs/p25_2_phase22_lock/A3_seed*_*.log | wc -l   # target 24
#
# Analyze:
#   python scripts/p25_2_phase22_lock_analyze.py
# ============================================================

set -e

# ── Verify init = RoMa v1 ──
FERN_PLY="data/nerf_llff_data/fern/3_views/dense/fused.ply"
FERN_V1="data/nerf_llff_data/fern/3_views/dense/fused.ply.romav1"
if ! cmp -s "$FERN_PLY" "$FERN_V1"; then
    echo "❌ fused.ply ≠ fused.ply.romav1"
    exit 1
fi
echo "✓ verified fused.ply = RoMa v1"

# ── Verify cleanup state (3 flag removed) ──
python -c "
import argparse
from arguments import ModelParams, OptimizationParams
parser = argparse.ArgumentParser()
ModelParams(parser); OptimizationParams(parser)
help_text = parser.format_help()
for flag in ['informed_crs_init', 'use_crs_pruning', 'use_r_visible']:
    if '--' + flag in help_text:
        print(f'❌ {flag} STILL in parser — cleanup KHÔNG complete')
        exit(1)
print('✓ post-cleanup state verified (3 trim flags removed)')
" || exit 1

# ── Phase 22 protocol (KHÔNG đổi) ──
PROTOCOL="--eval -r 8 --n_views 3 --random_background --iterations 10000 \
--densify_until_iter 5000 --densify_grad_threshold 0.0005 --gaussiansN 1 \
--sample_pseudo_interval 1 --start_sample_pseudo 500 --test_iterations 10000"

# ── Phase 22 A3-TRIM 8-module recipe ──
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

# Locked seeds (Phase 11+ project standard)
SEEDS=(42 137 9999)

# GPU split
GPU0_SCENES=(fern flower fortress horns)
GPU1_SCENES=(leaves orchids room trex)

LOG_ROOT="logs/p25_2_phase22_lock"
OUT_ROOT="output/p25_2_phase22_lock"
MASTER_LOG="${LOG_ROOT}/_master.log"
mkdir -p "$LOG_ROOT"

run_one() {
    local GPU=$1; local SEED=$2; local SC=$3
    mkdir -p "$LOG_ROOT" "$OUT_ROOT"

    local OUTDIR=${OUT_ROOT}/A3_seed${SEED}_${SC}
    local LOG=${LOG_ROOT}/A3_seed${SEED}_${SC}.log

    # Skip if already done (resume-safe)
    if [ -f "$LOG" ] && grep -q "Best test PSNR" "$LOG" 2>/dev/null; then
        return 0
    fi

    echo "[$(date +%H:%M:%S)] START [P25_2 seed${SEED}/${SC}] GPU${GPU}" | tee -a "$MASTER_LOG"
    CUDA_VISIBLE_DEVICES=${GPU} python -u train.py \
        --source_path data/nerf_llff_data/${SC} \
        -m ${OUTDIR} --seed ${SEED} \
        ${PROTOCOL} ${A3_TRIM} \
        2>&1 | tee ${LOG} > /dev/null
    echo "[$(date +%H:%M:%S)] DONE  [P25_2 seed${SEED}/${SC}]" | tee -a "$MASTER_LOG"
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

echo "[P25_2 LOCK] start $(date '+%Y-%m-%d %H:%M:%S')" | tee "$MASTER_LOG"
echo "  purpose: 4th N=24 measurement for Phase 22 PSNR lock" | tee -a "$MASTER_LOG"
echo "  recipe: A3-TRIM 8-module (10k iter, RoMa v1)" | tee -a "$MASTER_LOG"
echo "  seeds: ${SEEDS[*]}" | tee -a "$MASTER_LOG"
echo "  GPU0: ${GPU0_SCENES[*]}" | tee -a "$MASTER_LOG"
echo "  GPU1: ${GPU1_SCENES[*]}" | tee -a "$MASTER_LOG"
echo "  total: 24 runs × ~8min / 2 GPU = ~1.5h" | tee -a "$MASTER_LOG"

# Parallel 2 GPUs
run_gpu_loop 0 "${GPU0_SCENES[@]}" &
PID0=$!
run_gpu_loop 1 "${GPU1_SCENES[@]}" &
PID1=$!

wait $PID0 $PID1

echo "[P25_2 LOCK] done $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "next: python scripts/p25_2_phase22_lock_analyze.py" | tee -a "$MASTER_LOG"
