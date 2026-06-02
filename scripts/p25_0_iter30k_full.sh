#!/bin/bash
# ============================================================
# [CRSGaussian Phase 25_0] 30k iter scaling FULL — N=24 paired
# File: scripts/p25_0_iter30k_full.sh
#
# Mục đích: Sau smoke PASS (fern 30k = 23.79 ≈ 10k 23.84),
# verify full N=24 paired để khẳng định Phase 22 recipe scale lên
# 30k vẫn đạt PSNR tương đương 10k.
#
# Hypothesis: PSNR(30k) ≈ PSNR(10k) trên all 8 scenes
# (sparse-view 3-view saturates early, extra iter = overfit not gain).
#
# Narrative paper:
#   - 10k bucket: ours 21.89 vs FSGS 20.31 / CoR-GS 20.11 (clean win)
#   - 30k bucket: ours = X.XX (sẽ điền) vs Binocular3DGS 21.44
#   - Cross-budget: ours 10k 21.89 = compute efficiency story
#
# Recipe changes vs 10k:
#   - iterations 10000 → 30000
#   - densify_until_iter 5000 → 15000
#   - opacity_decay_factor 0.999 → 0.995 (per budget_scaling_rule)
#
# Cost: 24 runs × ~25 min / 2 GPU parallel = ~5h
#
# Pre-requisite:
#   conda activate corgs
#   cd ~/workspace/representation-3d/duyen/CoR-GS
#   chmod +x scripts/p25_0_iter30k_full.sh
#
# Usage:
#   tmux new -s p25
#   nohup ./scripts/p25_0_iter30k_full.sh > /tmp/p25_0_full.log 2>&1 &
#   disown
#   Ctrl-B D
#
# Monitor:
#   tail -f /tmp/p25_0_full.log
#   ls logs/p25_0_iter30k_full/A3_seed*_*.log | wc -l
#
# Analyze:
#   python scripts/p25_0_iter30k_analyze.py
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

# ── Phase 22 protocol MODIFIED for 30k ──
PROTOCOL_30K="--eval -r 8 --n_views 3 --random_background \
--iterations 30000 \
--densify_until_iter 15000 \
--densify_grad_threshold 0.0005 --gaussiansN 1 \
--sample_pseudo_interval 1 --start_sample_pseudo 500 \
--test_iterations 30000"

# ── A3-TRIM 8-module — opacity_decay SCALED for 30k ──
M_DEPTHCFG="--use_depth_prior --dav2_path ../Depth-Anything-V2 --crs_ema_decay 0.3 --crs_update_interval 100"
M_DCYCLE="--use_d_cycle --d_cycle_warmup 1000 --d_cycle_sigma 5.0 --d_cycle_update_freq 100"
M_SHFREEZE="--use_crs_modulated_sh_freeze --crs_freeze_start 1000 --crs_freeze_tau 0.5"
M_SHREL="--use_sh_reliability --sh_stability_warmup 1000 --sh_stability_ema_beta 0.95 --crs_w_s 0.33"
M_DROP="--use_dropansh --dropansh_pa 0.02 --dropansh_psh 0.2"
M_OPACITY_30K="--use_opacity_decay --opacity_decay_factor 0.995"   # SCALED
M_EFA="--use_lfcf --lfcf_init_scaling_max 1.5 --lfcf_init_scaling_min 1.0 \
--lfcf_last_scaling_max 1.0 --lfcf_pow 1.0 --lfcf_splitting_ub 1.0 \
--lfcf_interval_times 2 --lfcf_tolerance 1e-5 --lfcf_diffscale True --absdensify"

A3_TRIM_30K="${M_DEPTHCFG} ${M_DCYCLE} ${M_SHFREEZE} ${M_SHREL} ${M_DROP} ${M_OPACITY_30K} ${M_EFA}"

SEEDS=(42 137 9999)

# ── GPU split (8 scenes → 4 + 4) ──
GPU0_SCENES=(fern flower fortress horns)
GPU1_SCENES=(leaves orchids room trex)

LOG_ROOT="logs/p25_0_iter30k_full"
OUT_ROOT="output/p25_0_iter30k_full"
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

    echo "[$(date +%H:%M:%S)] START [P25_0 seed${SEED}/${SC}] GPU${GPU}" | tee -a "$MASTER_LOG"
    CUDA_VISIBLE_DEVICES=${GPU} python -u train.py \
        --source_path data/nerf_llff_data/${SC} \
        -m ${OUTDIR} --seed ${SEED} \
        ${PROTOCOL_30K} ${A3_TRIM_30K} \
        2>&1 | tee ${LOG} > /dev/null
    echo "[$(date +%H:%M:%S)] DONE  [P25_0 seed${SEED}/${SC}]" | tee -a "$MASTER_LOG"
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

echo "[P25_0 FULL 30k] start $(date '+%Y-%m-%d %H:%M:%S')" | tee "$MASTER_LOG"
echo "  recipe: A3-TRIM 8-module + 30k scaling" | tee -a "$MASTER_LOG"
echo "  iter: 30000 (vs Phase 22 10000)" | tee -a "$MASTER_LOG"
echo "  densify_until: 15000 (vs 5000)" | tee -a "$MASTER_LOG"
echo "  opacity_decay: 0.995 (vs 0.999)" | tee -a "$MASTER_LOG"
echo "  seeds: ${SEEDS[*]}" | tee -a "$MASTER_LOG"
echo "  GPU0: ${GPU0_SCENES[*]}" | tee -a "$MASTER_LOG"
echo "  GPU1: ${GPU1_SCENES[*]}" | tee -a "$MASTER_LOG"
echo "  total: 24 runs × ~25min / 2 GPU = ~5h" | tee -a "$MASTER_LOG"

run_gpu_loop 0 "${GPU0_SCENES[@]}" &
PID0=$!
run_gpu_loop 1 "${GPU1_SCENES[@]}" &
PID1=$!

wait $PID0 $PID1

echo "[P25_0 FULL 30k] done $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "next: python scripts/p25_0_iter30k_analyze.py" | tee -a "$MASTER_LOG"
