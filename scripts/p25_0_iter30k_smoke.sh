#!/bin/bash
# ============================================================
# [CRSGaussian Phase 25_0] 30k iter scaling SMOKE — 1 scene 1 seed
# File: scripts/p25_0_iter30k_smoke.sh
#
# Mục đích: Verify Phase 22 A3-TRIM recipe có scale lên 30k iter không.
# Phase 22 recipe calibrated cho 10k. Scale 30k cần:
#   - iterations 30000 (was 10000)
#   - densify_until_iter 15000 (proportional 3x)
#   - opacity_decay_factor 0.995 (was 0.999) — critical, per memory
#     budget_scaling_rule. 0.999^30000 → opacity collapse.
#     0.995 validated by Binocular3DGS at 30k.
# All other hyperparams unchanged (LFCF / SH-freeze / DropAnSH / D_cycle).
#
# Smoke target: fern scene seed 42 — Phase 22 N=24 anchor 23.84.
#   - PSNR ≥ 22.5 → 30k recipe works, proceed full N=24
#   - PSNR < 22.0 → recipe broken/overfit, debug
#
# Cost: ~25 min (3× of 10k smoke ~8 min)
#
# Pre-requisite:
#   conda activate corgs
#   cd ~/workspace/representation-3d/duyen/CoR-GS
#   chmod +x scripts/p25_0_iter30k_smoke.sh
#   # Cleanup Phase 24 đã xong + smoke verify PASS
#
# Usage:
#   ./scripts/p25_0_iter30k_smoke.sh
#   # Output: /tmp/p25_0_smoke.log
#
# Quick comparison sau khi xong:
#   grep "Best test PSNR" /tmp/p25_0_smoke.log
#   # Phase 22 fern reference: 23.84 N=24 mean (single seed ±1.3 dB noise)
#   # 30k expected: 22.5 - 24.5 (any in range = recipe works)
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

# ── Phase 22 protocol — MODIFIED for 30k ──
# Changes from Phase 22:
#   iterations 10000 → 30000
#   densify_until_iter 5000 → 15000
PROTOCOL_30K="--eval -r 8 --n_views 3 --random_background \
--iterations 30000 \
--densify_until_iter 15000 \
--densify_grad_threshold 0.0005 --gaussiansN 1 \
--sample_pseudo_interval 1 --start_sample_pseudo 500 \
--test_iterations 30000"

# ── A3-TRIM 8-module — opacity_decay_factor SCALED for 30k ──
# Phase 22 (10k): opacity_decay_factor 0.999
# Phase 25_0 (30k): opacity_decay_factor 0.995 (per budget_scaling_rule memory + Binocular3DGS)
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

SEED=42
SCENE=fern
OUT_DIR="/tmp/p25_0_smoke_fern_30k"
LOG="/tmp/p25_0_smoke.log"

echo "[P25_0 SMOKE] start $(date '+%Y-%m-%d %H:%M:%S')"
echo "  scene: ${SCENE}, seed: ${SEED}"
echo "  iter: 30000 (vs Phase 22 10000)"
echo "  densify_until: 15000 (vs 5000)"
echo "  opacity_decay: 0.995 (vs 0.999)"
echo "  log: ${LOG}"
echo "  output: ${OUT_DIR}"
echo "  Expected ~25 min on 1 GPU"

CUDA_VISIBLE_DEVICES=0 python -u train.py \
    --source_path data/nerf_llff_data/${SCENE} \
    -m ${OUT_DIR} --seed ${SEED} \
    ${PROTOCOL_30K} ${A3_TRIM_30K} \
    2>&1 | tee ${LOG}

echo ""
echo "[P25_0 SMOKE] done $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "=== RESULT ==="
grep "Best test PSNR" ${LOG} | tail -3
echo ""
echo "Phase 22 fern reference (10k, N=24 mean): 23.84"
echo "Verdict:"
echo "  PSNR ≥ 22.5  → 30k recipe works, proceed full N=24"
echo "  PSNR 21-22.5 → marginal, run 2 more seeds confirm"
echo "  PSNR < 21    → 30k recipe broken or overfit, debug"
