#!/bin/bash
# ============================================================
# [CRSGaussian Phase 24 POST-CLEANUP VERIFY] Full Phase 22 recipe re-test
# File: scripts/p24_postcleanup_verify.sh
#
# Mục đích: Sau khi remove 3 flag Phase 20 TRIM (informed_crs_init,
# use_crs_pruning, use_r_visible) + delete crs_init.py, verify Phase 22
# A3-TRIM 8-module recipe vẫn đạt project best 21.918 (±0.10 noise floor).
#
# Phase 22 anchor (pilot N=24, 2026-05-25): **21.918 PSNR** (RoMa v1 + A3-TRIM)
# Phase 24 c0_base re-run (N=24, 2026-05-29): 21.882 (drift −0.036 trong noise ✓)
# Post-cleanup mong đợi: tương đương Phase 24 c0_base (~21.88 ±0.10)
#
# Modes (set via env var SEEDS_MODE) — parallel 2 GPU:
#   single  → 1 seed × 8 scenes = 8 runs ~30min  (DEFAULT, quick verify)
#   multi   → 3 seeds × 8 scenes = 24 runs ~1.5h (full N=24 stat-sig)
#
# Pre-requisite:
#   conda activate corgs
#   cd ~/workspace/representation-3d/duyen/CoR-GS
#   chmod +x scripts/p24_postcleanup_verify.sh
#   # Cleanup đã xong (rm crs_init.py + test file + pyc cache)
#
# Usage:
#   # Quick verify (1 seed, ~1h):
#   nohup ./scripts/p24_postcleanup_verify.sh > /tmp/p24_postclean.log 2>&1 &
#   disown
#
#   # Full N=24 verify (3 seeds, ~3h):
#   SEEDS_MODE=multi nohup ./scripts/p24_postcleanup_verify.sh > /tmp/p24_postclean.log 2>&1 &
#   disown
#
# Monitor:
#   tail -f /tmp/p24_postclean.log
#   ls logs/p24_postcleanup_verify/A3_seed*_*.log | wc -l   # đếm done
#
# Analyze:
#   python scripts/p24_postcleanup_verify_analyze.py
# ============================================================

set -e

# ── Verify init = RoMa v1 (chống nhầm với MVS hoặc v2) ──
FERN_PLY="data/nerf_llff_data/fern/3_views/dense/fused.ply"
FERN_V1="data/nerf_llff_data/fern/3_views/dense/fused.ply.romav1"
if ! cmp -s "$FERN_PLY" "$FERN_V1"; then
    echo "❌ fused.ply ≠ fused.ply.romav1 — chạy scripts/p22_place_romav1_init.py trước!"
    exit 1
fi
echo "✓ verified fused.ply = RoMa v1"

# ── Verify cleanup đã xong (3 flag không còn trong arguments parser) ──
echo "Checking cleanup integrity..."
python -c "
import argparse
from arguments import ModelParams, OptimizationParams
parser = argparse.ArgumentParser()
ModelParams(parser)
OptimizationParams(parser)
help_text = parser.format_help()
for flag in ['informed_crs_init', 'use_crs_pruning', 'use_r_visible']:
    if '--' + flag in help_text:
        print(f'❌ {flag} STILL in parser — cleanup KHÔNG complete')
        exit(1)
print('✓ all 3 trim flags removed from parser')
" || exit 1

# ── Verify crs_init.py đã xóa ──
if [ -f utils/crs/crs_init.py ]; then
    echo "❌ utils/crs/crs_init.py STILL exists — chưa rm"
    exit 1
fi
echo "✓ utils/crs/crs_init.py removed"

# ── Phase 22 protocol (KHÔNG đổi) ──
PROTOCOL="--eval -r 8 --n_views 3 --random_background --iterations 10000 \
--densify_until_iter 5000 --densify_grad_threshold 0.0005 --gaussiansN 1 \
--sample_pseudo_interval 1 --start_sample_pseudo 500 --test_iterations 10000"

# ── Phase 22 A3-TRIM 8-module recipe = PROJECT BEST 21.918 ──
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

# ── Seeds: single (quick) hoặc multi (full N=24) ──
SEEDS_MODE=${SEEDS_MODE:-single}
if [ "$SEEDS_MODE" = "multi" ]; then
    SEEDS=(42 137 9999)
    echo "Mode: MULTI (3 seeds × 8 scenes = 24 runs ~1.5h parallel 2 GPU)"
else
    SEEDS=(42)
    echo "Mode: SINGLE (1 seed × 8 scenes = 8 runs ~30min parallel 2 GPU)"
fi

# ── GPU split (8 scenes → 4 + 4) ──
GPU0_SCENES=(fern flower fortress horns)
GPU1_SCENES=(leaves orchids room trex)

LOG_ROOT="logs/p24_postcleanup_verify"
OUT_ROOT="output/p24_postcleanup_verify"
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

    echo "[$(date +%H:%M:%S)] START [P24postclean seed${SEED}/${SC}] GPU${GPU}" | tee -a "$MASTER_LOG"
    CUDA_VISIBLE_DEVICES=${GPU} python -u train.py \
        --source_path data/nerf_llff_data/${SC} \
        -m ${OUTDIR} --seed ${SEED} \
        ${PROTOCOL} ${A3_TRIM} \
        2>&1 | tee ${LOG} > /dev/null
    echo "[$(date +%H:%M:%S)] DONE  [P24postclean seed${SEED}/${SC}]" | tee -a "$MASTER_LOG"
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

echo "[P24postclean] start $(date '+%Y-%m-%d %H:%M:%S')" | tee "$MASTER_LOG"
echo "  mode: ${SEEDS_MODE}" | tee -a "$MASTER_LOG"
echo "  seeds: ${SEEDS[*]}" | tee -a "$MASTER_LOG"
echo "  GPU0 scenes: ${GPU0_SCENES[*]}" | tee -a "$MASTER_LOG"
echo "  GPU1 scenes: ${GPU1_SCENES[*]}" | tee -a "$MASTER_LOG"
echo "  total: ${#SEEDS[@]} × 8 scenes = $((${#SEEDS[@]} * 8)) runs" | tee -a "$MASTER_LOG"
echo "  skip-if-exists ENABLED" | tee -a "$MASTER_LOG"

# Parallel 2 GPUs
run_gpu_loop 0 "${GPU0_SCENES[@]}" &
PID0=$!
run_gpu_loop 1 "${GPU1_SCENES[@]}" &
PID1=$!

wait $PID0 $PID1

echo "[P24postclean] done $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "next: python scripts/p24_postcleanup_verify_analyze.py" | tee -a "$MASTER_LOG"
