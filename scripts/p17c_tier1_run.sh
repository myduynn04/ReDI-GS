#!/bin/bash
# ============================================================
# [CRSGaussian Phase 17c — Tier 1] ∇depth noise pre-check runner
# Diagnostic-only, ~15 min, single GPU. KHÔNG đụng production code.
# Output: logs/p17c/tier1_noise_check.txt + tier1_scatter.png + tier1_boxplot.png
#
# Pre-registered config (LOCKED, không sửa khi verdict):
#   metric  = angle(n_raw, n_smooth_σ2px) via torch separable Gaussian (k=13)
#   bins    = r ≤ −0.7 / (−0.7,−0.5] / (−0.5,−0.3) / ≥ −0.3
#   compute = LOOSE (Tier 2 required if r ∈ (−0.7,−0.5], không skip)
#   smoke   = plane <1° AND noisy >15° required to PASS
# ============================================================
set -eo pipefail
mkdir -p logs/p17c

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export SCENES=${SCENES:-"fern flower fortress horns leaves orchids room trex"}
export SEEDS=${SEEDS:-"42 137 9999"}
export SEED_REF=${SEED_REF:-42}
export ITERATION=${ITERATION:-10000}
export DATA_ROOT=${DATA_ROOT:-data/nerf_llff_data}
export OUTPUT_ROOT=${OUTPUT_ROOT:-output/p13_lfcf}
export A3_LOG=${A3_LOG:-logs/p13_lfcf}
export C1_LOG=${C1_LOG:-logs/p17_c1}
export OUT_DIR=${OUT_DIR:-logs/p17c}

python scripts/p17c_tier1_noise_check.py 2>&1 | tee logs/p17c/tier1_noise_check.txt
echo ""
echo "Done. Verdict: cat logs/p17c/tier1_noise_check.txt | tail -25"
