#!/bin/bash
# ============================================================
# [CRSGaussian Phase 17c — Tier 2] noise trajectory pre-check runner
# Diagnostic-only, scan-first; ~10 min nếu có intermediate ckpts có sẵn,
# else exit + suggest re-train command.
# Output: logs/p17c/tier2_trajectory.txt + tier2_curves.png + tier2_drop_bar.png
#
# Pre-registered (LOCKED, không sửa khi verdict):
#   metric  = SAME as Tier 1 — angle(n_raw, n_smooth_σ2)
#   EARLY ≤ 2500 | LATE ≥ 6000
#   class   = STRONG ≥30% drop / WEAK ≥10% / NO <10%
#   verdict = ≥5/8 STRONG ∧ median elbow ∈ [3000,7000] → pilot at T∈{5000,7000}
#             ≥5/8 ANY-drop ∧ <5/8 STRONG → WEAK (user quyết)
#             else → SKIP → push Gate-phẳng / C1b
# ============================================================
set -eo pipefail
mkdir -p logs/p17c

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export SCENES=${SCENES:-"fern flower fortress horns leaves orchids room trex"}
export SEEDS=${SEEDS:-"42 137 9999"}
export SEED_REF=${SEED_REF:-42}
export DATA_ROOT=${DATA_ROOT:-data/nerf_llff_data}
export OUTPUT_ROOT=${OUTPUT_ROOT:-output/p13_lfcf}
export A3_LOG=${A3_LOG:-logs/p13_lfcf}
export C1_LOG=${C1_LOG:-logs/p17_c1}
export OUT_DIR=${OUT_DIR:-logs/p17c}
export EARLY_CUTOFF=${EARLY_CUTOFF:-2500}
export LATE_CUTOFF=${LATE_CUTOFF:-6000}

python scripts/p17c_tier2_noise_trajectory.py 2>&1 | tee logs/p17c/tier2_trajectory.txt
echo ""
echo "Done. Verdict tail: cat logs/p17c/tier2_trajectory.txt | tail -25"
