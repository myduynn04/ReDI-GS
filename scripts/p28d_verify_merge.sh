#!/bin/bash
# ============================================================
# Phase 28d — Verify CRS + SH freeze merge justification
# ============================================================
# Purpose: Test loo_no_shfreeze (CRS ON, SH freeze OFF) to verify
#          SH freeze is the dominant consumer of CRS.
#
# Expected: loo_no_shfreeze ≈ loo_no_crs (~21.94)
#           → CRS computation alone has no effect
#           → MERGE CRS+SH-freeze into 1 pillar justified
#
# Cell: 1 cell × 8 scenes × seed 42 = 8 runs
#   loo_no_shfreeze: full recipe MINUS --use_crs_modulated_sh_freeze
#
# References (Phase 27 seed 42 N=8):
#   trim_full  = 21.9207 (all CRS + SH freeze ON)
#   loo_no_crs = 21.9441 (all CRS OFF)
#
# GPU split: 4 scenes/GPU parallel
# Wall-clock: ~25 min (8 runs / 2 GPUs × ~6 min)
# ============================================================

set +e
cd ~/workspace/representation-3d/duyen/CoR-GS

# ============================================================
SEED=42
SCENES=(fern flower fortress horns leaves orchids room trex)
CELL="loo_no_shfreeze"
OUT_ROOT="output/p28d_verify_merge"
LOG_ROOT="logs/p28d_verify_merge"
LOCK_DIR="/tmp/p28d_locks"

mkdir -p "$LOG_ROOT"
rm -rf "$LOCK_DIR"
mkdir -p "$LOCK_DIR"

# ============================================================
# Base flags — full Phase 22 recipe MINUS sh_freeze
# Note: NO --use_crs_modulated_sh_freeze (this is the loo)
# Other CRS flags still ON (depth_prior, d_cycle, sh_reliability)
# ============================================================
BASE_FLAGS="--use_depth_prior --dav2_path ../Depth-Anything-V2 \
--crs_ema_decay 0.3 --crs_update_interval 100 \
--use_d_cycle --d_cycle_warmup 1000 --d_cycle_sigma 5.0 --d_cycle_update_freq 100 \
--use_sh_reliability --sh_stability_warmup 1000 --sh_stability_ema_beta 0.95 --crs_w_s 0.33 \
--use_dropansh --dropansh_pa 0.02 --dropansh_psh 0.2 \
--use_opacity_decay --opacity_decay_factor 0.999 \
--use_lfcf --lfcf_init_scaling_max 1.5 --lfcf_init_scaling_min 1.0 \
--lfcf_last_scaling_max 1.0 --lfcf_pow 1.0 --lfcf_splitting_ub 1.0 \
--lfcf_interval_times 2 --lfcf_tolerance 1e-5 --lfcf_diffscale True --absdensify"

# ============================================================
echo "================================================================"
echo "  Phase 28d — Verify CRS+SH-freeze MERGE (loo_no_shfreeze)"
echo "  Cell: $CELL"
echo "  Scenes: ${SCENES[@]}"
echo "  Started at $(date)"
echo "================================================================"

# ============================================================
run_one() {
    local GPU=$1; local SC=$2
    local OUTDIR="${OUT_ROOT}/${CELL}/A3_seed${SEED}_${SC}"
    local LOG="${LOG_ROOT}/${CELL}_${SC}.log"

    if [ -f "$LOG" ] && grep -q "Best test PSNR" "$LOG"; then
        local CACHED=$(grep "Best test PSNR" "$LOG" | grep -oE '[0-9]+\.[0-9]+' | head -1)
        echo "[$(date +%H:%M:%S)] SKIP cached ${SC} (PSNR=${CACHED}) GPU${GPU}"
        return
    fi

    echo "[$(date +%H:%M:%S)] ▶ START ${SC} GPU${GPU}"
    CUDA_VISIBLE_DEVICES=$GPU python train.py -s data/nerf_llff_data/${SC} -m "$OUTDIR" \
        --eval -r 8 --n_views 3 --random_background --iterations 10000 \
        --densify_until_iter 5000 --densify_grad_threshold 0.0005 --gaussiansN 1 \
        --sample_pseudo_interval 1 --start_sample_pseudo 500 --test_iterations 10000 \
        $BASE_FLAGS --seed $SEED > "$LOG" 2>&1

    local PSNR=$(grep "Best test PSNR" "$LOG" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    echo "[$(date +%H:%M:%S)] ✓ DONE  ${SC} PSNR=${PSNR} GPU${GPU}"
}

# ============================================================
# Worker — atomic mkdir-lock claim
# ============================================================
N_JOBS=${#SCENES[@]}

worker_front() {
    local GPU=$1
    while true; do
        local CLAIMED_IDX=""
        for IDX in $(seq 0 $((N_JOBS-1))); do
            if mkdir "${LOCK_DIR}/${IDX}" 2>/dev/null; then
                CLAIMED_IDX=$IDX
                break
            fi
        done
        [ -z "$CLAIMED_IDX" ] && break
        run_one $GPU "${SCENES[$CLAIMED_IDX]}"
    done
}

worker_back() {
    local GPU=$1
    while true; do
        local CLAIMED_IDX=""
        for IDX in $(seq $((N_JOBS-1)) -1 0); do
            if mkdir "${LOCK_DIR}/${IDX}" 2>/dev/null; then
                CLAIMED_IDX=$IDX
                break
            fi
        done
        [ -z "$CLAIMED_IDX" ] && break
        run_one $GPU "${SCENES[$CLAIMED_IDX]}"
    done
}

# Launch 2 workers parallel
worker_front 0 &
PID0=$!
worker_back 1 &
PID1=$!

echo "GPU 0 worker PID=$PID0 (front-to-back)"
echo "GPU 1 worker PID=$PID1 (back-to-front)"

wait $PID0 $PID1

echo ""
echo "================================================================"
echo "  Phase 28d FINISHED at $(date)"
echo "================================================================"

# ============================================================
# Auto-analysis
# ============================================================
python3 <<'PYEOF'
import re
from pathlib import Path

RX = re.compile(r'10000\s*\|\s*test\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*[\d.]+\s*\|\s*(\d+)')
SCENES = ['fern','flower','fortress','horns','leaves','orchids','room','trex']

# References (Phase 27 seed 42 N=8)
REF_TRIM      = {'fern': 23.829, 'flower': 21.443, 'fortress': 25.616, 'horns': 21.087,
                 'leaves': 19.395, 'orchids': 17.586, 'room': 22.968, 'trex': 23.441}
REF_NO_CRS    = {'fern': 23.819, 'flower': 21.457, 'fortress': 25.536, 'horns': 21.083,
                 'leaves': 19.438, 'orchids': 17.551, 'room': 23.036, 'trex': 23.633}
trim_mean    = sum(REF_TRIM.values()) / 8
no_crs_mean  = sum(REF_NO_CRS.values()) / 8

# Extract loo_no_shfreeze
results = {}
for sc in SCENES:
    log = Path(f"logs/p28d_verify_merge/loo_no_shfreeze_{sc}.log")
    if log.exists():
        m = RX.search(log.read_text(errors='ignore'))
        if m:
            results[sc] = {
                'psnr': float(m.group(1)),
                'ssim': float(m.group(2)),
                'lpips': float(m.group(3)),
                'ngauss': int(m.group(4)),
            }

if len(results) < 8:
    print(f"⚠️ Only {len(results)}/8 scenes completed")

# === Per-scene table ===
print("\n" + "="*80)
print("  Per-scene PSNR comparison")
print("="*80)
print(f"\n{'Scene':<10} {'trim_full':>10} {'no_shfreeze':>12} {'no_crs':>10} {'Δ_freeze':>10} {'Δ_crs':>10}")
print("-" * 70)
for sc in SCENES:
    nsf = results.get(sc, {}).get('psnr')
    row = [
        f"{sc:<10}",
        f"{REF_TRIM[sc]:>10.3f}",
        f"{nsf:>12.3f}" if nsf else f"{'---':>12}",
        f"{REF_NO_CRS[sc]:>10.3f}",
        f"{nsf - REF_TRIM[sc]:>+10.3f}" if nsf else f"{'---':>10}",
        f"{nsf - REF_NO_CRS[sc]:>+10.3f}" if nsf else f"{'---':>10}",
    ]
    print(" ".join(row))

# === Aggregate ===
psnrs = [results[sc]['psnr'] for sc in SCENES if sc in results]
if len(psnrs) == 8:
    nsf_mean = sum(psnrs) / 8
    print(f"\n{'='*60}")
    print(f"  AGGREGATE (mean across 8 scenes)")
    print(f"{'='*60}")
    print(f"  trim_full       = {trim_mean:.4f}")
    print(f"  loo_no_shfreeze = {nsf_mean:.4f}")
    print(f"  loo_no_crs      = {no_crs_mean:.4f}")
    print()
    print(f"  Δ (no_shfreeze − no_crs) = {nsf_mean - no_crs_mean:+.4f}")
    print(f"     → If ≈ 0 (within ±0.05): CRS computation has NO effect alone")
    print(f"     → If notably different: CRS has independent effect")
    print()

    diff = abs(nsf_mean - no_crs_mean)
    if diff < 0.03:
        verdict = "✅ MERGE JUSTIFIED — CRS+SH-freeze inseparable as 1 pillar"
        explanation = "SH freeze IS the dominant consumer. CRS computation alone = wash."
    elif diff < 0.06:
        verdict = "⚠️ WEAK SIGNAL — borderline merge"
        explanation = "Small independent CRS effect detected. Merge plausible but caveat needed."
    else:
        verdict = "❌ DO NOT MERGE — CRS has independent contribution"
        explanation = f"CRS computation alone contributes {nsf_mean - no_crs_mean:+.3f} dB."

    print(f"\n{'='*60}")
    print(f"  DECISION")
    print(f"{'='*60}")
    print(f"  {verdict}")
    print(f"  → {explanation}")

# === SSIM/LPIPS ===
ssims = [results[sc]['ssim'] for sc in SCENES if sc in results]
lpipss = [results[sc]['lpips'] for sc in SCENES if sc in results]
ngauss = [results[sc]['ngauss'] for sc in SCENES if sc in results]
if len(ssims) == 8:
    print(f"\n{'='*60}")
    print(f"  Full metrics — loo_no_shfreeze (mean 8 scenes)")
    print(f"{'='*60}")
    print(f"  PSNR    = {nsf_mean:.4f}")
    print(f"  SSIM    = {sum(ssims)/8:.4f}")
    print(f"  LPIPS   = {sum(lpipss)/8:.4f}")
    print(f"  N_gauss = {int(sum(ngauss)/8)}")
PYEOF
