#!/bin/bash
# ============================================================
# Phase 28c — Tau sweep on 3 diagnostic scenes
# ============================================================
# Purpose: Find optimal CRS freeze tau in [0.55, 0.64] range.
#          Approximate percentile-based freezing via uniform tau.
#
# Cells: 4 tau values × 3 scenes × seed 42 = 12 runs
#   tau55: --crs_freeze_tau 0.55   (≈ percentile 22-26%, conservative)
#   tau58: --crs_freeze_tau 0.58   (≈ percentile 26-32%, balanced)
#   tau61: --crs_freeze_tau 0.61   (≈ percentile 30-35%, moderate)
#   tau64: --crs_freeze_tau 0.64   (≈ percentile 38-44%, close to tau65)
#
# References (Phase 28 seed 42 N=8):
#   tau50 (trim_full): trex 23.441, fortress 25.616, fern 23.829
#   tau65 (winner):    trex 23.700, fortress 25.547, fern 23.876
#
# GPU split: dynamic mkdir-lock atomic claim (front/back meet in middle)
# Wall-clock: ~36 min (12 runs / 2 GPUs × ~6 min)
# ============================================================

set +e
cd ~/workspace/representation-3d/duyen/CoR-GS

# ============================================================
# Config
# ============================================================
SEED=42
SCENES=(trex fortress fern)
CELLS=(tau55 tau58 tau61 tau64)
TAUS=(0.55 0.58 0.61 0.64)
OUT_ROOT="output/p28c_tau_sweep"
LOG_ROOT="logs/p28c_tau_sweep"
LOCK_DIR="/tmp/p28c_locks"

mkdir -p "$LOG_ROOT"
rm -rf "$LOCK_DIR"
mkdir -p "$LOCK_DIR"

# ============================================================
# Base flags — full Phase 22 recipe
# ============================================================
BASE_FLAGS="--use_depth_prior --dav2_path ../Depth-Anything-V2 \
--crs_ema_decay 0.3 --crs_update_interval 100 \
--use_d_cycle --d_cycle_warmup 1000 --d_cycle_sigma 5.0 --d_cycle_update_freq 100 \
--use_crs_modulated_sh_freeze --crs_freeze_start 1000 \
--use_sh_reliability --sh_stability_warmup 1000 --sh_stability_ema_beta 0.95 --crs_w_s 0.33 \
--use_dropansh --dropansh_pa 0.02 --dropansh_psh 0.2 \
--use_opacity_decay --opacity_decay_factor 0.999 \
--use_lfcf --lfcf_init_scaling_max 1.5 --lfcf_init_scaling_min 1.0 \
--lfcf_last_scaling_max 1.0 --lfcf_pow 1.0 --lfcf_splitting_ub 1.0 \
--lfcf_interval_times 2 --lfcf_tolerance 1e-5 --lfcf_diffscale True --absdensify"

# ============================================================
# Build job list (12 jobs: cell_idx|scene_idx)
# ============================================================
JOBS=()
for CELL_IDX in 0 1 2 3; do
    for SC_IDX in 0 1 2; do
        JOBS+=("${CELL_IDX}|${SC_IDX}")
    done
done
N_JOBS=${#JOBS[@]}

echo "================================================================"
echo "  Phase 28c TAU SWEEP — ${N_JOBS} jobs"
echo "  Cells: ${CELLS[@]}"
echo "  Scenes: ${SCENES[@]}"
echo "  Seed: $SEED"
echo "  Started at $(date)"
echo "================================================================"

# ============================================================
# Run-one function
# ============================================================
run_one() {
    local GPU=$1; local CELL=$2; local TAU=$3; local SC=$4
    local OUTDIR="${OUT_ROOT}/${CELL}/A3_seed${SEED}_${SC}"
    local LOG="${LOG_ROOT}/${CELL}_${SC}.log"

    if [ -f "$LOG" ] && grep -q "Best test PSNR" "$LOG"; then
        local CACHED_PSNR=$(grep "Best test PSNR" "$LOG" | grep -oE '[0-9]+\.[0-9]+' | head -1)
        echo "[$(date +%H:%M:%S)] SKIP cached [${CELL}] ${SC} (PSNR=${CACHED_PSNR}) GPU${GPU}"
        return
    fi

    echo "[$(date +%H:%M:%S)] ▶ START [${CELL} tau=${TAU}] ${SC} GPU${GPU}"
    CUDA_VISIBLE_DEVICES=$GPU python train.py -s data/nerf_llff_data/${SC} -m "$OUTDIR" \
        --eval -r 8 --n_views 3 --random_background --iterations 10000 \
        --densify_until_iter 5000 --densify_grad_threshold 0.0005 --gaussiansN 1 \
        --sample_pseudo_interval 1 --start_sample_pseudo 500 --test_iterations 10000 \
        $BASE_FLAGS --crs_freeze_tau $TAU --seed $SEED > "$LOG" 2>&1

    local PSNR=$(grep "Best test PSNR" "$LOG" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    echo "[$(date +%H:%M:%S)] ✓ DONE  [${CELL}] ${SC} PSNR=${PSNR} GPU${GPU}"
}

# ============================================================
# Worker — atomic mkdir-lock claim
# GPU 0 scans front-to-back, GPU 1 scans back-to-front
# ============================================================
worker_front() {
    local GPU=$1
    while true; do
        local CLAIMED=""
        for IDX in $(seq 0 $((N_JOBS-1))); do
            if mkdir "${LOCK_DIR}/${IDX}" 2>/dev/null; then
                CLAIMED="${JOBS[$IDX]}"
                break
            fi
        done
        [ -z "$CLAIMED" ] && break

        IFS='|' read CELL_IDX SC_IDX <<< "$CLAIMED"
        run_one $GPU "${CELLS[$CELL_IDX]}" "${TAUS[$CELL_IDX]}" "${SCENES[$SC_IDX]}"
    done
}

worker_back() {
    local GPU=$1
    while true; do
        local CLAIMED=""
        for IDX in $(seq $((N_JOBS-1)) -1 0); do
            if mkdir "${LOCK_DIR}/${IDX}" 2>/dev/null; then
                CLAIMED="${JOBS[$IDX]}"
                break
            fi
        done
        [ -z "$CLAIMED" ] && break

        IFS='|' read CELL_IDX SC_IDX <<< "$CLAIMED"
        run_one $GPU "${CELLS[$CELL_IDX]}" "${TAUS[$CELL_IDX]}" "${SCENES[$SC_IDX]}"
    done
}

# ============================================================
# Launch 2 workers parallel
# ============================================================
worker_front 0 &
PID0=$!
worker_back 1 &
PID1=$!

echo "GPU 0 worker PID=$PID0 (front-to-back)"
echo "GPU 1 worker PID=$PID1 (back-to-front)"

wait $PID0 $PID1

echo ""
echo "================================================================"
echo "  Phase 28c FINISHED at $(date)"
echo "================================================================"

# ============================================================
# Auto-analysis
# ============================================================
python3 <<'PYEOF'
import re
from pathlib import Path

RX = re.compile(r'10000\s*\|\s*test\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*[\d.]+\s*\|\s*(\d+)')
SCENES = ['trex', 'fortress', 'fern']
CELLS = ['tau55', 'tau58', 'tau61', 'tau64']
TAUS = ['0.55', '0.58', '0.61', '0.64']

# References (Phase 27 trim_full + Phase 28 tau65 from seed 42)
REF = {
    'trex':     {'trim50': 23.441, 'tau65': 23.700},
    'fortress': {'trim50': 25.616, 'tau65': 25.547},
    'fern':     {'trim50': 23.829, 'tau65': 23.876},
}

# Extract results
results = {}
for cell in CELLS:
    results[cell] = {}
    for sc in SCENES:
        log = Path(f"logs/p28c_tau_sweep/{cell}_{sc}.log")
        if log.exists():
            m = RX.search(log.read_text(errors='ignore'))
            if m:
                results[cell][sc] = {
                    'psnr': float(m.group(1)),
                    'ssim': float(m.group(2)),
                    'lpips': float(m.group(3)),
                    'ngauss': int(m.group(4)),
                }

# === Per-scene table ===
print("\n" + "="*85)
print("  Per-scene PSNR — tau sweep")
print("="*85)
print(f"\n{'Scene':<10} {'tau=0.50':>10} {'tau=0.55':>10} {'tau=0.58':>10} {'tau=0.61':>10} {'tau=0.64':>10} {'tau=0.65':>10}")
print("-" * 75)
for sc in SCENES:
    row = [f"{sc:<10}", f"{REF[sc]['trim50']:>10.3f}"]
    for cell in CELLS:
        if sc in results.get(cell, {}):
            row.append(f"{results[cell][sc]['psnr']:>10.3f}")
        else:
            row.append(f"{'---':>10}")
    row.append(f"{REF[sc]['tau65']:>10.3f}")
    print(" ".join(row))

# === Mean 3-scene PSNR ===
print(f"\n{'='*60}")
print(f"  Mean PSNR across 3 scenes (trex + fortress + fern)")
print(f"{'='*60}")
print(f"{'Variant':<15} {'Mean PSNR':>12} {'vs tau50':>10} {'vs tau65':>10}")
print("-" * 50)

trim50_mean = sum(REF[sc]['trim50'] for sc in SCENES) / 3
tau65_mean = sum(REF[sc]['tau65'] for sc in SCENES) / 3
print(f"{'tau=0.50':<15} {trim50_mean:>12.4f} {'(ref)':>10} {tau65_mean - trim50_mean:>+10.4f}")

best_mean = -float('inf')
best_cell = None
for cell, tau in zip(CELLS, TAUS):
    psnrs = [results[cell][sc]['psnr'] for sc in SCENES if sc in results.get(cell, {})]
    if len(psnrs) == 3:
        m = sum(psnrs) / 3
        d50 = m - trim50_mean
        d65 = m - tau65_mean
        marker = ""
        if m > best_mean:
            best_mean = m
            best_cell = cell
        if d65 > 0.05:
            marker = " ⭐⭐"
        elif d65 > 0.02:
            marker = " ⭐"
        elif d65 > -0.02:
            marker = " ≈"
        elif d65 > -0.05:
            marker = " ✗"
        else:
            marker = " ✗✗"
        print(f"{f'tau={tau}':<15} {m:>12.4f} {d50:>+10.4f} {d65:>+10.4f}{marker}")

print(f"{'tau=0.65':<15} {tau65_mean:>12.4f} {tau65_mean - trim50_mean:>+10.4f} {'(ref)':>10}")

# === SSIM/LPIPS ===
print(f"\n{'='*70}")
print(f"  Full metrics (mean 3 scenes)")
print(f"{'='*70}")
print(f"{'Variant':<15} {'PSNR':>9} {'SSIM':>9} {'LPIPS':>9} {'N_gauss':>10}")
print("-" * 55)
for cell, tau in zip(CELLS, TAUS):
    psnrs = [results[cell][sc]['psnr'] for sc in SCENES if sc in results.get(cell, {})]
    ssims = [results[cell][sc]['ssim'] for sc in SCENES if sc in results.get(cell, {})]
    lpipss = [results[cell][sc]['lpips'] for sc in SCENES if sc in results.get(cell, {})]
    ngauss = [results[cell][sc]['ngauss'] for sc in SCENES if sc in results.get(cell, {})]
    if len(psnrs) == 3:
        print(f"{f'tau={tau}':<15} {sum(psnrs)/3:>9.4f} {sum(ssims)/3:>9.4f} {sum(lpipss)/3:>9.4f} {int(sum(ngauss)/3):>10}")

# === Decision ===
print(f"\n{'='*70}")
print(f"  DECISION")
print(f"{'='*70}")
if best_cell:
    d65 = best_mean - tau65_mean
    if d65 > 0.05:
        print(f"⭐⭐ BEST: {best_cell} (mean={best_mean:.4f}, beats tau65 by {d65:+.4f})")
        print(f"   → PROMISING — consider Cách A (true percentile) for novelty")
    elif d65 > 0.02:
        print(f"⭐ MARGINAL WIN: {best_cell} (mean={best_mean:.4f}, beats tau65 by {d65:+.4f})")
        print(f"   → Slight improvement, multi-seed verify recommended")
    elif d65 > -0.02:
        print(f"≈ WASH: best={best_cell} (mean={best_mean:.4f}, vs tau65 = {d65:+.4f})")
        print(f"   → tau65 likely already at sweet spot, skip Cách A")
    else:
        print(f"✗ WORSE: best variant {best_cell} below tau65 by {d65:.4f}")
        print(f"   → Tau direction may be saturated at 0.65, explore other directions")

print(f"\nDone at {Path('/tmp').stat().st_ctime}")
PYEOF
