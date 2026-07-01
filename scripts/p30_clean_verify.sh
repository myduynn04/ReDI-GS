#!/bin/bash
# ============================================================
# Phase 30 — Clean CRS Contribution Verify (loo_strict vs tau65)
# ============================================================
# Purpose: Run loo_no_crs_strict + tau65 cùng batch để get apples-to-apples
#          comparison, verify +0.019 CRS contribution number.
#
# Hypothesis: loo_no_crs_strict (no CRS consumer + R compute off) gives
#             clean baseline. Δ vs tau65 = pure CRS framework contribution.
#
# Cells: 2 cells × 8 scenes × seed 42 = 16 runs
#   loo_no_crs_strict:   No CRS consumer + --disable_r_signal
#   tau65 (re-run):      Full Phase 22 + tau=0.65 SH freeze
#
# Existing references (compare with):
#   tau65 (Phase 28):     21.9631  (single batch from Phase 28)
#   loo_no_crs (P27):     21.9441  (R compute ON, not used)
#
# GPU split: dynamic mkdir-lock (front/back meet in middle)
# Wall-clock: ~48 min (16 runs / 2 GPUs × ~6 min)
# ============================================================

set +e
cd ~/workspace/representation-3d/duyen/CoR-GS

# ============================================================
SEED=42
SCENES=(fern flower fortress horns leaves orchids room trex)
CELLS=(loo_no_crs_strict tau65_verify)
OUT_ROOT="output/p30_clean_verify"
LOG_ROOT="logs/p30_clean_verify"
LOCK_DIR="/tmp/p30_locks"

mkdir -p "$LOG_ROOT"
rm -rf "$LOCK_DIR"
mkdir -p "$LOCK_DIR"

# ============================================================
# COMMON: Phase 22 base (no CRS-specific flags, no SH freeze)
# All cells include: L_depth + DropAnSH + Opacity + EFA(LFCF+AbsGS)
# ============================================================
COMMON_FLAGS="--use_depth_prior --dav2_path ../Depth-Anything-V2 \
--crs_ema_decay 0.3 --crs_update_interval 100 \
--use_dropansh --dropansh_pa 0.02 --dropansh_psh 0.2 \
--use_opacity_decay --opacity_decay_factor 0.999 \
--use_lfcf --lfcf_init_scaling_max 1.5 --lfcf_init_scaling_min 1.0 \
--lfcf_last_scaling_max 1.0 --lfcf_pow 1.0 --lfcf_splitting_ub 1.0 \
--lfcf_interval_times 2 --lfcf_tolerance 1e-5 --lfcf_diffscale True --absdensify"

# ============================================================
# Per-cell flags
# ============================================================
get_cell_flags() {
    local CELL=$1
    case $CELL in
        loo_no_crs_strict)
            # NO CRS consumer + skip R compute
            echo "$COMMON_FLAGS --disable_r_signal"
            ;;
        tau65_verify)
            # Full Phase 28 best recipe (tau=0.65 SH freeze)
            echo "$COMMON_FLAGS \
--use_d_cycle --d_cycle_warmup 1000 --d_cycle_sigma 5.0 --d_cycle_update_freq 100 \
--use_crs_modulated_sh_freeze --crs_freeze_start 1000 --crs_freeze_tau 0.65 \
--use_sh_reliability --sh_stability_warmup 1000 --sh_stability_ema_beta 0.95 --crs_w_s 0.33"
            ;;
    esac
}

# ============================================================
# Build 16 jobs
# ============================================================
JOBS=()
for CELL_IDX in 0 1; do
    for SC_IDX in 0 1 2 3 4 5 6 7; do
        JOBS+=("${CELL_IDX}|${SC_IDX}")
    done
done
N_JOBS=${#JOBS[@]}

echo "================================================================"
echo "  Phase 30 — Clean CRS Contribution Verify"
echo "  Cells: ${CELLS[@]}"
echo "  Scenes: ${SCENES[@]}"
echo "  N_JOBS: $N_JOBS"
echo "  Started at $(date)"
echo "================================================================"

# ============================================================
run_one() {
    local GPU=$1; local CELL=$2; local SC=$3
    local OUTDIR="${OUT_ROOT}/${CELL}/A3_seed${SEED}_${SC}"
    local LOG="${LOG_ROOT}/${CELL}_${SC}.log"

    if [ -f "$LOG" ] && grep -q "Best test PSNR" "$LOG"; then
        local CACHED=$(grep "Best test PSNR" "$LOG" | grep -oE '[0-9]+\.[0-9]+' | head -1)
        echo "[$(date +%H:%M:%S)] SKIP cached [${CELL}] ${SC} (PSNR=${CACHED}) GPU${GPU}"
        return
    fi

    local CELL_FLAGS=$(get_cell_flags "$CELL")

    echo "[$(date +%H:%M:%S)] ▶ START [${CELL}] ${SC} GPU${GPU}"
    CUDA_VISIBLE_DEVICES=$GPU python train.py -s data/nerf_llff_data/${SC} -m "$OUTDIR" \
        --eval -r 8 --n_views 3 --random_background --iterations 10000 \
        --densify_until_iter 5000 --densify_grad_threshold 0.0005 --gaussiansN 1 \
        --sample_pseudo_interval 1 --start_sample_pseudo 500 --test_iterations 10000 \
        $CELL_FLAGS --seed $SEED > "$LOG" 2>&1

    local PSNR=$(grep "Best test PSNR" "$LOG" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    echo "[$(date +%H:%M:%S)] ✓ DONE  [${CELL}] ${SC} PSNR=${PSNR} GPU${GPU}"
}

# ============================================================
# Workers — atomic mkdir-lock
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
        run_one $GPU "${CELLS[$CELL_IDX]}" "${SCENES[$SC_IDX]}"
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
        run_one $GPU "${CELLS[$CELL_IDX]}" "${SCENES[$SC_IDX]}"
    done
}

worker_front 0 &
PID0=$!
worker_back 1 &
PID1=$!

echo "GPU 0 PID=$PID0 (front-to-back)"
echo "GPU 1 PID=$PID1 (back-to-front)"

wait $PID0 $PID1

echo ""
echo "================================================================"
echo "  Phase 30 DONE at $(date)"
echo "================================================================"

# ============================================================
# Auto-analysis — clean apples-to-apples CRS contribution
# ============================================================
python3 <<'PYEOF'
import re
from pathlib import Path

RX = re.compile(r'10000\s*\|\s*test\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*[\d.]+\s*\|\s*(\d+)')
SCENES = ['fern','flower','fortress','horns','leaves','orchids','room','trex']

def extract(log_path):
    p = Path(log_path)
    if not p.exists():
        return None
    m = RX.search(p.read_text(errors='ignore'))
    if not m:
        return None
    return {'psnr': float(m.group(1)), 'ssim': float(m.group(2)),
            'lpips': float(m.group(3)), 'ngauss': int(m.group(4))}

# Existing references
REF_TRIM = {'fern': 23.829, 'flower': 21.443, 'fortress': 25.616, 'horns': 21.087,
            'leaves': 19.395, 'orchids': 17.586, 'room': 22.968, 'trex': 23.441}
REF_NOCRS_P27 = {'fern': 23.819, 'flower': 21.457, 'fortress': 25.536, 'horns': 21.083,
                 'leaves': 19.438, 'orchids': 17.551, 'room': 23.036, 'trex': 23.633}
REF_TAU65_P28 = {'fern': 23.876, 'flower': 21.549, 'fortress': 25.547, 'horns': 21.052,
                 'leaves': 19.352, 'orchids': 17.480, 'room': 23.149, 'trex': 23.700}

def mean(d):
    if not d:
        return None
    vals = [v for v in d.values() if v is not None]
    return sum(vals) / len(vals) if vals else None

# Extract new results
NEW = {}
for cell in ['loo_no_crs_strict', 'tau65_verify']:
    NEW[cell] = {}
    for sc in SCENES:
        r = extract(f"logs/p30_clean_verify/{cell}_{sc}.log")
        if r:
            NEW[cell][sc] = r['psnr']

# === Per-scene table ===
print(f"\n{'='*100}")
print(f"  Phase 30 — Clean CRS Contribution Verify (same batch)")
print(f"{'='*100}")
print(f"\n{'Scene':<10} {'no_crs_strict':>13} {'tau65_verify':>13} {'Δ (CRS contrib)':>16}")
print(f"           {'(NO CRS)':>13} {'(WITH CRS)':>13}")
print("-" * 60)
for sc in SCENES:
    nc = NEW['loo_no_crs_strict'].get(sc)
    tv = NEW['tau65_verify'].get(sc)
    if nc and tv:
        d = tv - nc
        marker = " ✓" if d > 0.02 else ("" if abs(d) < 0.02 else " ✗")
        print(f"{sc:<10} {nc:>13.3f} {tv:>13.3f} {d:>+16.4f}{marker}")
    else:
        print(f"{sc:<10} {'---' if nc is None else f'{nc:.3f}':>13} "
              f"{'---' if tv is None else f'{tv:.3f}':>13} {'---':>16}")

# === Aggregate ===
nc_m = mean(NEW['loo_no_crs_strict'])
tv_m = mean(NEW['tau65_verify'])
print("-" * 60)
print(f"{'MEAN':<10} {nc_m:>13.4f} {tv_m:>13.4f} {tv_m - nc_m:>+16.4f}")

# === Cross-reference với existing data ===
print(f"\n{'='*80}")
print(f"  CROSS-REFERENCE — clean comparison vs existing data")
print(f"{'='*80}")
trim_m = mean(REF_TRIM)
noP27_m = mean(REF_NOCRS_P27)
tauP28_m = mean(REF_TAU65_P28)

print(f"\n{'Source':<35} {'PSNR':>10}    {'Note':<30}")
print("-" * 80)
print(f"{'Existing Phase 27 loo_no_crs':<35} {noP27_m:>10.4f}    {'R compute ON, not used'}")
print(f"{'NEW Phase 30 loo_no_crs_strict':<35} {nc_m:>10.4f}    {'R compute OFF (strict)'}")
print(f"  → Δ:                          {nc_m - noP27_m:>+10.4f}    Within noise?")
print()
print(f"{'Existing Phase 28 tau65':<35} {tauP28_m:>10.4f}    {'Single batch reference'}")
print(f"{'NEW Phase 30 tau65_verify':<35} {tv_m:>10.4f}    {'Re-run for batch parity'}")
print(f"  → Δ:                          {tv_m - tauP28_m:>+10.4f}    Reproducibility check")

# === CRS contribution analysis ===
print(f"\n{'='*80}")
print(f"  CRS CONTRIBUTION — Clean apples-to-apples (same batch)")
print(f"{'='*80}")
clean_delta = tv_m - nc_m
print(f"\n  loo_no_crs_strict (NO CRS):  {nc_m:.4f}")
print(f"  tau65_verify (WITH CRS):     {tv_m:.4f}")
print(f"  Δ CLEAN CRS contribution:    {clean_delta:+.4f}")
print()
print(f"  Existing reference (cross-batch):")
print(f"    tau65 (P28) − loo_no_crs (P27) = {tauP28_m - noP27_m:+.4f}")
print()
print(f"  Difference (clean - cross-batch): {clean_delta - (tauP28_m - noP27_m):+.4f}")

# === Verdict ===
print(f"\n{'='*80}")
print(f"  VERDICT")
print(f"{'='*80}")
if clean_delta > 0.05:
    print(f"  ⭐ CRS contribution = {clean_delta:+.4f} (POSITIVE significant)")
    print(f"     → CRS framework worth keeping in recipe")
elif clean_delta > 0.01:
    print(f"  ✓ CRS contribution = {clean_delta:+.4f} (positive marginal)")
    print(f"     → CRS framework contributes small but real benefit")
elif abs(clean_delta) < 0.01:
    print(f"  ≈ CRS contribution = {clean_delta:+.4f} (WASH)")
    print(f"     → CRS framework essentially neutral on dense init")
else:
    print(f"  ✗ CRS contribution = {clean_delta:+.4f} (NEGATIVE)")
    print(f"     → CRS framework hurts! Consider removing.")

# === Per-scene heterogeneity ===
print(f"\n  Per-scene contribution heterogeneity:")
wins = 0; losses = 0; wash = 0
for sc in SCENES:
    nc = NEW['loo_no_crs_strict'].get(sc)
    tv = NEW['tau65_verify'].get(sc)
    if nc and tv:
        d = tv - nc
        if d > 0.05: wins += 1
        elif d < -0.05: losses += 1
        else: wash += 1
print(f"    Wins (>+0.05):  {wins}/8 scenes")
print(f"    Losses (<-0.05): {losses}/8 scenes")
print(f"    Wash (±0.05):   {wash}/8 scenes")

print(f"\nDone.")
PYEOF
