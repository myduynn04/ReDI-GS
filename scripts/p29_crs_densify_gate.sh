#!/bin/bash
# ============================================================
# Phase 29 REPURPOSED — no_crs vs no_crs_strict verify
# ============================================================
# Original Phase 29 (CRS-modulated densification gate) REJECTED 2026-06-26
# Code reverted. Script repurposed for new test.
#
# Purpose: Verify R compute has no side effect on training when CRS not consumed.
#          Same-batch comparison: loo_no_crs (R ON, unused) vs loo_no_crs_strict (R OFF)
#
# Cells: 2 cells × 8 scenes × seed 42 = 16 runs
#   no_crs:        Full recipe (no CRS consumer) + R compute ON (Phase 27 reproduce)
#   no_crs_strict: Same + --disable_r_signal (R compute OFF)
#
# Expected: |Δ| < 0.05 dB (within noise) → R compute no side effect confirmed
#           If |Δ| > 0.05 → R compute has subtle effect, need investigate
#
# GPU split: dynamic mkdir-lock (front/back meet in middle)
# Wall-clock: ~48 min (16 runs / 2 GPUs × ~6 min)
# ============================================================

set +e
cd ~/workspace/representation-3d/duyen/CoR-GS

# ============================================================
SEED=42
SCENES=(fern flower fortress horns leaves orchids room trex)
CELLS=(no_crs no_crs_strict)
OUT_ROOT="output/p29_nocrs_verify"
LOG_ROOT="logs/p29_nocrs_verify"
LOCK_DIR="/tmp/p29_nocrs_locks"

mkdir -p "$LOG_ROOT"
rm -rf "$LOCK_DIR"
mkdir -p "$LOCK_DIR"

# ============================================================
# COMMON: Phase 22 base, NO CRS consumer (no SH freeze, no D_cycle, no S)
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
        no_crs)
            # R compute ON (default), no consumer (no SH freeze etc.)
            # = Phase 27 loo_no_crs reproduce
            echo "$COMMON_FLAGS"
            ;;
        no_crs_strict)
            # R compute OFF + no consumer = strict no-CRS baseline
            echo "$COMMON_FLAGS --disable_r_signal"
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
echo "  Phase 29 REPURPOSED — no_crs vs no_crs_strict"
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
echo "  Phase 29 REPURPOSED DONE at $(date)"
echo "================================================================"

# ============================================================
# Auto-analysis
# ============================================================
python3 <<'PYEOF'
import re
from pathlib import Path

RX = re.compile(r'10000\s*\|\s*test\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*[\d.]+\s*\|\s*(\d+)')
SCENES = ['fern','flower','fortress','horns','leaves','orchids','room','trex']

# Reference data
REF_NOCRS_P27 = {'fern': 23.819, 'flower': 21.457, 'fortress': 25.536, 'horns': 21.083,
                 'leaves': 19.438, 'orchids': 17.551, 'room': 23.036, 'trex': 23.633}
REF_TAU65_P28 = {'fern': 23.876, 'flower': 21.549, 'fortress': 25.547, 'horns': 21.052,
                 'leaves': 19.352, 'orchids': 17.480, 'room': 23.149, 'trex': 23.700}

def extract(p):
    pp = Path(p)
    if not pp.exists(): return None
    m = RX.search(pp.read_text(errors='ignore'))
    return float(m.group(1)) if m else None

def mean(d):
    vals = [v for v in d.values() if v is not None]
    return sum(vals) / len(vals) if vals else None

# Extract NEW results
NEW = {}
for cell in ['no_crs', 'no_crs_strict']:
    NEW[cell] = {sc: extract(f"logs/p29_nocrs_verify/{cell}_{sc}.log") for sc in SCENES}

# === Per-scene table ===
print(f"\n{'='*80}")
print(f"  Phase 29 REPURPOSED — no_crs vs no_crs_strict (same batch)")
print(f"{'='*80}")
print(f"\n{'Scene':<10} {'no_crs':>10} {'no_crs_strict':>14} {'Δ (R effect)':>14}")
print("-" * 60)
for sc in SCENES:
    nc = NEW['no_crs'].get(sc)
    ncs = NEW['no_crs_strict'].get(sc)
    if nc and ncs:
        d = ncs - nc
        marker = " ⚠️" if abs(d) > 0.05 else ""
        print(f"{sc:<10} {nc:>10.4f} {ncs:>14.4f} {d:>+14.4f}{marker}")
    else:
        print(f"{sc:<10} {nc if nc else '---':>10} {ncs if ncs else '---':>14} {'---':>14}")

# === Aggregate ===
nc_m = mean(NEW['no_crs'])
ncs_m = mean(NEW['no_crs_strict'])
print("-" * 60)
if nc_m and ncs_m:
    print(f"{'MEAN':<10} {nc_m:>10.4f} {ncs_m:>14.4f} {ncs_m - nc_m:>+14.4f}")

# === Cross-reference ===
print(f"\n{'='*80}")
print(f"  CROSS-REFERENCE vs existing data")
print(f"{'='*80}")
noP27_m = mean(REF_NOCRS_P27)
tauP28_m = mean(REF_TAU65_P28)
print(f"\nPhase 27 loo_no_crs (R ON, original):   {noP27_m:.4f}")
print(f"Phase 29-RP no_crs (R ON, fresh):       {nc_m:.4f}" if nc_m else "(pending)")
print(f"Phase 29-RP no_crs_strict (R OFF):      {ncs_m:.4f}" if ncs_m else "(pending)")
print()
if nc_m:
    print(f"Reproducibility (no_crs vs P27 ref):    {nc_m - noP27_m:+.4f}")
if nc_m and ncs_m:
    print(f"R compute side effect (strict - loose): {ncs_m - nc_m:+.4f}")
print()
print(f"Phase 28 tau65 (existing reference):    {tauP28_m:.4f}")
if ncs_m:
    print(f"CRS contribution (tau65 - no_crs_strict): {tauP28_m - ncs_m:+.4f}")
if nc_m:
    print(f"CRS contribution (tau65 - no_crs loose):  {tauP28_m - nc_m:+.4f}")

# === Verdict ===
if nc_m and ncs_m:
    diff = ncs_m - nc_m
    print(f"\n{'='*60}")
    print(f"  VERDICT — R compute side effect")
    print(f"{'='*60}")
    if abs(diff) < 0.02:
        print(f"  ≈ NO SIDE EFFECT (Δ={diff:+.4f} within tight noise)")
        print(f"  → R compute confirmed truly read-only")
        print(f"  → Phase 27 loo_no_crs measurement VALID")
    elif abs(diff) < 0.05:
        print(f"  ✓ MARGINAL EFFECT (Δ={diff:+.4f} within noise)")
        print(f"  → R compute likely no real effect, jitter only")
    else:
        print(f"  ⚠️ DETECTABLE EFFECT (Δ={diff:+.4f})")
        print(f"  → R compute may have subtle side effect")
        print(f"  → Investigate cause")

print(f"\nDone.")
PYEOF
