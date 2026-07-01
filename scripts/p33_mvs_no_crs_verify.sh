#!/bin/bash
# ============================================================
# Phase 33 — MVS init: no_CRS vs no_CRS_strict + reference
# ============================================================
# Purpose: Measure composite CRS framework contribution on SPARSE MVS init.
#          Phase 20 only measured per-component LOO (SH-CRS −0.30, D_cycle −0.15).
#          This phase measures COMPOSITE LOO (entire CRS framework removed).
#
# Hypothesis: Composite CRS contribution on MVS ≈ +0.30 to +0.50 dB
#             (vs +0.06 on dense init, 5-7× reduction).
#
# Cells: 3 cells × 8 scenes × seed 42 = 24 runs
#   mvs_full:        Full Phase 13 A3 recipe (with CRS, on MVS init)
#                     Should reproduce Phase 13 A3 ≈ 21.33
#   mvs_no_crs:      Remove CRS framework (R compute ON, not consumed)
#   mvs_no_crs_strict: Same + --disable_r_signal (R compute OFF)
#
# Expected:
#   mvs_full:           ~21.33  (Phase 13 A3 reference)
#   mvs_no_crs:         ~20.9-21.1  (CRS framework removed)
#   mvs_no_crs_strict:  ≈ mvs_no_crs (R compute no side effect)
#   Δ_CRS composite:    +0.20 to +0.40 dB
#
# ⚠️ VERIFY trước khi run:
#   1. MVS init mechanism: COLMAP sparse default? Or specific flag?
#   2. Phase 13 A3 recipe flags exact match
#   3. Data path không có RoMa preprocessed PLY (để dùng COLMAP)
#
# GPU split: dynamic mkdir-lock (parallel 2 GPU)
# Wall-clock: ~72 min (24 runs / 2 GPUs × ~6 min)
# ============================================================

set +e
cd ~/workspace/representation-3d/duyen/CoR-GS

# ============================================================
SEED=42
SCENES=(fern flower fortress horns leaves orchids room trex)
CELLS=(mvs_full mvs_no_crs mvs_no_crs_strict)
OUT_ROOT="output/p33_mvs_no_crs"
LOG_ROOT="logs/p33_mvs_no_crs"
LOCK_DIR="/tmp/p33_locks"

mkdir -p "$LOG_ROOT"
rm -rf "$LOCK_DIR"
mkdir -p "$LOCK_DIR"

# ============================================================
# ⚠️ DATA PATH — VERIFY before run
# Adjust if MVS data is in different path or needs specific flag
# ============================================================
DATA_ROOT="data/nerf_llff_data"   # MUST be COLMAP sparse (no RoMa preprocessed PLY)

# ============================================================
# BASE: Phase 13 A3 recipe on MVS init (KHÔNG có RoMa dense flag)
# Includes: L_depth + DropAnSH + Opacity + LFCF + AbsGS
# CRS framework flags managed per-cell
# ============================================================
COMMON_FLAGS="--use_depth_prior --dav2_path ../Depth-Anything-V2 \
--crs_ema_decay 0.3 --crs_update_interval 100 \
--use_dropansh --dropansh_pa 0.02 --dropansh_psh 0.2 \
--use_opacity_decay --opacity_decay_factor 0.999 \
--use_lfcf --lfcf_init_scaling_max 1.5 --lfcf_init_scaling_min 1.0 \
--lfcf_last_scaling_max 1.0 --lfcf_pow 1.0 --lfcf_splitting_ub 1.0 \
--lfcf_interval_times 2 --lfcf_tolerance 1e-5 --lfcf_diffscale True --absdensify"

# CRS framework flags (ON for mvs_full, OFF for no_crs cells)
CRS_FLAGS="--use_d_cycle --d_cycle_warmup 1000 --d_cycle_sigma 5.0 --d_cycle_update_freq 100 \
--use_crs_modulated_sh_freeze --crs_freeze_start 1000 --crs_freeze_tau 0.5 \
--use_sh_reliability --sh_stability_warmup 1000 --sh_stability_ema_beta 0.95 --crs_w_s 0.33"
# Note: tau=0.5 (default) for MVS since Phase 13 A3 used default tau (tau=0.65 was Phase 28 dense-init tuning)

# ============================================================
get_cell_flags() {
    local CELL=$1
    case $CELL in
        mvs_full)
            # Phase 13 A3 reference (with full CRS framework)
            echo "$COMMON_FLAGS $CRS_FLAGS"
            ;;
        mvs_no_crs)
            # No CRS framework, R compute ON (default in update_crs but unused)
            # update_crs NOT called because use_crs_modulated_sh_freeze=OFF
            # Actually: use_depth_prior=True → update_crs IS called → R IS computed (unused)
            echo "$COMMON_FLAGS"
            ;;
        mvs_no_crs_strict)
            # No CRS + skip R computation
            echo "$COMMON_FLAGS --disable_r_signal"
            ;;
    esac
}

# ============================================================
JOBS=()
for CELL_IDX in 0 1 2; do
    for SC_IDX in 0 1 2 3 4 5 6 7; do
        JOBS+=("${CELL_IDX}|${SC_IDX}")
    done
done
N_JOBS=${#JOBS[@]}

echo "================================================================"
echo "  Phase 33 — MVS init: no_CRS verify"
echo "  Cells: ${CELLS[@]}"
echo "  Scenes: ${SCENES[@]}"
echo "  N_JOBS: $N_JOBS"
echo "  Started at $(date)"
echo "================================================================"
echo ""
echo "  ⚠️ VERIFY: data path uses COLMAP sparse (no RoMa preprocessed PLY)"
echo "     Check: ls ${DATA_ROOT}/fern/sparse/0/  (should exist)"
echo "     Check: ls ${DATA_ROOT}/fern/*.ply       (should be empty/none)"
echo ""

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
    CUDA_VISIBLE_DEVICES=$GPU python train.py -s "${DATA_ROOT}/${SC}" -m "$OUTDIR" \
        --eval -r 8 --n_views 3 --random_background --iterations 10000 \
        --densify_until_iter 5000 --densify_grad_threshold 0.0005 --gaussiansN 1 \
        --sample_pseudo_interval 1 --start_sample_pseudo 500 --test_iterations 10000 \
        $CELL_FLAGS --seed $SEED > "$LOG" 2>&1

    local PSNR=$(grep "Best test PSNR" "$LOG" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    echo "[$(date +%H:%M:%S)] ✓ DONE  [${CELL}] ${SC} PSNR=${PSNR} GPU${GPU}"
}

# ============================================================
# Workers
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
echo "  Phase 33 DONE at $(date)"
echo "================================================================"

# ============================================================
# Auto-analysis
# ============================================================
python3 <<'PYEOF'
import re
from pathlib import Path

RX = re.compile(r'10000\s*\|\s*test\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*[\d.]+\s*\|\s*(\d+)')
SCENES = ['fern','flower','fortress','horns','leaves','orchids','room','trex']

def extract(p):
    pp = Path(p)
    if not pp.exists(): return None
    m = RX.search(pp.read_text(errors='ignore'))
    if not m: return None
    return {'psnr': float(m.group(1)), 'ssim': float(m.group(2)),
            'lpips': float(m.group(3)), 'ngauss': int(m.group(4))}

def mean(d, key='psnr'):
    vals = [v[key] for v in d.values() if v is not None]
    return sum(vals)/len(vals) if vals else None

results = {}
for cell in ['mvs_full', 'mvs_no_crs', 'mvs_no_crs_strict']:
    results[cell] = {}
    for sc in SCENES:
        r = extract(f"logs/p33_mvs_no_crs/{cell}_{sc}.log")
        if r: results[cell][sc] = r

# Per-scene table
print(f"\n{'='*80}")
print(f"  Phase 33 — MVS init: CRS framework composite contribution")
print(f"{'='*80}\n")
print(f"{'Scene':<10} {'mvs_full':>10} {'mvs_no_crs':>11} {'mvs_strict':>11} {'Δ_full_vs_noCRS':>16}")
print("-" * 65)
for sc in SCENES:
    f = results['mvs_full'].get(sc, {}).get('psnr')
    nc = results['mvs_no_crs'].get(sc, {}).get('psnr')
    ncs = results['mvs_no_crs_strict'].get(sc, {}).get('psnr')
    if f and nc:
        d = f - nc
        marker = " ⭐" if d > 0.10 else ""
        print(f"{sc:<10} {f:>10.4f} {nc:>11.4f} "
              f"{ncs if ncs else '---':>11} {d:>+16.4f}{marker}")
    else:
        print(f"{sc:<10} {'---':>10} {'---':>11} {'---':>11} {'---':>16}")

# Aggregate
f_m = mean(results['mvs_full'])
nc_m = mean(results['mvs_no_crs'])
ncs_m = mean(results['mvs_no_crs_strict'])

print("-" * 65)
if f_m and nc_m:
    d_composite = f_m - nc_m
    print(f"{'MEAN':<10} {f_m:>10.4f} {nc_m:>11.4f} {ncs_m if ncs_m else '---':>11} {d_composite:>+16.4f}")

# Cross-check
print(f"\n=== Cross-reference với existing data ===")
print(f"Phase 13 A3 (MVS, N=24 multi-seed): 21.330")
if f_m:
    print(f"Phase 33 mvs_full (single seed):     {f_m:.4f}  (Δ vs Phase 13: {f_m-21.330:+.4f})")
if nc_m:
    print(f"Phase 33 mvs_no_crs:                 {nc_m:.4f}")
if ncs_m:
    print(f"Phase 33 mvs_no_crs_strict:          {ncs_m:.4f}")
    if nc_m:
        print(f"  R compute side effect:           {ncs_m-nc_m:+.4f} (within noise expected)")

# Composite CRS contribution
print(f"\n=== CRS framework composite contribution on MVS ===")
if f_m and nc_m:
    print(f"  Δ_CRS_composite = mvs_full - mvs_no_crs = {f_m - nc_m:+.4f} dB")
if f_m and ncs_m:
    print(f"  Δ_CRS_strict   = mvs_full - mvs_no_crs_strict = {f_m - ncs_m:+.4f} dB")

print(f"\nReference:")
print(f"  Phase 20 SH-CRS LOO alone:   +0.30 dB")
print(f"  Phase 20 D_cycle LOO alone:  +0.15 dB")
print(f"  Expected composite (sum):    +0.45 dB (additive estimate)")
print(f"  Dense RoMa v1 composite:     +0.06 dB (Phase 29 verified)")

print(f"\nDone.")
PYEOF
