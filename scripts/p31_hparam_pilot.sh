#!/bin/bash
# ============================================================
# Phase 31 — Hyperparameter Sweep FULL (12 cells × 8 scenes)
# ============================================================
# Purpose: Test 12 CRS hyperparameter variants on full 8-scene LLFF.
#          Find candidates that beat Phase 28 tau65 baseline.
#
# All cells = Phase 28 tau65 best recipe + 1 hyperparameter changed.
# Reference baseline = Phase 28 tau65 (existing data 21.9631).
#
# Cells: 12 variants × 8 scenes × seed 42 = 96 runs
#
#   freeze_start (4 values): 500, 1500, 2000, 3000  (default 1000)
#     fs500:  --crs_freeze_start 500   (early freeze, CRS chưa stable)
#     fs1500: --crs_freeze_start 1500  (delay freeze)
#     fs2000: --crs_freeze_start 2000  (delay more)
#     fs3000: --crs_freeze_start 3000  (very late freeze)
#
#   ema_decay (4 values): 0.1, 0.5, 0.7, 0.9  (default 0.3)
#     ema01:  --crs_ema_decay 0.1     (very fast update, mostly new)
#     ema05:  --crs_ema_decay 0.5     (balanced)
#     ema07:  --crs_ema_decay 0.7     (smoother)
#     ema09:  --crs_ema_decay 0.9     (very smooth, mostly old)
#
#   w_s (4 values): 0.0, 0.15, 0.5, 0.7  (default 0.33)
#     ws00:   --crs_w_s 0.0   (drop S signal)
#     ws015:  --crs_w_s 0.15  (S de-weighted)
#     ws05:   --crs_w_s 0.5   (S amplified)
#     ws07:   --crs_w_s 0.7   (S dominant)
#
# Reference (Phase 28 tau65 8-scene mean): 21.9631
#
# GPU split: dynamic mkdir-lock front/back (parallel 2 GPU)
# Wall-clock: ~288 min = ~4.8h (96 runs / 2 GPUs × ~6 min)
#
# Code change: NONE (existing flags only)
# ============================================================

set +e
cd ~/workspace/representation-3d/duyen/CoR-GS

# ============================================================
SEED=42
SCENES=(fern flower fortress horns leaves orchids room trex)
CELLS=(fs500 fs1500 fs2000 fs3000 ema01 ema05 ema07 ema09 ws00 ws015 ws05 ws07)
OUT_ROOT="output/p31_hparam_sweep"
LOG_ROOT="logs/p31_hparam_sweep"
LOCK_DIR="/tmp/p31_locks"

mkdir -p "$LOG_ROOT"
rm -rf "$LOCK_DIR"
mkdir -p "$LOCK_DIR"

# ============================================================
# BASE: Phase 28 tau65 best recipe
# ============================================================
BASE_TAU65="--use_depth_prior --dav2_path ../Depth-Anything-V2 \
--crs_ema_decay 0.3 --crs_update_interval 100 \
--use_d_cycle --d_cycle_warmup 1000 --d_cycle_sigma 5.0 --d_cycle_update_freq 100 \
--use_crs_modulated_sh_freeze --crs_freeze_start 1000 --crs_freeze_tau 0.65 \
--use_sh_reliability --sh_stability_warmup 1000 --sh_stability_ema_beta 0.95 --crs_w_s 0.33 \
--use_dropansh --dropansh_pa 0.02 --dropansh_psh 0.2 \
--use_opacity_decay --opacity_decay_factor 0.999 \
--use_lfcf --lfcf_init_scaling_max 1.5 --lfcf_init_scaling_min 1.0 \
--lfcf_last_scaling_max 1.0 --lfcf_pow 1.0 --lfcf_splitting_ub 1.0 \
--lfcf_interval_times 2 --lfcf_tolerance 1e-5 --lfcf_diffscale True --absdensify"

# ============================================================
# Per-cell hyperparameter override (argparse last-wins)
# ============================================================
get_cell_flags() {
    local CELL=$1
    case $CELL in
        # freeze_start sweep
        fs500)    echo "$BASE_TAU65 --crs_freeze_start 500" ;;
        fs1500)   echo "$BASE_TAU65 --crs_freeze_start 1500" ;;
        fs2000)   echo "$BASE_TAU65 --crs_freeze_start 2000" ;;
        fs3000)   echo "$BASE_TAU65 --crs_freeze_start 3000" ;;
        # ema_decay sweep
        ema01)    echo "$BASE_TAU65 --crs_ema_decay 0.1" ;;
        ema05)    echo "$BASE_TAU65 --crs_ema_decay 0.5" ;;
        ema07)    echo "$BASE_TAU65 --crs_ema_decay 0.7" ;;
        ema09)    echo "$BASE_TAU65 --crs_ema_decay 0.9" ;;
        # w_s sweep
        ws00)     echo "$BASE_TAU65 --crs_w_s 0.0" ;;
        ws015)    echo "$BASE_TAU65 --crs_w_s 0.15" ;;
        ws05)     echo "$BASE_TAU65 --crs_w_s 0.5" ;;
        ws07)     echo "$BASE_TAU65 --crs_w_s 0.7" ;;
    esac
}

# ============================================================
# Build 96 jobs (12 cells × 8 scenes)
# ============================================================
JOBS=()
for CELL_IDX in $(seq 0 11); do
    for SC_IDX in 0 1 2 3 4 5 6 7; do
        JOBS+=("${CELL_IDX}|${SC_IDX}")
    done
done
N_JOBS=${#JOBS[@]}

echo "================================================================"
echo "  Phase 31 — Hyperparameter Sweep FULL (12 cells × 8 scenes)"
echo "  Cells: ${CELLS[@]}"
echo "  N_JOBS: $N_JOBS"
echo "  Estimated wall-clock: ~288 min (~4.8h)"
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
echo "  Phase 31 DONE at $(date)"
echo "================================================================"

# ============================================================
# Auto-analysis (full 8-scene comparison, 12 cells)
# ============================================================
python3 <<'PYEOF'
import re
from pathlib import Path

RX = re.compile(r'10000\s*\|\s*test\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*[\d.]+\s*\|\s*(\d+)')
SCENES = ['fern','flower','fortress','horns','leaves','orchids','room','trex']
CELLS = ['fs500', 'fs1500', 'fs2000', 'fs3000',
         'ema01', 'ema05', 'ema07', 'ema09',
         'ws00', 'ws015', 'ws05', 'ws07']

REF_TAU65_PSNR = {'fern': 23.876, 'flower': 21.549, 'fortress': 25.547, 'horns': 21.052,
                  'leaves': 19.352, 'orchids': 17.480, 'room': 23.149, 'trex': 23.700}
REF_TAU65_SSIM = {'fern': 0.7974, 'flower': 0.6947, 'fortress': 0.8368, 'horns': 0.7738,
                  'leaves': 0.7276, 'orchids': 0.5832, 'room': 0.8828, 'trex': 0.8659}
REF_TAU65_LPIPS = {'fern': 0.1408, 'flower': 0.2092, 'fortress': 0.1185, 'horns': 0.1840,
                   'leaves': 0.1642, 'orchids': 0.2034, 'room': 0.1247, 'trex': 0.1147}

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
for cell in CELLS:
    results[cell] = {}
    for sc in SCENES:
        r = extract(f"logs/p31_hparam_sweep/{cell}_{sc}.log")
        if r: results[cell][sc] = r

ref_psnr = sum(REF_TAU65_PSNR.values()) / 8
ref_ssim = sum(REF_TAU65_SSIM.values()) / 8
ref_lpips = sum(REF_TAU65_LPIPS.values()) / 8

# === Per-cell brief table ===
print(f"\n{'='*90}")
print(f"  Phase 31 — Hyperparameter Sweep FULL (12 cells × 8 scenes)")
print(f"  Reference Phase 28 tau65: PSNR={ref_psnr:.4f}")
print(f"{'='*90}\n")

print(f"{'Cell':<8} {'PSNR':>9} {'SSIM':>9} {'LPIPS':>9} {'N_gauss':>10} {'ΔPSNR':>10} {'Verdict':>10}")
print("-" * 70)
print(f"{'tau65':<8} {ref_psnr:>9.4f} {ref_ssim:>9.4f} {ref_lpips:>9.4f} {'(ref)':>10} {'(ref)':>10} {'baseline':>10}")
print("-" * 70)

best_cell, best_psnr = None, -float('inf')
for cell in CELLS:
    mp = mean(results[cell], 'psnr')
    ms = mean(results[cell], 'ssim')
    ml = mean(results[cell], 'lpips')
    mn = mean(results[cell], 'ngauss')
    if mp is None:
        print(f"{cell:<8} {'incomplete':>9}")
        continue
    d = mp - ref_psnr
    marker = ""
    if d > 0.05: marker = "⭐⭐ WIN"
    elif d > 0.02: marker = "⭐ marg"
    elif d > -0.02: marker = "≈ wash"
    elif d > -0.05: marker = "✗ hurt"
    else: marker = "✗✗ bad"
    if mp > best_psnr:
        best_psnr, best_cell = mp, cell
    print(f"{cell:<8} {mp:>9.4f} {ms:>9.4f} {ml:>9.4f} {int(mn):>10} {d:>+10.4f} {marker:>10}")

# === Per-group best ===
print(f"\n{'='*60}")
print(f"  Per-group winners")
print(f"{'='*60}")
groups = {
    'freeze_start': ['fs500','fs1500','fs2000','fs3000'],
    'ema_decay':    ['ema01','ema05','ema07','ema09'],
    'w_s':          ['ws00','ws015','ws05','ws07'],
}
for g, cells in groups.items():
    best_in_group = None
    best_in_group_psnr = -float('inf')
    print(f"\n  {g}:")
    for c in cells:
        m = mean(results[c], 'psnr')
        if m is None:
            print(f"    {c:<8} incomplete")
            continue
        d = m - ref_psnr
        print(f"    {c:<8} PSNR={m:.4f} (Δ {d:+.4f})")
        if m > best_in_group_psnr:
            best_in_group_psnr, best_in_group = m, c
    if best_in_group:
        d = best_in_group_psnr - ref_psnr
        print(f"  → BEST in group: {best_in_group} (Δ {d:+.4f})")

# === VERDICT ===
print(f"\n{'='*60}")
print(f"  OVERALL VERDICT")
print(f"{'='*60}")
if best_cell:
    d = best_psnr - ref_psnr
    if d > 0.05:
        print(f"⭐⭐ STRONG WIN: {best_cell} ({best_psnr:.4f}, +{d:.4f} vs tau65)")
        print(f"   → Multi-seed verify (N=24) recommended")
        print(f"   → Possible to stack with other group winners")
    elif d > 0.02:
        print(f"⭐ MARGINAL: {best_cell} ({best_psnr:.4f}, +{d:.4f})")
        print(f"   → Within noise floor, multi-seed needed")
    elif d > -0.02:
        print(f"≈ WASH: best={best_cell} ({d:+.4f})")
        print(f"   → tau65 defaults already optimal")
    else:
        print(f"✗ ALL HURT: best={best_cell} ({d:+.4f})")
        print(f"   → tau65 confirmed locally optimal")

print(f"\nDone.")
PYEOF
