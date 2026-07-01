#!/bin/bash
# ============================================================
# Phase 28c-extended — Tau sweep ABOVE 0.65 (5 scenes)
# ============================================================
# Purpose: Test tau > 0.65 to find true upper sweet spot.
#          Phase 28c only tested below (0.55-0.64) — missed upside direction.
#
# Cells: 3 tau values × 5 scenes × seed 42 = 15 runs
#   tau70: --crs_freeze_tau 0.70  (50-63% frozen, aggressive)
#   tau75: --crs_freeze_tau 0.75  (65-78% frozen, very aggressive)
#   tau80: --crs_freeze_tau 0.80  (78-90% frozen, extreme)
#
# Scenes (spread spectrum of tau65 effect):
#   trex      tau65 +0.259  BIG WIN (thin-structure)
#   flower    tau65 +0.106  moderate win (texture)
#   fern      tau65 +0.046  neutral
#   fortress  tau65 -0.069  moderate hurt (complex)
#   orchids   tau65 -0.106  BIG hurt (very complex)
#
# References (5-scene mean seed 42):
#   tau=0.50 = (23.441 + 21.443 + 23.829 + 25.616 + 17.586) / 5 = 22.383
#   tau=0.65 = (23.700 + 21.549 + 23.876 + 25.547 + 17.480) / 5 = 22.430 (+0.047)
#
# GPU split: dynamic mkdir-lock (front/back meet in middle)
# Wall-clock: ~45 min (15 runs / 2 GPUs × ~6 min)
# ============================================================

set +e
cd ~/workspace/representation-3d/duyen/CoR-GS

# ============================================================
SEED=42
SCENES=(trex flower fern fortress orchids)
CELLS=(tau70 tau75 tau80)
TAUS=(0.70 0.75 0.80)
OUT_ROOT="output/p28c_ext_tau_high"
LOG_ROOT="logs/p28c_ext_tau_high"
LOCK_DIR="/tmp/p28c_ext_locks"

mkdir -p "$LOG_ROOT"
rm -rf "$LOCK_DIR"
mkdir -p "$LOCK_DIR"

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

# Build 15 jobs
JOBS=()
for CELL_IDX in 0 1 2; do
    for SC_IDX in 0 1 2 3 4; do
        JOBS+=("${CELL_IDX}|${SC_IDX}")
    done
done
N_JOBS=${#JOBS[@]}

echo "================================================================"
echo "  Phase 28c-EXTENDED — Tau sweep ABOVE 0.65 (5 scenes)"
echo "  Cells: ${CELLS[@]}"
echo "  Scenes: ${SCENES[@]}"
echo "  Started at $(date)"
echo "================================================================"

run_one() {
    local GPU=$1; local CELL=$2; local TAU=$3; local SC=$4
    local OUTDIR="${OUT_ROOT}/${CELL}/A3_seed${SEED}_${SC}"
    local LOG="${LOG_ROOT}/${CELL}_${SC}.log"

    if [ -f "$LOG" ] && grep -q "Best test PSNR" "$LOG"; then
        local CACHED=$(grep "Best test PSNR" "$LOG" | grep -oE '[0-9]+\.[0-9]+' | head -1)
        echo "[$(date +%H:%M:%S)] SKIP cached [${CELL}] ${SC} (PSNR=${CACHED}) GPU${GPU}"
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

worker_front 0 &
PID0=$!
worker_back 1 &
PID1=$!

echo "GPU 0 PID=$PID0 (front-to-back)"
echo "GPU 1 PID=$PID1 (back-to-front)"

wait $PID0 $PID1

echo ""
echo "================================================================"
echo "  Phase 28c-EXTENDED DONE at $(date)"
echo "================================================================"

# ============================================================
# Auto-analysis — pull references from existing logs
# ============================================================
python3 <<'PYEOF'
import re
from pathlib import Path

RX = re.compile(r'10000\s*\|\s*test\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*[\d.]+\s*\|\s*(\d+)')
SCENES = ['trex', 'flower', 'fern', 'fortress', 'orchids']

def extract(log_path):
    p = Path(log_path)
    if not p.exists():
        return None
    m = RX.search(p.read_text(errors='ignore'))
    if not m:
        return None
    return float(m.group(1))

# Reference 1: tau=0.50 from Phase 27 trim_full
REF50 = {}
for sc in SCENES:
    REF50[sc] = extract(f"logs/p27_pillar/trim_full/A3_seed42_{sc}.log")

# Reference 2: tau=0.65 from Phase 28 tau65
REF65 = {}
for sc in SCENES:
    REF65[sc] = extract(f"logs/p28_crs_boost/tau65/A3_seed42_{sc}.log")

# New: tau70, tau75, tau80
NEW = {}
for cell, tau in [('tau70', '0.70'), ('tau75', '0.75'), ('tau80', '0.80')]:
    NEW[tau] = {}
    for sc in SCENES:
        NEW[tau][sc] = extract(f"logs/p28c_ext_tau_high/{cell}_{sc}.log")

# Combined
ALL = {'0.50': REF50, '0.65': REF65, '0.70': NEW['0.70'], '0.75': NEW['0.75'], '0.80': NEW['0.80']}

# === Per-scene table ===
print(f"\n{'='*80}")
print(f"  Phase 28c FULL — Tau sweep with references (5 scenes)")
print(f"{'='*80}")
print(f"\n{'Scene':<10}", end="")
for tau in ['0.50', '0.65', '0.70', '0.75', '0.80']:
    print(f"  tau={tau:>4}", end="")
print()
print("-" * 60)
for sc in SCENES:
    print(f"{sc:<10}", end="")
    for tau in ['0.50', '0.65', '0.70', '0.75', '0.80']:
        val = ALL[tau].get(sc)
        print(f"  {val:>8.3f}" if val else f"  {'---':>8}", end="")
    print()
print("-" * 60)
print(f"{'MEAN':<10}", end="")
for tau in ['0.50', '0.65', '0.70', '0.75', '0.80']:
    vals = [ALL[tau][sc] for sc in SCENES if ALL[tau].get(sc) is not None]
    if len(vals) == 5:
        print(f"  {sum(vals)/5:>8.4f}", end="")
    else:
        print(f"  {'---':>8}", end="")
print()

# === Δ analysis ===
print(f"\n{'='*70}")
print(f"  Δ vs tau=0.50 (5-scene mean)")
print(f"{'='*70}")
ref50 = sum(ALL['0.50'][sc] for sc in SCENES if ALL['0.50'].get(sc)) / 5 if all(ALL['0.50'].get(sc) for sc in SCENES) else None
ref65 = sum(ALL['0.65'][sc] for sc in SCENES if ALL['0.65'].get(sc)) / 5 if all(ALL['0.65'].get(sc) for sc in SCENES) else None

if ref50 and ref65:
    best_tau, best_psnr = '0.65', ref65
    print(f"  tau=0.50: PSNR={ref50:.4f}  (baseline)")
    print(f"  tau=0.65: PSNR={ref65:.4f}  Δ50={ref65-ref50:+.4f}")

    for tau in ['0.70', '0.75', '0.80']:
        vals = [ALL[tau][sc] for sc in SCENES if ALL[tau].get(sc) is not None]
        if len(vals) == 5:
            m = sum(vals)/5
            d50 = m - ref50
            d65 = m - ref65
            marker = ""
            if d65 > 0.05: marker = " ⭐⭐ BEAT tau65"
            elif d65 > 0.02: marker = " ⭐ marginal beat"
            elif d65 > -0.02: marker = " ≈ wash"
            elif d65 < -0.10: marker = " ✗✗ much worse"
            else: marker = " ✗ worse"
            if m > best_psnr:
                best_psnr = m
                best_tau = tau
            print(f"  tau={tau}: PSNR={m:.4f}  Δ50={d50:+.4f}  Δ65={d65:+.4f}{marker}")

    print(f"\n{'='*70}")
    print(f"  WINNER: tau={best_tau} (PSNR={best_psnr:.4f})")
    print(f"{'='*70}")

    if best_tau == '0.65':
        print(f"\n✗ Tau axis EXHAUSTED (tau=0.65 confirmed best)")
        print(f"  → Pivot to Hướng 2 (CRS-modulated densification)")
    elif float(best_tau) > 0.65:
        delta = best_psnr - ref65
        print(f"\n⭐ NEW BEST: tau={best_tau} beats tau=0.65 by {delta:+.4f}")
        print(f"  → Multi-seed verify (8 scenes × 2 more seeds) recommended")
PYEOF
