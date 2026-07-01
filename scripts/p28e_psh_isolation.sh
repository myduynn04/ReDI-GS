#!/bin/bash
# ============================================================
# Phase 28e — SH-dropout Isolation Test (psh=0)
# ============================================================
# Purpose: Isolate CRS contribution by removing overlapping
#          SH-dropout regularization (dropansh_psh=0).
#
# Hypothesis: CRS SH-freeze and DropAnSH SH-dropout overlap in mechanism.
#            Removing SH-dropout may amplify visible CRS contribution.
#
# Cells: 3 cells × 8 scenes × seed 42 = 24 runs
#   trim_psh0:    Full recipe + --dropansh_psh 0
#   nocrs_psh0:   loo_no_crs equivalent + --dropansh_psh 0
#   tau65_psh0:   tau=0.65 + --dropansh_psh 0
#
# Reference data (existing, do not re-run):
#   trim_full (psh=0.2)  = 21.9207
#   loo_no_crs (psh=0.2) = 21.9441
#   tau65 (psh=0.2)      = 21.9630
#
# Analysis logic:
#   CRS contribution with    SH-dropout: trim_full − loo_no_crs = -0.023
#   CRS contribution without SH-dropout: trim_psh0 − nocrs_psh0 = ???
#   If > -0.023 (less negative or positive) → overlap CONFIRMED
#
# GPU split: dynamic mkdir-lock (front/back meet in middle)
# Wall-clock: ~72 min (24 runs / 2 GPUs × ~6 min)
# ============================================================

set +e
cd ~/workspace/representation-3d/duyen/CoR-GS

# ============================================================
SEED=42
SCENES=(fern flower fortress horns leaves orchids room trex)
CELLS=(trim_psh0 nocrs_psh0 tau65_psh0)
OUT_ROOT="output/p28e_psh_iso"
LOG_ROOT="logs/p28e_psh_iso"
LOCK_DIR="/tmp/p28e_locks"

mkdir -p "$LOG_ROOT"
rm -rf "$LOCK_DIR"
mkdir -p "$LOCK_DIR"

# ============================================================
# Common flags for ALL cells (recipe base — DropAnSH anchor kept, psh=0)
# Note: --dropansh_psh 0 → drop only anchors, NO SH dropout
# ============================================================
COMMON_FLAGS="--use_depth_prior --dav2_path ../Depth-Anything-V2 \
--use_dropansh --dropansh_pa 0.02 --dropansh_psh 0 \
--use_opacity_decay --opacity_decay_factor 0.999 \
--use_lfcf --lfcf_init_scaling_max 1.5 --lfcf_init_scaling_min 1.0 \
--lfcf_last_scaling_max 1.0 --lfcf_pow 1.0 --lfcf_splitting_ub 1.0 \
--lfcf_interval_times 2 --lfcf_tolerance 1e-5 --lfcf_diffscale True --absdensify"

# CRS infrastructure (used by trim_psh0 and tau65_psh0, NOT nocrs_psh0)
CRS_FLAGS="--crs_ema_decay 0.3 --crs_update_interval 100 \
--use_d_cycle --d_cycle_warmup 1000 --d_cycle_sigma 5.0 --d_cycle_update_freq 100 \
--use_crs_modulated_sh_freeze --crs_freeze_start 1000 \
--use_sh_reliability --sh_stability_warmup 1000 --sh_stability_ema_beta 0.95 --crs_w_s 0.33"

# ============================================================
# Build per-cell flag dict
# ============================================================
get_cell_flags() {
    local CELL=$1
    case $CELL in
        trim_psh0)   echo "$COMMON_FLAGS $CRS_FLAGS --crs_freeze_tau 0.5" ;;
        nocrs_psh0)  echo "$COMMON_FLAGS" ;;
        tau65_psh0)  echo "$COMMON_FLAGS $CRS_FLAGS --crs_freeze_tau 0.65" ;;
    esac
}

# ============================================================
# Build job list (24 jobs: cell_idx|scene_idx)
# ============================================================
JOBS=()
for CELL_IDX in 0 1 2; do
    for SC_IDX in 0 1 2 3 4 5 6 7; do
        JOBS+=("${CELL_IDX}|${SC_IDX}")
    done
done
N_JOBS=${#JOBS[@]}

echo "================================================================"
echo "  Phase 28e — SH-dropout Isolation Test (psh=0)"
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
# Workers — atomic mkdir-lock claim
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
echo "  Phase 28e DONE at $(date)"
echo "================================================================"

# ============================================================
# Auto-analysis
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

# Existing reference data (psh=0.2)
REF_TRIM      = {'fern': 23.829, 'flower': 21.443, 'fortress': 25.616, 'horns': 21.087,
                 'leaves': 19.395, 'orchids': 17.586, 'room': 22.968, 'trex': 23.441}
REF_NO_CRS    = {'fern': 23.819, 'flower': 21.457, 'fortress': 25.536, 'horns': 21.083,
                 'leaves': 19.438, 'orchids': 17.551, 'room': 23.036, 'trex': 23.633}
REF_TAU65     = {'fern': 23.876, 'flower': 21.549, 'fortress': 25.547, 'horns': 21.052,
                 'leaves': 19.352, 'orchids': 17.480, 'room': 23.149, 'trex': 23.700}

def mean(d):
    return sum(d.values()) / len(d) if d else None

# Extract NEW results
NEW = {}
for cell in ['trim_psh0', 'nocrs_psh0', 'tau65_psh0']:
    NEW[cell] = {}
    for sc in SCENES:
        r = extract(f"logs/p28e_psh_iso/{cell}_{sc}.log")
        if r:
            NEW[cell][sc] = r['psnr']

# === Per-scene table ===
print(f"\n{'='*100}")
print(f"  Phase 28e — SH-dropout Isolation Test (psh=0)")
print(f"{'='*100}")
print(f"\n{'Scene':<10} {'trim_full':>10} {'trim_psh0':>10} {'loo_no_crs':>11} {'nocrs_psh0':>11} {'tau65':>9} {'tau65_psh0':>11}")
print("-" * 80)
for sc in SCENES:
    row = [f"{sc:<10}", f"{REF_TRIM[sc]:>10.3f}"]
    for cell in ['trim_psh0']:
        v = NEW[cell].get(sc)
        row.append(f"{v:>10.3f}" if v else f"{'---':>10}")
    row.append(f"{REF_NO_CRS[sc]:>11.3f}")
    for cell in ['nocrs_psh0']:
        v = NEW[cell].get(sc)
        row.append(f"{v:>11.3f}" if v else f"{'---':>11}")
    row.append(f"{REF_TAU65[sc]:>9.3f}")
    for cell in ['tau65_psh0']:
        v = NEW[cell].get(sc)
        row.append(f"{v:>11.3f}" if v else f"{'---':>11}")
    print(" ".join(row))

# === Means ===
trim_mean   = mean(REF_TRIM)
nocrs_mean  = mean(REF_NO_CRS)
tau65_mean  = mean(REF_TAU65)
trim_psh0_mean   = mean(NEW['trim_psh0']) if len(NEW['trim_psh0']) == 8 else None
nocrs_psh0_mean  = mean(NEW['nocrs_psh0']) if len(NEW['nocrs_psh0']) == 8 else None
tau65_psh0_mean  = mean(NEW['tau65_psh0']) if len(NEW['tau65_psh0']) == 8 else None

print("-" * 80)
print(f"{'MEAN':<10} {trim_mean:>10.4f} ", end="")
print(f"{trim_psh0_mean:>10.4f} " if trim_psh0_mean else f"{'---':>10} ", end="")
print(f"{nocrs_mean:>11.4f} ", end="")
print(f"{nocrs_psh0_mean:>11.4f} " if nocrs_psh0_mean else f"{'---':>11} ", end="")
print(f"{tau65_mean:>9.4f} ", end="")
print(f"{tau65_psh0_mean:>11.4f}" if tau65_psh0_mean else f"{'---':>11}")

# === CRS Contribution Analysis ===
print(f"\n{'='*70}")
print(f"  CRS CONTRIBUTION COMPARISON")
print(f"{'='*70}")
print(f"\nWith SH-dropout (psh=0.2, existing data):")
print(f"  trim_full     = {trim_mean:.4f}")
print(f"  loo_no_crs    = {nocrs_mean:.4f}")
print(f"  ΔCRS (trim − no_crs) = {trim_mean - nocrs_mean:+.4f}")
print(f"  tau65         = {tau65_mean:.4f}")
print(f"  Δtau65        = {tau65_mean - nocrs_mean:+.4f}")

if all([trim_psh0_mean, nocrs_psh0_mean, tau65_psh0_mean]):
    print(f"\nWithout SH-dropout (psh=0, NEW):")
    print(f"  trim_psh0     = {trim_psh0_mean:.4f}")
    print(f"  nocrs_psh0    = {nocrs_psh0_mean:.4f}")
    crs_iso = trim_psh0_mean - nocrs_psh0_mean
    print(f"  ΔCRS (trim_psh0 − nocrs_psh0) = {crs_iso:+.4f}")
    print(f"  tau65_psh0    = {tau65_psh0_mean:.4f}")
    tau65_iso = tau65_psh0_mean - nocrs_psh0_mean
    print(f"  Δtau65        = {tau65_iso:+.4f}")

    # === VERDICT ===
    print(f"\n{'='*70}")
    print(f"  VERDICT — Overlap Hypothesis")
    print(f"{'='*70}")
    delta_crs = crs_iso - (trim_mean - nocrs_mean)
    delta_tau65 = tau65_iso - (tau65_mean - nocrs_mean)
    print(f"\n  ΔCRS amplification (no-SH-dropout − with): {delta_crs:+.4f}")
    print(f"  Δtau65 amplification:                       {delta_tau65:+.4f}")

    if delta_crs > 0.05 or delta_tau65 > 0.05:
        print(f"\n  ⭐⭐ OVERLAP CONFIRMED — CRS contribution AMPLIFIED when SH-dropout removed")
        print(f"  → Implication: CRS and SH-dropout overlap. CRS sufficient regularizer.")
        print(f"  → Recipe simplification candidate: remove SH-dropout, keep CRS.")
    elif delta_crs > 0.02 or delta_tau65 > 0.02:
        print(f"\n  ⭐ PARTIAL OVERLAP — Slight amplification")
        print(f"  → Some overlap exists but not dominant.")
    elif abs(delta_crs) < 0.02 and abs(delta_tau65) < 0.02:
        print(f"\n  ≈ ORTHOGONAL — CRS and SH-dropout independent")
        print(f"  → Keep both. They contribute independently.")
    else:
        print(f"\n  ⚠️ ANTI-OVERLAP — CRS contribution WORSE without SH-dropout")
        print(f"  → They synergize. Keep both.")

    # === Absolute PSNR comparison ===
    print(f"\n{'='*70}")
    print(f"  ABSOLUTE PSNR DROP (lose SH-dropout regularization)")
    print(f"{'='*70}")
    print(f"  trim_full → trim_psh0:    {trim_psh0_mean - trim_mean:+.4f}")
    print(f"  loo_no_crs → nocrs_psh0:  {nocrs_psh0_mean - nocrs_mean:+.4f}")
    print(f"  tau65 → tau65_psh0:        {tau65_psh0_mean - tau65_mean:+.4f}")
    print(f"  (Expected: all negative, lose +0.4 to +0.6 dB SH dropout effect)")
PYEOF
