#!/bin/bash
# ============================================================
# [Phase 28] CRS BOOST experiments on RoMa v1 backbone
#
# DIAGNOSIS (from p28_diag fern run):
#   CRS distribution skewed HIGH on dense init:
#     mean=0.69, std=0.12, <0.35=0.14%, >0.65=66-69%
#   Root cause: R signal SATURATED (mean=0.89, std=0.08)
#     → R doesn't discriminate → CRS clustered HIGH
#     → SH freeze gate (tau=0.5) fires for only ~5-7% Gaussians
#     → CRS contribution wash (+0.023 dB Phase 27)
#
# GOAL: boost CRS contribution to ≥ +0.1 dB by fixing R saturation
#       and/or tuning gate threshold.
#
# 3 ABLATION CELLS (each Phase 22 recipe + 1 modification):
#
#   no_R          = trim_full + --disable_r_signal True
#                   → CRS = sigmoid(w_d × D + w_s × S)
#                   → eliminates R saturation problem
#                   → D has good spread (std=0.28) → discriminative CRS
#
#   tau65         = trim_full + --crs_freeze_tau 0.65
#                   → freeze Gaussians with CRS < 0.65 (≈31% currently)
#                   → more aggressive SH regularization
#
#   no_R_tau65    = combined both fixes
#
# REFERENCE (Phase 27 N=8 seed 42):
#   trim_full   = 21.9207
#   loo_no_crs  = 21.9441
#   Target: any cell > 22.04 → CRS contribution > 0.1 dB ⭐
#
# COST:
#   3 cells × 8 scenes × 1 seed × ~5-6 min/run = ~2.5h on 1 GPU
#   (Phase 27 trim_full re-run as control — optional, skip if cached)
#
# Usage:
#   conda activate corgs
#   cd ~/workspace/representation-3d/duyen/CoR-GS
#   chmod +x scripts/p28_crs_boost.sh
#   mkdir -p logs/p28_crs_boost
#   tmux new -s p28
#   nohup bash scripts/p28_crs_boost.sh > logs/p28_crs_boost/_driver.log 2>&1 &
#   disown
#   Ctrl-B D
#
# Monitor:
#   tail -f logs/p28_crs_boost/_driver.log
#   ls logs/p28_crs_boost/*/A3_seed42_*.log 2>/dev/null | wc -l   # target 24
# ============================================================
set +e

# ── Verify RoMa v1 init ──
FERN_PLY="data/nerf_llff_data/fern/3_views/dense/fused.ply"
FERN_V1="data/nerf_llff_data/fern/3_views/dense/fused.ply.romav1"
if [ ! -f "$FERN_V1" ]; then
    echo "❌ $FERN_V1 chưa được sinh — chạy p22_run_all_scenes.sh trước"
    exit 1
fi
if ! cmp -s "$FERN_PLY" "$FERN_V1"; then
    echo "❌ fused.ply ≠ fused.ply.romav1 — chạy p22_place_romav1_init.py trước"
    exit 1
fi
echo "✓ verified RoMa v1 init"

# ── Phase 22 protocol ──
PROTOCOL="--eval -r 8 --n_views 3 --random_background --iterations 10000 \
--densify_until_iter 5000 --densify_grad_threshold 0.0005 --gaussiansN 1 \
--sample_pseudo_interval 1 --start_sample_pseudo 500 --test_iterations 10000"

# ── Phase 22 mechanism flag groups (UNCHANGED from trim_full) ──
M_DEPTHCFG="--use_depth_prior --dav2_path ../Depth-Anything-V2 --crs_ema_decay 0.3 --crs_update_interval 100"
M_DCYCLE="--use_d_cycle --d_cycle_warmup 1000 --d_cycle_sigma 5.0 --d_cycle_update_freq 100"
M_SHREL="--use_sh_reliability --sh_stability_warmup 1000 --sh_stability_ema_beta 0.95 --crs_w_s 0.33"
M_DROP="--use_dropansh --dropansh_pa 0.02 --dropansh_psh 0.2"
M_OPACITY="--use_opacity_decay --opacity_decay_factor 0.999"
M_EFA="--use_lfcf --lfcf_init_scaling_max 1.5 --lfcf_init_scaling_min 1.0 \
--lfcf_last_scaling_max 1.0 --lfcf_pow 1.0 --lfcf_splitting_ub 1.0 \
--lfcf_interval_times 2 --lfcf_tolerance 1e-5 --lfcf_diffscale True --absdensify"

# ── SH FREEZE variants ──
M_SHFREEZE_TAU50="--use_crs_modulated_sh_freeze --crs_freeze_start 1000 --crs_freeze_tau 0.5"
M_SHFREEZE_TAU65="--use_crs_modulated_sh_freeze --crs_freeze_start 1000 --crs_freeze_tau 0.65"

# ── 3 BOOST cells ──
declare -A CELLS=(
    # Cell 1: Disable R signal (root cause fix)
    [no_R]="${M_DEPTHCFG} ${M_DCYCLE} ${M_SHFREEZE_TAU50} ${M_SHREL} --disable_r_signal ${M_DROP} ${M_OPACITY} ${M_EFA}"

    # Cell 2: Tau 0.65 (aggressive gate)
    [tau65]="${M_DEPTHCFG} ${M_DCYCLE} ${M_SHFREEZE_TAU65} ${M_SHREL} ${M_DROP} ${M_OPACITY} ${M_EFA}"

    # Cell 3: Combined (both fixes)
    [no_R_tau65]="${M_DEPTHCFG} ${M_DCYCLE} ${M_SHFREEZE_TAU65} ${M_SHREL} --disable_r_signal ${M_DROP} ${M_OPACITY} ${M_EFA}"
)

# Run order — lowest risk first
CONFIG_ORDER=(no_R tau65 no_R_tau65)

# Seed 42 only (priority)
if [ -n "${SEEDS_OVERRIDE}" ]; then
    SEEDS=(${SEEDS_OVERRIDE})
else
    SEEDS=(42)
fi
echo "✓ Seeds: ${SEEDS[@]}"
echo "✓ Cells: ${CONFIG_ORDER[@]}"

# Default GPU
GPU=${GPU:-0}
echo "✓ GPU: ${GPU}"

SCENES=(fern flower fortress horns leaves orchids room trex)

LOG_ROOT="logs/p28_crs_boost"
OUT_ROOT="output/p28_crs_boost"
mkdir -p "$LOG_ROOT"

# ── Helpers ──
is_cached() {
    local LOG=$1
    local OUTDIR=$2
    [ -f "$LOG" ] || return 1
    grep -q "Best test PSNR" "$LOG" 2>/dev/null || return 1
    [ -d "${OUTDIR}/point_cloud" ] || return 1
    return 0
}

run_one() {
    local CONFIG=$1; local SEED=$2; local SC=$3; local FLAGS=$4

    local OUTDIR=${OUT_ROOT}/${CONFIG}/A3_seed${SEED}_${SC}
    local LOGDIR=${LOG_ROOT}/${CONFIG}
    local LOG=${LOGDIR}/A3_seed${SEED}_${SC}.log
    mkdir -p "$LOGDIR"

    if is_cached "$LOG" "$OUTDIR"; then
        local PSNR=$(grep "Best test PSNR" "$LOG" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
        echo "[$(date +%H:%M:%S)] ⏩ SKIP cached ${CONFIG}/seed${SEED}_${SC} (PSNR=${PSNR})"
        return 0
    fi

    if [ -f "$LOG" ]; then
        echo "[$(date +%H:%M:%S)] ↻ RETRY ${CONFIG}/seed${SEED}_${SC}"
        rm -f "$LOG"
    fi

    echo "[$(date +%H:%M:%S)] ▶ START [${CONFIG}] seed${SEED}_${SC} GPU${GPU}"
    CUDA_VISIBLE_DEVICES=${GPU} python train.py \
        -s data/nerf_llff_data/${SC} -m ${OUTDIR} \
        ${PROTOCOL} ${FLAGS} --seed ${SEED} \
        > "$LOG" 2>&1 || echo "  ⚠ FAIL [${CONFIG}] seed${SEED}_${SC}"

    if grep -q "Best test PSNR" "$LOG" 2>/dev/null; then
        local PSNR=$(grep "Best test PSNR" "$LOG" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
        echo "[$(date +%H:%M:%S)] ✓ DONE  [${CONFIG}] seed${SEED}_${SC} (PSNR=${PSNR})"
    else
        echo "[$(date +%H:%M:%S)] ❌ FAIL [${CONFIG}] seed${SEED}_${SC} — check $LOG"
    fi
}

# ── MAIN: sequential cells × scenes × seeds ──
echo ""
echo "================================================================"
echo " Phase 28 CRS BOOST — START $(date '+%Y-%m-%d %H:%M:%S')"
echo " Diagnosis: R signal saturated → CRS clustered HIGH → gate barely fires"
echo " Goal: CRS contribution ≥ +0.1 dB via fixing R saturation"
echo "================================================================"

for CONFIG in "${CONFIG_ORDER[@]}"; do
    FLAGS="${CELLS[$CONFIG]}"
    echo ""
    echo "================================================================"
    echo " [$(date '+%H:%M:%S')] CONFIG = ${CONFIG}"
    echo "   FLAGS: ${FLAGS}"
    echo "================================================================"

    for SEED in "${SEEDS[@]}"; do
        for SC in "${SCENES[@]}"; do
            run_one "$CONFIG" "$SEED" "$SC" "$FLAGS"
        done
    done

    N_DONE=$(ls ${LOG_ROOT}/${CONFIG}/A3_seed*.log 2>/dev/null | wc -l)
    EXPECT=$((${#SEEDS[@]} * 8))
    echo "  [${CONFIG}] produced ${N_DONE} / ${EXPECT} logs"
done

echo ""
echo "================================================================"
echo " Phase 28 ALL DONE $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

# ── Quick analysis ──
python3 <<'PYEOF'
import re
from pathlib import Path

RX = re.compile(r'10000\s*\|\s*test\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)')
SCENES = ["fern","flower","fortress","horns","leaves","orchids","room","trex"]

def mean(folder, prefix="A3_seed42_"):
    psnrs, ssims, lpipss = [], [], []
    for sc in SCENES:
        log = Path(folder) / f"{prefix}{sc}.log"
        if not log.exists(): continue
        m = RX.search(log.read_text(errors='ignore'))
        if m:
            psnrs.append(float(m.group(1)))
            ssims.append(float(m.group(2)))
            lpipss.append(float(m.group(3)))
    if not psnrs: return None
    return (sum(psnrs)/len(psnrs), sum(ssims)/len(ssims), sum(lpipss)/len(lpipss), len(psnrs))

# References from Phase 27
REF_FULL = (21.9207, 0.7689, 0.1588)
REF_NOCRS = (21.9441, 0.7701, 0.1578)

print(f"\n{'='*70}")
print(f"  Phase 28 CRS BOOST — Results vs Phase 27 references")
print(f"{'='*70}")
print(f"\nReferences (Phase 27 N=8 seed 42):")
print(f"  trim_full   = PSNR {REF_FULL[0]:.4f}  SSIM {REF_FULL[1]:.4f}  LPIPS {REF_FULL[2]:.4f}")
print(f"  loo_no_crs  = PSNR {REF_NOCRS[0]:.4f}  SSIM {REF_NOCRS[1]:.4f}  LPIPS {REF_NOCRS[2]:.4f}")
print(f"  Current CRS contribution = {REF_FULL[0] - REF_NOCRS[0]:+.4f} dB (wash)")

print(f"\n{'Cell':<14} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'Δ_full':>9} {'Δ_no_crs':>10} {'CRS_contrib':>12}")
print("-" * 80)
for cell in ["no_R", "tau65", "no_R_tau65"]:
    r = mean(f"logs/p28_crs_boost/{cell}")
    if not r:
        print(f"{cell:<14} no data")
        continue
    p, s, l, n = r
    d_full = p - REF_FULL[0]
    d_no_crs = p - REF_NOCRS[0]
    contrib = p - REF_NOCRS[0]   # vs loo_no_crs = CRS contribution
    marker = "⭐ HIT!" if contrib > 0.10 else ("✓ better" if contrib > 0.02 else "wash")
    print(f"{cell:<14} {p:>8.4f} {s:>8.4f} {l:>8.4f} {d_full:>+9.4f} {d_no_crs:>+10.4f} {contrib:>+12.4f} {marker}")

print(f"\nInterpretation:")
print(f"  CRS_contrib = (variant PSNR) - (loo_no_crs PSNR)")
print(f"  = how much CRS framework would contribute if we use this variant")
print(f"  Target: CRS_contrib > 0.10 → CRS becomes meaningful pillar")
PYEOF
