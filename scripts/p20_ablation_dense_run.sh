#!/bin/bash
# ============================================================
# [CRSGaussian Phase 20] Dense-init ablation — LOO trên A3-TRIM
# File: scripts/p20_ablation_dense_run.sh  (TẠO MỚI — keep local)
#
# LOO ablation trên backbone PDCNet+ dense init (Phase 18) + baseline =
# A3-TRIM (Phase 20 trim verdict 2026-05-23 N=24: bỏ `informed_crs_init`
# + `use_crs_pruning` + `use_r_visible` — verified ≈0 đóng góp, no harm).
#
# Khác MVS ablation (p20_ablation_run.sh):
#   - Backbone = PDCNet+ dense (~28× COLMAP-MVS), KHÔNG phải MVS
#   - Baseline = A3-TRIM-FULL (CRS_TRIM, 3 module trim đã verified-dead)
#   - Output → logs/p20_ablation_dense/<config>/ (tách khỏi MVS)
#
# Configs (env CONFIG chọn):
#   base              3DGS thuần (0 module)
#   trim_full         A3-TRIM-FULL = CRS_TRIM + DropAnSH + opacity + EFA
#   trim_no_efa       trim_full − LFCF − AbsGS
#   trim_no_drop      trim_full − DropAnSH
#   trim_no_opacity   trim_full − opacity decay
#   trim_no_dcycle    trim_full − D_cycle
#   trim_no_shcrs     trim_full − SH-modulated freeze − SH reliability
#   trim_no_depthcrs  trim_full − depth & toàn bộ CRS_TRIM (cascade)
#
# ⚠️ PHẢI place PDCNet+ dense init TRƯỚC khi chạy:
#   SCENES="fern flower fortress horns leaves orchids room trex" \
#       python scripts/p18_gate2_place_dense_init.py
#
# Usage (2-GPU split):
#   mkdir -p logs/p20_ablation_dense
#   GPU=0 CONFIG=trim_full SCENES_OVERRIDE="fern flower fortress horns" \
#       SEEDS_OVERRIDE="42" \
#       nohup bash scripts/p20_ablation_dense_run.sh \
#       > logs/p20_ablation_dense/trim_full_gpu0.log 2>&1 & disown
#
# Phân tích:
#   ABL_ROOT=logs/p20_ablation_dense REF_LABEL=trim_full \
#       FULL_LOG=logs/p18_pilot \
#       python scripts/p20_ablation_analyze.py
#   (Δ-vs-trim_full = đóng góp riêng mỗi block trên dense-TRIM backbone)
# ============================================================
set -eo pipefail
GPU=${GPU:-0}
CONFIG=${CONFIG:?phải set CONFIG (base/trim_full/trim_no_efa/trim_no_drop/trim_no_opacity/trim_no_dcycle/trim_no_shcrs/trim_no_depthcrs)}
LOG_DIR=${LOG_DIR:-logs/p20_ablation_dense/${CONFIG}}
OUT_DIR_VAL=${OUT_DIR:-output/p20_ablation_dense/${CONFIG}}
mkdir -p ${LOG_DIR} ${OUT_DIR_VAL}

# ── PROTOCOL — cố định MỌI config (y MVS script) ──
PROTOCOL="--eval -r 8 --n_views 3 --random_background --iterations 10000 \
--densify_until_iter 5000 --densify_grad_threshold 0.0005 --gaussiansN 1 \
--sample_pseudo_interval 1 --start_sample_pseudo 500 --test_iterations 10000"

# ── Module flag-strings (verbatim từ p20_ablation_run.sh, KHÔNG có
#    M_CRSINIT / M_CRSPRUNE / M_RVIS — đã trim) ──
M_DEPTHCFG="--use_depth_prior --dav2_path ../Depth-Anything-V2 --crs_ema_decay 0.3 --crs_update_interval 100"
M_DCYCLE="--use_d_cycle --d_cycle_warmup 1000 --d_cycle_sigma 5.0 --d_cycle_update_freq 100"
M_SHFREEZE="--use_crs_modulated_sh_freeze --crs_freeze_start 1000 --crs_freeze_tau 0.5"
M_SHREL="--use_sh_reliability --sh_stability_warmup 1000 --sh_stability_ema_beta 0.95 --crs_w_s 0.33"
M_DROP="--use_dropansh --dropansh_pa 0.02 --dropansh_psh 0.2"
M_OPACITY="--use_opacity_decay --opacity_decay_factor 0.999"
M_EFA="--use_lfcf --lfcf_init_scaling_max 1.5 --lfcf_init_scaling_min 1.0 \
--lfcf_last_scaling_max 1.0 --lfcf_pow 1.0 --lfcf_splitting_ub 1.0 \
--lfcf_interval_times 2 --lfcf_tolerance 1e-5 --lfcf_diffscale True --absdensify"

# A3-TRIM = CRS framework (D_cycle + SH-freeze + S_stab) — bỏ 3 module verified-dead
CRS_TRIM="${M_DEPTHCFG} ${M_DCYCLE} ${M_SHFREEZE} ${M_SHREL}"
INDEP_FULL="${M_DROP} ${M_OPACITY} ${M_EFA}"

case "${CONFIG}" in
  base)             FLAGS="" ;;
  trim_full)        FLAGS="${CRS_TRIM} ${INDEP_FULL}" ;;
  trim_no_efa)      FLAGS="${CRS_TRIM} ${M_DROP} ${M_OPACITY}" ;;
  trim_no_drop)     FLAGS="${CRS_TRIM} ${M_OPACITY} ${M_EFA}" ;;
  trim_no_opacity)  FLAGS="${CRS_TRIM} ${M_DROP} ${M_EFA}" ;;
  trim_no_dcycle)   FLAGS="${M_DEPTHCFG} ${M_SHFREEZE} ${M_SHREL} ${INDEP_FULL}" ;;
  trim_no_shcrs)    FLAGS="${M_DEPTHCFG} ${M_DCYCLE} ${INDEP_FULL}" ;;
  trim_no_depthcrs) FLAGS="${INDEP_FULL}" ;;
  *) echo "❌ CONFIG không hợp lệ: ${CONFIG}"; exit 1 ;;
esac

run_one() {
    local SC=$1; local SEED=$2
    local OUTDIR=${OUT_DIR_VAL}/A3_seed${SEED}_${SC}
    local LOG=${LOG_DIR}/A3_seed${SEED}_${SC}.log
    echo "[$(date +%H:%M:%S)] START [${CONFIG}] seed${SEED}_${SC} (GPU${GPU}) [DENSE-TRIM]"
    CUDA_VISIBLE_DEVICES=${GPU} python -u train.py \
        --source_path data/nerf_llff_data/${SC} \
        -m ${OUTDIR} --seed ${SEED} \
        ${PROTOCOL} ${FLAGS} \
        2>&1 | tee ${LOG}
    echo "[$(date +%H:%M:%S)] DONE  [${CONFIG}] seed${SEED}_${SC} [DENSE-TRIM]"
}

SCENES=(${SCENES_OVERRIDE:-fern flower fortress horns leaves orchids room trex})
SEEDS=(${SEEDS_OVERRIDE:-42 137 9999})

echo "=== Phase 20 DENSE ablation (A3-TRIM baseline) — CONFIG=${CONFIG} (GPU${GPU}) ==="
echo "    ⚠️ init phải = PDCNet+ dense (place trước nếu chưa)"
echo "    FLAGS: ${FLAGS:-<none — BASE>}"
echo "    SCENES=${SCENES[*]} SEEDS=${SEEDS[*]}"
for SEED in "${SEEDS[@]}"; do
    for S in "${SCENES[@]}"; do
        run_one $S $SEED
    done
done
echo "[GPU${GPU}] Phase 20 DENSE ablation CONFIG=${CONFIG} complete."
