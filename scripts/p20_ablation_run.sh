#!/bin/bash
# ============================================================
# [CRSGaussian Phase 20] Ablation study runner — LOO trên A3-FULL
# File: scripts/p20_ablation_run.sh  (TẠO MỚI — keep local)
#
# Bảng ablation kiểu Binocular3DGS Table 4 — đo đóng góp BIÊN mỗi module
# bằng Leave-One-Out: A3-FULL, và A3-FULL trừ từng block.
#
# Full factorial 2^11 bất khả → LOO (chuẩn cho method nhiều thành phần).
# Module phụ thuộc: use_depth_prior = prerequisite cả CRS block →
# config `no_depthcrs` tắt depth là cascade tắt toàn bộ CRS (note rõ).
#
# PROTOCOL = setup cố định MỌI config (KHÔNG đổi → cô lập đóng góp module).
# Mỗi config = PROTOCOL + tập module. Init = MVS (restore trước khi chạy);
# dòng "dense init" của bảng = reuse logs/p18_pilot, KHÔNG chạy ở đây.
#
# Configs (env CONFIG chọn):
#   base         3DGS thuần (0 module)
#   full         A3-FULL  (= reuse logs/p13_lfcf — chạy lại nếu muốn fresh)
#   no_efa       full − LFCF − AbsGS
#   no_drop      full − DropAnSH
#   no_opacity   full − opacity decay
#   no_dcycle    full − D_cycle
#   no_shcrs     full − CRS-modulated SH freeze − SH reliability
#   no_crsprune  full − CRS pruning − informed CRS₀ init
#   no_rvis      full − R_visible
#   no_depthcrs  full − depth-prior & TOÀN BỘ CRS block (cascade)
#
# Usage (2-GPU split):
#   SCENES="fern flower fortress horns leaves orchids room trex" RESTORE=1 \
#       python scripts/p18_gate2_place_dense_init.py        # init = MVS
#   mkdir -p logs/p20_ablation
#   GPU=0 CONFIG=no_efa SCENES_OVERRIDE="fern flower fortress horns" \
#       nohup bash scripts/p20_ablation_run.sh > logs/p20_ablation/no_efa_gpu0.log 2>&1 & disown
#   ...
# Output: output/p20_ablation/<config>/A3_seed<sd>_<scene>/
# Log:    logs/p20_ablation/<config>/A3_seed<sd>_<scene>.log
# ============================================================
set -eo pipefail
GPU=${GPU:-0}
CONFIG=${CONFIG:?phải set CONFIG (base/full/no_efa/no_drop/no_opacity/no_dcycle/no_shcrs/no_crsprune/no_rvis/no_crsprune_rvis/no_prune_rvis/no_depthcrs)}
LOG_DIR=${LOG_DIR:-logs/p20_ablation/${CONFIG}}
OUT_DIR_VAL=${OUT_DIR:-output/p20_ablation/${CONFIG}}
mkdir -p ${LOG_DIR} ${OUT_DIR_VAL}

# ── PROTOCOL — cố định MỌI config ──
PROTOCOL="--eval -r 8 --n_views 3 --random_background --iterations 10000 \
--densify_until_iter 5000 --densify_grad_threshold 0.0005 --gaussiansN 1 \
--sample_pseudo_interval 1 --start_sample_pseudo 500 --test_iterations 10000"

# ── Module flag-strings (verbatim từ A3 recipe p18b_cyclic_triage.sh) ──
M_DEPTHCFG="--use_depth_prior --dav2_path ../Depth-Anything-V2 --crs_ema_decay 0.3 --crs_update_interval 100"
M_CRSINIT="--informed_crs_init --crs_init_use_view False --crs_init_w_reproj 0.4 --crs_init_w_depth 0.6 --crs_init_w_view 0.0"
M_CRSPRUNE="--use_crs_pruning"
M_DCYCLE="--use_d_cycle --d_cycle_warmup 1000 --d_cycle_sigma 5.0 --d_cycle_update_freq 100"
M_RVIS="--use_r_visible"
M_SHFREEZE="--use_crs_modulated_sh_freeze --crs_freeze_start 1000 --crs_freeze_tau 0.5"
M_SHREL="--use_sh_reliability --sh_stability_warmup 1000 --sh_stability_ema_beta 0.95 --crs_w_s 0.33"
M_DROP="--use_dropansh --dropansh_pa 0.02 --dropansh_psh 0.2"
M_OPACITY="--use_opacity_decay --opacity_decay_factor 0.999"
M_EFA="--use_lfcf --lfcf_init_scaling_max 1.5 --lfcf_init_scaling_min 1.0 \
--lfcf_last_scaling_max 1.0 --lfcf_pow 1.0 --lfcf_splitting_ub 1.0 \
--lfcf_interval_times 2 --lfcf_tolerance 1e-5 --lfcf_diffscale True --absdensify"

# CRS block đầy đủ (mọi thứ cần depth)
CRS_FULL="${M_DEPTHCFG} ${M_CRSINIT} ${M_CRSPRUNE} ${M_DCYCLE} ${M_RVIS} ${M_SHFREEZE} ${M_SHREL}"
INDEP_FULL="${M_DROP} ${M_OPACITY} ${M_EFA}"

case "${CONFIG}" in
  base)        FLAGS="" ;;
  full)        FLAGS="${CRS_FULL} ${INDEP_FULL}" ;;
  no_efa)      FLAGS="${CRS_FULL} ${M_DROP} ${M_OPACITY}" ;;
  no_drop)     FLAGS="${CRS_FULL} ${M_OPACITY} ${M_EFA}" ;;
  no_opacity)  FLAGS="${CRS_FULL} ${M_DROP} ${M_EFA}" ;;
  no_dcycle)   FLAGS="${M_DEPTHCFG} ${M_CRSINIT} ${M_CRSPRUNE} ${M_RVIS} ${M_SHFREEZE} ${M_SHREL} ${INDEP_FULL}" ;;
  no_shcrs)    FLAGS="${M_DEPTHCFG} ${M_CRSINIT} ${M_CRSPRUNE} ${M_DCYCLE} ${M_RVIS} ${INDEP_FULL}" ;;
  no_crsprune) FLAGS="${M_DEPTHCFG} ${M_DCYCLE} ${M_RVIS} ${M_SHFREEZE} ${M_SHREL} ${INDEP_FULL}" ;;
  no_rvis)     FLAGS="${M_DEPTHCFG} ${M_CRSINIT} ${M_CRSPRUNE} ${M_DCYCLE} ${M_SHFREEZE} ${M_SHREL} ${INDEP_FULL}" ;;
  # [Phase 20 trim-check] tắt ĐỒNG THỜI CRS-prune + informed-init + R_visible
  # (3 module Phase 20 1-seed gợi ý ≈0 đóng góp). Test paired N=24 vs FULL
  # trên MVS — tăng-thật → bỏ luôn 3 module; ≈/giảm → giữ (synergy phép thử).
  no_crsprune_rvis) FLAGS="${M_DEPTHCFG} ${M_DCYCLE} ${M_SHFREEZE} ${M_SHREL} ${INDEP_FULL}" ;;
  # [Phase 20 trim-check] GIỮ informed_crs_init, chỉ tắt prune + R_visible.
  # So với no_crsprune_rvis ⟹ tách vai trò riêng của informed_init.
  no_prune_rvis)    FLAGS="${M_DEPTHCFG} ${M_CRSINIT} ${M_DCYCLE} ${M_SHFREEZE} ${M_SHREL} ${INDEP_FULL}" ;;
  no_depthcrs) FLAGS="${INDEP_FULL}" ;;
  *) echo "❌ CONFIG không hợp lệ: ${CONFIG}"; exit 1 ;;
esac

run_one() {
    local SC=$1; local SEED=$2
    local OUTDIR=${OUT_DIR_VAL}/A3_seed${SEED}_${SC}
    local LOG=${LOG_DIR}/A3_seed${SEED}_${SC}.log
    echo "[$(date +%H:%M:%S)] START [${CONFIG}] seed${SEED}_${SC} (GPU${GPU})"
    CUDA_VISIBLE_DEVICES=${GPU} python -u train.py \
        --source_path data/nerf_llff_data/${SC} \
        -m ${OUTDIR} --seed ${SEED} \
        ${PROTOCOL} ${FLAGS} \
        2>&1 | tee ${LOG}
    echo "[$(date +%H:%M:%S)] DONE  [${CONFIG}] seed${SEED}_${SC}"
}

SCENES=(${SCENES_OVERRIDE:-fern flower fortress horns leaves orchids room trex})
SEEDS=(${SEEDS_OVERRIDE:-42 137 9999})

echo "=== Phase 20 ablation — CONFIG=${CONFIG} (GPU${GPU}) ==="
echo "    FLAGS: ${FLAGS:-<none — BASE>}"
echo "    SCENES=${SCENES[*]} SEEDS=${SEEDS[*]}"
for SEED in "${SEEDS[@]}"; do
    for S in "${SCENES[@]}"; do
        run_one $S $SEED
    done
done
echo "[GPU${GPU}] Phase 20 ablation CONFIG=${CONFIG} complete."
