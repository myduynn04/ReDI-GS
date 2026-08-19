#!/bin/bash
# ============================================================
# [CRSGaussian Phase 26] Ablation runner — A1 (V_stability) / C1
# (density-frequency modulation) / A2 (temporal parameter reg)
# File: scripts/p26_ablation_run.sh (NEW — mirrors scripts/trainer.sh)
#
# Leave-one-IN trên top của trim_full (production hiện tại, xem README /
# scripts/trainer.sh): mỗi CONFIG mới = trim_full + đúng 1 module Phase 26,
# để đo Δ so với baseline production thay vì so với 3DGS trần.
#
# Configs mới (ngoài các CONFIG cũ của trainer.sh vẫn hoạt động y hệt):
#   trim_full_a1       trim_full + V_stability signal (KHÔNG bật geom_freeze)
#   trim_full_a1freeze trim_full + V_stability + CRS-modulated geom freeze
#   trim_full_c1       trim_full + density-frequency modulation (DropAnSH p_sh)
#   trim_full_c1_full  trim_full_c1 + opacity-decay modulation
#   trim_full_a2       trim_full + temporal parameter regularization loss
#   trim_full_a1c1a2   trim_full + cả 3 module (A1freeze + C1_full + A2)
#
# Prerequisite: giống trainer.sh — fused.ply đã swap sang dense init mong
# muốn (RoMa v1 production) qua scripts/place_init.py.
#
# Usage (1 GPU, so từng module với trim_full baseline, 1 seed trước khi
# multi-seed đầy đủ — theo đúng thứ tự "đo trước, quyết sau" đã thống nhất):
#   GPU=0 CONFIG=trim_full        SEEDS_OVERRIDE="42" \
#       bash scripts/p26_ablation_run.sh
#   GPU=0 CONFIG=trim_full_a1     SEEDS_OVERRIDE="42" \
#       bash scripts/p26_ablation_run.sh
#   GPU=0 CONFIG=trim_full_c1     SEEDS_OVERRIDE="42" \
#       bash scripts/p26_ablation_run.sh
#   GPU=0 CONFIG=trim_full_a2     SEEDS_OVERRIDE="42" \
#       bash scripts/p26_ablation_run.sh
#
# Sau khi có tín hiệu rõ ràng (Δ đáng kể, không noise) mới chạy multi-seed
# đầy đủ (SEEDS_OVERRIDE="42 137 9999") + toàn bộ 8 scene.
# ============================================================
set -eo pipefail
GPU=${GPU:-0}
CONFIG=${CONFIG:?phải set CONFIG (trim_full/trim_full_a1/trim_full_a1freeze/trim_full_c1/trim_full_c1_full/trim_full_a2/trim_full_a1c1a2/base)}
LOG_DIR=${LOG_DIR:-logs/p26_ablation/${CONFIG}}
OUT_DIR_VAL=${OUT_DIR:-output/p26_ablation/${CONFIG}}
mkdir -p ${LOG_DIR} ${OUT_DIR_VAL}

# Protocol cố định — giống hệt trainer.sh để so sánh công bằng.
PROTOCOL="--eval -r 8 --n_views 3 --random_background --iterations 10000 \
--densify_until_iter 5000 --densify_grad_threshold 0.0005 --gaussiansN 1 \
--sample_pseudo_interval 1 --start_sample_pseudo 500 --test_iterations 10000"

# ── Module flag-strings (production, y hệt trainer.sh) ──
M_DEPTHCFG="--use_depth_prior --dav2_path ../Depth-Anything-V2 --crs_ema_decay 0.3 --crs_update_interval 100"
M_DCYCLE="--use_d_cycle --d_cycle_warmup 1000 --d_cycle_sigma 5.0 --d_cycle_update_freq 100"
M_SHFREEZE="--use_crs_modulated_sh_freeze --crs_freeze_start 1000 --crs_freeze_tau 0.65"
M_SHREL="--use_sh_reliability --sh_stability_warmup 1000 --sh_stability_ema_beta 0.95 --crs_w_s 0.33"
M_DROP="--use_dropansh --dropansh_pa 0.02 --dropansh_psh 0.2"
M_OPACITY="--use_opacity_decay --opacity_decay_factor 0.999"
M_EFA="--use_lfcf --lfcf_init_scaling_max 1.5 --lfcf_init_scaling_min 1.0 \
--lfcf_last_scaling_max 1.0 --lfcf_pow 1.0 --lfcf_splitting_ub 1.0 \
--lfcf_interval_times 2 --lfcf_tolerance 1e-5 --lfcf_diffscale True --absdensify"

CRS_TRIM="${M_DEPTHCFG} ${M_DCYCLE} ${M_SHFREEZE} ${M_SHREL}"
INDEP_FULL="${M_DROP} ${M_OPACITY} ${M_EFA}"
TRIM_FULL="${CRS_TRIM} ${INDEP_FULL}"

# ── [Phase 26] Module flag-strings MỚI ──
# A1: V_stability signal thêm vào CRS formula (crs_w_v mặc định 0.33, tự
#     normalize phần weight còn lại — xem crs_module.py update_crs()).
M_A1_SIGNAL="--use_v_stability --v_stability_warmup 1000 --crs_w_v 0.33"
# A1freeze: thêm geometric freeze (zero xyz/scaling/rotation grad khi V thấp)
# — cần M_A1_SIGNAL làm prerequisite (compute_V_stability).
M_A1_FREEZE="--use_crs_modulated_geom_freeze --geom_freeze_start 1000 --geom_freeze_tau 0.5"

# C1: density-as-frequency modulation cho DropAnSH SH-dropout probability.
M_C1_DROPOUT="--use_density_freq_modulate --density_freq_method voxel --density_freq_strength 1.0"
# C1_full: thêm modulation cho opacity-decay factor (cần use_opacity_decay bật).
M_C1_OPACITY="--opacity_decay_freq_modulate"

# A2: temporal parameter regularization loss (CRS-weighted velocity penalty).
M_A2="--use_temporal_reg --temporal_reg_start_iter 1000 --temporal_ema_beta 0.9 \
--lambda_temporal_xyz 0.01 --lambda_temporal_shape 0.01 --temporal_crs_weighted True"

case "${CONFIG}" in
  base)               FLAGS="" ;;
  trim_full)          FLAGS="${TRIM_FULL}" ;;
  trim_full_a1)       FLAGS="${TRIM_FULL} ${M_A1_SIGNAL}" ;;
  trim_full_a1freeze) FLAGS="${TRIM_FULL} ${M_A1_SIGNAL} ${M_A1_FREEZE}" ;;
  trim_full_c1)        FLAGS="${TRIM_FULL} ${M_C1_DROPOUT}" ;;
  trim_full_c1_full)   FLAGS="${TRIM_FULL} ${M_C1_DROPOUT} ${M_C1_OPACITY}" ;;
  trim_full_a2)        FLAGS="${TRIM_FULL} ${M_A2}" ;;
  trim_full_a1c1a2)    FLAGS="${TRIM_FULL} ${M_A1_SIGNAL} ${M_A1_FREEZE} ${M_C1_DROPOUT} ${M_C1_OPACITY} ${M_A2}" ;;
  *) echo "ERROR: invalid CONFIG: ${CONFIG}"; exit 1 ;;
esac

# run_one KHÔNG dùng `set -e` bên trong — 1 (scene,seed) lỗi (train.py
# crash, OOM...) không được làm mất các (scene,seed) còn lại trong cùng
# CONFIG. Lỗi được log vào FAIL_LOG thay vì abort cả script — quan trọng
# cho chạy unattended (Kaggle treo máy) qua kaggle_run_all.sh.
FAIL_LOG=${LOG_DIR}/_failed_runs.txt
> "${FAIL_LOG}"

run_one() {
    local SC=$1; local SEED=$2
    local OUTDIR=${OUT_DIR_VAL}/${CONFIG}_seed${SEED}_${SC}
    local LOG=${LOG_DIR}/${CONFIG}_seed${SEED}_${SC}.log
    echo "[$(date +%H:%M:%S)] START [${CONFIG}] seed${SEED}_${SC} (GPU${GPU})"
    set +e
    CUDA_VISIBLE_DEVICES=${GPU} python -u train.py \
        --source_path data/nerf_llff_data/${SC} \
        -m ${OUTDIR} --seed ${SEED} \
        ${PROTOCOL} ${FLAGS} \
        2>&1 | tee ${LOG}
    local RC=${PIPESTATUS[0]}
    set -e
    if [ "${RC}" -ne 0 ]; then
        echo "[$(date +%H:%M:%S)] LỖI  [${CONFIG}] seed${SEED}_${SC} (RC=${RC}) — xem ${LOG}"
        echo "${CONFIG} seed${SEED} ${SC} RC=${RC}" >> "${FAIL_LOG}"
    else
        echo "[$(date +%H:%M:%S)] DONE  [${CONFIG}] seed${SEED}_${SC}"
    fi
}

SCENES=(${SCENES_OVERRIDE:-fern flower fortress horns leaves orchids room trex})
SEEDS=(${SEEDS_OVERRIDE:-42 137 9999})

echo "=== Phase 26 ablation CONFIG=${CONFIG} GPU=${GPU} ==="
echo "    Prerequisite: fused.ply đã swap sang dense init mong muốn (place_init.py)."
echo "    FLAGS: ${FLAGS:-<none — BASE>}"
echo "    SCENES=${SCENES[*]} SEEDS=${SEEDS[*]}"
for SEED in "${SEEDS[@]}"; do
    for S in "${SCENES[@]}"; do
        run_one $S $SEED
    done
done
echo "[GPU${GPU}] Phase 26 ablation CONFIG=${CONFIG} complete."

if [ -s "${FAIL_LOG}" ]; then
    echo "CẢNH BÁO: một số run lỗi (xem ${FAIL_LOG}):"
    cat "${FAIL_LOG}"
    exit 1   # exit code phản ánh có lỗi, nhưng đã chạy hết mọi (scene,seed)
fi
