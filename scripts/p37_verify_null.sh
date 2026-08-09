#!/bin/bash
# ============================================================
# [CRSGaussian Phase 37] ĐỐI CHỨNG — A1 chạy HAI LẦN, cùng seed
# File: scripts/p37_verify_null.sh  (KEEP LOCAL — server-only)
#
# VÌ SAO CẦN
#   p37_verify_inert.sh báo "BUFFER KHÔNG TRƠ" vì A2 − A1 = −953 Gaussian.
#   Kết luận đó dựa trên GIẢ ĐỊNH CHƯA ĐO: "cùng seed ⇒ N_gauss trùng khít".
#   Dự án đã biết atomicAdd gây ±1.3 dB PSNR trên 1 scene. Gradient không
#   tất định ⇒ quyết định densify không tất định ⇒ N_gauss trôi.
#
#   Script này đo PHÂN PHỐI NULL: chạy ĐÚNG MỘT CẤU HÌNH ba lần, cùng seed,
#   cùng init, không đổi một cờ nào. Mọi chênh lệch quan sát được LÀ nhiễu
#   theo định nghĩa.
#
#   Rồi so:  |A2 − A1| = 953  vs  spread của ba lần chạy giống hệt nhau.
#     nằm trong  → buffer TRƠ, verdict cũ là báo động giả
#     vượt hẳn   → buffer thật sự đổi hành vi, phải soi code
#
#   Đây đúng bài học của R1 v1 — đừng đặt ngưỡng khi chưa có đối chứng.
#
# ⚠ Dùng nhánh A1 (init _p37, flag OFF) làm cấu hình lặp, vì đó chính là
#   baseline mà A2 được so với.
#
# Chi phí: 3 run × ~5 phút = ~15 phút trên 1 GPU.
#
# Usage:
#   conda activate corgs
#   cd ~/workspace/representation-3d/duyen/CoR-GS
#   bash scripts/p37_verify_null.sh 2>&1 | tee logs/p37/verify_null.log
# ============================================================
set -u

SCENE=${SCENE:-fern}
SEED=${SEED:-42}
GPU=${GPU:-0}
N_REP=${N_REP:-3}

DATA=data/nerf_llff_data/${SCENE}/3_views/dense
PLY=${DATA}/fused.ply
PLY_ORIG=${DATA}/fused.ply.romav1
PLY_P37=${DATA}/fused.ply.romav1_p37
SAVE=${DATA}/fused.ply.p37null_backup

LOG_DIR=logs/p37
OUT_ROOT=output/p37_verify_null
mkdir -p "${LOG_DIR}" "${OUT_ROOT}"

# Recipe COPY NGUYÊN VĂN từ p37_verify_inert.sh — KHÔNG đổi một cờ.
PROTOCOL="--eval -r 8 --n_views 3 --random_background --iterations 10000 \
--densify_until_iter 5000 --densify_grad_threshold 0.0005 --gaussiansN 1 \
--sample_pseudo_interval 1 --start_sample_pseudo 500 --test_iterations 10000"
M_DEPTHCFG="--use_depth_prior --dav2_path ../Depth-Anything-V2 --crs_ema_decay 0.3 --crs_update_interval 100"
M_DCYCLE="--use_d_cycle --d_cycle_warmup 1000 --d_cycle_sigma 5.0 --d_cycle_update_freq 100"
M_SHFREEZE="--use_crs_modulated_sh_freeze --crs_freeze_start 1000 --crs_freeze_tau 0.65"
M_SHREL="--use_sh_reliability --sh_stability_warmup 1000 --sh_stability_ema_beta 0.95 --crs_w_s 0.33"
M_DROP="--use_dropansh --dropansh_pa 0.02 --dropansh_psh 0.2"
M_OPACITY="--use_opacity_decay --opacity_decay_factor 0.999"
M_EFA="--use_lfcf --lfcf_init_scaling_max 1.5 --lfcf_init_scaling_min 1.0 \
--lfcf_last_scaling_max 1.0 --lfcf_pow 1.0 --lfcf_splitting_ub 1.0 \
--lfcf_interval_times 2 --lfcf_tolerance 1e-5 --lfcf_diffscale True --absdensify"
A3_TRIM="${M_DEPTHCFG} ${M_DCYCLE} ${M_SHFREEZE} ${M_SHREL} ${M_DROP} ${M_OPACITY} ${M_EFA}"

restore_ply () {
  if [ -f "${SAVE}" ]; then
    cp "${SAVE}" "${PLY}"; rm -f "${SAVE}"
    echo "[P37 NULL] đã khôi phục ${PLY}"
  fi
}
trap restore_ply EXIT INT TERM

echo "============================================================"
echo "[P37 NULL] đối chứng — ${N_REP} lần chạy GIỐNG HỆT NHAU"
echo "  scene=${SCENE} seed=${SEED} GPU=${GPU}  $(date '+%F %T')"
echo "  cấu hình = nhánh A1 (init _p37, flag P37 TẮT)"
echo "============================================================"

for f in "${PLY}" "${PLY_ORIG}" "${PLY_P37}"; do
  [ -f "$f" ] || { echo "❌ thiếu $f"; exit 1; }
done
cp "${PLY}" "${SAVE}"
cp "${PLY_P37}" "${PLY}"
echo "✓ đặt init = fused.ply.romav1_p37"

for r in $(seq 1 ${N_REP}); do
  OUTDIR=${OUT_ROOT}/null_rep${r}_${SCENE}_seed${SEED}
  LOG=${LOG_DIR}/verify_NULL_rep${r}_${SCENE}.log
  echo
  echo "--- lần ${r}/${N_REP} ---"
  CUDA_VISIBLE_DEVICES=${GPU} python -u train.py \
      --source_path data/nerf_llff_data/${SCENE} \
      -m "${OUTDIR}" --seed ${SEED} \
      ${PROTOCOL} ${A3_TRIM} \
      > "${LOG}" 2>&1
  if [ $? -ne 0 ]; then
    echo "❌ lần ${r} FAIL — xem ${LOG}"; tail -20 "${LOG}"; exit 1
  fi
  grep -E "Best test PSNR|Final #Gaussians" "${LOG}" | sed 's/^/    /'
done

echo
echo "============================================================"
echo "[P37 NULL] PHÂN PHỐI NULL vs HIỆU ỨNG QUAN SÁT"
echo "============================================================"
python - "${LOG_DIR}" "${SCENE}" "${N_REP}" <<'PY'
import os, re, sys, statistics as st
log_dir, scene, nrep = sys.argv[1], sys.argv[2], int(sys.argv[3])

def read(p):
    if not os.path.isfile(p):
        return None, None
    t = open(p, encoding="utf-8", errors="ignore").read()
    m = re.search(r"Best test PSNR:\s*([0-9.]+)", t)
    n = re.search(r"Final #Gaussians:\s*(\d+)", t)
    return (float(m.group(1)) if m else None, int(n.group(1)) if n else None)

nulls = []
for r in range(1, nrep + 1):
    ps, ng = read(os.path.join(log_dir, f"verify_NULL_rep{r}_{scene}.log"))
    if ng is not None:
        nulls.append((ps, ng))
        print(f"  null lần {r}:  PSNR={ps:.4f}  N={ng}")

if len(nulls) < 2:
    print("\n❌ Cần ít nhất 2 lần chạy null. Dừng.")
    sys.exit(0)

ns = [n for _, n in nulls]
ps = [p for p, _ in nulls]
span_n = max(ns) - min(ns)
span_p = max(ps) - min(ps)
print()
print(f"  PHÂN PHỐI NULL (chạy giống hệt nhau):")
print(f"    N_gauss  : min={min(ns)} max={max(ns)} span={span_n} "
      f"({100.0*span_n/max(st.mean(ns),1):.2f}% của trung bình)")
print(f"    PSNR     : min={min(ps):.4f} max={max(ps):.4f} span={span_p:.4f}")

# Hiệu ứng quan sát được từ verify_inert
p1, n1 = read(os.path.join(log_dir, f"verify_A1_p37_off_{scene}.log"))
p2, n2 = read(os.path.join(log_dir, f"verify_A2_p37_on_w0_{scene}.log"))
print()
if n1 is None or n2 is None:
    print("  ⚠ thiếu log A1/A2 — chạy p37_verify_inert.sh trước để so.")
    sys.exit(0)

dn, dp = abs(n2 - n1), abs(p2 - p1)
print(f"  HIỆU ỨNG QUAN SÁT (A2 − A1):  |ΔN| = {dn}   |ΔPSNR| = {dp:.4f}")
print()
print("=" * 60)
if dn <= span_n and dp <= span_p:
    print("  ✅ BUFFER TRƠ — hiệu ứng NẰM TRONG phân phối null.")
    print("     Verdict 'KHÔNG TRƠ' của verify_inert là BÁO ĐỘNG GIẢ,")
    print("     do ngưỡng 'N_gauss phải trùng khít' chưa từng được đo.")
    print("     → Đi tiếp S2.")
elif dn <= 2 * span_n:
    print("  🟡 KHÔNG PHÂN BIỆT ĐƯỢC — hiệu ứng cùng bậc độ lớn với null,")
    print(f"     nhưng lớn hơn span quan sát ({dn} vs {span_n}).")
    print("     3 lần chạy là mẫu nhỏ, span thật có thể rộng hơn.")
    print("     → Tăng N_REP=5, HOẶC chấp nhận và dựa vào p37_verify_trace.py")
    print("       (tách muộn = nhiễu, tách ngay lần densify đầu = bug).")
else:
    print("  🔴 HIỆU ỨNG VƯỢT HẲN NULL — buffer thật sự đổi hành vi.")
    print(f"     |ΔN| = {dn} so với span null chỉ {span_n}.")
    print("     Soi lại densification_postfix + prune_points + scene/__init__.")
print("=" * 60)
PY

echo
echo "[P37 NULL] xong. fused.ply sẽ được khôi phục khi thoát."
