#!/bin/bash
# ============================================================
# [CRSGaussian Phase 37] VERIFY buffer _q_init TRƠ — 3 nhánh trên fern
# File: scripts/p37_verify_inert.sh  (KEEP LOCAL — server-only)
#
# MỤC ĐÍCH (Quy tắc 11 — verify flag-OFF trước khi tin bất cứ kết quả nào)
#   Code Phase 37 đụng 5 file production. Phải chứng minh hai điều TÁCH BẠCH,
#   nếu không thì mọi số của S2/S3 sau này đều bị nhiễm mà không biết.
#
#   A0  init GỐC (fused.ply.romav1)  + flag OFF
#       → chứng minh CODE MỚI KHÔNG ĐỘNG GÌ khi tắt. So với anchor Phase 28.
#
#   A1  init _p37                    + flag OFF
#       → cô lập ẢNH HƯỞNG CỦA INIT. Đây cũng là "R2 rẻ" cho 1 scene:
#         init dựng lại có cho PSNR tương đương không.
#
#   A2  init _p37                    + flag ON, w_Q = 0
#       → chứng minh BUFFER TRƠ THẬT. A2 phải BẰNG A1.
#         Đây là nhánh quan trọng nhất. Buffer đi qua prune/densify, được
#         dump ra q_init.npz, nhưng KHÔNG được đụng vào logit khi w_Q=0.
#
#   Đọc kết quả:
#     |A0 − 23.88| nhỏ  → code sạch
#     |A2 − A1|    ~0    → buffer trơ  ← QUAN TRỌNG NHẤT
#     A1 vs A0           → chênh lệch do init, thông tin phụ
#
# ⚠ SÀN NHIỄU: atomicAdd cho ±1.3 dB trên MỘT scene MỘT seed. Nên |A2−A1|
#   nhỏ KHÔNG chứng minh trơ tuyệt đối. Bằng chứng mạnh hơn là N_gauss cuối
#   và số iter — script in cả hai. Trơ thật thì N_gauss phải BẰNG NHAU
#   CHÍNH XÁC (cùng seed, cùng init, cùng luồng RNG).
#
# ⚠ SCRIPT CÓ SWAP fused.ply. Dùng cp (không mv), backup trước, trap khôi
#   phục kể cả khi lỗi. fused.ply.romav1 và fused.ply.romav1_p37 KHÔNG BAO
#   GIỜ bị ghi.
#
# Pre-requisite:
#   conda activate corgs
#   cd ~/workspace/representation-3d/duyen/CoR-GS
#
# Usage:
#   bash scripts/p37_verify_inert.sh 2>&1 | tee logs/p37/verify_inert.log
# ============================================================
set -u

SCENE=${SCENE:-fern}
SEED=${SEED:-42}
GPU=${GPU:-0}
DATA=data/nerf_llff_data/${SCENE}/3_views/dense
PLY=${DATA}/fused.ply
PLY_ORIG=${DATA}/fused.ply.romav1
PLY_P37=${DATA}/fused.ply.romav1_p37
SAVE=${DATA}/fused.ply.p37verify_backup

LOG_DIR=logs/p37
OUT_ROOT=output/p37_verify
mkdir -p "${LOG_DIR}" "${OUT_ROOT}"

# ── Phase 22 protocol + A3-TRIM recipe — COPY NGUYÊN VĂN từ p24_postcleanup_verify.sh ──
# KHÔNG sửa một cờ nào. Khác biệt duy nhất giữa 3 nhánh là init và cờ P37.
# crs_freeze_tau = 0.65 (mặc định hiện tại, Phase 28 tuned) → anchor fern 23.88.
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

ANCHOR_FERN=23.88   # Phase 28 tau65 seed42, docs/28

# ── An toàn: luôn khôi phục fused.ply, kể cả khi Ctrl-C hoặc lỗi ──
restore_ply () {
  if [ -f "${SAVE}" ]; then
    cp "${SAVE}" "${PLY}"
    rm -f "${SAVE}"
    echo "[P37 VERIFY] đã khôi phục ${PLY} về trạng thái ban đầu"
  fi
}
trap restore_ply EXIT INT TERM

echo "============================================================"
echo "[P37 VERIFY] scene=${SCENE} seed=${SEED} GPU=${GPU}  $(date '+%F %T')"
echo "============================================================"

# ── Tiền kiểm ──
for f in "${PLY}" "${PLY_ORIG}" "${PLY_P37}"; do
  [ -f "$f" ] || { echo "❌ thiếu $f"; exit 1; }
done
if ! cmp -s "${PLY}" "${PLY_ORIG}"; then
  echo "❌ fused.ply hiện KHÔNG phải RoMa v1 gốc."
  echo "   Chạy p22_place_romav1_init.py trước để về trạng thái chuẩn."
  exit 1
fi
echo "✓ fused.ply = fused.ply.romav1 (trạng thái Phase 22 chuẩn)"
cp "${PLY}" "${SAVE}"
echo "✓ đã backup → ${SAVE}"

SIDECAR=${DATA}/fused.romav1.qinit.npz
[ -f "${SIDECAR}" ] || { echo "❌ thiếu sidecar ${SIDECAR}"; exit 1; }
echo "✓ sidecar có mặt"

# ============================================================
# run_arm — chạy 1 nhánh
#   $1 tên nhánh   $2 file ply nguồn để đặt vào fused.ply   $3 cờ P37 thêm
# ============================================================
run_arm () {
  local NAME=$1; local SRC=$2; local EXTRA=$3
  local OUTDIR=${OUT_ROOT}/${NAME}_${SCENE}_seed${SEED}
  local LOG=${LOG_DIR}/verify_${NAME}_${SCENE}.log

  echo
  echo "------------------------------------------------------------"
  echo "[${NAME}] init=$(basename ${SRC})  extra='${EXTRA}'"
  echo "------------------------------------------------------------"
  cp "${SRC}" "${PLY}"

  CUDA_VISIBLE_DEVICES=${GPU} python -u train.py \
      --source_path data/nerf_llff_data/${SCENE} \
      -m "${OUTDIR}" --seed ${SEED} \
      ${PROTOCOL} ${A3_TRIM} ${EXTRA} \
      > "${LOG}" 2>&1
  local RC=$?
  if [ ${RC} -ne 0 ]; then
    echo "❌ [${NAME}] train FAIL (rc=${RC}) — xem ${LOG}"
    tail -25 "${LOG}"
    return 1
  fi
  echo "✓ [${NAME}] xong → ${LOG}"
  grep -E "Best test PSNR|Final #Gaussians" "${LOG}" | sed 's/^/    /'
  return 0
}

# A0 — init GỐC, flag OFF. Chứng minh code mới không đụng gì khi tắt.
run_arm A0_orig_off  "${PLY_ORIG}" "" || exit 1

# A1 — init _p37, flag OFF. Cô lập ảnh hưởng của init.
run_arm A1_p37_off   "${PLY_P37}"  "" || exit 1

# A2 — init _p37, flag ON, w_Q=0. Buffer chạy nhưng phải TRƠ.
run_arm A2_p37_on_w0 "${PLY_P37}" \
  "--use_roma_qinit --qinit_w_cert 0.0 --qinit_w_reproj 1.0 --qinit_w_in_crs 0.0 --qinit_log_interval 2000" || exit 1

# ── Tổng hợp ──
echo
echo "============================================================"
echo "[P37 VERIFY] KẾT QUẢ"
echo "============================================================"
python - "${LOG_DIR}" "${SCENE}" "${ANCHOR_FERN}" <<'PY'
import re, sys, os
log_dir, scene, anchor = sys.argv[1], sys.argv[2], float(sys.argv[3])
arms = ["A0_orig_off", "A1_p37_off", "A2_p37_on_w0"]
res = {}
for a in arms:
    p = os.path.join(log_dir, f"verify_{a}_{scene}.log")
    if not os.path.isfile(p):
        continue
    t = open(p, encoding="utf-8", errors="ignore").read()
    m = re.search(r"Best test PSNR:\s*([0-9.]+)", t)
    n = re.search(r"Final #Gaussians:\s*(\d+)", t)
    res[a] = (float(m.group(1)) if m else None, int(n.group(1)) if n else None)

print(f"{'nhánh':<16} {'PSNR':>9} {'N_gauss':>10}")
print("-" * 38)
for a in arms:
    if a in res:
        ps, ng = res[a]
        print(f"{a:<16} {ps if ps is not None else float('nan'):>9.4f} "
              f"{ng if ng is not None else -1:>10}")
    else:
        print(f"{a:<16} {'THIẾU':>9} {'-':>10}")

print()
ok = True
if "A0_orig_off" in res and res["A0_orig_off"][0] is not None:
    d = res["A0_orig_off"][0] - anchor
    v = "✅" if abs(d) < 0.5 else "⚠"
    print(f"{v} A0 vs anchor Phase 28 ({anchor}): Δ = {d:+.4f}")
    print("   (sàn nhiễu 1-scene 1-seed là ±1.3 dB → chỉ bắt được lỗi TO)")

if all(k in res for k in ("A1_p37_off", "A2_p37_on_w0")):
    p1, n1 = res["A1_p37_off"]; p2, n2 = res["A2_p37_on_w0"]
    dp = p2 - p1
    print()
    print(f"🔑 A2 − A1 (BUFFER CÓ TRƠ KHÔNG):  ΔPSNR = {dp:+.6f}   "
          f"ΔN_gauss = {n2 - n1:+d}")
    if n1 == n2 and abs(dp) < 1e-4:
        print("   ✅ TRƠ TUYỆT ĐỐI — N_gauss trùng khít và PSNR giống hệt.")
        print("      Buffer không đụng gì vào training. Đi tiếp S2 được.")
    elif n1 == n2:
        print("   🟡 N_gauss TRÙNG nhưng PSNR lệch.")
        print("      Cùng seed + cùng init + cùng N ⇒ lệch này gần như chắc")
        print("      chắn là atomicAdd. Chấp nhận được nếu |Δ| < 0.05.")
        ok = abs(dp) < 0.05
    else:
        print("   🔴 N_gauss KHÁC NHAU — BUFFER KHÔNG TRƠ.")
        print("      Cùng seed, cùng init, cùng cờ mà số Gaussian khác nghĩa là")
        print("      code P37 đã làm đổi luồng RNG hoặc đổi quyết định densify.")
        print("      DỪNG. Đừng chạy S2 — mọi số sau đó sẽ bị nhiễm.")
        ok = False

if all(k in res for k in ("A0_orig_off", "A1_p37_off")):
    p0 = res["A0_orig_off"][0]; p1 = res["A1_p37_off"][0]
    print()
    print(f"ℹ A1 − A0 (ảnh hưởng của init dựng lại): Δ = {p1 - p0:+.4f}")
    print("   1 scene 1 seed, KHÔNG kết luận được gì về reproducibility.")
    print("   Chỉ để biết init _p37 có lệch thô bạo hay không.")

print()
print("VERDICT:", "✅ ĐI TIẾP S2" if ok else "🔴 DỪNG, SỬA TRƯỚC")
PY

echo
echo "[P37 VERIFY] xong. fused.ply sẽ được khôi phục khi script thoát."
