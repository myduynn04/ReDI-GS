#!/usr/bin/env bash
# ============================================================
# [CRSGaussian Phase 37] Dò dấu vết RoMa v1 để dựng lại env
# File: scripts/p37_diag_romav1_env.sh  (KEEP LOCAL — server-only)
#
# Bối cảnh: 2026-08-06 phát hiện env `roma_v1` không còn tồn tại.
#   8 file fused.ply.romav1 CÒN NGUYÊN (Phase 22 an toàn), nhưng
#   không chạy lại preprocess được → chặn H1 (docs/37) VÀ chặn
#   reproducibility của Phase 22.
#
# Script này KHÔNG cài gì, KHÔNG sửa gì. Chỉ đọc và in.
# Mục tiêu: trả lời "romatch từng cài từ đâu" để chọn đúng cách dựng lại.
#
# Usage:
#   bash scripts/p37_diag_romav1_env.sh 2>&1 | tee logs/p37/diag_env.log
# ============================================================
set -u

DUYEN="${DUYEN:-$HOME/workspace/representation-3d/duyen}"
mkdir -p logs/p37

hr () { printf '%s\n' "------------------------------------------------------------"; }

echo "============================================================"
echo "[P37 DIAG] RoMa v1 env forensics — $(date '+%F %T')"
echo "  DUYEN = ${DUYEN}"
echo "============================================================"

# ── 1. Có những thư mục gì trong workspace ──
echo
echo "[1] Thư mục trong ${DUYEN}"
hr
ls -1d "${DUYEN}"/*/ 2>/dev/null | sed 's|.*/\([^/]*\)/$|  \1|'

# ── 2. Cấu trúc repo romav2 — có package romatch không ──
echo
echo "[2] Repo romav2 — cấu trúc gốc"
hr
if [ -d "${DUYEN}/romav2" ]; then
  ls -1 "${DUYEN}/romav2" | head -40
  echo
  echo "  → file cài đặt:"
  for f in setup.py pyproject.toml setup.cfg requirements.txt; do
    [ -f "${DUYEN}/romav2/${f}" ] && echo "     ✅ ${f}" || echo "     —  ${f}"
  done
  echo
  echo "  → package python cấp 1-2 (tìm thư mục có __init__.py):"
  find "${DUYEN}/romav2" -maxdepth 2 -name "__init__.py" 2>/dev/null \
    | sed "s|${DUYEN}/romav2/||" | head -20
  echo
  echo "  → có định nghĩa roma_outdoor / RoMaV2 / use_custom_corr ở đâu:"
  grep -rn --include=*.py -E "def roma_outdoor|class RoMaV2|use_custom_corr" \
    "${DUYEN}/romav2" 2>/dev/null | head -15
  echo
  echo "  → git state:"
  git -C "${DUYEN}/romav2" log --oneline -3 2>/dev/null || echo "     (không phải git repo)"
  git -C "${DUYEN}/romav2" remote -v 2>/dev/null | head -2
else
  echo "  ❌ ${DUYEN}/romav2 KHÔNG tồn tại"
fi

# ── 3. Có bản romatch nào còn sót trên đĩa không ──
echo
echo "[3] Tàn tích romatch trên đĩa"
hr
echo "  → thư mục tên romatch (ngoài site-packages đã xóa):"
find "$HOME" -maxdepth 6 -type d -name "romatch" 2>/dev/null | head -10
echo "  → egg-link / dist-info còn sót trong các env khác:"
find "$HOME/miniconda3/envs" -maxdepth 4 -iname "romatch*" 2>/dev/null | head -10
echo "  → pip cache:"
find "$HOME/.cache/pip" -iname "*romatch*" 2>/dev/null | head -5

# ── 4. Weights — thứ đắt nhất phải tải lại nếu mất ──
echo
echo "[4] Weights RoMa + DINOv2 (tránh phải tải lại)"
hr
for d in "$HOME/.cache/torch/hub/checkpoints" "$HOME/.cache/huggingface" "${DUYEN}/romav2"; do
  echo "  ${d}:"
  find "${d}" -maxdepth 3 \( -iname "*roma*" -o -iname "*dinov2*" -o -iname "*dinov3*" \) \
       \( -name "*.pth" -o -name "*.pt" -o -name "*.safetensors" \) 2>/dev/null \
    | head -8 | while read -r f; do echo "     $(du -h "$f" 2>/dev/null | cut -f1)  $f"; done
done

# ── 5. Env romav2 — nền để clone ──
echo
echo "[5] Env romav2 — dùng làm nền clone"
hr
conda run -n romav2 python -c "import sys, torch; print(f'  python {sys.version.split()[0]}'); print(f'  torch  {torch.__version__}  cuda={torch.version.cuda}  avail={torch.cuda.is_available()}')" 2>&1 | head -5
echo "  → package liên quan:"
conda run -n romav2 pip list 2>/dev/null | grep -iE "^(roma|dkm|torch|torchvision|xformers|einops|kornia|timm|numpy|opencv|plyfile)" | sed 's/^/     /'

# ── 6. RoMa v2 import thế nào — manh mối cho v1 ──
echo
echo "[6] RoMa v2 import bằng cách nào (manh mối cho v1)"
hr
echo "  → thử import từ thư mục repo (cwd-based, KHÔNG cài):"
(cd "${DUYEN}/romav2" 2>/dev/null && conda run -n romav2 python -c "
try:
    import romatch; print('     ✅ romatch import OK từ cwd repo:', romatch.__file__)
except Exception as e:
    print('     ❌ romatch:', type(e).__name__, e)
for name in ('roma', 'romav2', 'roma_v2'):
    try:
        m = __import__(name); print(f'     ✅ {name} OK:', getattr(m,'__file__','<ns>'))
    except Exception as e:
        print(f'     —  {name}: {type(e).__name__}')
" 2>&1 | grep -v "^$")

# ── 7. Script Phase 21/22 gọi gì ──
echo
echo "[7] Script Phase 21/22 — API thực tế đã dùng"
hr
grep -n -E "conda activate|^from |^import |roma_outdoor|RoMaV2|use_custom_corr|PYTHONPATH" \
  scripts/p21_roma_preprocess.py scripts/p22_romav1_preprocess.py 2>/dev/null \
  | grep -vE "^\S+:[0-9]+:import (os|sys|time|struct|collections|numpy|cv2)" | head -25

echo
echo "============================================================"
echo "[P37 DIAG] XONG — đọc mục [2] và [6] trước."
echo "  Nếu [6] import romatch OK từ cwd repo romav2:"
echo "     → RoMa v1 và v2 CÙNG một codebase, chỉ khác API entrypoint."
echo "       Dựng lại = clone env romav2 + pip install -e repo. KHÔNG cần internet."
echo "  Nếu [6] fail nhưng [2] có setup.py/pyproject:"
echo "     → cài từ repo có sẵn. Vẫn KHÔNG cần internet cho source."
echo "  Nếu cả hai fail:"
echo "     → phải clone lại repo RoMa. Cần internet 1 lần."
echo "  Mục [4] cho biết weights còn cache không — đây là phần tải lâu nhất."
echo "============================================================"
