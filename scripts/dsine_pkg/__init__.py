# ============================================================
# [CRSGaussian Phase 16] VENDORED DSINE package.
# Nguồn: dn-splatter (WACV 2025, github.com/maturk/dn-splatter)
#        dn_splatter/scripts/dsine/{rotations,submodules,dsine,
#        dsine_predictor}.py — copy NGUYÊN VĂN 2026-05-19.
# Lý do vendor: dn_splatter/__init__.py:1-7 import nerfstudio dataparsers
#   ⟹ KHÔNG thể `import dn_splatter` trong env 3DGS (vỡ env, rủi ro
#   Phase-10A). 4 file dsine TỰ-CHỨA (torch/numpy/torchvision + geffnet
#   + jaxtyping) → tách ra dùng offline.
# Sửa DUY NHẤT: `from dn_splatter.scripts.dsine.X import` → `from .X import`
#   (3 dòng: dsine.py ×2, dsine_predictor.py ×1). KHÔNG đổi logic/số.
# ============================================================
