# Codebase Map — File nào làm gì

> Đọc file này trước khi đụng vào bất kỳ repo nào.

---

## CoR-GS — BASE CODE (sửa ở đây)

### File structure
```
CoR-GS/
├── train.py                    ← Main training loop
├── render.py                   ← Rendering + evaluation
├── metrics.py                  ← Standard metrics (PSNR/SSIM/LPIPS)
├── metrics_dtu.py              ← DTU-specific evaluation (GFS extension HOLD)
├── convert.py                  ← COLMAP preprocessing
├── scene/
│   ├── gaussian_model.py       ← GaussianModel class ← SỬA NHIỀU NHẤT
│   ├── cameras.py              ← Camera representation
│   ├── dataset_readers.py      ← Data loading
│   └── scene.py                ← Scene wrapper
├── gaussian_renderer/
│   └── __init__.py             ← render() function
├── utils/
│   ├── loss_utils.py           ← Loss functions ← THÊM adaptive_depth_loss
│   ├── general_utils.py        ← Utilities
│   └── graphics_utils.py       ← Geometry helpers
├── arguments/
│   └── __init__.py             ← Argument parsing
├── scripts/
│   ├── run_llff.sh             ← Run LLFF training
│   └── run_dtu.sh              ← Run DTU training
└── submodules/
    ├── diff-gaussian-rasterization-confidence/  ← Custom rasterizer
    └── simple-knn/
```

### Hàm quan trọng cần đọc kỹ trước khi sửa

**gaussian_model.py:**
```
__init__()              → xem tất cả attributes hiện có
get_opacity             → property pattern để học theo
get_xyz                 → property pattern để học theo
densification_postfix() → nơi thêm Gaussian mới vào model
densify_and_clone()     → clone small Gaussians
densify_and_split()     → split large Gaussians
compute_prune_mask()    → ← THÊM CRS CONDITION VÀO ĐÂY
densify_and_prune()     → entry point, gọi compute_prune_mask
capture() / restore()   → checkpoint save/load
```

**train.py:**
```
training()              → main loop
  - Tìm: gaussians.densify_and_prune()  → thêm warmup check quanh đây
  - Tìm: depth_loss                      → thay bằng adaptive_depth_loss
  - Tìm: pseudo view sampling            → tái dụng cho L_pseudo_depth
```

**metrics_dtu.py:** [HOLD — không mở rộng cho đến khi được yêu cầu]
```
evaluate()              → entry point
                          (GFS extension tạm gác)
```

---

## FSGS — CHỈ ĐỌC (lấy pseudo-view generation)

### Cần đọc
```
FSGS/train.py:
  - Tìm: start_sample_pseudo, end_sample_pseudo, sample_pseudo_interval
  - Đoạn code tạo custom_cam (pseudo camera giữa 2 training views)
  - Đoạn code compute depth loss trên pseudo views
  - Copy pattern này vào CoR-GS/train.py

FSGS/utils/camera_utils.py (nếu có):
  - Camera interpolation logic
```

### Không cần đọc
- scene/gaussian_model.py (khác với CoR-GS)
- gaussian_renderer/ (tương tự nhưng khác API)

---

## DNGaussian — CHỈ ĐỌC (lấy depth normalization)

### Cần đọc
```
DNGaussian/utils/loss_utils.py (hoặc tương đương):
  - Tìm: global_local_depth_normalization() hoặc depth_norm
  - Hiểu: normalize depth globally (toàn scene) và locally (per-patch)
  - Copy function này vào CoR-GS/utils/loss_utils.py
  - Dùng để stabilize D_i trong warmup phase
```

---

## LoopSparseGS — CHỈ ĐỌC (lấy DAR loss)

### Cần đọc
```
LoopSparseGS/train.py:
  - Tìm: DAR (Depth Alignment Regularization) hoặc sliding_window_pearson
  - Tìm: patch_length parameter
  - Hiểu: tính Pearson correlation trong sliding window thay vì toàn ảnh
  - Copy logic này vào CoR-GS/crs_module.py cho compute_depth_consistency()

LoopSparseGS/loop.py:
  - Không cần — chỉ cần DAR loss từ train.py
```

---

## DepthRegularizedGS — CHỈ ĐỌC (lấy weighted alignment)

### Cần đọc
```
Bất kỳ file nào có scale alignment:
  - Tìm: argmin_{s,t} || w(p) * D_sparse(p) - (s * D_dense(p) + t) ||^2
  - w(p) = 1 / reprojection_error(p)
  - Đây là ~10-20 dòng code
  - Copy vào CoR-GS/depth_alignment.py
```

---

## File sẽ tạo mới trong CoR-GS/

### depth_alignment.py (tạo mới)
```python
# Input:
#   D_dense: Tensor [H, W] — DepthAnything V2 output (relative depth)
#   D_sparse: Tensor [H, W] — COLMAP projected depth (có nhiều zero pixels)
#   reprojection_errors: Tensor [H, W] — COLMAP reprojection error per pixel
# Output:
#   D_aligned: Tensor [H, W] — aligned depth ở scale của scene

def weighted_scale_alignment(D_dense, D_sparse, reprojection_errors):
    # w(p) = 1 / reprojection_error(p)
    # Chỉ dùng pixels có D_sparse > 0 (valid COLMAP points)
    # Solve weighted least squares: min ||W(D_sparse - s*D_dense - t)||^2
    # Return: s* * D_dense + t*
    pass
```

### crs_module.py (tạo mới)
```python
# Các function cần implement:

def init_crs_from_colmap(reprojection_errors, num_gaussians):
    # CRS₀ cho COLMAP points = 1 - normalize(reprojection_error)
    # CRS₀ cho densified points (sinh sau) = 0.5
    # Return: Tensor [N] — initial CRS scores
    pass

def compute_depth_consistency(gaussians, cameras, depth_prior_dict):
    # D_i = 1 - |d_render_i - d_prior_i| / depth_range
    # depth_prior_dict: {camera_id: aligned_depth_tensor}
    # Return: Tensor [N] — per-Gaussian depth consistency
    pass

def compute_reprojection_consistency(gaussians, cameras, render_fn):
    # R_i = 1 - mean(|c_i - warp(c_j→i, depth_j)|) / 255
    # Return: Tensor [N] — per-Gaussian reprojection consistency
    pass

def update_crs(gaussians, cameras, depth_prior_dict, render_fn, w1=0.5, w2=0.5, ema=0.9):
    # CRS_i = sigmoid(w1*D_i + w2*R_i)
    # EMA: crs_new = ema * crs_old + (1-ema) * crs_computed
    # Cập nhật gaussians._crs_score in-place
    pass
```

### depth_model.py (tạo mới)
```python
# Wrapper cho DepthAnything V2
# Compatible với training loop của CoR-GS

class DepthModel:
    def __init__(self, model_type='vitl', device='cuda'):
        # Load DepthAnything V2
        pass
    
    def predict(self, image):
        # image: Tensor [3, H, W] hoặc numpy [H, W, 3]
        # Return: Tensor [H, W] relative depth
        pass
```