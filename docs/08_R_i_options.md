# R_i Implementation Options — CRSGaussian
> Tài liệu tham chiếu khi cần thay đổi approach R_i.
> Hiện tại: Hướng 1 (GT color pairwise) đang được dùng.
> Nếu signal yếu sau T2.7 → thử Hướng 3, rồi Hướng 2.

---

## Hướng 1 — GT Color Pairwise (ĐANG DÙNG)

**Ý tưởng:** Project Gaussian xuống từng camera, lấy màu ảnh thật
tại pixel đó, so sánh màu giữa các cameras.

```python
# Với mỗi Gaussian g:
# 1. Project xyz → pixel (px, py) trên từng camera
# 2. Lookup GT color: cam.original_image[:, py, px] → (3,) RGB [0,1]
# 3. Tính pairwise L1 diff giữa tất cả cặp cameras visible
# 4. R_g = 1 - mean(pairwise_diffs)
# 5. Visible < 2 cameras → R_g = 0.5 (neutral)

def compute_reprojection_consistency(xyz, cameras):
    # xyz: (N, 3) GPU
    # cameras: list Camera với .original_image (3,H,W) [0,1] cuda
    # return: (N,) R_i range [0,1]
```

**Ưu điểm:**
- Đơn giản, không dependency
- Novel — chưa paper nào làm per-Gaussian GT color pairwise
- Dễ implement và debug

**Nhược điểm:**
- Bị ảnh hưởng bởi lighting (bóng, specular)
- Textureless regions → màu giống nhau dù Gaussian sai → false negative

**Research support:**
- Chưa có paper làm đúng hướng này
- Gần nhất: MVS cổ điển dùng NCC patch comparison (Furukawa & Ponce 2010)
- ICO-GS (arxiv 2603.02893) làm feature-based thay color-based

**Khi nào switch sang hướng khác:**
- T2.7: R_i.std() < 0.05 (signal flat, không phân biệt floater)
- Hoặc: ablation A2 (R_i only) không improve so với A0

---

## Hướng 3 — Geometric Reprojection (DỰ PHÒNG 1)

**Ý tưởng:** Warp pixel từ camera A sang camera B dùng depth,
so sánh pixel coordinates thay vì color.

```python
# Với mỗi Gaussian g visible từ camera A và B:
# 1. Project xyz → pixel_A (từ cam A), pixel_B (từ cam B)
# 2. Từ pixel_A + depth_A, warp sang cam B → pixel_A_warped
# 3. Reprojection error = |pixel_A_warped - pixel_B| / image_size
# 4. R_g = 1 - mean(reprojection_errors)

def warp_pixel(pixel_src, depth_src, cam_src, cam_dst):
    # Unproject: pixel + depth → 3D world
    # Re-project: 3D world → pixel trong cam_dst
    pass
```

**Ưu điểm:**
- Robust với lighting thay đổi
- Thuần geometry, không bị color noise
- Không cần external matcher (khác SCGaussian)

**Nhược điểm:**
- Cần depth prior chính xác để warp
- Nếu depth alignment sai → warp sai → R_i sai
- Phụ thuộc vào --use_depth_prior flag

**Research support:**
- SCGaussian (NeurIPS 2024): dùng geometric reprojection nhưng cần GIM matcher
- Ta không cần matcher vì đã có Gaussian 3D position làm anchor

**Khi nào dùng:**
- Hướng 1 signal yếu (R_i.std() < 0.05 sau T2.7)
- Hoặc: scene có lighting variation lớn (outdoor, DTU với varying illumination)
- Yêu cầu: --use_depth_prior=True (cần depth map để warp)

**Implementation note:**
- Tái dụng warp_pixels() từ docs/05_prompts.md T2.4 section
- depth_prior_dict phải có sẵn (gated bởi use_depth_prior)

---

## Hướng 2 — Feature-based (DỰ PHÒNG 2)

**Ý tưởng:** Thay GT color bằng deep features (DINO/VGG),
so sánh feature vectors thay vì raw color.

```python
# Với mỗi Gaussian g:
# 1. Extract DINO features cho tất cả training images (precompute)
# 2. Project xyz → pixel trên từng camera
# 3. Lookup feature vector tại pixel đó
# 4. So sánh cosine similarity giữa feature vectors

def compute_feature_consistency(xyz, cameras, feature_extractor):
    # Cần precompute features: {cam.uid: feature_map (C,H,W)}
    pass
```

**Ưu điểm:**
- Robust nhất với lighting, specular, imaging artifacts
- Feature mang nhiều thông tin hơn RGB
- ICO-GS (arxiv 2603.02893) validate approach này

**Nhược điểm:**
- Cần DINO/VGG model → thêm dependency nặng
- Precompute features tốn VRAM
- Phức tạp hơn implement và debug

**Research support:**
- ICO-GS: dùng feature-based multi-view consistency
- DINOv2 features được dùng rộng rãi cho geometric tasks

**Khi nào dùng:**
- Cả Hướng 1 và Hướng 3 đều không đủ signal
- Hoặc: scene outdoor với lighting thay đổi mạnh
- Yêu cầu: install DINOv2, precompute feature maps

---

## Decision Log

| Date | Quyết định | Lý do |
|------|-----------|-------|
| 2026-03 | Chọn Hướng 1 | Đơn giản nhất, novel nhất, không dependency |
| — | Hướng 3 nếu H1 yếu | Robust hơn với lighting, không cần external model |
| — | Hướng 2 nếu H3 yếu | Robust nhất nhưng nặng nhất |

---

## Ablation cho paper

```
A_R1: R_i dùng GT color pairwise (Hướng 1)
A_R3: R_i dùng geometric reprojection (Hướng 3)
A_R2: R_i dùng feature-based (Hướng 2) — nếu cần
```

So sánh A_R1 vs A_R3 → justify tại sao chọn hướng nào trong paper.