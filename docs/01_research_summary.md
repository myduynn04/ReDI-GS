# CRSGaussian — Research Summary
### Confidence-Reliability Guided Sparse-View 3D Gaussian Splatting with Geometry-Aware Evaluation

> Tổng hợp nghiên cứu · Few-Shot Novel View Synthesis · Tháng 3, 2026
> **Version 3.0** — Cập nhật: CRS₀ informed init, position constraint từ T_densify=500, GFS HOLD, SOTA context

---

## 1. Mục tiêu nghiên cứu

### 1.1 Bài toán

Few-shot Novel View Synthesis (NVS) — tổng hợp góc nhìn mới từ **chỉ 3–8 ảnh đầu vào** — là bài toán nền tảng với ứng dụng trong AR/VR, robotics, autonomous driving. 3D Gaussian Splatting (3DGS) là SOTA về tốc độ (real-time rendering), nhưng hiệu năng sụt giảm nghiêm trọng với sparse views vì hai vấn đề cốt lõi:

| Vấn đề | Mô tả | Bằng chứng |
|---|---|---|
| **Floater** | Gaussian đặt sai vị trí 3D — đúng từ training views, sai từ novel views | AD-GS 2025: root cause là densification không có position constraint |
| **Overdensification** | Số Gaussian tăng không kiểm soát → overfit training views | DropoutGS CVPR 2025: 10k Gaussian overfit nặng hơn 1k Gaussian |

### 1.2 Mục tiêu cụ thể

1. **Method** — Thiết kế CRS (Confidence-Reliability Score): per-Gaussian dynamic score tích hợp depth consistency + reprojection consistency, dùng để điều phối depth loss, densification, và pruning.
2. ~~**Evaluation**~~ [HOLD] — GFS metric tạm gác. Chỉ implement sau khi C1+C2 có số experiment support.
3. **Kết quả** — Target: >21.0 PSNR LLFF 3-view để competitive, >21.35 để beat SOTA (D2GS); duy trì real-time rendering (>100 FPS).

### 1.3 Ba contribution chính

```
C1 — CRS Module                                              ← FOCUS
     Unified per-Gaussian confidence score (depth consistency + reprojection consistency).
     Dynamic, in-loop, single-model. Khác CoR-GS (2x memory), khác TIDI-GS (không per-Gaussian).
     CRS₀: COLMAP points = 1-normalize(reproj_error), densified = 0.5

C2 — CRS-guided Densification Control                        ← FOCUS
     (a) Adaptive depth loss weight theo CRS từng Gaussian
     (b) Position constraint cho Gaussian mới khi splitting — BẬT TỪ T_densify=500
     (c) Multi-signal pruning (CRS + opacity + isolation)

C3 — Geometric Fidelity Score (GFS)                           ← HOLD
     Depth RMSE + Floater Ratio. Tạm gác — chỉ implement sau khi C1+C2 có kết quả.
```

---

## 2. Định hướng nghiên cứu

### 2.1 Landscape — ai đã làm gì

| Paper / Venue | Position constraint | Count control | Unified quality signal | Geometry metric |
|---|---|---|---|---|
| FSGS (ECCV 24) | ~ depth loss | ✗ | ✗ | ✗ |
| DNGaussian (CVPR 24) | ~ depth norm | ✗ | ✗ | ✗ |
| SCGaussian (NeurIPS 24) | ✓ ray-bound | ✗ | ✗ | ✗ |
| CoR-GS (ECCV 24) | ✗ | co-pruning | ~ 2-field disagreement | ✗ |
| LoopSparseGS (IEEE TIP 25) | ~ SfM+DAR | SFS split | ✗ | ✗ |
| DropGaussian (CVPR 25) | ✗ | dropout | ✗ | ✗ |
| AD-GS (2025) | ✗ | alternating | ✗ | ✗ |
| VGNC (ACM MM 25) | ✗ | validation-guided | ✗ | ✗ |
| TIDI-GS (2025) | ~ uncertainty depth | ✗ | ✗ | ✗ |
| **CRSGaussian (đề xuất)** | **depth + reprojection** | **CRS-guided** | **✓ unified** | ~~GFS~~ HOLD |

### 2.2 Gap thực sự — bằng chứng từ literature

**Gap 1 — Không ai kết hợp depth consistency + reprojection consistency trong một unified score:**
- CoR-GS dùng rendering disagreement nhưng cần 2x memory (2 fields song song)
- FSGS/DNGaussian/LoopSparseGS chỉ dùng depth prior — bị scale ambiguity
- **CRS** = single-model, kết hợp cả hai, lightweight

**Gap 2 — Position constraint trong densification chưa được giải quyết đúng:**
- AD-GS (2025) xác định root cause: Gaussian mới khi splitting được sample tự do → vòng lặp tự khuếch đại floater
- SCGaussian dùng ray-bound nhưng cần matching prior nặng (GIM/LoFTR)
- **CRS** dùng depth prior có sẵn để constraint position — không cần external matching model

**Gap 3 — Metric floater chưa được propose đầy đủ:**
- TIDI-GS (2025) và CGF 2024 paper đều xác nhận PSNR/SSIM không capture floater artifacts
- Chưa có paper nào propose + validate metric mới cho floater trong sparse-view 3DGS

### 2.3 Lý do KHÔNG làm theo các hướng khác

| Hướng bị loại | Lý do |
|---|---|
| CoR-GS style (2 fields) | 2x memory, 2x training time — không thực tế |
| VGNC style (NVS model validation) | Cần train NVS generative model — quá nặng |
| Diffusion prior (SDS loss) | Oversmoothing, slow, không giải quyết geometry trực tiếp |
| Feed-forward 3DGS (PixelSplat) | Generalizable nhưng kém per-scene — khác hướng |

---

## 3. Chi tiết Methodology

### 3.1 Pipeline tổng quan

```
Input: 3 sparse views + COLMAP poses
         ↓
[Stage 1] Initialization
  - DepthAnything V2 depth prediction (hoặc DPT — xem ablation T0.8/T0.9)
  - Weighted scale alignment (depth prior → COLMAP points)  ← từ Chung et al.
  - Initialize Gaussians từ SfM points
  - CRS₀ cho COLMAP points = 1 - normalize(reprojection_error)  ← MỚI (v3.0)
  - CRS₀ cho densified points = 0.5 (neutral default)
         ↓
[Stage 2a] Pre-densification (iter 0 → T_densify = 500)
  - Chỉ photometric loss (L1 + SSIM)
  - Chưa densify, chưa dùng CRS
  - Gaussians settle sơ bộ
         ↓
[Stage 2b] Early densification (iter T_densify=500 → T_warmup=2000)
  - Photometric + fixed depth loss
  - Densification BẬT + position constraint BẬT  ← MỚI (v3.0)
  - KHÔNG dùng CRS cho loss weighting (signal còn noisy)
  - Lý do: AD-GS (2025) chứng minh floater hình thành ngay khi densify bắt đầu
         ↓
[Stage 3] CRS Computation (sau mỗi 100 iter, từ iter 2000)
  - Tính D_i (depth consistency)
  - Tính R_i (reprojection consistency)
  - Update CRS_i = sigmoid(w1*D_i + w2*R_i)
         ↓
[Stage 4] CRS-guided Optimization (iter 2000+)
  - Adaptive depth loss weight theo CRS
  - Position constraint vẫn tiếp tục
  - Multi-signal pruning (CRS + opacity + isolation)
         ↓
[Stage 5] Evaluation
  - Standard: PSNR / SSIM / LPIPS
  - GFS (Depth RMSE + Floater Ratio) — [HOLD]
```

### 3.2 Weighted Depth Scale Alignment ← BỔ SUNG MỚI

Trước khi tính D_i, depth prior từ DepthAnything V2 cần được align scale với scene thực tế. Thay vì naive least squares, dùng **weighted alignment** (từ Chung et al., CVPR Workshop 2024):

```python
# Align scale s và offset t của monocular depth về COLMAP sparse depth
# w(p) = 1 / reprojection_error(p)  → SfM points tin cậy được tin nhiều hơn

s*, t* = argmin_{s,t}  Σ_{p ∈ D_sparse}  || w(p) · D_sparse(p) - (s · D_dense(p) + t) ||²

D_aligned = s* · D_DepthAnythingV2(I) + t*
```

**Tại sao quan trọng cho D_i:** Nếu scale alignment sai → D_i bị systematic error từ iter đầu → CRS không phản ánh quality thực. Weighted alignment giúp D_i ổn định hơn ở scenes có textureless regions (một số SfM points kém tin cậy).

**Khác Chung et al.:** Họ dùng ZoeDepth (metric depth, domain-sensitive), ta dùng DepthAnything V2 (relative depth, general-purpose) nên bước alignment này càng critical hơn.

### 3.3 Công thức CRS

#### Hai tín hiệu

```
D_i = 1 - |d_render_i - d_prior_i| / depth_range
    → Depth consistency: rendered depth có khớp D_aligned không?

R_i = 1 - mean( |c_i - warp(c_j → i, depth_j)| ) / 255
    → Reprojection consistency: color của Gaussian có nhất quán cross-view không?
      (tính với tất cả j ≠ i trong sparse view set — tối đa 2 views còn lại)

CRS_i = sigmoid( w1 * D_i + w2 * R_i )
      w1 = 0.5, w2 = 0.5  (ablate: [0.3, 0.7] và [0.7, 0.3])
      Update mỗi 100 iterations sau T_warmup
```

#### Tại sao bỏ Opacity Stability (O_i)?

> **Bằng chứng (DropoutGS CVPR 2025):** Floater có thể có opacity **cao** và **ổn định** vì đang fit training view tốt. Dùng O_i sẽ bỏ sót đúng loại floater nguy hiểm nhất. D_i + R_i là đủ và principled hơn.

#### Tại sao dùng delayed activation (T_warmup)?

> **Vấn đề:** Iter 0–2000, Gaussians chưa settle → D_i và R_i là noise → CRS noise → weight lại loss sai → làm training tệ hơn baseline.  
> **Giải pháp:** Delayed activation — consistent với cách FSGS/CoR-GS enable pseudo-views sau warmup.

### 3.4 Ba ứng dụng của CRS

#### (a) Adaptive Depth Loss

```python
lambda_depth_i = lambda_base * (2.0 - CRS_i)

# CRS thấp (0.0–0.4) → lambda = 1.6 * base  → constraint mạnh → kéo về surface
# CRS cao  (0.7–1.0) → lambda = 0.6 * base  → constraint nhẹ → không bị depth noise

L_depth_adaptive = sum_i( lambda_depth_i * L_Pearson(d_render_i, d_prior_i) )
```

#### (b) Position Constraint trong Densification — BẬT TỪ T_densify=500

```python
# Root cause (AD-GS 2025): Gaussian mới sample tự do → floater tự khuếch đại
# QUAN TRỌNG: BẬT ngay từ T_densify=500, KHÔNG đợi T_warmup=2000
# Lý do: floater hình thành và khuếch đại ngay khi densification bắt đầu

for attempt in range(3):
    G_new.position = G_i.position + noise
    depth_new = project_to_depth(G_new.position, camera)

    if abs(depth_new - d_prior[pixel]) < epsilon_depth:
        accept(G_new)   # nằm gần surface → OK
        break
else:
    skip_densification(G_i)   # không tìm được vị trí hợp lệ

# epsilon_depth = 0.05 * depth_range  (ablate: 0.02, 0.05, 0.10)
# T_densify ∈ {500, 1000}  (ablate)
```

#### (c) Multi-signal Pruning

```python
# Mở rộng từ compute_prune_mask() của CoR-GS — thêm 1 dòng:

def should_prune(G_i):
    return (
        G_i.crs < tau_crs           # tau_crs = 0.2  ← MỚI
        and G_i.opacity < epsilon   # epsilon = 0.005 (như 3DGS gốc)
        and knn_distance(G_i) > tau_isolated   # isolated trong 3D space
    )
```

### 3.5 Full Loss Function

```
L = λ1 * L_L1 + λ2 * L_SSIM + L_depth_adaptive + λ3 * L_pseudo_depth

  λ1 = 0.8  (photometric L1)
  λ2 = 0.2  (D-SSIM)
  λ3 = 0.05 (pseudo-view depth, sau T_warmup)
  lambda_base = 0.05 (depth loss base weight)

L_depth_adaptive = Σᵢ [ lambda_base * (2 - CRS_i) * Pearson(d_render_i, d_prior_i) ]
L_pseudo_depth   = Pearson loss trên pseudo-views (kế thừa từ CoR-GS/FSGS)
```

### 3.6 Geometric Fidelity Score (GFS) — [HOLD - deprioritized]

> **Tạm gác hoàn toàn.** Chỉ implement sau khi C1+C2 có kết quả experiment support.
> Lý do: contribution rủi ro, cần chứng minh CRS hoạt động trước.

<details>
<summary>Chi tiết GFS (collapsed — chỉ mở khi cần)</summary>

#### Motivation

> TIDI-GS (2025): *"Standard metrics like PSNR, SSIM, and LPIPS do not fully capture the interactive failure modes common to 3DGS."*
> CGF 2024: SSIM và LPIPS perform **tệ hơn PSNR** trong NVS perceptual evaluation.

#### Hai thành phần GFS

```
GFS = { Depth RMSE, Floater Ratio }

Depth RMSE:
  - So rendered depth với GT depth scan (DTU dataset)
  - Masked evaluation (exclude background theo DTU protocol)
  - Build trên metrics_dtu.py có sẵn trong CoR-GS

Floater Ratio:
  - Surface reference = D_aligned median ± 2σ
  - Floater = Gaussian nằm ngoài surface band
  - Floater Ratio = count(floaters) / total_gaussians
```

#### Cách trình bày trong paper

- **Table 1:** PSNR / SSIM / LPIPS — compare với SOTA
- **Table 2:** Depth RMSE + Floater Ratio — chứng minh geometry improvement
- **Figure:** Scatter plot PSNR vs Depth RMSE → chứng minh không correlate → PSNR không đủ

</details>

---

## 4. Thực nghiệm

### 4.1 Datasets

| Dataset | Setting | Mục đích | Metric |
|---|---|---|---|
| **LLFF** | 3 views, 504×378 | Primary benchmark | PSNR / SSIM / LPIPS |
| **DTU** | 3 views, 15 scenes | Geometry evaluation (có GT depth) | + Depth RMSE + Floater Ratio |
| **Blender** | 8 views, 400×400 | Object-level, synthetic | PSNR / SSIM / LPIPS |
| Mip-NeRF360 | 24 views | Optional — complex outdoor | PSNR / SSIM / LPIPS |

### 4.2 Baselines

| Method | Lý do so sánh |
|---|---|
| 3DGS (Kerbl 2023) | Vanilla baseline |
| FSGS (ECCV 2024) | Direct predecessor, pseudo-view generation |
| DNGaussian (CVPR 2024) | Depth normalization approach |
| SCGaussian (NeurIPS 2024) | Best position-constraint method hiện tại |
| LoopSparseGS (IEEE TIP 2025) | Comprehensive pipeline gần nhất |
| CoR-GS (ECCV 2024) | Codebase base — best consistency-based method |
| DropGaussian (CVPR 2025) | Latest CVPR 2025 baseline |

### 4.3 Ablation Study

| Config | D_i | R_i | Position constraint | Weighted align | Mục đích |
|---|---|---|---|---|---|
| A0 | ✗ | ✗ | ✗ | ✗ | Baseline (CoR-GS không có CRS) |
| A1 | ✓ | ✗ | ✗ | ✓ | Depth-only CRS |
| A2 | ✗ | ✓ | ✗ | ✓ | Reproj-only CRS |
| A3 | ✓ | ✓ | ✗ | ✓ | Full CRS, no position constraint |
| **A4** | **✓** | **✓** | **✓** | **✓** | **Full CRSGaussian** |
| A5 | ✓ | ✓ | ✓ (T_warmup=0) | ✓ | Chứng minh cần delayed activation |
| A6 | ✓ | ✓ | ✓ | ✗ (naive) | Chứng minh cần weighted depth alignment |

**Hyperparameter sensitivity:**
- `T_densify` ∈ {500, 1000}
- `T_warmup` ∈ {1000, 2000, 3000}
- `epsilon_depth` ∈ {0.02, 0.05, 0.10}
- `tau_crs` ∈ {0.15, 0.20, 0.25}
- `w1 : w2` ∈ {0.3:0.7, 0.5:0.5, 0.7:0.3}
- `CRS update interval` ∈ {50, 100, 200}
- `CRS₀ init` — Alt-1 through Alt-4 (xem ablation)

### 4.4 Timeline

| Giai đoạn | Tuần | Công việc |
|---|---|---|
| Setup & baseline | 1–2 | Clone CoR-GS, chạy LLFF + DTU 3-view baseline, visualise floater, DPT vs DAV2 ablation |
| Weighted alignment | 2 | Implement `depth_alignment.py`, so sánh naive vs weighted (ablation A6) |
| CRS module | 3 | Implement D_i + R_i + EMA update, CRS₀ informed init, log CRS distribution |
| Adaptive depth loss | 4 | Kết nối CRS vào loss (ablation A3) |
| Full pipeline | 5 | Position constraint (từ T_densify=500) + multi-signal pruning (A4) |
| ~~GFS metric~~ | ~~6–7~~ | ~~HOLD — tạm gác~~ |
| Writing | 6–10 | Full experiments, figures, paper writing |

---

## 5. Codebase

### 5.1 Quyết định cuối: CoR-GS làm base

```
Repository : https://github.com/jiaw-z/CoR-GS
Paper      : CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization (ECCV 2024)
Stars      : 129  |  License: MIT  |  Tested: LLFF, Mip-NeRF360, DTU, Blender
```

**So sánh lý do chọn CoR-GS so với các ứng viên khác:**

| Tiêu chí | CoR-GS | FSGS | SCGaussian | LoopSparseGS | DNGaussian |
|---|---|---|---|---|---|
| R_i (reprojection) | ✓ có sẵn | ✗ viết mới | ✓ có sẵn | ✗ viết mới | ✗ viết mới |
| Multi-signal pruning | ✓ extend 1 dòng | ✗ | ✗ | ~ | ✗ |
| DTU evaluation | ✓ metrics_dtu.py | ✗ | ✗ | ✗ | ✓ |
| Custom rasterizer | ✓ confidence submodule | ✓ | ✓ | ✓ | ✓ |
| Thêm CRS attribute | ✓ đơn giản | ✓ đơn giản | ✗ phức tạp (2 loại GS) | ✓ đơn giản | ✓ đơn giản |
| Training time | ~10 phút | ~10 phút | ~1 phút | ~35–45 phút | ~1 phút |
| External dependency nặng | ✗ | ✗ | ✓ GIM/LoFTR | ✗ | ✗ |

**Code thực tế từ `gaussian_model.py` của CoR-GS — lý do chính:**
```python
# Hàm có sẵn, chỉ cần thêm 1 dòng để có CRS-guided pruning:
def compute_prune_mask(self, max_grad, min_opacity, ...):
    prune_mask = (self.get_opacity < min_opacity).squeeze()
    prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
    # Thêm:
    # crs_mask = (self.crs_score < tau_crs).squeeze()
    # prune_mask = torch.logical_or(prune_mask, crs_mask)
```

### 5.2 Các file cần tạo / sửa

| File | Loại | Nội dung |
|---|---|---|
| `depth_alignment.py` | **Mới** | Weighted scale alignment — Eq. 4 từ Chung et al. Align DepthAnything V2 với COLMAP sparse depth. |
| `crs_module.py` | **Mới** | Tính D_i, R_i, CRS_i với EMA. Hook vào training loop mỗi 100 iter sau T_warmup. |
| `scene/gaussian_model.py` | **Sửa** | Thêm `crs_score` attribute. Extend `compute_prune_mask()` (+1 dòng). Sửa `densification()` thêm position constraint. |
| `train.py` | **Sửa** | Thêm warmup logic, adaptive depth loss, gọi crs_module mỗi 100 iter. |
| `metrics_dtu.py` | ~~Mở rộng~~ | ~~Thêm Depth RMSE và Floater Ratio~~ [HOLD] |
| `depth_model.py` | **Mới** | Wrapper DepthAnything V2 thay DPT/MiDaS. |

### 5.3 Chiến lược lấy code từ 4 repo

| Lấy từ | Dùng cho | Phần cụ thể |
|---|---|---|
| **CoR-GS** (base) | Toàn bộ pipeline | train.py, gaussian_model.py, compute_prune_mask(), diff-gaussian-rasterization-confidence, metrics_dtu.py, scripts/ |
| **FSGS** | Pseudo-view generation | Camera interpolation logic — sạch hơn CoR-GS, dùng cho L_pseudo_depth |
| **LoopSparseGS** | D_i computation | DAR sliding window Pearson loss — tính depth consistency per-patch |
| **DNGaussian** | Depth normalization | Global-local normalization để stabilize D_i trong warmup phase |

---

## 6. Tài liệu tham khảo

### Nhóm A — Core methods (bắt buộc đọc kỹ)

| # | Tên | Nội dung chính | Venue | Đọc phần |
|---|---|---|---|---|
| [1] | **3DGS** — Kerbl et al. | 3D Gaussian Splatting for Real-Time Radiance Field Rendering. Baseline foundation. | ACM TOG 2023 | Sec 4 (densification) |
| [2] | **FSGS** — Zhu et al. | Real-Time Few-Shot View Synthesis using Gaussian Splatting. Pseudo-view generation, Pearson depth loss. | ECCV 2024 | Sec 3.2–3.4 |
| [3] | **DNGaussian** — Li et al. | Global-Local Depth Normalization. Hard-Soft depth regularization. Depth normalization trick. | CVPR 2024 | Sec 3 toàn bộ |
| [4] | **SCGaussian** — Peng et al. | Structure Consistent với matching prior. Ray-based Gaussian, cross-view reprojection. | NeurIPS 2024 | Sec 3.2–3.3 |
| [5] | **LoopSparseGS** — Bao et al. | PGI + DAR + SFS. Lấy DAR sliding window Pearson loss code cho D_i. | IEEE TIP 2025 | Sec 3.3 (DAR) |
| [6] | **CoR-GS** — Zhang et al. | Sparse-View 3DGS via Co-Regularization. **Codebase base.** Rendering disagreement → R_i. `metrics_dtu.py`. | ECCV 2024 | Sec 3–4 + toàn bộ code |

### Nhóm B — Evidence và landscape

| # | Tên | Lý do quan trọng | Venue |
|---|---|---|---|
| [7] | **AD-GS** | Root cause floater = unconstrained densification. Justification cho position constraint. | arXiv 2025 |
| [8] | **DropGaussian** (Park et al.) | Prior-free dropout. PSNR 20.76 LLFF 3-view — competitive baseline cần compare. | CVPR 2025 |
| [9] | **DropoutGS** (Xu et al.) | Floater có opacity cao + ổn định. Justification trực tiếp cho bỏ O_i khỏi CRS. | CVPR 2025 |
| [10] | **VGNC** — Lin et al. | Validation-guided Gaussian count control. Bằng chứng external validation > in-loop. | ACM MM 2025 |
| [11] | **TIDI-GS** | PSNR/SSIM không đủ cho NVS evaluation. Justification cho GFS metric. | arXiv 2025 |
| [12] | **SparseGS** — Xiong et al. | Mode-selection depth, floater pruning heuristic. Reference cho GFS design. | arXiv 2023 |
| [13] | **Pixel-GS** — Zhang et al. | Pixel-aware gradient cho densification. Reference cho position constraint. | ECCV 2024 |
| [14] | **Taming 3DGS** | Score-based densification, saliency-aware. Reference cho principled densification. | SIGGRAPH Asia 2024 |

### Nhóm B+ — Technique tham khảo (không nhất thiết cite)

| # | Tên | Lý do | Venue |
|---|---|---|---|
| [—] | **Depth-Regularized Optimization** — Chung et al. | Weighted scale alignment (Eq. 4): dùng reprojection error làm weight khi align depth scale. Adopt technique này vào `depth_alignment.py`, nhưng paper CVPR Workshop 2024 yếu hơn các bài cùng nhóm — có thể bỏ qua trong reference list nếu cần tiết kiệm slot. | CVPR Workshop 2024 |

### Nhóm C — Metric và evaluation

| # | Tên | Lý do | Venue |
|---|---|---|---|
| [15] | **Perceptual Quality Assessment of NeRF** — Liang et al. | SSIM và LPIPS tệ hơn PSNR ở NVS. Justification chính cho GFS proposal. | CGF 2024 |
| [16] | **BOGausS** | Confidence estimation cho 3DGS. Related work gần nhất với CRS — phân biệt: họ post-hoc static, mình in-loop dynamic. | EUSIPCO 2025 |
| [17] | **PUP 3D-GS** — Hanson et al. | Principled uncertainty pruning. Related work — phân biệt: post-hoc compression vs in-loop sparse-view. | CVPR 2025 |

### Nhóm D — NeRF-based baselines (chỉ để compare)

| # | Tên | Venue |
|---|---|---|
| [18] | **SparseNeRF** — Wang et al. | ICCV 2023 |
| [19] | **FreeNeRF** — Yang et al. | CVPR 2023 |
| [20] | **RegNeRF** — Niemeyer et al. | CVPR 2022 |
| [21] | **DietNeRF** — Jain et al. | ICCV 2021 |

### Nhóm E — Depth estimation (tools)

| # | Tên | Dùng cho | Venue |
|---|---|---|---|
| [22] | **DepthAnything V2** — Yang et al. | Depth prior chính trong training loop | CVPR 2024 |
| [23] | **Metric3D v2** — Hu et al. | Optional: metric depth cho initialization | IEEE TPAMI 2024 |
| [24] | **DPT** — Ranftl et al. | Reference (baseline dùng, ta thay bằng DepthAnything V2) | ICCV 2021 |

---

## 7. Tóm tắt một trang

### Câu một dòng
> *"CRSGaussian giải quyết floater và overdensification trong sparse-view 3DGS bằng cách gán confidence score động cho mỗi Gaussian, dùng nó để điều phối depth constraint và densification, đồng thời đề xuất metric mới để đánh giá geometry quality mà PSNR/SSIM bỏ sót."*

### Điểm khác biệt với các paper liên quan nhất

| So với | CRSGaussian khác ở đâu |
|---|---|
| **CoR-GS** (base) | Single-model — không cần 2 fields, 2x rẻ hơn. R_i từ cross-view warp thay 2-field disagreement. |
| **FSGS** | Thêm reprojection consistency + position constraint + weighted depth alignment |
| **VGNC** | Dùng pseudo-views có sẵn thay NVS generative model — lightweight hơn nhiều |
| **DNGaussian** | Adaptive per-Gaussian weight thay fixed normalization |
| **SCGaussian** | Không cần external matching model (GIM/LoFTR) |

### Những gì KHÔNG làm (và lý do)

- **Không dùng diffusion prior** — oversmoothing, chậm, không giải quyết geometry trực tiếp
- **Không dùng Opacity Stability** — floater có opacity cao và ổn định (bằng chứng từ DropoutGS)
- **Không freeze shape** — root cause là position sai, không phải shape sai (bằng chứng từ AD-GS)
- **Không dùng Metric3Dv2 trong training loop** — fragile trên outdoor, DepthAnything V2 là đủ
- **Không dùng FSGS làm base** — thiếu R_i và DTU eval, phải viết thêm ~2 tuần không cần thiết
- **Không dùng LoopSparseGS làm base** — 35–45 phút/run, ablation study sẽ mất hàng trăm giờ GPU
- **Không adopt toàn bộ Chung et al.** — paper CVPR Workshop, kết quả thấp hơn FSGS/DNGaussian; chỉ lấy weighted alignment technique

---

## 8. Changelog

| Version | Thay đổi |
|---|---|
| v1.0 | Bản gốc — FSGS làm codebase chính |
| v2.0 | **[Codebase]** Chuyển sang CoR-GS: có `metrics_dtu.py`, rendering disagreement → R_i, `compute_prune_mask()` extensible. **[Methodology]** Thêm weighted scale alignment (Sec 3.2) từ Chung et al. — stabilize D_i. **[Ablation]** Thêm A6 (naive vs weighted alignment). **[Timeline]** Thêm tuần 2 cho depth alignment. **[Files]** Thêm `depth_alignment.py`. **[References]** Thêm Chung et al. vào Nhóm B+. |
| **v3.0** | **[CRS₀]** Informed init: COLMAP points = 1-normalize(reproj_error), densified = 0.5. **[Pipeline]** Stage 2 chia thành 2a (pre-densify, 0→500) và 2b (early densify+position constraint, 500→2000). Position constraint BẬT từ T_densify=500, không đợi T_warmup. **[GFS]** HOLD — C3 tạm gác, focus C1+C2. **[SOTA]** Thêm bảng SOTA 2025 (D2GS 21.35 dẫn đầu). Target >21.0 competitive, >21.35 beat. **[Ablation]** Thêm T_densify, CRS update interval, CRS₀ init variants. **[Depth]** Cần ablation DPT vs DepthAnything V2 trước khi quyết định. |

---

*Document version: 3.0 · Cập nhật: Tháng 3, 2026*