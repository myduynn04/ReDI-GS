# CRSGaussian — Context for Claude Code

> Đọc file này trước khi làm bất kỳ thứ gì.
> Đọc thêm docs/01_research_summary.md để hiểu đầy đủ methodology.

---

## Dự án là gì

CRSGaussian giải quyết hai vấn đề trong sparse-view 3D Gaussian Splatting:
- **Floater**: Gaussian đặt sai vị trí 3D vì densification không có position constraint
- **Overdensification**: Quá nhiều Gaussian → overfit training views

Giải pháp: **CRS (Confidence-Reliability Score)** — per-Gaussian dynamic signal
tích hợp depth consistency (D_i) + reprojection consistency (R_i), dùng để
điều phối depth loss, densification, và pruning.

Ba contribution chính (focus hiện tại: C1 + C2):
1. **CRS Module** — D_i + R_i, update mỗi 100 iter sau T_warmup
2. **CRS-guided Densification** — adaptive depth loss + position constraint (từ T_densify=500) + multi-signal pruning
3. ~~**GFS Metric**~~ — [HOLD] Depth RMSE + Floater Ratio — tạm gác

---

## Codebase structure

```
CoR-GS/          ← BASE — mọi thay đổi implement vào đây
FSGS/            ← CHỈ ĐỌC — lấy pseudo-view generation logic
DNGaussian/      ← CHỈ ĐỌC — lấy depth normalization trick
LoopSparseGS/    ← CHỈ ĐỌC — lấy DAR sliding window Pearson loss
DepthRegularizedGS/ ← CHỈ ĐỌC — lấy weighted scale alignment
SCGaussian/      ← KHÔNG DÙNG CODE — chỉ đọc paper
```

---

## File sẽ tạo mới trong CoR-GS/

| File | Mô tả |
|------|-------|
| `depth_alignment.py` | Weighted scale alignment: align DepthAnything V2 với COLMAP sparse depth |
| `crs_module.py` | Tính D_i, R_i, CRS_i với EMA. Gọi mỗi 100 iter sau T_warmup |
| `depth_model.py` | Wrapper DepthAnything V2 thay DPT/MiDaS |

## File sửa trong CoR-GS/

| File | Sửa gì |
|------|--------|
| `scene/dataset_readers.py` | Extract reprojection errors từ COLMAP — KHÔNG discard field thứ 3 |
| `scene/gaussian_model.py` | Thêm `_crs_score` attribute (KHÔNG dùng `confidence`), extend `compute_prune_mask()`, thêm position constraint vào `densify_and_prune()` |
| `train.py` | Thêm warmup logic, gọi crs_module mỗi 100 iter, thêm depth loss mới, adaptive depth loss sau T_warmup |
| `metrics_dtu.py` | ~~Thêm Depth RMSE và Floater Ratio~~ [HOLD] |
| `utils/loss_utils.py` | Thêm pearson_depth_loss() và adaptive_depth_loss() |

---

## Công thức cốt lõi

```python
# ── CRS₀ Initialization — Informed (updated 2026-04) ──
# Triết lý mới: CRS₀ có ý nghĩa hình học ngay từ đầu, không neutral 0.5.
# Lý do: iter 500-1000 densification chạy "mù" khi CRS₀=0.5 → floater sinh tự do
#         → CRS chỉ detect được sau T_warmup=1000 → quá muộn.
#
# COLMAP Gaussians — 3 signals kết hợp:
#   q_reproj = 1 - clip(reproj_error / τ_r, 0, 1)       # τ_r=2.5
#   q_depth  = 1 - clip(|d_DAV2 - d_COLMAP| / range, 0, 1)
#   q_view   = (n_obs - 1) / max(N_train - 1, 1)
#   Q_i = w_r*q_reproj + w_d*q_depth + w_v*q_view        # default equal 1/3
#   ℓᵢ⁽⁰⁾ = γ * (Q_i - 0.5)                              # γ=5.0
#   CRS₀ = sigmoid(ℓᵢ⁽⁰⁾)   [lưu logit ℓᵢ⁽⁰⁾ vào _crs_score]
#
# Densified Gaussians — conservative inherit:
#   CRS₀_child = clip(η * CRS_parent, 0, 0.5)            # η=0.7
#   Child không bao giờ bắt đầu trên neutral → phải "earn" CRS cao
#
# Ablation-friendly: mỗi component bật/tắt độc lập qua arguments.
# Master switch: --informed_crs_init (default False → giữ behavior cũ)
# Xem docs/09_informed_crs_init_plan.md cho chi tiết implementation.
#
# Behavior cũ (khi informed_crs_init=False):
self._crs_score = torch.zeros(N, 1)   # logit space → sigmoid(0) = 0.5

# ── Depth consistency ──
# Chỉ tính cho visible Gaussians (visibility_filter = radii > 0)
# Gaussian bị occlude → giữ D_i = 0.5 (neutral)
D_i = 1 - abs(d_render_i - d_prior_i) / depth_range   # range [0, 1]
# d_render_i: tính từ project 3D position qua camera matrix
# KHÔNG dùng rendered depth per-pixel (alpha-weighted average)

# ── Reprojection consistency ──
R_i = 1 - mean(abs(c_i - warp(c_j→i, depth_j))) / 255   # range [0, 1]

# ── CRS — scale logit để tận dụng full range sigmoid ──
# VẤN ĐỀ: sigmoid(0.5*D + 0.5*R) với D,R ∈ [0,1] → CRS ∈ [0.5, 0.73]
#          tau_crs=0.2 không bao giờ trigger → pruning không hoạt động
# GIẢI PHÁP: scale × 5.0 → logit ∈ [-2.5, 2.5] → CRS ∈ [0.08, 0.92]
crs_logit = 5.0 * (w1 * D_i + w2 * R_i - 0.5)
CRS_i = sigmoid(crs_logit)   # range [0.08, 0.92]
# w1=w2=0.5, ablate [0.3,0.7] và [0.7,0.3]
# Hằng số 5.0: ablate {3.0, 5.0, 8.0} nếu cần
# Update mỗi 100 iter SAU T_warmup

# ── Ngưỡng CRS ──
# Chọn sau khi có histogram thực nghiệm từ T2.7
tau_crs      = 0.35   # ngưỡng prune: CRS < 0.35 AND opacity thấp AND isolated
tau_densify  = 0.45   # ngưỡng chặn sinh con: CRS < 0.45 → skip densify
# tau_densify > tau_crs: chặn sinh con trước, xóa sau — logic đúng thứ tự
# Ablate: tau_crs ∈ {0.25, 0.35, 0.45}, tau_densify ∈ {0.40, 0.45, 0.50}

# ── Fixed depth loss ──
# CoR-GS gốc KHÔNG có depth loss — đây là thêm hoàn toàn mới
# Depth loss = correction (kéo Gaussian về đúng depth)
# CRS = elimination (prune floater qua Phase 4)
# Hai cơ chế tách biệt, không overlap
L_depth = lambda_base * pearson_depth_loss(rendered_depth, depth_prior)
# lambda_base = 0.05, ablate {0.01, 0.05, 0.10}

# ── depth_range ──
# depth_range = median(far) - median(near) across tất cả training cameras
# Dùng median (robust với outlier), tính một lần ở đầu training

# ── Position constraint — DISABLED ──
# Thực nghiệm: epsilon=0.05*depth_range quá hẹp vì DAV2 noise
# → reject Gaussians hợp lệ → PSNR giảm 3 dB. Xem decisions_log.
# Depth loss + CRS pruning đủ kiểm soát floater.

# ── Pruning — Option C, chỉ sau T_warmup ──
# T2.7 phát hiện: floater có opacity=0.90 → AND(CRS, opacity<0.005) vô hiệu
# Option C: tách CRS pruning thành kênh riêng, giữ legacy opacity pruning
prune = (CRS_i < tau_crs AND knn_dist > tau_isolated) OR (opacity < 0.005)
```

---

## Quy tắc bắt buộc — KHÔNG được vi phạm

1. **KHÔNG sửa** file trong FSGS/, DNGaussian/, LoopSparseGS/, DepthRegularizedGS/, SCGaussian/
2. **LUÔN đọc file gốc trước** khi sửa — không đoán API
3. **KHÔNG implement nhiều hơn 1 task** trong 1 session
4. **Sau mỗi task**: chạy test nhỏ để confirm không break baseline
5. **Tóm tắt hiểu biết** về file gốc trước khi đề xuất implementation
6. **Đợi confirm** trước khi bắt đầu code — đừng tự ý code ngay
7. **GFS metric (C3) — HOLD**
8. **KHÔNG dùng `confidence` attribute** của CoR-GS cho CRS — thêm `_crs_score` riêng
9. **CRS pruning chỉ active sau T_warmup**

---

## Quy tắc chú thích code — BẮT BUỘC mọi thay đổi

Khi thêm hoặc sửa bất kỳ đoạn code nào, Claude Code PHẢI làm đủ 3 việc:

### Việc 1 — Header block ở đầu mỗi function/class thêm mới
```python
# ============================================================
# [CRSGaussian] Task: T2.3 — compute_depth_consistency
# File: CoR-GS/crs_module.py  (file mới tạo)
# Mục đích: Tính D_i per-Gaussian — đo Gaussian có đứng đúng
#           độ sâu so với depth prior không
# Được gọi từ: train.py, sau T_warmup, mỗi 100 iter
# ============================================================
```

### Việc 2 — Inline comment tại mỗi logic quan trọng
```python
# Chỉ update visible Gaussians (radii > 0 sau rasterization)
# Lý do: Gaussian bị occlude không nên bị penalize — depth của
#        nó không thể so sánh với depth prior tại pixel đó
vis = out['visibility_filter']

# Dùng 3D projection, KHÔNG dùng rendered depth per-pixel
# Lý do: rendered depth là alpha-weighted avg của nhiều Gaussians,
#        không phải depth chính xác của Gaussian i
d_proj = project_to_depth(xyz, cam)

# Gaussian không visible từ bất kỳ camera nào → neutral 0.5
# Không update vì không có thông tin để đánh giá
D[~valid] = 0.5
```

### Việc 3 — Báo cáo tóm tắt sau khi implement
```
File đã sửa/tạo:
  - CoR-GS/crs_module.py  (TẠO MỚI) — thêm compute_depth_consistency()
  - CoR-GS/train.py       (SỬA dòng 145-160) — thêm CRS update block

Chú thích đã thêm:
  - Header block tại compute_depth_consistency()
  - Inline comment tại: visibility filter, 3D projection,
    neutral default, EMA update, torch.no_grad()
```

---

## Phát hiện quan trọng từ code review

| Phát hiện | Impact |
|-----------|--------|
| CoR-GS **KHÔNG có depth loss** | Phải thêm hoàn toàn mới, không sửa loss cũ |
| `confidence` attribute dùng cho rasterizer | Thêm `_crs_score` riêng, không repurpose |
| COLMAP reprojection errors bị **discard** | Sửa dataset_readers.py trước T1.2 |
| CoR-GS train 2 fields, **default gaussiansN=1** | CRSGaussian chạy default, không cần flag |
| **sigmoid(D,R) range quá hẹp [0.5, 0.73]** | Phải scale logit × 5.0 trước sigmoid |
| Co-pruning dùng **Open3D evaluate_registration** (nearest neighbor, không phải ICP) | Verify PASS ở T0.0 |
| Single-field **OOM** khi không có co-pruning | Gaussian tăng vô hạn → cần CRS pruning thay thế |
| DAV2 output **inverse depth** (giá trị lớn = gần) | Alignment cho scale âm (-0.09) — đúng kỳ vọng |
| CoR-GS co-reg/co-prune **không dùng được** cho single-field R_i | R_i phải viết mới: cross-view warp + color compare |

---

## Task hiện tại

Xem **docs/03_task_queue.md** — làm theo thứ tự, không nhảy cóc.

```
Phase 0 — DONE
  T0.0  ✓ Grep verify PASS
  T0.1  ✓ Environment OK
  T0.2  ✓ DAV2 vitl installed + utils/depth/depth_model.py
  T0.3  ✓ LLFF 8 scenes
  T0.X  ✓ SKIP — đọc errors trực tiếp từ bin
  T0.5  ✓ Baseline 2-field: Avg PSNR=20.11
  T0.5b   Partial: fern PSNR=21.15, others OOM (no co-pruning)

Phase 1 — DONE
  T1.1  ✓ SKIP — closed-form WLS thay Adam
  T1.2  ✓ utils/depth/depth_alignment.py
  T1.3  ✓ Tích hợp vào train.py (--use_depth_prior flag)
  T1.4  ✓ Depth map verified (scale=-0.09, range [16.9, 47.6])

Phase 2 — DONE
  T2.1  ✓ Đọc CoR-GS disagreement → không refactor, viết mới R_i
  T2.2  ✓ _crs_score attribute trong gaussian_model.py (5 chỗ sửa)
  T2.3  ✓ D_i function
  T2.4  ✓ R_i function (GT color pairwise)
  T2.5  ✓ update_crs() — EMA logit space, scale=5.0
  T2.6  ✓ Hook vào train.py — LOG ONLY
  T2.7  ✓ Validate CRS distribution — bimodal confirmed

Phase 3 — DONE
  T3.1  ✓ pearson_depth_loss()
  T3.3  ✓ Fixed depth loss λ=0.05 → PSNR 22.35 (+1.2)

Phase 4 — IN PROGRESS
  T4.1  ✓ Position constraint — DISABLED (PSNR -3dB)
  T4.2  ✓ CRS pruning Option C
  T5.x  ← NEXT: Informed CRS₀ Initialization (Phase 5)
```

---

## Lưu ý kỹ thuật

- **T_warmup = 1000** — ablate {500, 1000, 2000}
- **T_densify = 500** — iter bắt đầu densification + position constraint
- **depth_range = median(far) - median(near)** — tính một lần
- **D_i dùng 3D projection + visibility filter**
- **CRS scale factor = 5.0** — CRS range [0.08, 0.92]
- **tau_crs = 0.35, tau_densify = 0.45** — provisional, xác nhận sau T2.7
- **CRS₀ — Informed init** (updated 2026-04): COLMAP Gaussians dùng q_reproj+q_depth+q_view, densified dùng conservative inherit. Gated bởi --informed_crs_init
- CRS update interval = 100 iter — ablate {50, 100, 200}

---

## Framing paper

1. **Per-Gaussian dynamic quality signal** — mới, chưa ai update in-loop điều phối cả 3 quyết định
2. **R_i per-Gaussian** — chưa ai làm, CoR-GS dùng 2-field (khác hoàn toàn)
3. **Single-model thay 2-field** — 2x rẻ hơn CoR-GS
4. **Interpretable** — biết Gaussian nào fail và tại sao