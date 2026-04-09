# CRSGaussian — Research Summary
### Confidence-Reliability Guided Sparse-View 3D Gaussian Splatting

> Tổng hợp nghiên cứu · Few-Shot Novel View Synthesis · Tháng 4, 2026
> **Version 4.0** — Cập nhật: Informed CRS₀ (Phase 5 DONE), position constraint DISABLED,
> depth loss fixed lambda, CRS pruning Option C, EMA/interval/weight ablation results.

---

## 1. Mục tiêu nghiên cứu

### 1.1 Bài toán

Few-shot Novel View Synthesis (NVS) — tổng hợp góc nhìn mới từ **chỉ 3–8 ảnh đầu vào**. 3D Gaussian Splatting (3DGS) là SOTA về tốc độ (real-time rendering), nhưng sụt giảm nghiêm trọng với sparse views:

| Vấn đề | Mô tả | Bằng chứng |
|---|---|---|
| **Floater** | Gaussian đặt sai vị trí 3D — đúng từ training views, sai từ novel views | AD-GS 2025, T2.7: floater CRS=0.316, opacity=0.903 |
| **Overdensification** | Số Gaussian tăng không kiểm soát → overfit training views | DropoutGS CVPR 2025, T3.3: N giảm 65% khi có depth loss |

### 1.2 Ba contribution chính

```
C1 — CRS Module                                              ← DONE (Phase 2)
     Per-Gaussian dynamic quality score: depth consistency (D_i) + reprojection
     consistency (R_i). Single-model, in-loop. EMA update trên logit space.

C2 — CRS-guided Densification Control                        ← IN PROGRESS
     (a) Fixed depth loss (Pearson, λ=0.05)                   ← DONE (Phase 3)
     (b) Position constraint                                  ← DISABLED (PSNR -3dB)
     (c) CRS pruning Option C: (CRS<0.35 AND isolated) OR legacy  ← DONE (Phase 4)
     (d) Informed CRS₀: 3-signal geometry prior               ← DONE (Phase 5)

C3 — Geometric Fidelity Score (GFS)                           ← HOLD
     Tạm gác — focus C1+C2 trước.
```

---

## 2. Methodology — Cập nhật theo thực nghiệm

### 2.1 Pipeline tổng quan (updated v4.0)

```
Input: 3 sparse views + COLMAP poses
         ↓
[Pre-training] Depth alignment
  - DepthAnything V2 (vitl) predict relative depth
  - Closed-form WLS align với COLMAP sparse depth
  - weight = 1/reprojection_error
  - Output: aligned_depth_dict + depth_range
         ↓
[Pre-training] Informed CRS₀ Initialization       ← MỚI (Phase 5)
  - q_reproj = 1 - clip(reproj_error / τ_r, 0, 1)
  - q_depth  = 1 - clip(|d_DAV2 - d_COLMAP| / depth_range, 0, 1)
  - q_view   = (n_obs - 1) / max(N_train - 1, 1)
  - Q_i = w_r*q_reproj + w_d*q_depth + w_v*q_view   (auto-normalize)
  - CRS₀ = sigmoid(γ * (Q_i - 0.5))                 γ=5.0
  - Densified Gaussians: neutral 0.5 (inherit DISABLED — hại khi kết hợp pruning)
         ↓
[Iter 0 → 500] Pre-densification
  - Photometric loss (L1 + SSIM) + fixed depth loss
  - Chưa densify, chưa prune
         ↓
[Iter 500 → T_warmup] Densification without CRS
  - Densification bắt đầu (clone/split/prune legacy)
  - CRS₀ informed → densification không blind
  - CRS update chưa bắt đầu
         ↓
[Iter T_warmup → 10000] Full CRS active
  - CRS update mỗi crs_update_interval iter (D_i + R_i → EMA)
  - CRS pruning: (CRS < 0.35 AND isolated) OR (opacity < 0.005)
  - Densification tiếp tục + CRS pruning loại floater
```

### 2.2 Công thức CRS — cập nhật

```python
# ── CRS₀ Initialization (Phase 5) ──
# Dùng geometry info có sẵn, thay vì neutral 0.5
Q_i = w_r*q_reproj + w_d*q_depth + w_v*q_view
crs_logit_init = 5.0 * (Q_i - 0.5)
CRS₀ = sigmoid(crs_logit_init)   # range [0.08, 0.92]

# ── CRS Update (mỗi crs_update_interval iter sau T_warmup) ──
D_i = 1 - |d_projected - d_prior| / depth_range        # [0, 1]
R_i = 1 - mean(pairwise L1 color diff cross-view)      # [0, 1]
crs_logit_new = 5.0 * (0.5*D_i + 0.5*R_i - 0.5)
_crs_score = ema * _crs_score_old + (1-ema) * crs_logit_new

# ── CRS Pruning (Option C) ──
prune = (CRS < 0.35 AND knn_dist > 0.1*extent) OR (opacity < 0.005)
```

### 2.3 Loss Function (simplified v4.0)

```
L = 0.8 * L_L1 + 0.2 * L_SSIM + 0.05 * L_Pearson_depth

# Depth loss = fixed lambda, KHÔNG adaptive
# CRS điều khiển pruning, KHÔNG điều khiển loss weight
# Hai cơ chế tách biệt: correction (depth loss) + elimination (CRS pruning)
```

### 2.4 Thay đổi so với v3.0

| Thành phần | v3.0 (kế hoạch) | v4.0 (thực nghiệm) | Lý do |
|---|---|---|---|
| **Position constraint** | BẬT từ T_densify=500 | **DISABLED** | PSNR -3 dB vì DAV2 noise → reject Gaussians hợp lệ |
| **Adaptive depth loss** | CRS weight loss | **Fixed lambda** | Depth loss (correction) + CRS pruning (elimination) tách biệt sạch hơn |
| **CRS₀** | 1-normalize(reproj) | **3-signal: q_reproj + q_depth + q_view** | Tận dụng đầy đủ geometry info có sẵn |
| **CRS₀ densify inherit** | Planned | **DISABLED** | C8 experiment: inherit + pruning → xóa nhầm child hữu ích (-1.67 dB) |
| **Pruning logic** | AND(CRS, opacity, isolated) | **Option C: (CRS AND isolated) OR opacity** | T2.7: floater opacity=0.90 → AND logic vô hiệu |
| **T_warmup** | 2000 | **1000** (default, ablation pending) | 10k training → T_warmup=2000 chiếm 20%, quá conservative |
| **EMA decay** | 0.9 | **0.3** (best from ablation) | ema=0.9 lag quá, CRS không kịp adapt |
| **CRS update interval** | 100 | **100** (confirmed best) | Ablation {25,50,75,100}: int=100 ổn định nhất |

---

## 3. Kết quả thực nghiệm (tháng 4, 2026)

### 3.1 Baselines đã chạy

| Config | PSNR fern (10k) | Ghi chú |
|---|---|---|
| CoR-GS 2-field (T0.5) | 22.43 | gaussiansN=2, coreg, coprune |
| Single-field baseline (T0.5b) | 21.15 | gaussiansN=1, 3k iter |
| Depth loss only (T3.3) | 22.35 | +1.2 dB vs single-field baseline |

### 3.2 Informed CRS₀ Ablation — fern 3-view

| Config | Informed | Inherit | Pruning | PSNR @10k |
|---|:---:|:---:|:---:|:---:|
| C0 | No | No | No | 21.92 |
| C0+prune | No | No | Yes | 21.99 |
| C7+prune | **Yes** | No | Yes | **22.11** |
| C8+prune | Yes | Yes | Yes | 20.32 (broken) |

**C7+prune (informed, no inherit) = best.** Inherit + pruning → xóa nhầm.

### 3.3 EMA Ablation — 8 LLFF scenes (int=100)

| EMA | AVG PSNR @10k | Drop (peak→10k) |
|:---:|:---:|:---:|
| 0.9 | 20.21 | -0.43 |
| 0.7 | 20.30 | -0.10 |
| 0.5 | 20.25 | -0.02 |
| **0.3** | **20.33** | **-0.02** |
| 0.2 | 20.26 | -0.14 |
| 0.1 | 20.20 | -0.24 |

**ema=0.3 = best:** AVG cao nhất + ổn định nhất.

### 3.4 Interval Ablation — 8 LLFF scenes

| Config | AVG @10k | Drop |
|---|:---:|:---:|
| ema=0.3, int=100 | **20.33** | **-0.02** |
| ema=0.5, int=75 | 20.38 | -0.06 |
| ema=0.3, int=50 | 20.30 | -0.12 |

**ema=0.3/int=100 = best balance** (AVG cao, drop gần 0).

### 3.5 Weight Ablation — 8 LLFF scenes (ema=0.3, int=100)

| Config | w_reproj | w_depth | w_view | AVG @10k |
|---|:---:|:---:|:---:|:---:|
| WE | 0.3 | 0.5 | 0.2 | **20.29** |
| WB | 0.6 | 0.1 | 0.3 | 20.27 |
| WC (equal) | 0.33 | 0.33 | 0.34 | 20.24 |
| WF (no view) | 0.5 | 0.5 | 0 | 20.23 |

**Spread chỉ 0.07 dB — weights gần như không ảnh hưởng.**
q_depth quan trọng nhất (flower +0.6 dB khi depth weight cao).
q_view gần vô dụng cho 3-view (chỉ 3 giá trị discrete).
Đang test thêm: WG(0.4/0.6), WH(0.6/0.4), WI(0.3/0.7), WJ(0.7/0.3) + W0 (no informed init).

### 3.6 Phát hiện quan trọng

| Phát hiện | Bằng chứng | Impact |
|---|---|---|
| Floater có opacity CAO (0.90) | T2.7 histogram | AND(CRS, opacity) pruning vô hiệu → Option C |
| Position constraint hại hơn lợi | T4.1: PSNR -3 dB | DAV2 noise → reject Gaussians hợp lệ |
| Depth loss = correction đủ mạnh | T3.3: +1.2 dB, N -65% | Không cần adaptive weight |
| CRS inherit + pruning = toxic | C8: -1.67 dB vs C7 | Child CRS thấp → bị prune nhầm |
| EMA 0.9 quá chậm | Ablation: drop -0.43 vs -0.02 | CRS lag behind, không adapt kịp |
| CRS₀ informed giúp iter sớm | C7 +0.33 dB @1k vs CoR-GS | Densification 500-1000 không blind |

---

## 4. Hyperparameters chốt (pending final ablation)

| Param | Giá trị | Status |
|---|---|---|
| EMA decay | **0.3** | Confirmed (8-scene ablation) |
| Update interval | **100** | Confirmed |
| T_warmup | **1000** | Default, ablation pending |
| CRS₀ weights | **0.33/0.33/0.34** (equal) | ~confirmed, final test running |
| gamma | **5.0** | Confirmed (T5.8) |
| tau_r | **2.5** | Default |
| Depth loss lambda | **0.05** | Fixed |
| tau_crs | **0.35** | Default |
| tau_isolated | **0.1** | Default |
| Informed CRS₀ | **ON** | Confirmed |
| Densify inherit | **OFF** | Confirmed (harmful) |
| Position constraint | **OFF** | Confirmed (harmful) |

---

## 5. SOTA Context (tháng 3/2026)

| Method | PSNR LLFF 3-view | Venue |
|---|---|---|
| CoR-GS | 20.45 | ECCV 2024 |
| DropGaussian | 20.76 | CVPR 2025 |
| LoopSparseGS | 20.85 | TIP 2025 |
| CuriGS | 21.10 | arXiv 2025 |
| HBSplat | 21.13 | arXiv 2025 |
| D2GS | 21.35 | arXiv 2025 |
| NexusGS | ~21.47 | CVPR 2025 Highlight |
| **CRSGaussian (hiện tại, best config, 8-scene AVG)** | **~20.33** | — |

**Gap với SOTA:** ~1 dB. Bottleneck chính: **overfit** (train 38 dB vs test 20 dB = gap 18 dB).
CoR-GS 2-field kiểm soát overfit tốt hơn nhờ co-prune. CRS pruning hiện tại chưa đủ aggressive.

---

## 6. Next Steps

| Priority | Task | Mục đích |
|---|---|---|
| 1 | Chốt weight ablation (WG-WJ + W0 đang chạy) | Final weights cho paper |
| 2 | T_warmup ablation (0, 250, 500, 750, 1000, 1500, 2000) | Tối ưu khi nào bắt đầu pruning |
| 3 | Giải quyết overfit gap (18 dB) | Bottleneck lớn nhất — cần aggressive pruning hoặc regularization mới |
| 4 | Multi-scene full eval (8 LLFF scenes) | Bảng chính cho paper |
| 5 | DTU evaluation | Benchmark thứ 2 |

---

## 7. Changelog

| Version | Thay đổi |
|---|---|
| v1.0 | Bản gốc — FSGS làm codebase chính |
| v2.0 | Chuyển sang CoR-GS. Thêm weighted scale alignment. |
| v3.0 | CRS₀ informed init (1 signal), position constraint plan, GFS HOLD, SOTA context. |
| **v4.0** | **[Phase 2 DONE]** CRS module: D_i + R_i + EMA update. T2.7 validated: floater CRS=0.316 vs surface 0.820. **[Phase 3 DONE]** Fixed depth loss +1.2 dB. **[Phase 4 DONE]** Position constraint DISABLED (-3dB). CRS pruning Option C. **[Phase 5 DONE]** Informed CRS₀ 3-signal (q_reproj + q_depth + q_view). C7+prune best (+0.12 dB). Inherit DISABLED (toxic with pruning). **[Ablation]** EMA=0.3 best. Interval=100 best. Weights: equal ≈ depth-heavy (spread <0.07 dB). **[Key finding]** Overfit gap 18 dB = bottleneck chính, không phải CRS init hay weights. |

---

*Document version: 4.0 · Cập nhật: Tháng 4, 2026*
