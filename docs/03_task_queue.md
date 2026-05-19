# Task Queue — CRSGaussian Implementation

> Làm theo thứ tự. KHÔNG nhảy cóc.
> Tick [x] sau khi xong và verify.
> Ghi kết quả số vào cột "Kết quả".

---

## Trạng thái legend
- `[ ]` Chưa làm
- `[~]` Đang làm
- `[x]` Xong + verified
- `[!]` Bị block — cần giải quyết trước

---

## PHASE 0 — Setup & Baseline

| # | Task | Kết quả | Ghi chú |
|---|------|---------|---------|
| T0.0 | **Grep verify ICP/gs1 không hardcoded** | [x] Pass | 18 matches gaussiansN, 10 gs1, 3 coprune, 1 open3d — tất cả CONDITIONAL (gated bởi args). Single-field gaussiansN=1 an toàn. |
| T0.1 | Cài environment CoR-GS | [x] Done | Server đã có conda env |
| T0.2 | Cài DepthAnything V2 + viết depth_model.py | [x] Done | Checkpoint vitl 1.3GB tại ../Depth-Anything-V2/checkpoints/. Wrapper tại utils/depth/depth_model.py |
| T0.3 | Chuẩn bị LLFF dataset | [x] Done | 8 scenes đã có trên server |
| T0.4 | Chuẩn bị DTU dataset | [ ] | Để sau |
| T0.X | ~~Sửa dataset_readers.py~~ | [x] SKIP | Không cần — depth_alignment.py đọc errors trực tiếp từ points3D.bin |
| T0.5 | **Baseline CoR-GS — LLFF 8 scenes 3-view (gaussiansN=2, --coreg, --coprune)** | [x] Avg PSNR=20.11 SSIM=0.704 LPIPS=0.201 | Per-scene: fern 22.29, flower 19.86, fortress 23.46, horns 19.02, leaves 16.75, orchids 15.59, room 21.47, trex 22.43 |
| T0.5b | **Single-field baseline — LLFF 3-view (gaussiansN=1, no coreg, no coprune)** | fern: PSNR=21.15 SSIM=0.678 LPIPS=0.232 | Chỉ fern chạy được, các scene khác OOM (không có co-pruning kiểm soát Gaussian count) |
| T0.6 | **Baseline CoR-GS — DTU scan24 3-view** | PSNR=? SSIM=? | |
| T0.6b | **Single-field baseline — DTU scan24 3-view** | PSNR=? SSIM=? | Tương tự T0.5b cho DTU |
| T0.7 | Visualize floater từ depth map | | Confirm floater visible trên cả 2-field và 1-field |
| T0.8 | **Chạy CoR-GS với DPT (baseline gốc)** | PSNR=? | Depth estimator ablation |
| T0.9 | **Chạy CoR-GS với DepthAnything V2** | PSNR=? | So sánh với T0.8 — nếu ≈ thì giữ DPT |

### Kết quả cần ghi sau Phase 0

| Run | gaussiansN | coreg | coprune | PSNR | SSIM | LPIPS |
|-----|-----------|-------|---------|------|------|-------|
| T0.5 — CoR-GS gốc | 2 | ✓ | ✓ | 20.11 | 0.704 | 0.201 |
| T0.5b — Single-field | 1 | ✗ | ✗ | ? | ? | ? |
| Gap (T0.5 - T0.5b) | — | — | — | ? | ? | ? |

> Gap > 0.3 dB → co-reg/co-prune quan trọng → CRS phải compensate
> Gap < 0.15 dB → single-field đủ mạnh → CRS có path rõ ràng hơn

---

## PHASE 1 — Depth Alignment

> **Prerequisite:** T0.2 (depth_model.py + DAV2) phải xong trước T1.2. T0.X bỏ — đọc errors trực tiếp từ bin trong depth_alignment.py.

| # | Task | Kết quả | Ghi chú |
|---|------|---------|---------|
| T1.1 | Đọc DepthRegularizedGS weighted alignment code | [x] SKIP | Đã đọc và hiểu approach. Quyết định dùng closed-form WLS thay Adam — ghi decisions_log. |
| T1.2 | Tạo `utils/depth/depth_alignment.py` | [x] Done | Closed-form WLS, weight=1/reproj_err. Scale âm (DAV2 inverse depth) — đúng kỳ vọng. |
| T1.3 | Tích hợp vào `train.py` | [x] Done | --use_depth_prior flag. Test: baseline không break (PSNR 20.94 vs 20.94). DAV2+align chạy OK: scale≈-0.09, depth_range=30.56 |
| T1.4 | Test: so sánh depth map trước/sau alignment | [x] Pass | Raw [0,328] → Aligned [16.9, 47.6]. Scale âm flip đúng chiều. Depth structure giữ nguyên. debug_depth/ verified trực quan. |
| T1.5 | Ablation A6: naive vs weighted | [ ] | Chạy fern scene — để sau khi có depth loss (Phase 3) |

> **Phase 1 Summary:** DAV2 vitl predict relative depth → closed-form WLS align với COLMAP (weight=1/reproj_err) → metric depth. aligned_depth_dict + depth_range sẵn sàng cho Phase 2 (D_i) và Phase 3 (depth loss). Baseline không bị break.

---

## PHASE 2 — CRS Module

| # | Task | Kết quả | Ghi chú |
|---|------|---------|---------|
| T2.1 | Đọc CoR-GS rendering disagreement logic | [x] Done | Không refactor — co-reg/co-prune cần 2 fields, không dùng cho single-field R_i. Chỉ tái dùng: render() trả về color (N,3) + visibility_filter. R_i viết mới. |
| T2.2 | Thêm `_crs_score` attribute vào gaussian_model.py | [x] Done | 5 chỗ sửa: create_from_pcd (init=0), get_crs property, densification_postfix (neutral), prune_points, capture/restore (backward-compatible). Test PASS: PSNR 20.94, 104 it/s. |
| T2.3 | Tạo `CoR-GS/crs_module.py` — D_i function | [x] Done | GPU projection, multi-camera avg, neutral 0.5 invisible. File: utils/crs/crs_module.py. Unit test 8/8 PASS. |
| T2.4 | Tạo `CoR-GS/crs_module.py` — R_i function | [x] Done | Hướng 1: GT color pairwise. File: utils/crs/crs_module.py. Unit test 6/6 PASS. |
| T2.5 | Tạo `CoR-GS/crs_module.py` — update_crs() | [x] Done | EMA trên logit space, scale=5.0, w1=w2=0.5 default. Unit test 6/6 PASS. Surface=0.74, floater=0.46, invisible=0.50. |
| T2.6 | Hook vào train.py — LOG ONLY (không đổi loss) | [x] Done | PSNR 21.10 (baseline 21.15, no regression). CRS range [0.14, 0.90]. <0.35=4572 (2.2%), >0.65=54679 (25.9%). Signal separation confirmed. |
| T2.7 | Validate: floater có CRS thấp hơn surface? | [x] Done | Floater CRS=0.316 opacity=0.903, surface CRS=0.820 opacity=0.432. Histogram bimodal (log). Floaters_only.ply confirmed. Pruning AND logic cần sửa → Option C ở T4.2. |

---

## PHASE 3 — Depth Loss

> **Nhắc lại:** CoR-GS gốc KHÔNG có depth loss. Đây là thêm hoàn toàn mới.
> **Quyết định:** Depth loss dùng fixed lambda. CRS chỉ điều khiển densify/prune (Phase 4). Xem decisions_log.

| # | Task | Kết quả | Ghi chú |
|---|------|---------|---------|
| T3.1 | Thêm `pearson_depth_loss()` vào loss_utils.py | [x] Done | Pearson correlation, scale/shift invariant, differentiable. Unit test 8/8 PASS. |
| ~~T3.2~~ | ~~Thêm `adaptive_depth_loss()` vào loss_utils.py~~ | REMOVED | Bỏ — depth loss fixed lambda, CRS tách biệt cho Phase 4. Xem decisions_log. |
| T3.3 | Tích hợp fixed depth loss vào train.py | [x] Done | PSNR 22.35 (+1.2 vs baseline 21.15). Vượt 2-field CoR-GS (22.29). N=73k (-65% vs no-depth 211k). Gap train-test giảm 14→11 dB. |
| T3.4 | **Ablation: depth loss only vs baseline** | [x] Done | Kết quả T3.3 = ablation A1 (depth loss only). PSNR 22.35 vs A0=21.15. |

---

## PHASE 4 — Position Constraint + Full Pruning

| # | Task | Kết quả | Ghi chú |
|---|------|---------|---------|
| T4.1 | ~~Position constraint~~ | [x] DISABLED | Implemented + tested, nhưng TẮT sau ablation: PSNR -3 dB vì DAV2 noise. Xem decisions_log 2026-04. |
| T4.2 | Sửa densify_and_prune() — CRS pruning Option C | [x] Done | (CRS<0.35 AND isolated) OR legacy. Params in OptimizationParams. Unit test 5/5 PASS. |
| T4.3 | **Ablation A4: Full CRSGaussian** | PSNR=? vs A0=? | LLFF fern |
| T4.4 | **Ablation A5: T_warmup=0** | PSNR=? | Confirm need warmup |

---

## PHASE 5 — Informed CRS₀ Initialization

> **Triết lý:** CRS₀ có ý nghĩa hình học ngay từ đầu → densification iter 500-1000 không blind.
> **Prerequisite:** Phase 1 DONE (aligned depth), Phase 2 DONE (CRS module), T4.2 DONE (pruning).
> **Xem chi tiết:** docs/09_informed_crs_init_plan.md

| # | Task | Kết quả | Ghi chú |
|---|------|---------|---------|
| T5.1 | Thêm arguments Informed CRS₀ vào arguments/__init__.py | [x] Done | 11 args mới trong ModelParams. Master switch --informed_crs_init default False. str2bool cho bool default=True. |
| T5.2 | Tạo utils/crs/crs_init.py — compute_informed_crs0() | [x] Done | 4 functions: q_reproj, q_depth, q_view, compute_informed_crs0. + _match_reproj_errors helper. |
| T5.3 | Sửa gaussian_model.py — create_from_pcd() nhận informed_crs0 | [x] Done | Optional param, backward-compatible. |
| T5.4 | Sửa gaussian_model.py — densification_postfix() inherit CRS | [x] Done | Conservative inherit: clip(η*parent, 0, 0.5). Gated --crs_densify_inherit. Chain qua split/clone/prune. |
| T5.5 | Sửa train.py — hook informed CRS₀ + eta | [x] Done | Option C: overwrite sau Scene(). Log Q distribution. eta truyền vào densify. |
| T5.6 | Tạo tests/test_informed_crs_init.py — 6 unit tests | [x] Done | 6/6 PASS trên server. |
| T5.7 | Verify baseline không break (informed_crs_init=False) | [ ] | Chạy trên server: PSNR phải = baseline cũ. |
| T5.8 | Log Q distribution (100 iter, --informed_crs_init) | [ ] | Validate γ=5.0 phù hợp với Q distribution thực tế. |

---

## PHASE 5b — Anti-Overfit: Pseudo-View Loss

> **Mục đích:** Giảm train-test gap (~16-18 dB) bằng cách ép Gaussians học geometry/appearance ở novel views, không chỉ training views.
> **Bottleneck đã xác định:** Densify_until ablation chứng minh dừng densify sớm KHÔNG giảm overfit (drop AVG chỉ ±0.05 dB) → root cause là model overfit photometric ở 3 training views, cần signal regularization mới.

### Sub-phase 5b.1 — Pseudo Depth Loss (Approach 1) — KHÔNG HOẠT ĐỘNG

| # | Task | Kết quả | Ghi chú |
|---|------|---------|---------|
| T5b1.1 | Tạo `utils/depth/depth_warping.py` — forward warp DAV2 depth | [x] Done | 3 functions: `find_nearest_training_cam`, `forward_warp_depth`, `compute_warp_coverage`. Convention `cam.R = C2W rot`, `cam.T = W2C trans`. |
| T5b1.2 | Verify camera convention (round-trip + gradient flow) | [x] PASS | 4/4 tests: round-trip 7e-15, coverage 86%, scale ratio 0.875, gradient norm 1.46e-2. |
| T5b1.3 | 6 unit tests synthetic | [x] PASS | 6/6 tests trong `tests/test_pseudo_depth_warp.py`. |
| T5b1.4 | Hook `--use_pseudo_depth_loss` vào train.py | [x] Done | Gated, default OFF. Tái dùng `RenderDict["depth_pseudo_co_gs0"]`. |
| T5b1.5 | Ablation 7 configs × 8 LLFF scenes (B0/PM/PL/PH/PU5/PE/PT) | [x] Done | **AVG @10k: PL=20.288 vs B0=20.230 (+0.058)** — không cải thiện, gain trong noise floor. |
| T5b1.6 | Verify H1-H4 hypotheses | [x] **H4 = ROOT CAUSE** | H1/H2/H3 đã verify OK ở session trước. **H4: pseudo cams chỉ cách training cams 0.3-3.68° (max!), 0/10000 cam pass threshold 5°**. Pseudo loss = duplicate training depth signal. |
| T5b1.7 | Ablation H4 fix? | SKIP | F1 (filter) bất khả thi vì max angle 3.68° < threshold. F2 (perturb) phức tạp + reference DAV2 redundant nên không đáng đầu tư. → Switch sang Approach 2. |

### Sub-phase 5b.2 — Pseudo Photometric Consistency (Approach 2)

> **Triết lý khác:** Reference KHÔNG phải DAV2 (đã dùng training depth loss → redundant) mà là **GT IMAGE** warped từ training cam. Detect floater qua **PARALLAX**: floater 3D ở vị trí sai → khi nhìn từ pseudo cam (dù chỉ 3°), parallax shift `~ depth × tan(3°)` đủ để L1 loss detect.

| # | Task | Kết quả | Ghi chú |
|---|------|---------|---------|
| T5b2.1 | Thêm `warp_image_forward()` vào `utils/depth/depth_warping.py` | [x] Done | Scatter RGB thay vì depth, collision nearest-wins. |
| T5b2.2 | Thêm 3 args vào ModelParams | [x] Done | `use_pseudo_photo_loss`, `lambda_pseudo_photo`, `pseudo_photo_start_iter`. Default OFF → không phá baseline. |
| T5b2.3 | Hook vào train.py — loss block sau pseudo depth block | [x] Done | Gated bởi 5 điều kiện AND. Tái dùng `RenderDict["image_pseudo_co_gs0"]`. L1 masked loss với valid mask broadcast (3,H,W). |
| T5b2.4 | Tạo `scripts/ablation_pseudo_photo.sh` | [x] Done | 5 configs: B0, AP2_005 (λ=0.005), AP2_01, AP2_02, AP2_05. Lambda thấp vì warp có boundary artifacts. |
| T5b2.5 | Smoke test fern (5 runs) | [ ] | Verify không crash, có log `loss/pseudo_photo`. |
| T5b2.6 | Full 8 LLFF scenes (40 runs, parallel 2 GPU) | [ ] | Sau khi smoke test pass. |

### Sub-phase 5b.3 — Pseudo DAV2 Self-Reference (Approach 1') — HOLD

> Chạy DAV2 trên rendered image tại pseudo cam → reference depth phụ thuộc content render thực. FSGS-style. **Hold cho đến khi Approach 2 cho kết quả** vì overhead DAV2 inference per-iter cao (+1.3GB GPU memory + ~50ms/call), risk OOM.

---

## PHASE 6 — GFS Metric [HOLD - deprioritized]

> **Tạm gác hoàn toàn.** Focus 100% vào C1 (CRS Module) và C2 (CRS-guided Densification) trước.
> Không implement metrics_dtu.py extension, compute_depth_rmse(), compute_floater_ratio().

| # | Task | Kết quả | Ghi chú |
|---|------|---------|---------|
| ~~T6.1~~ | ~~Mở rộng metrics_dtu.py — Depth RMSE~~ | HOLD | |
| ~~T6.2~~ | ~~Mở rộng metrics_dtu.py — Floater Ratio~~ | HOLD | |
| ~~T6.3~~ | ~~Validate: CRSGaussian vs CoR-GS trên DTU~~ | HOLD | |
| ~~T6.4~~ | ~~Scatter plot PSNR vs Depth RMSE~~ | HOLD | |

---

## PHASE 7 — Full Ablation Study

### Component ablation

| Config | D_i | R_i | Pos.const | W.align | CRS₀ | T_warmup | PSNR (fern) | PSNR (DTU avg) |
|--------|-----|-----|-----------|---------|------|----------|-------------|----------------|
| A0 — CoR-GS 2-field (T0.5) | ✗ | ✗ | ✗ | ✗ | N/A | N/A | 22.29 | |
| A0b — Single-field (T0.5b) | ✗ | ✗ | ✗ | ✗ | N/A | N/A | 21.15 | |
| A1 — Depth loss only (T3.3) | ✗ | ✗ | ✗ | ✓ | 0.5 | 1000 | **22.35** | |
| A2 — R_i only | ✗ | ✓ | ✗ | ✓ | f(reproj)/0.5 | 1000 | | |
| A3 — Full CRS, no pos | ✓ | ✓ | ✗ | ✓ | f(reproj)/0.5 | 1000 | | |
| A4 — Full CRSGaussian | ✓ | ✓ | ✓ (T_densify=500) | ✓ | f(reproj)/0.5 | 1000 | | |
| A5 — No warmup | ✓ | ✓ | ✓ | ✓ | f(reproj)/0.5 | 0 | | |
| A6 — Naive align | ✓ | ✓ | ✓ | ✗ | f(reproj)/0.5 | 1000 | | |

### CRS₀ Informed Initialization ablation (updated 2026-04)

| Config | q_reproj | q_depth | q_view | Inherit | PSNR (fern) | Ghi chú |
|--------|----------|---------|--------|---------|-------------|---------|
| C0 | ✗ | ✗ | ✗ | ✗ | | Baseline — informed_crs_init=False |
| C1 | ✓ | ✗ | ✗ | ✗ | | Chỉ reproj |
| C2 | ✗ | ✓ | ✗ | ✗ | | Chỉ depth |
| C3 | ✗ | ✗ | ✓ | ✗ | | Chỉ view |
| C4 | ✓ | ✓ | ✗ | ✗ | | reproj + depth |
| C5 | ✓ | ✗ | ✓ | ✗ | | reproj + view |
| C6 | ✗ | ✓ | ✓ | ✗ | | depth + view |
| C7 | ✓ | ✓ | ✓ | ✗ | | Tất cả, equal weights |
| C8 | ✓ | ✓ | ✓ | ✓ | | Tất cả + densify inherit |

### Hyperparameter sensitivity

| Param | Default | Values | PSNR (fern) |
|-------|---------|--------|-------------|
| T_densify | 500 | {500, 1000} | |
| T_warmup | **1000** | {500, 1000, 2000} | ← default đổi từ 2000 |
| epsilon_depth | 0.05 | {0.02, 0.05, 0.10} | |
| w1/w2 | 0.5/0.5 | {0.5/0.5, 0.7/0.3, 0.3/0.7, 0.6/0.4, 0.4/0.6, 0.8/0.2, 0.2/0.8} | |
| CRS scale factor | 5.0 | {3.0, 5.0, 8.0} | |
| EMA decay | 0.9 | {0.8, 0.9, 0.95} | |
| CRS update interval | 100 | {50, 100, 200} | |
| tau_crs | 0.35 | {0.25, 0.35, 0.45} | ← updated từ 0.20 |

> **Quy trình tìm bộ trọng số tốt nhất:**
> 1. Chạy fern 3-view với default (w1/w2=0.5/0.5, scale=5.0, ema=0.9)
> 2. Grid search w1/w2 trước (7 bộ) — giữ cố định scale, ema
> 3. Chọn w1/w2 tốt nhất → grid search scale (3 giá trị)
> 4. Chọn scale tốt nhất → grid search ema (3 giá trị)
> 5. Verify bộ tốt nhất trên 2-3 scenes LLFF khác (flower, room)

---

## PHASE 6 — CRS Diagnostic & Mechanism Exhaustion (2026-04 → 2026-04-30)

> **Triệu chứng kích hoạt:** Phase 2c CRS isolation cho thấy D1-noCRS-O999 = 21.21 dB > D1-O999
> = 21.13 dB → CRS pruning REDUNDANT với opacity decay. Paper crisis trigger.

### Phase 6.1 — Tier A diagnostic suite

| # | Task | Kết quả | Ghi chú |
|---|------|---------|---------|
| T6.1.1 | Implement utils/crs/tier_a_diag.py — A1/A3/A4 functions | [x] Done | BC bimodality, synthetic floater, occlusion test |
| T6.1.2 | Run Tier A trên 8 LLFF scenes | [x] Done | 8 scenes × 4 iter checkpoints (1100/3000/5000/10000) |
| T6.1.3 | Verdict tổng hợp | [x] D, R bimodal (BC > 0.555 in 6/8). D-R orthogonal. A3 floater discrimination ✓. **A4 occlusion contamination 36.5%** | Signal có shape, R noisy |

### Phase 6.2 — CRS mechanism variants exhaustive test

| # | Variant | AVG ΔPSNR vs B0 | Verdict |
|---|---------|-----------------|---------|
| T6.2.1 | 6 CRS pruning ablations (tau sweep) | ≤ +0.05 | dead |
| T6.2.2 | C1 occlusion-aware R (depth-test) | −0.064 | worse |
| T6.2.3 | C1.5 depth_range-relative tolerance sweep | best −0.038 | dead |
| T6.2.4 | F-invisible visibility-streak (out-of-frustum 1.3% catch) | marginal | dead |
| T6.2.5 | D1G CRS-gated densification | +0.028 | REDUNDANT vs DECAY (synergy −0.132) |
| T6.2.6 | Stack D1G + DECAY 0.999 | +0.082 (vs DECAY +0.186 alone) | sub-additive |
| T6.2.7 | hC Hybrid RC × D fusion (proxy RC) | **−0.079** | worse than B0 |
| T6.2.8 | RNRC L3 differentiable α-coupling | +0.074 | STACK +0.163 < DECAY +0.186 |

**Pattern**: Δ ceiling +0.07 ± 0.10 dB across 7 mechanisms với D+R signal → **signal saturation**.

### Phase 6.3 — Literature survey & breakthrough proposals

| # | Task | Kết quả |
|---|------|---------|
| T6.3.1 | Survey 8-15 sparse-view 3DGS papers (2024-2026) | [x] Done — 17 papers reviewed |
| T6.3.2 | Identify saturated vs unexplored axes | Saturated: opacity, densification, prune. Unexplored: loss-path, SH-gating, feature MPC |
| T6.3.3 | 4 candidate breakthrough proposals | A (loss reweighter), B (SH gating), C (CRS-MPC), D (pseudo-view α) |
| T6.3.4 | Top recommendation | **Loss-path mechanism + signal upgrade (D_cycle)** |

---

## PHASE 7 — CRS Tier 2-min: D_cycle + Loss Reweighter (DONE)

> **Mục tiêu**: Last attempt cho CRS line với dual upgrade (signal + mechanism).
> **Commitment**: Δ ≥ +0.20 breakthrough, ≥ +0.10 solid, < +0.10 → pivot recipe paper.
> **Compute target**: <+15% training time, +0 GB GPU memory bump, baseline render FPS.
> **Xem chi tiết**: docs/11_crs_diagnostic_redesign.md

### Phase 7.1 — Implementation (Python only, NO CUDA)

| # | Task | Kết quả | Ghi chú |
|---|------|---------|---------|
| T7.1.1 | arguments/__init__.py — flags use_d_cycle + use_loss_reweight + 4 hyperparams | [ ] | Default OFF, ablation-friendly |
| T7.1.2 | utils/crs/crs_module.py — compute_D_cycle() | [ ] | Cycle warping qua training view pairs, dùng rendered depth |
| T7.1.3 | utils/crs/crs_module.py — render_crs_map() (color-swap trick) | [ ] | Alpha-composite CRS_i without CUDA modification |
| T7.1.4 | utils/crs/crs_module.py — modify update_crs() để dùng D_cycle khi flag on | [ ] | Warmup logic (D_DAV2 trước iter 1000) |
| T7.1.5 | train.py — apply per-pixel loss weight với CRS_pix | [ ] | Stop-gradient, cache mỗi 100 iters |

### Phase 7.2 — Smoke tests (BẮT BUỘC trước ablation)

| # | Test | Pass criterion |
|---|------|----------------|
| T7.2.1 | Smoke 1: Flag OFF byte-identical | PSNR fern khớp B0 ± 0.05 dB, không log "[Tier2-min]" |
| T7.2.2 | Smoke 2: D_cycle only ON | Training stable, D_cycle_med > 0, log appear sau iter 1000 |
| T7.2.3 | Smoke 3: Loss reweighter only ON | CRS_map_mean ∈ [0.3, 0.8], Loss_w_mean ∈ [0.5, 1.0] |
| T7.2.4 | Smoke 4: Full Tier 2-min ON | Training stable, all logs hợp lý |

### Phase 7.3 — Ablation matrix (24 NEW runs, B0 reuse)

| Tag | use_d_cycle | use_loss_reweight | Mục đích |
|-----|-------------|-------------------|----------|
| B0 | (REUSE) | (REUSE) | baseline |
| DCYCLE | ON | OFF | signal upgrade alone (vẫn dùng prune) |
| LWEIGHT | OFF | ON | mechanism upgrade alone (D+R cũ) |
| **TIER2MIN** | **ON** | **ON** | **full Tier 2-min (combined)** |

8 scenes × 3 NEW configs = 24 runs ~1.5-2h trên 2 GPU.

### Phase 7.4 — Compute & efficiency reporting (REQUIRED)

| Metric | Target |
|--------|--------|
| Training time / scene | < 12 phút (baseline ~10) |
| Peak GPU memory | bằng baseline (no foundation model) |
| Final N_gaussians | report per scene |
| Render FPS | bằng baseline (CRS không touch inference) |
| Model storage (PLY) | bằng baseline |

### Phase 7.5 — Verdict & results

**Phase 7 Original (Phase 5 weak backbone, B0=20.234):**
- DCYCLE +0.125 (FIRST CRS variant clean positive across 9 attempts)
- LWEIGHT −0.007 (mechanism alone neutral)
- TIER2MIN combined +0.035 (synergy −0.083 redundant)
- Verdict 🔴 STOP per +0.15 threshold trên backbone này

**Phase 7 Stage 1 (D1-O999 strong backbone, OLD=21.178):**
- TIER1_DC_GATE −0.015 (D_cycle FLIPS NEGATIVE on strong backbone)
- TIER1_DC_LW +0.050 (combined within noise)
- vs No-CRS (21.21) +0.018 (within noise)
- Verdict 🟡 H_both_dead (signal + mechanism saturated với D+R intact)
- **Hypothesis cho Phase 8: R contamination 36.5% diluting D_cycle → fix R first**

---

## PHASE 8 — Formula Redesign + SH Path (DONE 🎉 BREAKTHROUGH)

> **Trigger:** Phase 7 Stage 1 hypothesis — R contamination dilutes D_cycle. Fix R + add S signal +
> CRS-modulated SH freeze mechanism.

### Phase 8.1 — Implementation (4 NEW files + modifications)

| # | Task | File | Status |
|---|------|------|--------|
| T8.1.1 | R_visible (visibility-aware reprojection) | utils/crs/crs_module.py modify | [x] Done |
| T8.1.2 | S_stability EMA variance signal | utils/crs/sh_stability.py NEW | [x] Done |
| T8.1.3 | CRS-modulated SH freeze | utils/crs/sh_freeze.py NEW | [x] Done |
| T8.1.4 | Multi-component CRS formula update | utils/crs/crs_module.py update_crs | [x] Done |
| T8.1.5 | train.py hooks (S update + SH freeze) | train.py modify | [x] Done |
| T8.1.6 | 5-config ablation runner | scripts/p8_master.sh | [x] Done |
| T8.1.7 | Attribution analyzer | scripts/p8_analyze.py | [x] Done |

### Phase 8.2 — Ablation results (5 configs × 8 scenes = 40 runs)

| Config | Components | AVG | Δ vs OLD |
|--------|-----------|-----|----------|
| OLD | D_DAV2 + R_old | 21.178 | 0 |
| FIX_R_DAV2 | + R_visible (R fix only) | 21.068 | **−0.111** ❌ R alone HURTS |
| FIX_R_DC | + D_cycle (with clean R) | 21.169 | +0.102 ✅ D vindicated |
| FIX_RS | + S_stability | 21.146 | −0.023 ⚪ S adds nothing |
| **FULL** | + CRS-mod SH freeze | **21.335** | **+0.156** |

**vs No-CRS reference (21.21):** FULL = +0.125 dB (first CRS variant beat no-CRS!)

**Compute:** FULL +3.7% slowdown, GPU memory unchanged, render FPS unchanged.

### Phase 8.3 — Attribution insights

| Component | Δ alone | Verdict |
|-----------|---------|---------|
| Δ_R (R_visible) | −0.111 | ❌ Hurts alone (data loss > noise reduction) |
| Δ_D (D_cycle in clean R) | +0.102 | ✅ Works when R cleaned |
| Δ_S (S_stability) | −0.023 | ⚪ Neutral, doesn't help |
| **Δ_M (CRS-mod SH freeze)** | **+0.189** | **✅ BIGGEST WINNER** |

🎉 **First CRS contribution defendable** sau 9 prior attempts ceiling +0.07 dB.

---

## PHASE 9 — Simplification + Cross-backbone confirmation (DONE)

> **Goal:** Test 3 hypotheses từ Phase 8 attribution + close DOC-GS/BinocularGS gap.
> Best Phase 9 variant target ≥21.45 dB (close BinocularGS 21.44).

### Phase 9.1 — Test 1: D1-O999 simplification (3 NEW configs + FULL reuse)

| # | Task | Hypothesis | Status |
|---|------|-----------|--------|
| T9.1.1 | FULL_NoS = D + R + CRS-mod-freeze | H2: S adds nothing | [ ] |
| T9.1.2 | D_ONLY_FREEZE = D + CRS-mod-freeze | H1: drop R helps | [ ] |
| T9.1.3 | D_ONLY_GATE = D + CRS prune (no SH freeze) | Isolate D signal alone | [ ] |

### Phase 9.2 — Test 2: A1+B1β cross-backbone (1 NEW config)

| # | Task | Hypothesis | Status |
|---|------|-----------|--------|
| T9.2.1 | A1B1_BASELINE = A1+B1β baseline (verify ~20.96) | reference | [ ] |
| T9.2.2 | A1B1_BEST = A1+B1β (no global freeze) + Phase 8 components | H3: SH freeze universal mechanism | [ ] |

### Phase 9.3 — Implementation needs

| # | Task | Status |
|---|------|--------|
| T9.3.1 | `--disable_r_signal` flag (D-only formula support) | [ ] |
| T9.3.2 | `--disable_global_sh_freeze` flag (replace Track A1 with CRS-mod) | [ ] |
| T9.3.3 | scripts/p9_master.sh + scripts/p9_analyze.py | [ ] |

### Phase 9.4 — Result (executed 2026-05-08)

**Test 1 — D1-O999 simplification:**
| Config | AVG | Δ vs FULL |
|--------|-----|-----------|
| **FULL (Phase 8)** | **21.335** | 0 BEST |
| D_ONLY_GATE | 21.242 | −0.093 |
| FULL_NoS | 21.200 | −0.135 |
| D_ONLY_FREEZE | 21.159 | −0.176 |

**Test 2 — A1+B1β cross-backbone:**
| Config | AVG | Δ |
|--------|-----|---|
| A1B1_BASELINE | 20.932 | reference |
| A1B1_BEST | 20.983 | +0.051 |

**Verdict:**
- ❌ H1 REJECTED — R contributes (Δ_NoR = −0.041)
- ❌ H2 REJECTED — S synergize với mechanism (Δ_NoS = −0.135)
- 🟡 H3 PARTIAL — SH freeze works on A1+B1β but smaller (+0.051 vs +0.189 on D1-O999)

**Key insight:** Sequential delta (Phase 8) ≠ leave-one-out (Phase 9). Phase 8 said "S adds nothing alone" (−0.023), Phase 9 says "removing S hurts" (−0.135). All 4 components (D, R, S, mechanism) **synergize** in FULL recipe.

→ **Phase 8 FULL recipe LOCKED at 21.335 dB. Don't simplify. CRS axis exhausted.**

---

## PHASE 10 — DUSt3R Dense Init — DONE (FAILED, axis DEAD)

**Status (2026-05-07):** Pivot khỏi initial-PC axis. DUSt3R/MASt3R/foundation-model dense init nói chung loại bỏ.

| # | Task | Status | Kết quả |
|---|------|--------|---------|
| T10.1a | Implement DUSt3R wrapper + precompute cache | [x] Done | 8 scenes pre-computed |
| T10.1b | Run AUGMENT/REPLACE × 8 scenes ablation | [x] Done | AUGMENT Δ=−0.898, REPLACE Δ=−3.529 |
| T10.1c | Diagnostic — FILTER/DENSIFY/BOTH × 2 scenes | [x] Done | Best Δ=−0.074 (orchids FILTER), still < +0.05 noise floor |
| T10.1d | Decision Phase 10A | [x] DEAD | All hyperparameter combos fail. Systematic, not tuning. |

**Verdict:** DUSt3R dense init không phải orthogonal lever. Ceiling ≈ −0.07 dB even with optimal filter.

**Cleanup:**
- ✅ DUSt3R env removed
- ✅ Checkpoint + source + cache deleted (~3-5 GB freed)
- ⏸ Code Phase 10A giữ default OFF cho paper reference

Xem **decisions_log [2026-05-07] Phase 10A — DUSt3R Dense Init FAIL hard**.

---

## PHASE 11 — Loss-axis + Anti-overfit Exploration (CURRENT)

**Strategy update 2026-05-09**: Multi-seed protocol (3 seeds × 8 scenes paired) cho mọi step. Min detectable Δ ≈ ±0.10 dB. Variance discovery (atomicAdd ±1.3 dB single-scene) → paired comparison cancel common-mode noise.

**Pre-flight (DONE 2026-05-09):**
- cuDNN+atomicAdd non-determinism discovered → multi-seed locked
- timm migration cho DINOv2 (Python 3.8 compatible)
- Variance band documented (decisions_log [2026-05-09])

| # | Step | Lever | Cost | Verdict | Status |
|---|------|-------|------|---------|--------|
| T11.1 | **Step 1** — CRS × Covisibility reweight (depth-based) | Per-pixel weight cov_norm × CRS_pix on L_phot | 0.5d + 5h test | 🟡 **MARGINAL** (cross-batch Δ +0.014) | [x] **DONE** keep code OFF |
| T11.2 | Step 2 — Same-view perceptual DINO | Cosine distance feat_render vs feat_GT same view | 0.5d + 3h | ❌ **REJECTED** (Δ −0.046 ± 0.056) | [x] **DONE** cleanup pending |
| T11.3 | Stack (S1+S2) — synergy test | Both flags ON simultaneously | 3h | ❌ **REJECTED** (no synergy, Δ_Synergy −0.063) | [x] **DONE** cleanup script |
| T11.4 | **Step 4** — Cross-view MPC | Anti-overfit: geometric DINO warp | 1d + 3h | ❌ **REJECTED** (Δ −0.042, incomplete s9999 4 scenes) | [x] **DONE** cleanup pending |
| T11.5 | **Step 5** — TV depth edge-preserving | Anti-overfit: smooth depth field | 0.5d + 3h | ❌ **REJECTED** (Δ −0.026 ± 0.051) | [x] **DONE** cleanup pending |

**Phase 11 LOSS-AXIS EXHAUSTED — 6/6 attempts REJECTED:**
- All 6 mechanism classes tested multi-seed N=24 paired, none significant
- Phase 11 Step 1 batch 2 A1 mean = 21.186 (highest, single batch lucky)
- No A1 mean robustly > 21.20 across batches
- Ceiling ≈ 21.18 ± 0.05 multi-seed → Phase 8 FULL recipe practical optimum

**Code regression confirmed (commit 0511edd May 9):**
- Phase 8 paper 21.335 NOT reproducible from current code
- 5 batches consistent A0 ≈ 21.16-21.20 (gap −0.155 dB vs paper)
- Commit message: "mất config phase 8 full được 21.335"
- Unable to bisect (no git snapshot at Phase 8 ablation working tree)
- Paired Δ within-batch CANCELS regression → Phase 11 verdicts VALID
- Accept current baseline; paper 21.335 reproducible từ saved PLYs

## PHASE 12 — DONE REJECT (2026-05-12) → Pivot Phase 13

**Verdict**: CRS-pull 3/3 configs REJECT seed 42 (Δ_A1=−0.027, Δ_A2=−0.033, Δ_A3=−0.096, all 95% CI cross 0). CRS axis 9/9 EXHAUSTED. Pivot frequency-axis EFA-GS LFCF (Phase 13).

Cleanup completed (Rule 13): `utils/loss/crs_pull.py` deleted, scripts annotated DEPRECATED, 15 flags removed.

---

## PHASE 13 — EFA-GS LFCF + AbsGS port (🎯🎯 DONE COMMIT WORTHY 2026-05-13)

**Design doc**: `docs/13_efa_gs_lfcf_design.md` (1100+ lines, Section 15 FINAL appended)

### FINAL N=24 RESULTS (3 seeds × 8 scenes paired, 2026-05-13)

| Config | N | PSNR | Δ vs A0 | 95% CI | Verdict |
|--------|---|------|---------|--------|---------|
| A0 (Phase 8 FULL baseline) | 24 | 21.166 | — | — | reference |
| A1 (LFCF alone) | 8 | 21.187 | +0.015 | [−0.133, +0.163] | ❌ NOT SIG |
| A2 (LFCF no diffscale) | 8 | 21.152 | −0.020 | [−0.149, +0.110] | ❌ NOT SIG |
| **A3 (LFCF + AbsGS)** | **24** | **21.330** | **+0.164** | **[+0.101, +0.227]** | **🎯 SIG WINNER** |
| A4 (AbsGS alone) | 24 | 21.244 | +0.078 | [+0.009, +0.147] | 🎯 SIG MARGINAL |

**4 decision criteria ALL PASS** (Section 15.6):
1. Multi-seed N=24 paired Δ ≥ +0.10 ✓ (+0.164)
2. 95% CI excludes 0 strict ✓ ([+0.101, +0.227])
3. Per-scene robustness ≥ 6/8 wins ✓ (7/8 + 1 neutral, ZERO hại)
4. Per-seed consistency all ≥ +0.10 ✓ (seeds 42/137/9999: +0.159, +0.195, +0.137)

**Phase 13 = first CRS-axis breakthrough in 10/10 attempts.** Phase 13 A3 N=24 (21.330) ≈ Phase 8 paper 1-sample (21.335) NHƯNG multi-seed reproducible.

### Tasks completed

| # | Task | Status | Result |
|---|------|--------|--------|
| T13.0 | Implementation (Phase 13 port) | [x] DONE | utils/densify/lfcf.py + 8 sửa gaussian_model.py + 3 sửa train.py + 9 flags |
| T13.1 | Round 1 ablation (seed 42, 5 configs) | [x] DONE 2026-05-12 | A3 Δ=+0.159 SIG seed 42 |
| T13.2 | Round 2 multi-seed verify (seeds 137+9999, A0/A3/A4) | [x] DONE 2026-05-13 | Pooled N=24 Δ_A3=+0.164 SIG |
| T13.3 | Pooled N=24 analyzer + verdict | [x] DONE | 🎯🎯 COMMIT WORTHY |

### Next steps (post-commit, ranked by priority)

| # | Task | Cost | Type |
|---|------|------|------|
| **T13.4** | **Lock Phase 13 A3 as new FULL recipe** (change defaults) | 30 min | code config |
| T13.5 | Optional Phase 13.1 tolerance sweep (~3.5h, per Section 10.4) | 3.5h | sensitivity check |
| T13.6 | Optional Direction A λ scaler_max sweep (~1-2h) | 1-2h | robustness check |
| **T13.7** | **Continue PSNR optimization** — Phase 13.1 done + Phase 13.2 next | sequential | mech extension |

---

## PHASE 13.1 — LFCF intensity sweep (DONE, A3 base CONFIRMED optimal 2026-05-13)

**Verdict**: A3 base (scaler=1.5, interval=2) is stable operating point. KHÔNG có variant intensity beats A3.

### Sweep results (seed 42, 8 scenes, N=8 paired)

| Variant | Config | Test PSNR | Train PSNR | Gap | Δ_test vs A3 | Verdict |
|---------|--------|-----------|------------|-----|--------------|---------|
| Base (A3) | scaler=1.5, int=2 | 21.331 | 34.235 | 12.905 | 0 (ref) | ⭐ optimal |
| M | scaler=1.8, int=2 | 21.318 | 34.123 | 12.805 | −0.013 | ⚪ neutral |
| H | scaler=2.0, int=1 | 19.637 | 23.279 | 3.642 | **−1.694** | 📉 catastrophic |
| X | scaler=2.5, int=1 | 19.627 | 23.282 | 3.655 | **−1.703** | 📉 catastrophic |

**Key findings**:
- scaler dimension INSENSITIVE (1.5 → 1.8 neutral, 2.0 ≈ 2.5 saturated)
- interval dimension CRITICAL (interval=2 OK, interval=1 catastrophic)
- H/X = MODEL COLLAPSE (train −11dB >> test −1.7dB), NOT anti-overfit
- Root cause: interval=1 disables standard `densify_and_clone`, model undergrowth
- A3 (scaler=1.5, interval=2) is stable, not lucky local optimum

### Tasks completed

| # | Task | Status | Result |
|---|------|--------|--------|
| T13.1.1 | Sweep script + analyzer (M/H/X variants) | [x] DONE | scripts/p13_1_sweep.sh + analyze.py |
| T13.1.2 | Run sweep 24 runs ~2h | [x] DONE 2026-05-13 | All variants ≤ A3 base |
| T13.1.3 | Analyze + verdict | [x] DONE | A3 optimal, no further intensity sweep |

---

## PHASE 13.2 — Sequential mechanism testing (CURRENT)

**Strategy**: Sau Phase 13 + 13.1 commit A3 21.330, continue PSNR optimization qua **sequential focused single-mech testing** (replace Path 1 stack approach).

**Target**: Best path to push +0.05~+0.20 over A3.

**Rationale Path 1 stack REJECT**:
- Project track record: 15 mechs tested, 1 win → base rate ~7%
- Phase 11 historical synergy −0.063 (stack 2 mechs hurt)
- Path 1 estimate 40-50% P(+0.2) too optimistic; honest 15-25%
- Path 1 stack obscures attribution per mech
- Solution: SEQUENTIAL focused tests, each with stop conditions, clean attribution

### Step-by-step plan

| Step | Direction | Cost | Evidence | Stop condition |
|------|-----------|------|----------|----------------|
| **T13.2.0** | **FFT spectrum analysis on GT (pre-Gap C)** | 30 min | data-driven σ_blur choice | Output σ ablation range |
| **T13.2.1** | **Gap C FALA** (Frequency-Annealed Loss Annealing) | 1 ngày code + 3h test | First-principles, σ informed by 13.2.0 | Δ ≥ +0.10 → multi-seed verify |
| T13.2.2 | DWTGS port (Gap A — novel-view HF sparsity) | 1.5 ngày code + 3h test | Paper claim +0.3-0.4 PSNR sparse-view | Δ ≥ +0.10 → multi-seed verify |
| T13.2.3 | Visibility-based prune (T12.2 from Phase 12 plan) | 0.5 ngày + 3h test | Phase 12 plan untested | Δ ≥ +0.10 → multi-seed verify |
| T13.2.4 (conditional) | Tier 3 architecture (Hierarchical Gaussians, BinocularGS-like) | 2-3 tuần | Different mechanism class | Per step verdict |

### Excluded directions (evidence-based)

- ❌ T1.4 + T1.5 (hyperparam tweaks — base rate <10%)
- ❌ T2.3 Cross-view MPC (Phase 11 Step 4 already failed Δ=−0.042, same setting)
- ❌ T3.1 Mip-Splatting (TaT regression −0.94 + CUDA conflict, Phase 13 Section 6 rejected)
- ❌ T3.4 Foundation model priors (Phase 10A DUSt3R failed catastrophically)
- ❌ Path 1 stack approach (low attribution + negative synergy risk)

### Current focus — T13.2.0' Render-vs-GT diagnostic DONE 🎯 (verdict OVERTURNS initial Gap C plan)

**T13.2.0 GT FFT spectrum analysis ✅ DONE 2026-05-13 morning**
- σ recommendations: conservative 0.58 / balanced 1.08 / aggressive 2.57 px
- Per-scene variance 6× (leaves 0.91 → fortress 5.89)

**T13.2.0' Render-vs-GT diagnostic ✅ DONE 2026-05-13 evening — KEY VERDICT**

3 critical findings từ diagnostic (`scripts/p13_2_spectrum_diagnostic.py`):

1. **Universal HF deficit 8/8 scenes**: A3 HF rel_Δ mean = −0.192 → render đạt chỉ **64% HF energy GT**. KHÔNG scene nào SPURIOUS.

2. **A3 mechanism = SPATIAL, NOT spectral**: 6/8 scenes A3 produces HF ÍT HƠN A0. horns Δ_PSNR=+0.362 (highest) nhưng imp_HF=−0.014. A3 PSNR gain qua spatial alignment, KHÔNG qua HF amplitude.

3. **Literature direction WRONG SIGN**:
   - ❌ DWTGS HF-sparsity: assume OVER-produces HF → REFUTED
   - ❌ Standard FALA blur GT: would worsen deficit → WRONG SIGN
   - ✅ HF-emphasis loss (high-pass GT + L1): CORRECT DIRECTION
   - ✅ FALA-reversed (sharpen GT): alternative correct sign

**Pattern tally (seed 42)**: 4/8 MISSING_HF + 4/8 WEAK_SIGNAL, 0/8 SPURIOUS.

**Pilot plan**: orchids × seed 42 × λ_HF ∈ {0.05, 0.10, 0.20} = 3 runs ~21 min. Target orchids (HF deficit largest −0.255, PSNR lowest 16.97).

**Risk acknowledged**:
- Spectrum close không guarantee PSNR up (fern: A3≈A0 spectrum, +0.144 PSNR)
- 3-view sparse fundamental limit có thể không cứu được bằng loss-side trick
- Single seed 42, cần seed 137/9999 verify nếu pilot win

### Updated Phase 13.2 sequential plan (post-diagnostic)

```
Phase 13.2 — Sequential mech testing (UPDATED)
├── T13.2.0   GT FFT spectrum analysis            ✅ DONE
├── T13.2.0'  Render-vs-GT diagnostic             ✅ DONE — verdict UNIVERSAL HF DEFICIT
├── T13.2.1   Gap C decision (overturned):
│   ├── ❌ DWTGS HF-sparsity     — REJECT (wrong sign per diagnostic)
│   ├── ❌ Standard FALA blur    — REJECT (wrong sign per diagnostic)
│   ├── ✅ HF-emphasis loss      — PILOT orchids 3 runs (~21 min) ← NEXT
│   └── 🟡 Per-scene adaptive σ  — defer to Round 2 nếu pilot win
├── T13.2.2   ⏸ DWTGS port      SKIPPED (wrong sign confirmed)
├── T13.2.3   Visibility prune (geometry axis)    pending HF-emphasis verdict
└── T13.2.4   Tier 3 architecture                 last resort
```

**Files updated** (new):
- `scripts/p13_2_spectrum_diagnostic.py` — diagnostic tool (~400 lines)
- `logs/p13_2_diagnostic/*.png` — per-scene spectrum plots
- Output: SUMMARY available in conversation

**Next concrete step**: Implement HF-emphasis loss (Sobel high-pass + L1) + smoke pilot 3 runs orchids → analyze → multi-seed if win.

### T13.2.1 λ_HF calibration ✅ DONE (2026-05-14)

**Tool**: `scripts/p13_2_lambda_calibration.py` measure L_main + L_HF magnitudes trên A3 baseline renders.

**Result**:
- L_main aggregate = 0.0550 (= L1+SSIM main photometric)
- L_HF aggregate = 0.0873 (= mean |∇²I_render − ∇²I_GT| Laplacian L1)
- **R = L_main / L_HF = 0.631** (calibration ratio)

**Calibrated 4 λ_HF levels** (% L_main contribution × R):

| Level | Target % L_main | λ_HF |
|---|---|---|
| safety | 3% | **0.019** |
| gentle | 10% | **0.063** |
| moderate | 30% | **0.189** |
| strong | 100% | **0.631** |

Replace earlier guess values {0.05, 0.10, 0.20} (underestimate 1.3-3.2× per level).

### Updated pilot matrix — 12 runs, early-stop order

```
       trex      horns     orchids
       --------- --------- ---------
0.019  S_trex    S_horns   S_orchids
0.063  G_trex    G_horns   G_orchids
0.189  M_trex    M_horns   M_orchids
0.631  X_trex    X_horns   X_orchids
```

**Order**: orchids × 4 first (~30 min) → IF any ≥+0.05 → horns × 4 → IF any ≥+0.05 → trex × 4.
**Worst case**: 12 runs ~1.5h. **Best case** (orchids all regress): 4 runs ~30 min.

**Parseval ceiling**: theoretical max PSNR gain ~+0.10-0.20 dB (HF carry ~12% total energy).

**Next concrete step**: Implement HF-emphasis loss (Laplacian 3×3 + L1) + 4-flag arg + train.py hook + pilot script với 4 calibrated λ values.

### T13.2.1 HF-emphasis pilot ❌ DONE REJECT (2026-05-14 evening)

**Run scope**: 24 runs = 4 λ × 2 timing × 3 scenes × seed 42

**Results — INVERSE prediction pattern**:

| Scene | Mean Δ vs A3 (8 configs) | Verdict |
|---|---|---|
| trex (low HF deficit) | **+0.18** | ⭐ helps consistently |
| orchids (high HF deficit) | +0.02 | ⚪ null (Parseval/data limit) |
| **horns** (mid HF deficit, A3 best win) | **−0.31** | ❌ **8/8 configs negative** |

**Statistical sig**: horns 8/8 negative → p=0.004 (NOT noise).

**Phase conflict CONFIRMED**: scenes có A3 spatial gain CAO → HF emphasis HẠI mạnh. Spatial + spectral axes ORTHOGONAL in concept, INTERFERE in practice.

**Decision**: 
- ❌ Reject HF-emphasis L1 Laplacian (mechanism dead trên A3 backbone)
- ❌ Skip generalizations: FALA-reversed, Sobel variants, FFT-domain HF L1, DWTGS — all same class, all expected fail
- ❌ Multi-seed verify SKIPPED (pattern clear, save 24 GPU-hours)
- ✅ **All loss-side frequency-axis mechanisms EXHAUSTED**

**Cleanup pending** (Rule 13): delete utils/loss/hf_emphasis.py + scripts/p13_2_hf_pilot.* + lambda_calibration.py + revert train.py + remove args. KEEP logs/p13_2_hf/ negative result record + design doc Section 20.

### T13.2.3 — Visibility-based prune (CURRENT NEXT)

**Mechanism**: Force Gaussians visible từ ≥2 train views → geometric constraint, NOT photometric amplitude.

**Why different from HF-emphasis**:
- Geometry axis, NOT frequency axis
- Constraint on Gaussian existence, NOT on Gaussian param values
- KHÔNG conflict A3 spatial mechanism (different intervention point)

**Phase 12 plan T12.2 — never executed**. Fresh untested mechanism.

**Cost**: ~0.5 ngày code + 3h test.

**Probability**: 20-25% Δ ≥ +0.05 trên A3.

**Fallback if T13.2.3 fails**: Tier 3 architecture (Hierarchical Gaussians BinocularGS-like, 2-3 tuần) hoặc accept 21.330 ceiling.

### Reference targets (Phase 13 A3 anchor):
- Phase 13 A3 N=24 = **21.330** (committed)
- DOC-GS 21.38 (gap −0.05, closing)
- BinocularGS 21.44 (gap −0.11)
- ICO-GS SOTA 22.20 (gap −0.87, future work via Tier 3)

### Paper narrative (Section 15.5)

**Main story — REVERSAL pattern**: Phase 12 CRS-pull worst failure modes (horns −0.241, fortress −0.246, flower −0.180) ↔ Phase 13 best wins (+0.362, +0.101, +0.150) — symmetric mechanism reversal:
- Phase 12: pull centroid → HẠI thin geometry
- Phase 13 diffscale: isotropify → PROTECT thin geometry

**Secondary story — Synergy +0.071** (74% over linear): AbsGS catch + LFCF gate + diffscale shape combo.

**Comprehensive ablation**: Phase 11 6/6 + Phase 12 3/3 + Phase 13 4 configs = 13 mechanism classes evaluated, defendable.

**Comparison vs literature**:
- Phase 13 A3 N=24 = 21.330 (multi-seed reproducible)
- DOC-GS 21.38 (gap −0.05, closing)
- BinocularGS 21.44 (gap −0.11, closing)
- ICO-GS SOTA 22.20 (gap −0.87, future work)

### Key files

| File | Status | Role |
|------|--------|------|
| `utils/densify/lfcf.py` | Implemented | Pure functions: tolerance compare, diffscale, depth-aware split prob |
| `scene/gaussian_model.py` | Extended | LFCF mode gate trong `densify_and_prune` + LFCF attrs prune/postfix |
| `arguments/__init__.py` | Extended | 9 LFCF flags + uncomment `absdensify` |
| `train.py` | Extended | LFCF hook + uncomment `--absdensify` CLI |
| `scripts/p13_lfcf_multiseed.sh` | Done | 5 configs ablation |
| `scripts/p13_lfcf_multiseed_analyze.py` | Done | 5-config attribution + 4 paired Δ |

---

## PHASE 12 archive — Improvement attempts pre-Phase-13

User stance (pre-Phase-13 pivot): KHÔNG writeup vội. Try untouched directions.

### Cleanup (parallel với Track 1)

| Item | Action | Status |
|------|--------|--------|
| Phase 11 Step 2/4/5 rejected modules | Delete files, hooks, flags | [~] cleanup prompt drafted |
| Phase 5b rejected modules | Delete pseudo_depth/photo | [~] |
| DINOv2 wrapper (only used by Step 2/3/4) | Delete | [~] |
| Step 1 Covisibility (MARGINAL) | KEEP code default OFF | preserved |
| Phase 7 LWEIGHT | KEEP reference | preserved |
| Phase 10A code | KEEP for now (separate decision) | preserved |

### Improvement attempts (ranked)

| # | Direction | Probability ≥+0.10 | Cost | Class |
|---|-----------|---------------------|------|-------|
| **T12.1** | **Iter budget 15k test** | ~25-30% | 0 code | Quick CLI flag test |
| T12.2 | Visibility-based prune | ~20-25% | 0.5 ngày | Anti-overfit, different from CRS |
| **T12.3** | **Mip-Splatting anti-aliasing** | **~30-40%** | 2-3 ngày | Rendering trick, highest probability |
| T12.4 | Anisotropy regularizer | ~15-20% | 0.5 ngày | Anti-overfit shape |
| T12.5 | Soft scale regularizer | ~15-20% | 0.3 ngày | Anti-overfit no position move |
| T12.6 | CRS-pull (user idea) | ~15% | 1-2 ngày | Position-axis, CRS reliability risk |
| T12.7 | Density-aware densify | ~15-20% | 1 ngày | Untested precedent |

### Decision tree

```
Phase A: T12.1 (Iter 15k) — instant
  Δ ≥ +0.10 → scale multi-seed → COMMIT if confirm
  < +0.05    → Phase B

Phase B: T12.2 (Visibility prune) — 0.5 ngày
  Δ ≥ +0.10 → confirm → COMMIT
  Δ < +0.05  → Phase C

Phase C: T12.3 (Mip-Splatting) — 2-3 ngày
  Δ ≥ +0.10 → COMMIT, strong paper claim
  < +0.05    → ACCEPT ceiling, writeup with comprehensive negative results

Phase D (fallback): writeup
  Phase 8 FULL 21.335 paper baseline
  6 Phase 11 + 3 Phase 12 negative results documented
  Methodology contribution: multi-seed paired methodology
```

---

## PHASE 12 — Full Experiments (deferred until Phase 11 done)

| # | Task | Status |
|---|------|--------|
| T12.1 | LLFF 8 scenes × baselines (with best CRS variant) | (pending Phase 11 verdict) |
| T12.2 | DTU 15 scenes × baselines | (pending) |
| T12.3 | Blender 8 scenes × baselines | (pending) |
| T12.4 | Compile Table 1 (PSNR/SSIM/LPIPS + compute metrics) | (pending) |
| T12.5 | Compile Table 2 (Depth RMSE / Floater Ratio) [HOLD] | |

---

## Notes / Blockers

*(Ghi vào đây khi gặp vấn đề)*

---

## Future Work / Ideas

- **Selective CRS update**: Hiện tại update_crs() tính D_i+R_i cho TẤT CẢ N Gaussians mỗi lần. Gaussians đã ổn định (không di chuyển) có D_i/R_i giống hệt → update thừa. Ý tưởng: chỉ update Gaussians mới sinh + Gaussians có gradient lớn. Tiết kiệm ~80% compute ở iter muộn. Chưa cần vì overhead hiện tại nhỏ (vài ms/lần, N=90k, K=3).

---

## PHASE 13.2 — CURRENT STATE (2026-05-14)

> **Phase 12 task table ở trên (T12.1 iter-15k / T12.3 Mip-Splatting decision tree) = VOID.**
> Superseded bởi Phase 13 A3 commit + Phase 13.2. T12.1 iter-budget LOẠI VĨNH VIỄN
> (10k iter LOCKED cho clean paper comparison). Mip-Splatting REJECTED (Phase 13).
> Chi tiết verdict: xem `04_decisions_log.md` các entry [2026-05-14].

### Baseline locked
- **A3 (LFCF+AbsGS) = test 21.330 N=24, Δ=+0.164 SIG** — Phase 13 FULL recipe committed.
- A3 train 34.21 / test 21.33 / **overfit gap 12.88 dB** (= dominant measured signal).

### Đã REJECT trong 13.2 (KHÔNG re-propose)
| Task | Verdict |
|------|---------|
| [x] T13.2.0 GT FFT spectrum analysis | DONE — informed σ, nhưng Gap C sau đó bị overturn |
| [x] T13.2.0' Render-vs-GT diagnostic | DONE — universal HF deficit, A3=spatial-not-spectral |
| [x] T13.2.1 HF-emphasis pilot (Laplacian, 24 runs) | ❌ REJECT — INVERSE pattern, phase-amplitude conflict |
| [x] T13.2.3 Covis-weighted pre-flight | ❌ REJECT — covis degenerate (multi=0 4/8), no substrate |
| [x] T13.2.5 GDAGS A/B full-8 (2026-05-18) | ❌ REJECT — Δtest≈0 «noise, ΔN+51%/Δtrain+2.08=overfit, horns −0.605 phá A3-best |
| ⏸ DWTGS / FALA-blur / FALA-reversed | SKIP — wrong-sign / same-class predicted fail |
| ⏸ Tier 3 architecture (đổi primitive) | DEPRIORITIZED — train=34 chứng minh primitive KHÔNG phải limit |
| ⏸ Densification-axis (mọi capacity-add) | EXHAUSTED — 3rd confirm 3-view capacity ceiling (HF/D3/GDAGS) |

### Synthesis chốt
- **Cross-view consistency DEAD** trong 3-view wide-baseline (hợp nhất 5 thất bại: pseudo-view/DUSt3R/MPC/CRS-pull/covis). Loại trước lớp cross-view + SOTA Binocular3DGS/NexusGS/SCGaussian.
- **HF-deficit-as-bottleneck REFUTED** — là triệu chứng overfit, không phải architectural limit.

### Phase 13.2.4-5 DONE (2026-05-17)
- [x] **T13.2.4 Bottleneck decompose** — verified 8-scene. H3≈0.5% (geometry SOLVED), H1=21% irreducible, H4=63% appearance, H8≈0 (exposure refuted), chroma/specular refuted (room hypothesis bác). Lỗi = 3-view appearance ambiguity, không phải mechanism post-hoc.
- [x] **OVER-CLAIM corrected**: "accept ceiling" sai — bottleneck post-hoc MÙ với training-dynamics. AbsGS (+0.164 densify-axis) = bằng chứng axis viable. Co-Adapt dropout family đã exhausted (D3 −0.48). AbsGS > LFCF (LFCF alone +0.015).
- [x] **T13.2.5 GDAGS Gate-2** — verified mechanism (GCR=grads/grads_abs, KHÔNG orthogonal = policy A/B trên trục AbsGS). Gate-2 full-8 ✅ TRACTION (8/8 non-degenerate). Standalone, no production touch.

### Phase 13.2.5 GDAGS — DONE ❌ REJECTED (2026-05-18)
- [x] **Implement use_gdags** flag default OFF, helper `utils/densify/gdags.py`, gate mirror absdensify, LFCF KHÔNG đụng, train.py KHÔNG sửa — đúng contract
- [x] **B1 verify flag-OFF = A3 byte-identical** — ✅ PASS (N reldiff <5%; PSNR=atomicAdd noise, N là primary)
- [x] **A/B pilot full-8 seed42** (paired vs logs/p13_lfcf/A3_seed42): **Δtest_mean=+0.0149 (« ±0.10 noise floor) · ΔN=+51% · Δtrain=+2.08 · horns −0.605 (phá A3 best-win)** → ❌ OVER-DENSIFY→OVERFIT
- [x] **REJECT GDAGS → LOCK A3 21.330.** 3rd independent confirmation **3-view capacity ceiling** (HF-pilot −0.31 / D3 −0.48 / GDAGS ≈0+horns−0.605). Densification-axis EXHAUSTED.
- [ ] **Cleanup pending (Quy tắc 13)** — chờ user approve: revert arguments/gaussian_model 5 chỗ, delete utils/densify/gdags.py + scripts/p13_2_gdags_*, rm output/p13_2_gdags/. GIỮ logs + design Section 20. Gộp HF-emphasis cleanup.

### Phase 13.2 — NEXT DECISION (2026-05-18)
> Densification-axis exhausted (GDAGS). Loss-axis exhausted (Phase 11 6/6 + HF-pilot).
> CRS-axis exhausted (9/9). Cross-view class structurally dead (5 fail). Capacity-add =
> memorize ×3 confirm. Bottleneck verified = 3-view appearance ambiguity (H4=63%, H1=21%).
- **Survivor axis = external-prior injection (chưa đụng)**: `dn-splatter` monocular **NORMAL** prior. Phase 8 chỉ dùng DAV2 **depth** — normal là tín hiệu external-prior orthogonal, KHÔNG capacity-add, KHÔNG cross-view, KHÔNG loss-only-frequency. Chưa từng test trong toàn bộ project.
- **Alternative = accept 21.330** — giờ defensible bằng: verified bottleneck (appearance ambiguity) + 3 capacity-axis fail + cross-view structural-dead + loss/CRS/freq exhausted. Gap tới DOC-GS −0.05 / BinocularGS −0.11 (đã rất sát tier).
- **Pending user**: chọn (a) pre-flight dn-splatter normal-prior (Gate-style trước implement) hay (b) chốt accept 21.330. KHÔNG re-propose: densify / loss-freq / CRS / cross-view / dropout / foundation / iter-budget.
- [ ] Caveat: GDAGS policy-A/B KHÔNG +feature, kỳ vọng modest, có thể ≈/< AbsGS
- [ ] Cleanup HF-emphasis reject files (Quy tắc 13) — chờ approve

### Phase 14 — L_consist (Binocular3DGS) port — PLAN LOCKED, đang Bước-0 (2026-05-18)
> User pivot: research Binocular3DGS in-workspace. Verified L_consist = self-supervised
> single-view stereo loss → thoát cross-view-dead + capacity-ceiling. Orthogonal vs
> d_cycle (no_grad-score ≠ differentiable-loss). Chi tiết: decisions_log [2026-05-18] Phase 14.
- **2×2 factorial**: A=A3(reuse) · B=A3+Lc · C=A3−Dcyc(→D_DAV2) · D=A3−Dcyc+Lc. (D_cycle-off = clean toggle; test JUSTIFIED vì decisions_log:948 "D_cycle flips negative on strong backbone", chưa A/B lại trên A3.)
- [~] **Bước 0** — A3 re-verify (server ĐANG CHẠY): re-run A3 horns+orchids seed42, LOG_DIR=logs/verify_a3_restore (KHÔNG đè baseline), so N vs logs/p13_lfcf. <5% ✅ / ≥20% ❌ STOP.
- [x] **Bước 0** — A3 re-verify DONE ✅ PASS: horns N reldiff 2.80% / orchids 0.06% (<5%). PSNR drop (−0.31/−0.07) « ±1.3 noise; orchids identical-N+PSNR-drop = airtight proof = atomicAdd, KHÔNG code-regression. Baseline reuse hợp lệ.
- [~] **Bước 1** — saturation pre-check WRITTEN `scripts/p14_lconsist_precheck.py` (no-train, standalone, verified-from-code). substrate = res(B│valid∧~Lamb) − res(B→0│cùng region); UNMASKED=upper-bound (UB≈0→reject chắc). Chờ chạy server.
- [x] **Bước 1** — pre-check: auto-verdict "reject" = FALSE-REJECT (MARGIN_ABS=0.010 tôi bịa, uncalibrated). Data thật: substrate mọc ~tuyến tính theo B 8/8 scene, 4/8 substantial (fortress/orchids/horns/leaves). KHÔNG obvious-no-go → KHÔNG reject, đi tiếp (pilot là arbiter thật).
- [x] **Bước 2** — verify-from-code production hook DONE (flag→OptimizationParams; hook SAU L_depth train.py:427; bg per-iter; scene.cameras_extent in-scope; disable_dropout API ok).
- [x] **Bước 3** — implement DONE: arguments +5 flag (use_lconsist OFF); `utils/loss/binocular_consistency.py` NEW (differentiable, #1/#2/#3 fixed, faithful Binocular); train.py +hook ≤14 dòng gated. Flag-OFF=A3 by construction. C=0 code.
- [~] **Bước 4** — pilot runner + analyzer + pre-pilot verify (flag-OFF N=A3 + flag-ON 1-scene smoke).
- [x] **Bước 5** — pilot DONE. **B (A3+Lc) ❌ REJECT**: ΔB=+0.041«±0.10 saturate + **horns −0.318 catastrophe (3rd-confirm fragility: HF/GDAGS/Lc)** + +36% cost. **C: D_cycle CONFIRMED beneficial trên A3** (ΔC=−0.135, horns−0.925) → giữ D_cycle (Phase-7 flip không replicate). D reject (D−B=−0.045, 2 catastrophe).
- [x] **Dense-init Gate DONE** `scripts/p14_denseinit_gate.py`: auto-GO=false-accept (cherry-pick R8); substrate sụp 22→8→3.5% theo R; cross-val leaves NO_HELP 95.6%↔H1 92.6%. KHÔNG đóng hẳn (Gate ALREADY-mask ambiguity) → cần refine nearest-sparse-dist.

### Phase 15 — Untried orthogonal PHYSICAL axes (2026-05-18)
> User push-back ĐÚNG: Phase-11 "loss-axis dead=ceiling" → user đẩy frequency → +0.164=21.330.
> Claim-exhausted có tiền sử SAI khi có trục-vật-lý orthogonal chưa thử. KHÔNG accept 21.330 vội.
> Info-ceiling chỉ giải thích xào-cùng-trục saturate; trục-vật-lý-MỚI rút thêm recoverable-signal
> = đúng điều frequency/AbsGS làm. Docs caveat: H4 có mảnh reducible-overfit chưa tách.
> **Phase 14 L_consist REJECT+CLEANED (2026-05-18)** — production reverted (arguments+train.py),
> A3/Phase-13 byte-clean, build mới TRÊN A3. Chi tiết decisions_log [2026-05-18] Phase 14 CLEANED.
- [x] Recon DONE. Đã đóng: density-dropout (D3 −0.32 verified), anisotropy-blunt (Q4 77%-flat verified 8/8).
- [x] (a) Anisotropy diagnostic `p15_aniso_diag.py` DONE — blunt s_max/s_min CLOSED (77% legitimate-flat). Gap: lfcf_diffscale chỉ densify-time (verified) nhưng blunt-ratio sai form.
- ❌ ~~(b) Density-aware dropout~~ REJECT Phase-2d 2026-04-21 (D3 −0.32). KHÔNG re-propose.
- **3-arm Phase-15 (user "test cả 3", trên A3-clean):**
  - [ ] **A** blunt `s_max/s_min` L_aniso — pilot = **control falsify Q4** (diagnostic misfire nhiều) + thử. EV thấp nhưng meta-value.
  - [ ] **B** targeted `s_max-excess-vs-scene` — form data CHỈ vào (né 77%-flat). EV khá hơn A.
  - [ ] **C1** dn-splatter DSINE monocular-normal — verified: external estimator → **nặng-preprocess (class dense-init)**, NHƯNG loss-integration (recipe-risk<init-replace), duy nhất +info-mới. Setup-then-pilot (verify DSINE-weights workspace trước).
  - [~] **C2** depth→normal self-consist — DEPRIORITIZE (no-new-info, predicted-saturate ≈ L_consist-class).
- [ ] Implement A+B (Quy tắc 11/12, default OFF=A3-identical) → pilot 2-GPU full-8 single-seed reuse-A3, per-scene catastrophe-guard. C1 verify+plan song song.
- ⚠️ Lesson tích lũy: "untried" verify vs FULL memory-file + decisions_log (KHÔNG index/code-presence); verify-from-code mechanism TRƯỚC implement (C over-claim "nhẹ" đã sửa bằng đọc code); diagnostic project misfire nhiều → empirical control đáng giá.