# CRSGaussian — Context for Claude Code

> **THỨ TỰ ĐỌC**:
> 1. **`docs/00_code_session_rules.md`** ← BẮT BUỘC đọc trước khi code (env mapping, coding rules, common pitfalls)
> 2. File này (CLAUDE.md) — context dự án + công thức cốt lõi
> 3. `docs/01_research_summary.md` — methodology đầy đủ
> 4. `docs/03_task_queue.md` — task hiện tại

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

**Status hiện tại (cập nhật 2026-05-08): 🎉 Phase 8 BREAKTHROUGH — first CRS contribution defendable**

**Phase 6 (Apr 2026)** — 8 CRS variants tested, all marginal: 6 pruning ablations, C1 occlusion-aware,
F-invisible, D1G gate-densify, hC hybrid RC×D, RNRC L3 differentiable. Ceiling +0.07 dB.
Hypothesis: signal D+R bottleneck, not mechanism class.

**Phase 7 Tier 2-min (May 2026)**:
- **Phase 5 weak backbone**: DCYCLE alone +0.125 (FIRST CRS variant với clean positive),
  LWEIGHT alone neutral, combined synergy −0.083 redundant
- **D1-O999 strong backbone (Stage 1)**: D_cycle FLIPS NEGATIVE (−0.015), combined +0.018 vs no-CRS
  (within noise). Hypothesis: R contamination 36.5% diluting D_cycle.

**🎉 Phase 8 (May 2026) — Formula Redesign + SH Path BREAKTHROUGH:**
- Implementation: R_visible (visibility-aware), S_stability EMA, CRS-modulated SH freeze
- **FULL = 21.335 dB** vs OLD 21.178 (+0.156) vs No-CRS 21.21 (**+0.125** ⭐ first beat!)
- Attribution:
  - Δ_R (R_visible alone) = **−0.111** (surprising — data loss > noise reduction)
  - Δ_D (D_cycle in clean R) = +0.102 (vindicated — R noise was diluting)
  - Δ_S (S_stability) = −0.023 (neutral)
  - **Δ_M (CRS-modulated SH freeze) = +0.189** (BIGGEST WINNER)
- Compute: +3.7% training slowdown, GPU memory + render FPS unchanged

**Phase 9 (May 2026) — Simplification + Cross-backbone DONE:**
- Test 1 results: ALL simplifications HURT
  - FULL = 21.335 (BEST), drop S → 21.200 (−0.135), drop R+S → 21.159 (−0.176)
- Test 2 results: A1B1_BEST 20.983 (+0.051 vs A1B1_BASELINE 20.932)
- ❌ H1 REJECTED (drop R), ❌ H2 REJECTED (drop S), 🟡 H3 PARTIAL (SH freeze works smaller on A1+B1β)
- **Key insight**: Sequential Δ ≠ leave-one-out. All 4 components (D, R, S, mechanism) **synergize**
- → **Phase 8 FULL recipe LOCKED at 21.335 dB. CRS axis exhausted.**

**Phase 10A (May 2026) — DUSt3R Dense Init FAIL hard, axis DEAD:**
- Implementation: DUSt3R clone + precompute cache + 16-run AUGMENT/REPLACE × 8 scenes
- Result: AUGMENT Δ=−0.898, REPLACE Δ=−3.529 (catastrophic)
- 6-run diagnostic (FILTER/DENSIFY/BOTH × orchids+leaves) → ceiling ≈ −0.07 dB even with optimal filter
- ❌ All 3 hypotheses REJECTED → systematic failure, not tuning
- Root cause: DUSt3R alignment residual error + Phase 8 recipe calibrated cho ~3K sparse init
- → **Foundation-model dense init nói chung BỎ HẲN** (DUSt3R, MASt3R same class)
- Cleanup: env + checkpoint + cache + source removed (~5GB freed). Code Phase 10A giữ default OFF.

**Current state (2026-05-08): Phase 11 — Loss-axis Exploration (sequential strategy)**
- CRS axis exhausted ở 21.335. Initial-PC axis dead. Loss-axis là direction còn lại.
- 4 candidates xếp theo cost: Step 1 covisibility (cheapest, in progress) → Step 2 perceptual → Step 3 R_feature → Step 4 cross-view MPC (conditional)
- Strategy: sequential evaluation, abort early if win. Best case 1 ngày, worst 4 ngày.
- Pivot ready (if all fail): regularization / depth fine-tune / accept ceiling

**Reference targets (LLFF 3-view):**
| Method | PSNR | Note |
|--------|------|------|
| Phase 8 FULL | 21.335 | first CRS variant defendable, current best |
| No-CRS Tier1 | 21.21 | D1-noCRS-O999 |
| DOC-GS | 21.38 | gap −0.045 (within noise) |
| BinocularGS | 21.44 | gap −0.105 |
| **ICO-GS (SOTA)** | **22.20** | **gap −0.865** |

**To break SOTA:** Phase 11 loss-axis với external supervision (covisibility / perceptual / R_feature / cross-view MPC). Honest probability ~40% best-single ≥ +0.20.

Xem **docs/11_crs_diagnostic_redesign.md** cho full arc + Phase 7-9 details.

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

### Quy tắc 10 — TAG code mới với marker `[CRSGaussian ...]`

Mọi đoạn code thêm/sửa PHẢI có comment marker `[CRSGaussian]` hoặc `[CRSGaussian Tx.x]` để dễ grep ngược tìm lại sau này.

**Ví dụ:**
```python
# ── [CRSGaussian Pseudo-photo] ... ──
if dataset.use_pseudo_photo_loss:
    ...

# ============================================================
# [CRSGaussian] Task: T5b2 — warp_image_forward
# File: utils/depth/depth_warping.py
# ============================================================
def warp_image_forward(...):
    ...
```

**Lý do:** train.py + các file khác đã >800 dòng. Khi review/debug cần grep `[CRSGaussian` để biết đoạn nào là CRSGaussian thêm vs CoR-GS gốc.

### Quy tắc 11 — Thiết kế MODULE BẬT/TẮT cho mọi feature mới

Mọi ý tưởng mới PHẢI có:
- **Master switch** trong `arguments/__init__.py` (default `False`)
- **Gating** trong code: `if dataset.<feature_flag> and ...` (5 điều kiện AND nếu cần)
- **Default OFF** đảm bảo baseline không thay đổi khi không bật flag
- **Verify**: chạy với flag OFF → kết quả phải giống baseline cũ

**Ví dụ pattern:**
```python
# arguments/__init__.py
self.use_my_new_feature = False  # master switch
self.lambda_my_feature  = 0.05
self.my_feature_start_iter = 1000

# train.py
if (dataset.use_my_new_feature
        and dataset.use_depth_prior        # prerequisites
        and iteration >= dataset.my_feature_start_iter
        and <other_conditions>):
    # ... feature code chỉ chạy khi bật
```

**Lý do:** Cho phép ablation A/B clean (B0 vs B0+feature), dễ rollback nếu feature hại, dễ combine nhiều features.

### Quy tắc 12 — TÁCH FILE cho ý tưởng mới, IMPORT vào train.py

KHÔNG nhồi logic mới (>30 dòng) trực tiếp vào train.py. Thay vào đó:

1. **Tạo file mới** trong subdir thích hợp:
   - `utils/depth/<feature>.py` cho depth-related
   - `utils/crs/<feature>.py` cho CRS-related
   - `utils/loss/<feature>.py` cho loss-related (tạo dir mới nếu cần)
   - `utils/regularizer/<feature>.py` cho regularization
2. **Function chính** trả về loss tensor (có gradient nếu cần)
3. **train.py chỉ chứa**:
   - Import: `from utils.<dir>.<feature> import compute_<feature>_loss`
   - Gating block với `if dataset.use_<feature>:`
   - Gọi `L = compute_<feature>_loss(...)` (1-3 dòng)
   - `LossDict["loss_gs0"] += L`
   - TB log

**Ví dụ pattern đúng:**
```python
# utils/loss/pseudo_photo.py  (TẠO MỚI)
def compute_pseudo_photo_loss(rendered_img_P, nearest_cam, pseudo_cam,
                                aligned_depth_dict, lambda_weight):
    """[CRSGaussian Pseudo-photo] Compute pseudo-view photometric loss."""
    warped_gt, valid_gt = warp_image_forward(...)
    if int(valid_gt.sum().item()) < 100:
        return None, 0.0  # skip
    mask3 = valid_gt.unsqueeze(0).expand(3, -1, -1).float()
    L = lambda_weight * l1_loss_mask(rendered_img_P, warped_gt, mask3)
    coverage = float(valid_gt.float().mean().item())
    return L, coverage

# train.py  (CHỈ 5-10 dòng cho mỗi feature)
from utils.loss.pseudo_photo import compute_pseudo_photo_loss

if dataset.use_pseudo_photo_loss and iteration >= dataset.pseudo_photo_start_iter \
        and pseudo_cam_co is not None and "image_pseudo_co_gs0" in RenderDict:
    nearest_cam = find_nearest_training_cam(pseudo_cam_co, allCameras)
    if nearest_cam and nearest_cam.uid in aligned_depth_dict:
        L_photo, cov = compute_pseudo_photo_loss(
            RenderDict["image_pseudo_co_gs0"], nearest_cam, pseudo_cam_co,
            aligned_depth_dict, dataset.lambda_pseudo_photo)
        if L_photo is not None:
            LossDict["loss_gs0"] += L_photo
            if tb_writer and iteration % 100 == 0:
                tb_writer.add_scalar('loss/pseudo_photo', float(L_photo.item()), iteration)
```

**Lý do:**
- train.py hiện 833 dòng — khó review, khó merge
- Logic riêng biệt → dễ unit test
- Mỗi ý tưởng = 1 file → dễ tìm, dễ xoá nếu fail
- Import line + gate block là **interface clean** giữa main loop và feature

### Quy tắc 13 — DỌN DẸP khi hướng đi bị reject

Sau khi test một hướng (feature mới, ablation, diagnostic) và **quyết định KHÔNG dùng** (kết quả không improve, hoặc trade-off không đáng), PHẢI dọn dẹp để không tốn dung lượng và giảm noise khi grep/review sau này.

**Khi hướng bị REJECT, cần xóa:**

1. **Script ablation/test** trong `scripts/` — `.sh` và `.py` wrappers chỉ dùng cho hướng đó
2. **Module implementation** trong `utils/.../<feature>.py` nếu feature tách file riêng
3. **Flags trong `arguments/__init__.py`** nếu chỉ feature đó dùng
4. **Gating block** trong `train.py`/renderer — revert về baseline
5. **Output dirs** `output/<ablation_name>/` (nặng, nhiều GB)
6. **Log dirs** `logs/<ablation_name>/` (nhẹ nhưng vẫn nên dọn)

**KHÔNG xóa:**
- `docs/<task>_results.md` — giữ để ghi nhớ đã test gì + lý do reject
- Log cuối của ablation cuối (`summary` output) — reference cho paper discussion

**Quy trình reject + cleanup:**

```bash
# 1. Ghi chú kết quả + lý do vào docs/decisions_log.md
echo "## 2026-04-XX — DropAnSH rejected" >> docs/decisions_log.md
echo "Test 3 scenes: D1/D2/D3 ≤ B0 (không improve). Skip Phase 2." >> docs/decisions_log.md

# 2. Xóa files
rm scripts/ablation_dropansh_phase1.sh
rm scripts/smoke_test_dropansh.py
rm utils/regularizer/dropansh.py

# 3. Revert code production
# - Remove gate block trong gaussian_renderer/__init__.py
# - Remove 8 flags trong arguments/__init__.py (PipelineParams)
# - Remove bookkeeping trong train.py (nếu thêm riêng cho feature)
# → Verify bằng grep: grep -r "DropAnSH" . (phải empty)

# 4. Xóa output + logs
rm -rf output/ablation_dropansh
rm -rf output/verify_dropansh_off_*
rm -rf logs/ablation_dropansh
rm -f logs/verify_dropansh_off*.log
```

**Khi ACCEPT hướng:**
- Giữ tất cả (module + flags + gating)
- Script ablation vẫn giữ trong `scripts/` để reproduce
- Update default config nếu muốn always-on

**Lý do quy tắc này:**
- Mỗi experiment thất bại để lại ~100-500MB output checkpoints + PLY
- Qua 3-5 rounds test → tổng dung lượng dư 2-5GB
- Code dead không dùng làm noise cho grep + confuse future Claude sessions
- `grep -r "[CRSGaussian ...]"` trả về code không còn relevant

**Tự đề xuất cleanup**: Sau mỗi task có verdict "không đi tiếp", Claude PHẢI **chủ động hỏi user có xóa không** + liệt kê files/dirs sẽ xóa để user approve trước khi rm.

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

Phase 4 — DONE
  T4.1  ✓ Position constraint — DISABLED (PSNR -3dB)
  T4.2  ✓ CRS pruning Option C

Phase 5 — DONE
  T5.1-T5.6 ✓ Informed CRS₀ (3-signal: q_reproj+q_depth+q_view)
  Best WG: w_reproj=0.4, w_depth=0.6, w_view=0, ema=0.3, AVG=20.36

Phase 5b — Anti-overfit (Pseudo-View Loss) — DONE/SUPERSEDED
  T5b1  ✗ Pseudo DEPTH loss — NOT WORKING (pseudo cams not novel)
  T5b2  ✗ Pseudo PHOTOMETRIC loss — không vượt baseline đáng kể

Phase 6 — CRS Diagnostic & Mechanism Exhaustion — DONE
  T6.1  ✓ Tier A diagnostic suite (BC bimodality, A3 floater, A4 occlusion 36.5%)
  T6.2  ✓ 8 CRS mechanism variants tested, all ceiling +0.07 ± 0.10 dB
  T6.3  ✓ Literature survey 17 papers → loss-path + signal upgrade selected

Phase 7 — CRS Tier 2-min: D_cycle + Loss Reweighter — DONE
  T7.1  ✓ Phase 5 weak backbone: DCYCLE +0.125 (first CRS positive)
  T7.2  ✓ Stage 1 D1-O999 strong: D_cycle FLIPS −0.015, +0.018 vs no-CRS noise
        → R contamination hypothesis cho Phase 8

Phase 8 — Formula Redesign + SH Path — DONE 🎉
  T8.1  ✓ R_visible + S_stability + CRS-modulated SH freeze implemented
  T8.2  ✓ 5-config × 8 scenes ablation (40 runs)
        FULL = 21.335 (+0.125 vs No-CRS) — first CRS contribution defendable
        Attribution: Δ_R=−0.111, Δ_D=+0.102, Δ_S=−0.023, Δ_M=+0.189 (winner)

Phase 9 — Simplification + Cross-backbone — DONE (2026-05-08)
  T9.1  ✓ Test 1: ALL simplifications HURT, FULL recipe LOCKED at 21.335
  T9.2  ✓ Test 2: A1B1_BEST +0.051 (smaller gain on A1+B1β)
  T9.3  ✓ H1/H2 REJECTED, H3 PARTIAL — components synergize

Phase 10 — DUSt3R Dense Init — DONE (FAILED, axis DEAD, 2026-05-07)
  T10.1 [x] DUSt3R wrapper + 16-run AUGMENT/REPLACE → Δ=−0.898/−3.529
  T10.1c [x] Diagnostic 6-run FILTER/DENSIFY/BOTH → ceiling −0.07
  → Foundation-model dense init BỎ HẲN. Cleanup done.

Phase 11 — Loss-axis Exploration — CURRENT (sequential, 2026-05-08)
  T11.1 [~] Step 1: CRS × Covisibility reweight (depth-based) — IN PROGRESS
  T11.2 [ ] Step 2: Same-view perceptual (DINOv2)
  T11.3 [ ] Step 3: R_feature replace R_visible
  T11.4 [ ] Step 4: Cross-view MPC (conditional)
  Goal: break Phase 8 ceiling 21.335. Best-single probability ≥+0.20 ≈ 40%.

Phase 12 — Full Experiments (deferred)
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