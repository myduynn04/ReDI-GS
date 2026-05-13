# Phase 13 — EFA-GS LFCF Port Design Document

> **Trạng thái:** Draft for user review (2026-05-12)
> **Tác giả:** Planning session
> **Mục đích:** Giải thích đầy đủ Phase 13 EFA-GS LFCF port để user **duyệt TRƯỚC** khi implement. Không có code production trong tài liệu này.

---

## Section 1 — Tóm tắt 1 trang

**Vấn đề:**
- Phase 8 FULL ceiling 21.16 multi-seed (paper 21.335 không reproduce sau commit `0511edd`).
- Test view blur nặng ở thin structures (horns, fortress, flower); train PSNR ~33 vs test ~21 → gap 12 dB.
- 9 attempts trước REJECT toàn bộ: Phase 11 Steps 1-5 + Stack (loss-axis), Phase 12 CRS-pull A1/A2/A3 (position-axis).

**Ý tưởng tiếp theo:**
EFA-GS LFCF (Low-Frequency Component First) — tấn công **densify-axis** (axis CHƯA THỬ trong CRSGaussian). Replace standard `densify_and_clone/split` với tolerance-based decision + diffscale isotropify.

**Cost:**
- **Code:** 2-2.5 ngày (no CUDA, pure Python/PyTorch) — chi tiết Section 9.4
- **Server:** ~6.5h experiments (Round 1 + 2) hoặc ~10h (+ Phase 13.1 sweep nếu winner)
- **End-to-end timeline (xem Section 8.5):**
  - Reject path: ~4 ngày (code + Round 1 + cleanup + pivot writeup)
  - Winner path: ~7 ngày (code + Round 1 + Round 2 + Phase 13.1 + paper writeup start)

**Probability commit-worthy:** **45-50%** (last CRS-axis attempt, must be honest).

**Strategic significance:** Đây là attempt cuối trên CRS axis. Nếu REJECT → pivot paper writeup "comprehensive ablation methodology" thay vì "flagship CRSGaussian 22+ dB".

---

## Section 2 — Vấn đề & Diagnose

### 2.1 Số liệu thực tế

```
Train PSNR: ~33 dB    ← scene fit pixel-perfect trên 3 training views
Test PSNR:  ~21 dB    ← novel view blur nặng
GAP:        ~12 dB    ← OVERFIT cực nặng, không phải floater đơn thuần
```

### 2.2 9 attempts đã fail trên Phase 8 FULL backbone

| Attempt | Axis | Δ paired | Verdict |
|---|---|---|---|
| Phase 11 Step 1 (Covisibility reweight) | Loss | +0.014 | 🟡 marginal, keep OFF |
| Phase 11 Step 2 (Perceptual DINO) | Loss | −0.046 | ❌ REJECT |
| Phase 11 Stack (S1+S2) | Loss | −0.007 | ❌ REJECT no synergy |
| Phase 11 Step 4 (Cross-view MPC) | Loss | −0.042 | ❌ REJECT |
| Phase 11 Step 5 (TV depth) | Loss | −0.026 | ❌ REJECT |
| Phase 12 CRS-pull A1 (full) | Position | −0.027 | ❌ REJECT |
| Phase 12 CRS-pull A2 (replace Phase 4) | Position | −0.033 | ❌ REJECT |
| Phase 12 CRS-pull A3 (pull-only) | Position | −0.096 | ❌ REJECT (gần SIG-NEG) |

→ **CRS axis 9/9 attempts ≈ noise** (paired Δ |x| < 0.10 dB, all 95% CI cross 0).

### 2.3 Bottleneck thực sự

**Phase 11 + 12 đều attack symptom (test view blur, floater) chứ không attack ROOT CAUSE.**

Root cause: **Densification spawn HF floater bừa bãi → mô hình memorize 3 training views ở pixel level**. Khi project sang novel view, Gaussians sai vị trí 3D → blur/floater visible.

### 2.4 5 architectural gaps của Phase 8 FULL

| Gap | Mô tả | Phase 8 có fix? |
|---|---|---|
| **Gap 1** — Densify spawn HF bừa bãi | Standard rule `grad > τ → split` → spawn vô tận trên HF detail của 3 views | ❌ Không attack |
| **Gap 2** — Rasterizer no anti-aliasing | Sub-pixel Gaussian project khác camera = aliased differently → train view memorize, test view mismatch | ❌ |
| **Gap 3** — Loss-function không phân biệt frequency | L1+SSIM treat HF noise và LF geometry như nhau → grad từ HF noise drive densify | ❌ |
| **Gap 4** — Split trigger không phân biệt "WHY grad cao" | Grad cao do (A) legit detail hoặc (B) noise/aliasing — split bừa | ❌ |
| **Gap 5** — Scale anisotropy unrestricted | Gaussian elongate thành "needle" theo direction grad cao → test view nhìn cạnh khác = needle sai/mất | ❌ |

**Phase 8 FULL chỉ có post-hoc handle** (prune low-CRS, SH freeze, opacity decay) — đã muộn vì floater đã spawn rồi.

---

## Section 3 — Tần số (Frequency) là gì

### 3.1 Phân loại tần số trong rendering

- **Low-frequency (LF):** Cấu trúc tổng thể, biến đổi chậm theo không gian.
  - Ví dụ: bức tường phẳng, bầu trời, mặt nước, hình khối chính.
- **High-frequency (HF):** Chi tiết, biến đổi nhanh.
  - Ví dụ: texture lá cây, edge sắc nét, lông horns, masonry brick.

### 3.2 Tại sao sparse-view dễ overfit HF

Với 3 training views, mô hình có **đủ data để fit HF pixel của 3 views chính xác** nhưng **không đủ để constrain 3D structure giữa các views**. Kết quả:

```
3 views train → grad cao tại HF pixels → spawn Gaussians fit HF
              → 3D structure giữa views không constrain → Gaussian drift
              → novel view = composition của Gaussians sai vị trí
              → blur/floater visible
```

### 3.3 "Low-Frequency First" — triết lý

Học **LF (geometry chung) trước**, **HF (detail) sau** — và CHỈ học HF khi confident là signal thật (không phải noise/aliasing).

Tương tự: human visual system perceive LF first (silhouette), HF later (texture). Cũng tương tự coarse-to-fine reconstruction trong nhiều NeRF variants.

---

## Section 4 — Cơ chế EFA-GS LFCF

### 4.1 Decision logic (tolerance comparison)

Mỗi `interval_times × densification_interval` iter (TaT default: 200 iter), thay vì standard clone+split:

```python
# Pseudo-code (logic chính)
selected_now = (grad_norm > threshold)        # Gaussians grad cao hiện tại
prev_selected = self.prev_selected_pts_mask    # Gaussians grad cao kỳ trước

# Phân thành 3 nhóm:
new_selected   = NOT prev_selected AND selected_now   # mới pick lần này
intersection   = prev_selected     AND selected_now   # pick liên tục 2 kỳ
                                                       # → đáng nghi: signal thật hay stuck noise?

# Cho new_selected: chưa có grad history → DEFAULT ENLARGE
#                   (chưa biết signal hay noise, give chance học LF)
enlarged_mask = new_selected

# Cho intersection: dùng tolerance compare:
decent = (curr_grad <= prev_grad - tolerance)   # grad ĐANG GIẢM?

if decent:
    → Model học signal thật, gradient giảm → SPLIT (chia nhỏ để fit HF detail)
    splitted_mask[intersection] = True

if NOT decent:
    → Grad stuck/oscillate (noise hoặc aliasing) → ENLARGE
    enlarged_mask[intersection] = True
```

**Insight (cơ chế):** Tolerance check phân loại gradient delta để chọn enlarge vs split. Standard 3DGS không có khái niệm này — luôn split khi grad cao.

#### 4.1.1 Lưu ý quan trọng về `tolerance = 1e-5`

**Authors KHÔNG chọn 1e-5 để tune signal-vs-noise detection**. Chosen value mainly là **fix hardware floating-point reproducibility** (per README lines 200-204):

> Different GPU architectures (A100 vs V100) introduce slight FP variations (~1e-6) due to FMA, Tensor Core behavior, and non-deterministic PyTorch operators. Strict comparison `grad > prev_grad` is too sensitive — even tiny variations trigger different execution paths and divergent training. Tolerance 1e-5 mitigates this.

**Evidence (TanksandTemples V100):**

| Variant | PSNR |
|---|---|
| EFA-GS strict comparison (no tolerance) | 21.67 |
| EFA-GS tolerance-based (tolerance=1e-5) | **21.72** (+0.05) |

→ Tolerance là **reproducibility fix với bonus PSNR side-effect**, không phải core LFCF mechanism.

**Implication CỰC KỲ QUAN TRỌNG cho CRSGaussian:**

Per memory `project_3dgs_variance_floor.md`: CRSGaussian có **atomicAdd non-determinism → variance ±1.3 dB single-scene**. Đây là **CÙNG ROOT CAUSE** với vấn đề EFA-GS authors giải quyết bằng tolerance!

→ Tolerance mechanism có thể **mitigate atomicAdd variance** của CRSGaussian, không chỉ là LFCF mechanism. Đây là argument bổ sung mạnh cho Phase 13.

**Decision về tolerance value:**
- Round 1 dùng **1e-5 (TaT default)** — sticking with author-validated, không tune sớm
- Tolerance sweep = **Phase 13.1 contingent post-win** (xem Section 10.4)

### 4.2 Diffscale (volume-preserving isotropify)

Khi enlarge Gaussian với multiplier `mult`:

```
Sort 3 scales ASC: [s_min, s_mid, s_max]

Áp dụng:
  s_min × mult^(+1)      → enlarge trục nhỏ nhất
  s_mid × mult^(-1/3)    → shrink trục giữa nhẹ
  s_max × mult^(-2/3)    → shrink trục lớn nhất MẠNH

Volume sau enlarge:
  V_new = (s_min × mult) × (s_mid × mult^(-1/3)) × (s_max × mult^(-2/3))
        = s_min × s_mid × s_max × mult^(1 - 1/3 - 2/3)
        = V_old × mult^0
        = V_old   ✓ VOLUME PRESERVED
```

**Diffscale là volume-preserving isotropify**: Gaussian giữ nguyên thể tích nhưng trở nên "tròn hơn" (anisotropy giảm).

#### 4.2.1 Méo có phải xấu? Khi nào diffscale can thiệp

**Câu hỏi natural:** "Edge thật trong scene dài/mỏng → Gaussian needle (méo) có khi đúng hơn Gaussian tròn. Sao lại ép tròn?"

**Trả lời ngắn:** Diffscale KHÔNG ép mọi Gaussian phải tròn vĩnh viễn. Nó chỉ là **intervention tạm thời cho Gaussian méo + đang stuck**.

**Phân biệt 2 loại méo (anisotropy):**

| Loại | Đặc điểm | LFCF làm gì | Diffscale active? |
|---|---|---|---|
| **Méo tốt** (signal-aligned) | Gaussian needle bám đúng edge thật, scale ví dụ `(0.1, 0.1, 5)`. Gradient ĐANG GIẢM (decent) → model học đúng. | **SPLIT** (chia nhỏ fit chi tiết) | ❌ KHÔNG — diffscale chỉ áp dụng cho enlarge path |
| **Méo xấu** (optimization artifact) | Gaussian needle pathological, scale ví dụ `(0.01, 0.05, 20)`. Gradient stuck/oscillate → đang cố fit noise hoặc bị kéo lệch. | **ENLARGE + diffscale** (reset tư thế, để optimizer thử lại) | ✅ CÓ |

**Insight quan trọng:** Diffscale CHỈ active trên **enlarge path** (gradient stuck). Gaussian méo nhưng grad decent → đi vào **split path** → diffscale không touch shape nó.

**Tại sao "tròn hơn" giúp thoát stuck:**

```
Loss surface của Gaussian quá méo (needle 20:1):

         \              /
          \            /          ← hẻm núi hẹp
           \          /             gradient nhảy mạnh theo 1 chiều
            \________/              dễ oscillate, overshoot

Sau diffscale → optimization landscape smoother:

         \                /
          \              /        ← thung lũng rộng
           \            /           dễ trượt về minimum
            \__________/
```

**Volume-preserving giải quyết 2 failure mode khác:**

| Approach | Vấn đề |
|---|---|
| Chỉ enlarge mọi chiều (mult > 1 trên tất cả axes) | Volume nổ → Gaussian thành blob to → blur, mất sharpness |
| Chỉ shrink chiều lớn | Volume co mạnh → Gaussian mất coverage → spawn floater mới |
| **Diffscale (volume-preserving)** | Redistribute scale giữa các trục, total "mass" giữ nguyên. Giống bóp bóng nước: bóp 1 chiều → chiều khác phình, lượng nước không đổi |

**Critical: Diffscale là reversible**

Sau khi diffscale reset Gaussian từ méo `(0.05, 0.1, 15)` → `(0.5, 0.4, 4)`, training tiếp tục. Nếu edge dài là **signal thật**, optimizer sẽ **re-elongate** Gaussian lại theo đúng hướng:

```
iter 5000:  (0.05, 0.1, 15)   ← stuck, diffscale active
iter 5200:  (0.5, 0.4, 4)     ← sau diffscale, tròn hơn, isotropic
iter 5500:  (0.3, 0.3, 7)     ← optimizer tự kéo dài lại
iter 6000:  (0.2, 0.3, 8)     ← anisotropic theo signal đúng (NOT pathological)
```

**Analogy:** Reset tư thế khi đang chạy sai form — không phải tư thế cũ luôn sai, mà vì lúc đó nó khiến bạn kẹt. Sau reset, nếu form cũ thực sự tối ưu, bạn sẽ tự về form đó.

→ Diffscale không áp đặt "tròn là tốt hơn méo". Nó nói: "khi méo + stuck → tạm reshape để optimizer thử lại từ tư thế tốt hơn".

### 4.3 Tại sao Diffscale quan trọng cho ta

**Phase 12 CRS-pull HẠI 3 scenes có thin structures:**
- `horns`: Δ = −0.241 (thin antlers)
- `fortress`: Δ = −0.246 (complex masonry edges)
- `flower`: Δ = −0.180 (thin petals)

**Lý do failure:** Gaussian elongated theo direction grad cao → "needle" mảnh. Novel view nhìn cạnh khác → needle:
- Có thể xuất hiện như stray line không có trong GT (floater)
- Có thể disappear do projection alignment (blur/missing geometry)

**Diffscale fix:** Isotropify Gaussian → không có "needle" → robust với arbitrary viewing angle.

→ Diffscale là **direct test cho Phase 12 failure mode**.

### 4.4 Probabilistic split (depth-aware)

Khi gradient decent (signal thật → split candidate), KHÔNG split tất cả:

```python
# Depth-dependent split probability
interval_coef = normalize(depth_to_camera)    # 0=closest, 1=deepest

splitting_lb_now = 1 - (iter - densify_from) / (densify_until - densify_from)
# splitting_lb decays linearly từ 1.0 → 0 over densify range
# → Early training: split aggressive
# → Late training: split conservative

prob_split = interval_coef × (splitting_ub - splitting_lb_now) + splitting_lb_now
whether_split = uniform(0,1) <= prob_split

splitted_mask &= whether_split    # lottery cuối
```

**Insight 1 (depth-aware):** Deeper Gaussian projected onto fewer pixels → smaller sampling rate → cần split nhiều hơn để compensate.

**Insight 2 (schedule):** Late training → splitting_lb_now → 0 → chỉ deep Gaussians split, shallow Gaussians enlarge thay split. Mostly enlarge ở cuối → fewer total Gaussians → less overfit capacity.

### 4.5 Tóm tắt 4 cơ chế EFA-GS

| Mục | Vấn đề của standard 3DGS | Cách giải quyết | Kết quả |
|---|---|---|---|
| **4.1 Decision logic (tolerance compare)** | Gradient cao là split luôn → dễ split nhầm vào noise/artifact | So sánh grad hiện tại với grad kỳ trước để phân biệt **signal thật / stuck noise** | Chỉ split khi Gaussian thật sự đang học tốt |
| **4.2 Diffscale (volume-preserving isotropify)** | Gaussian stuck thường bị quá méo (*needle-like*), khó tối ưu | Reshape Gaussian: tăng trục nhỏ, giảm trục lớn, **giữ nguyên volume** | Gaussian ổn định hơn, dễ thoát local bad shape |
| **4.3 Tác dụng Diffscale trên thin structures** | Needle Gaussian fit tốt training view nhưng fail novel view (floater / mất chi tiết) | Làm Gaussian isotropic hơn để bớt phụ thuộc orientation | Robust hơn với góc nhìn mới |
| **4.4 Probabilistic split (depth-aware)** | Split tất cả candidate → số Gaussian bùng nổ, dễ overfit | Split theo xác suất dựa trên **depth + training schedule** | Giữ model gọn, chỉ tăng complexity nơi thật cần |

---

## Section 5 — 3 Bottlenecks EFA-GS giải quyết

| Bottleneck CRSGaussian | EFA-GS fix | Mechanism |
|---|---|---|
| **Densify spawn HF floater bừa bãi (Gap 1)** | Tolerance-based decision | Grad delta phân biệt signal (decent → split) vs noise (stuck → enlarge thay split) |
| **Thin structures blur test view (Phase 12 failure)** | Diffscale isotropify | Shrink major axis, enlarge minor — bóp anisotropy, volume-preserved |
| **Aliasing HF noise (Gap 4)** | Enlarge thay split khi grad stuck | Gaussian phủ rộng hơn 1 pixel → cover sub-pixel noise bằng spatial smoothing |

→ **3 bottlenecks attacked simultaneously bằng cùng 1 mechanism.** Đây là điểm khác biệt với Phase 11/12 (mỗi attempt attack 1 mechanism, không đa-axis).

---

## Section 6 — Tại sao chọn EFA-GS, KHÔNG Mip-Splatting

### 6.1 So sánh compatibility

| | EFA-GS LFCF | Mip-Splatting |
|---|---|---|
| Implementation | Pure Python / PyTorch | Custom CUDA kernel |
| Conflict với CoR-GS rasterizer | ❌ Không | ✅ MERGE 2 CUDA forks |
| Engineering cost | 1.5-2 ngày | 3-4 ngày + risk high |
| Touch `submodules/diff-gaussian-rasterization-confidence/` | KHÔNG | CẦN (Rule 5 violation potential) |

### 6.2 Evidence base trên regime tương tự

| Regime | EFA-GS gain | Mip-Splatting gain |
|---|---|---|
| TanksandTemples (~21 dB, sparse-ish) | +0.17 ~ +0.22 ✅ | **−0.94** ❌ |
| MipNeRF 360 (~27 dB, dense) | −0.06 ~ −0.09 | +0.34 |
| LLFF 3-view của ta (~21 dB, very sparse) | predicted positive | predicted negative |

**TaT là regime gần nhất** với LLFF 3-view (low PSNR ceiling + forward-facing sparse cameras).

### 6.3 Tại sao Mip-Splatting tệ trên sparse-view

```
filter_3D = distance / focal_length × √0.2
```

- LLFF 3 views gần nhau (forward-facing cluster) → distance từ Gaussian đến cam khá đồng đều
- focal_length lớn → filter_3D nhỏ → mechanism contributes ít
- BUT filter_3D vẫn enforce **min projected size** → Gaussians nhỏ bị bumped up → over-smoothed
- Loss tăng → model fight back → spawn thêm → cycle vô hạn

TaT result −0.94 dB **confirm** mechanism conflict với sparse-view densification.

### 6.4 Verdict

EFA-GS = **higher probability + lower cost + lower risk**. Mip-Splatting để dành cho **Phase 14** (hypothetical, nếu Phase 13 win mạnh và muốn stack frequency axis).

---

## Section 7 — Bonus: AbsGS dormant code

### 7.1 Hiện trạng

CRSGaussian đã có **AbsGS (Absolute Gradient Densification) infrastructure complete** nhưng **dormant**:

| File:line | Trạng thái |
|---|---|
| `scene/gaussian_model.py:155` | `self.absdensify = False` (hardcoded) |
| `scene/gaussian_model.py:154` | `# self.absdensify = args.absdensify` (commented) |
| `scene/gaussian_model.py:688, 746` | `if self.absdensify:` gates (active code path) |
| `train.py:1053` | `# parser.add_argument("--absdensify", ...)` (CLI commented) |
| `arguments/__init__.py:225` | `self.absdensify = False` (flag exists, default False) |

### 7.2 AbsGS là gì

**AbsGS** = densify trigger thêm 1 channel:
- Standard: `grad_norm = norm(viewspace_point_tensor.grad[:, :2])` — image-plane (x, y) gradient
- AbsGS: `grad_abs_norm = norm(viewspace_point_tensor.grad[:, 2:])` — channels (2, 3) là **absolute gradient values** (per-direction signed) thay vì norm

Selection union: `selected = (grad >= τ) OR (grad_abs >= Q)` với `Q = quantile(grads_abs, 1 - ratio_grad)` adaptive.

**Insight:** AbsGS bắt được Gaussian có gradient lớn theo 1 chiều specific (asymmetric) mà standard norm miss. Better densify criterion trên scenes có anisotropic texture.

### 7.3 Phase 8 FULL chạy với AbsGS OFF

Ceiling 21.16 multi-seed **KHÔNG có AbsGS active**. Có nghĩa: bật AbsGS = "free experiment" — chưa ai test trên Phase 8 backbone.

### 7.4 Cách bật

Uncomment 2 lines:
1. `scene/gaussian_model.py:154` — restore `self.absdensify = args.absdensify`
2. `train.py:1053` — restore `parser.add_argument("--absdensify", action="store_true")`

→ Phase 13 ablation matrix sẽ test cả LFCF alone, AbsGS alone, và LFCF+AbsGS combo (Section 8).

---

## Section 8 — Ablation matrix (5 configs)

### 8.1 Configs — Round 1 (5 configs × 8 scenes × seed 42 = 40 runs)

| Config | `use_lfcf` | `scaler_max` | `interval_times` | `diffscale` | `absdensify` | Mục đích |
|---|---|---|---|---|---|---|
| **A0** | False | — | — | — | False | Phase 8 FULL baseline (reference ≈ 21.16, verify reproduce) |
| **A1** ⭐ | True | 1.5 | 2 | ON | False | LFCF gentle (TaT defaults) — core LFCF test |
| **A2** | True | 1.5 | 2 | **OFF** | False | LFCF không diffscale — isolate diffscale contribution |
| **A3** | True | 1.5 | 2 | ON | **True** | LFCF + AbsGS combo — synergy test |
| **A4** | False | — | — | — | **True** | AbsGS standalone (no LFCF) — bonus axis test |

**Flags chung cho A1/A2/A3** (TaT defaults từ `EFA-GS/3DGS/scripts/run_tat.py`):
- `scaler_min = 1.0`, `pow = 1.0`, `splitting_ub = 1.0`, `splitting_lb = 1.0` (decays linear)
- `tolerance = 1e-5` (FP-stability default, fixed Round 1)
- `last_scaling_multiplier_max = 1.0` (decay target)

**Run command pattern (planning level):**

```bash
# A0 (baseline) — không bật flag mới
python train.py --source_path data/{scene} --model_path output/p13/A0_s42_{scene} \
    [Phase 8 FULL flags như production hiện tại] --seed 42

# A1 (core LFCF) — bật use_lfcf, giữ AbsGS off
python train.py ... \
    --use_lfcf --lfcf_init_scaling_max 1.5 --lfcf_interval_times 2 \
    --lfcf_diffscale True --seed 42

# A2 (no diffscale)
python train.py ... --use_lfcf --lfcf_init_scaling_max 1.5 \
    --lfcf_interval_times 2 --lfcf_diffscale False --seed 42

# A3 (LFCF + AbsGS)
python train.py ... --use_lfcf --lfcf_init_scaling_max 1.5 \
    --lfcf_interval_times 2 --lfcf_diffscale True --absdensify --seed 42

# A4 (AbsGS only)
python train.py ... --absdensify --seed 42
```

### 8.1.1 Expected output (format dự kiến, sau analyzer)

```
=== Phase 13 LFCF Round 1 analyzer ===
LOG_DIR=logs/p13_lfcf  SEEDS=['42']
SCENES=['fern','flower','fortress','horns','leaves','orchids','room','trex']

=== Per-scene PSNR (A0/A1/A2/A3/A4 — seed 42) ===
scene       A0_s42     A1_s42     A2_s42     A3_s42     A4_s42
----------------------------------------------------------------------
fern        ?          ?          ?          ?          ?
flower      ?          ?          ?          ?          ?
fortress    ?          ?          ?          ?          ?
horns       ?          ?          ?          ?          ?         ← Phase 12 hại most
leaves      ?          ?          ?          ?          ?
orchids     ?          ?          ?          ?          ?
room       ?          ?          ?          ?          ?
trex        ?          ?          ?          ?          ?

=== 8-scene avg + Δ vs A0 (paired N=8 per Δ) ===
config         8-avg PSNR    Δ vs A0    SEM    95% CI       Verdict
A0 (Phase8)    ?             —          —      —            baseline
A1 (LFCF)      ?             ?          ?      [?, ?]       ?
A2 (no diff)   ?             ?          ?      [?, ?]       ?
A3 (+AbsGS)    ?             ?          ?      [?, ?]       ?
A4 (AbsGS)     ?             ?          ?      [?, ?]       ?

=== Attribution (Section 8.2) ===
LFCF full effect           Δ_A1            = ?
Diffscale alone            Δ_A1 - Δ_A2     = ?
AbsGS bonus on LFCF        Δ_A3 - Δ_A1     = ?
AbsGS standalone           Δ_A4            = ?
LFCF × AbsGS synergy       Δ_A3 - Δ_A1 - Δ_A4 = ?

=== Verdict per Section 10.1 ===
Best Δ vs A0: ?
Decision: ?
```

### 8.1.2 Configs cho contingency steps (preview)

**Phase 13.1 Tolerance sweep (CHỈ kích hoạt nếu Round 2 winner)** — best config từ Round 1 với tolerance varied:

| Config | `tolerance` | Other flags | Mục đích |
|---|---|---|---|
| **T-tight** | 5e-6 | best Round 1 config | Stricter signal/noise (gần strict comparison) |
| **T-base** ⭐ | 1e-5 | best Round 1 config | Author default (verify reproduce) |
| **T-loose** | 5e-5 | best Round 1 config | Looser — nhiều enlarge hơn |
| **T-very-loose** | 1e-4 | best Round 1 config | Test boundary với atomicAdd variance scale |

Run: 4 × 8 scenes × seed 42 = 32 runs ~3.5h.

**λ sweep contingency (CHỈ nếu Round 1 marginal +0.05 ≤ Δ < +0.10)** — sweep scaler_max trên 2 scenes diagnostic:

| Config | `scaler_max` | Scenes | Mục đích |
|---|---|---|---|
| **S-1.3** | 1.3 | horns, fortress | Conservative enlarge |
| **S-1.5** ⭐ | 1.5 (TaT) | horns, fortress | Default baseline |
| **S-1.8** | 1.8 | horns, fortress | Stronger enlarge |
| **S-2.0** | 2.0 | horns, fortress | Aggressive enlarge |

Run: 4 × 2 scenes × seed 42 = 8 runs ~30 min.

→ Phase 12 hại horns (−0.241) và fortress (−0.246) nặng nhất → 2 scenes này nhạy nhất với densify changes → diagnostic value cao.

### 8.2 Attribution matrix (5 deltas)

| Δ compute | Tells us |
|---|---|
| Δ_A1 = A1 − A0 | LFCF full effect (core question) |
| Δ_A1 − Δ_A2 | Diffscale alone contribution (isolate isotropify) |
| Δ_A3 − Δ_A1 | AbsGS bonus on top of LFCF |
| Δ_A4 = A4 − A0 | AbsGS standalone value |
| Δ_A3 − (Δ_A1 + Δ_A4) | LFCF × AbsGS synergy (positive = combine helps, negative = redundant) |

### 8.3 Run budget

| Round | Configs | Scenes | Seeds | Runs | Time (2 GPU) |
|---|---|---|---|---|---|
| Round 1 | 5 | 8 | 1 (seed 42) | 40 | ~4.5h |
| Round 2 (if winner) | 1 (best) + A0 | 8 | 2 (seed 137, 9999) | 16 | ~2h |

**Total max:** ~6.5h server time.

### 8.4 Hyperparameter source

TaT defaults từ `EFA-GS/3DGS/scripts/run_tat.py`:
- `init_scaling_multiplier_max = 1.5`
- `init_scaling_multiplier_min = 1.0`
- `interval_times = 2` (LFCF mỗi 200 iter)
- `diffscale = True`
- `pow = 1.0`, `splitting_ub = 1.0`, `splitting_lb = 1.0` (decays linear)
- `tolerance = 1e-5`

**KHÔNG self-tune** trong main plan. Nếu marginal → λ sweep contingency (Section 10.3). Tolerance fixed 1e-5 trong Round 1 — sweep CHỈ KHI Phase 13 winner (Section 10.4 Phase 13.1).

### 8.5 End-to-end test flow — Phase 13 hướng test thực tế

Đây là **timeline thực tế** từ lúc bắt đầu đến quyết định cuối, tích hợp tất cả pieces (Section 8 configs + Section 9 implementation + Section 10 decision tree + Section 11 probability + Section 12 cleanup).

```
═══════════════════════════════════════════════════════════════════════
 STEP 1 — IMPLEMENTATION (Day 1-2, code session local Windows)
═══════════════════════════════════════════════════════════════════════
 ├── Code port theo Section 9.1:
 │     • NEW utils/densify/lfcf.py
 │     • MODIFY scene/gaussian_model.py (densify_and_prune extend)
 │     • MODIFY arguments/__init__.py (10 LFCF flags + uncomment absdensify)
 │     • MODIFY train.py (LFCF hook block)
 │     • NEW scripts/p13_lfcf_multiseed.{sh,py}
 ├── Resolve 4 conflicts (Section 9.2):
 │     prune_points coverage / CRS inherit / opacity_decay / SH freeze
 └── Smoke test no-GPU: import + mock densify → PASS
              │
              ▼
═══════════════════════════════════════════════════════════════════════
 STEP 2 — ROUND 1 ABLATION (Day 3, ~4.5h server Linux)
═══════════════════════════════════════════════════════════════════════
 ├── 5 configs × 8 scenes × 1 seed (42) = 40 runs
 │     A0 = Phase 8 FULL baseline (verify reproduce 21.16)
 │     A1 = LFCF gentle (TaT defaults)        ⭐ core test
 │     A2 = LFCF no diffscale                 (isolate diffscale)
 │     A3 = LFCF + AbsGS                      (synergy test)
 │     A4 = AbsGS only                        (standalone)
 ├── 2 GPU parallel split (memory feedback_use_both_gpus)
 ├── Output: logs/p13_lfcf/seed42/*.log, output/p13_lfcf/*/
 └── Run analyzer p13_lfcf_multiseed_analyze.py:
       • Per-config 8-scene avg PSNR
       • 5 attribution deltas (Section 8.2)
       • Verdict per Section 10.1
              │
              ▼   BRANCH theo best Δ_A1 vs A0
              ▼
┌─────────────────────┬──────────────────────┬──────────────────────────┐
│ 🎯 Δ ≥ +0.10        │ 🟡 +0.05 ≤ Δ < +0.10 │ ❌ Δ < +0.05             │
│ WINNER path         │ MARGINAL path        │ REJECT path              │
└─────────────────────┴──────────────────────┴──────────────────────────┘
         │                    │                          │
         ▼                    ▼                          ▼
═══════════════════════════════════════════════════════════════════════
 STEP 3a — ROUND 2 MULTI-SEED VERIFY (Day 4, ~2h)
═══════════════════════════════════════════════════════════════════════
 ├── Best config + A0, seeds {137, 9999} × 8 scenes = 16 runs
 ├── Pooled N = 24 paired (seed 42 từ Round 1 + 137, 9999)
 └── Multi-seed Δ verdict:
       • Δ ≥ +0.10 multi-seed → COMMIT-WORTHY → STEP 4
       • Δ < +0.10 (Round 1 lucky sample) → demote MARGINAL → STEP 3b

═══════════════════════════════════════════════════════════════════════
 STEP 3b — λ SWEEP CONTINGENCY (Day 4, ~2h)
═══════════════════════════════════════════════════════════════════════
 ├── scaler_max ∈ {1.3, 1.5, 1.8, 2.0} × 2 scenes (horns + fortress)
 │     (Phase 12 hại 2 scenes này nhất → diagnostic)
 ├── 4 sweeps × 2 scenes × seed 42 = 8 runs ~30min
 └── Verdict:
       • Best sweep Δ ≥ +0.10 → full Round 2 verify → STEP 4
       • Sweep flat hoặc all < +0.10 → REJECT → STEP 3c

═══════════════════════════════════════════════════════════════════════
 STEP 3c — CLEANUP (Day 4, 1h)
═══════════════════════════════════════════════════════════════════════
 ├── Per Rule 13 (Section 12):
 │     rm utils/densify/lfcf.py
 │     rm scripts/p13_lfcf_*.{sh,py}
 │     rm -rf output/p13_lfcf/
 │     Revert arguments/__init__.py + train.py + gaussian_model.py
 ├── KEEP logs/p13_lfcf/ (paper appendix negative result)
 ├── Document Phase 13 REJECT trong docs/04_decisions_log.md
 └── 🛑 PIVOT paper writeup: "comprehensive ablation methodology"
       (Section 13 mental prep — 10 mechanism classes evaluated)

              ▼ (chỉ nếu STEP 3a hoặc 3b SUCCESS)
═══════════════════════════════════════════════════════════════════════
 STEP 4 — PHASE 13.1 TOLERANCE SWEEP (Day 5, ~3.5h)
═══════════════════════════════════════════════════════════════════════
 Gate: Round 2 Δ ≥ +0.10 (commit-worthy CONFIRMED multi-seed)
 ├── 4 tolerance × seed 42 × 8 scenes = 32 runs
 │     T-tight  = 5e-6  | T-base = 1e-5 (default)
 │     T-loose  = 5e-5  | T-very-loose = 1e-4
 ├── Lý do: validate 1e-5 trên LLFF 3-view regime (authors test dense view)
 │   + check tolerance có mitigate CRSGaussian atomicAdd ±1.3 dB variance
 └── Verdict Section 10.4:
       • Best tolerance Δ ≥ +0.05 vs T-base → multi-seed verify (16 runs)
       • Flat → keep T-base 1e-5, log curve cho paper appendix
              │
              ▼
═══════════════════════════════════════════════════════════════════════
 STEP 5 — FINAL COMMIT (Day 6-7)
═══════════════════════════════════════════════════════════════════════
 ├── Update docs:
 │     • docs/04_decisions_log.md — Phase 13 WIN entry
 │     • docs/03_task_queue.md — Phase 13 DONE, paper writeup START
 │     • CLAUDE.md — new ceiling, Phase 8 FULL → Phase 13 recipe
 │     • This doc — append "RESULTS" section với multi-seed numbers
 ├── Lock final config:
 │     • scaler_max, interval_times, diffscale, absdensify confirmed
 │     • tolerance final value (default hoặc best từ sweep)
 │     • A* config-final-name (e.g., A1-final, A3-final)
 ├── Generate paper figures + ablation table
 └── 🎉 Paper writeup START — frequency-axis contribution defendable
```

**Timeline tóm tắt theo branch:**

| Day | Activity | Duration | Branch áp dụng |
|---|---|---|---|
| 1-2 | Step 1 (implementation + smoke test) | ~2 ngày | Mọi case |
| 3 | Step 2 Round 1 (40 runs) | ~4.5h | Mọi case |
| 4 | Step 3a/b/c branch theo verdict | 1-2h | Tùy verdict |
| 5 | Step 4 Phase 13.1 tolerance sweep | ~3.5h | CHỈ winner path |
| 6-7 | Step 5 Final commit + writeup start | 2 ngày | CHỈ winner path |

**Total max** (winner path): **7 ngày** từ start tới paper writeup ready.
**Total min** (reject path): **4 ngày** (implement + Round 1 + cleanup + pivot writeup).

**Probability allocation cuối cùng:**
- ~25-30% chance **winner path** → 7 ngày → paper writeup ngay với contribution mạnh
- ~25-30% chance **marginal-survive-λ-sweep** → 6 ngày
- ~40-50% chance **REJECT path** → 4 ngày → pivot methodology writeup (vẫn dùng được Round 1 data)

→ **Worst case 4 ngày + writeup** vẫn productive vì cleanup nhanh + Round 1 data feed vào paper appendix "exhaustive ablation".

---

## Section 9 — Implementation plan

### 9.1 Files cần modify/create (cao-level)

| # | File | Action | Mô tả |
|---|---|---|---|
| 1 | `utils/densify/lfcf.py` | **NEW** | LFCF logic + helpers (compute_3D_interval, normalize_interval, calculate_training_percent_powered, compute_lfcf_changes) |
| 2 | `scene/gaussian_model.py` | **MODIFY** | Extend `densify_and_prune` với LFCF mode gate; mở rộng `prune_points` + `densification_postfix` cover CRS attrs; track `prev_lff_xyz_grad` + `prev_selected_pts_mask` |
| 3 | `arguments/__init__.py` OptimizationParams | **MODIFY** | Add ~10 LFCF flags (use_lfcf, lfcf_init_scaling_max/min, lfcf_last_scaling_max, lfcf_pow, lfcf_splitting_ub/lb, lfcf_interval_times, lfcf_tolerance, lfcf_diffscale); uncomment `absdensify` line 154 + train.py:1053 |
| 4 | `train.py` | **MODIFY** | LFCF hook trong densify block — build `lfcf_opts` dict, pass kwargs xuống `densify_and_prune` (gate qua `opt.use_lfcf`) |
| 5 | `scripts/p13_lfcf_multiseed.sh` | **NEW** | 5 configs × 8 scenes × 1 seed (Round 1), parallel 2 GPU split |
| 6 | `scripts/p13_lfcf_multiseed_analyze.py` | **NEW** | 5-config attribution per Section 8.2 |

### 9.2 4 conflicts với Phase 8 FULL cần address

**Conflict 1 — `prune_points` merge với EFA-GS `prune_points_lff`**

**Bối cảnh:** Phase 13 KHÔNG gọi `self.prune_points(mask)` cũ nữa — phải dùng method MỚI `prune_points_lff(mask)` từ EFA-GS (cần track thêm `prev_selected_pts_mask`, `lff_xyz_grad_accum`, `prev_lff_xyz_grad`, `lff_denom` không tồn tại trong prune_points cũ).

**Vấn đề:** EFA-GS `prune_points_lff` chỉ biết về 9 base attrs + 4 EFA-GS-new attrs. KHÔNG biết về CRSGaussian-specific attrs đã có sẵn trong Phase 8 FULL.

**CRSGaussian attrs hiện đang prune** (grep `prune_points` trong `scene/gaussian_model.py` line 537-569):

| Loại | Attrs (verify via grep, KHÔNG trust list này) |
|---|---|
| Base 6 tensors | `_xyz`, `_features_dc`, `_features_rest`, `_opacity`, `_scaling`, `_rotation` |
| Gradient tracking (4) | `xyz_gradient_accum`, `xyz_gradient_accum_abs`, `xyz_gradient_accum_abs_max`, `denom` |
| Screen (1) | `max_radii2D` |
| CRS core (2) | `confidence`, `_crs_score` |
| Lifecycle (1) | `spawn_iter` |
| Phase 8 mechanisms (2) | `_rc_smooth`, `_crs_rnrc` |

**Total ≥ 16 attrs** (có thể nhiều hơn — Phase 8 FULL có thể add thêm sau commit gần nhất).

**Plus EFA-GS-new attrs cần add:**
- `prev_selected_pts_mask` (track Gaussians được pick lần trước)
- `lff_xyz_grad_accum`, `lff_denom` (LFCF-specific grad accumulator, độc lập với standard accum)
- `prev_lff_xyz_grad` (grad từ LFCF kỳ trước cho tolerance compare)

**⚠️ Quy tắc cho execution session:**
```bash
# Trước khi viết prune_points_lff, PHẢI grep toàn bộ attrs:
grep -n "self\.[a-z_]* = self\.[a-z_]*\[valid_points_mask\]" \
    CRSGaussian/scene/gaussian_model.py
grep -n "self\._[a-z_]* = self\._[a-z_]*\[valid_points_mask\]" \
    CRSGaussian/scene/gaussian_model.py
```
→ Lấy ground-truth danh sách attrs, KHÔNG trust liệt kê trong doc này (có thể stale).

**Approach merge khuyến nghị:**
1. Copy `prune_points_lff` từ EFA-GS gốc làm template
2. Copy TẤT CẢ CRSGaussian-specific attrs từ `prune_points` cũ vào method mới
3. Smoke test: tạo dummy GaussianModel với mock data, call `prune_points_lff(random_mask)` → KHÔNG crash + tất cả attr.shape[0] == new_N

**Conflict 2 — `densification_postfix` cần CRS inherit (eta T5.5)**

CRSGaussian split/clone truyền `parent_crs_logits` + `eta` cho conservative CRS inherit. EFA-GS LFCF split path KHÔNG có CRS handling. → CẦN extend `densification_postfix_lff` accept `parent_crs_logits` + `eta` kwarg, apply identical CRS inherit logic.

**Conflict 3 — `opacity_reset` vs `opacity_decay`**

EFA-GS dùng standard `opacity_reset_interval = 3000` (hard reset to 0.01). CRSGaussian dùng `opacity_decay_factor = 0.995` mỗi iter (soft decay). → KHÔNG xung đột (opacity update là post-densify step độc lập). LFCF chỉ replace clone/split, không động opacity path. **Verify:** với `--use_opacity_decay True`, opacity_reset gate trong train.py phải False (Phase 8 FULL config).

**Conflict 4 — CRS-mod SH freeze interaction với LFCF enlarge**

Phase 8 FULL có CRS-modulated SH freeze (low-CRS Gaussians zero SH grad). Khi LFCF ENLARGE 1 Gaussian (mới-selected) — SH coefficients được giữ nguyên (chỉ scaling change). → KHÔNG conflict (SH freeze gate dùng CRS, không liên quan scale). **Verify:** sau LFCF iter, CRS scores của enlarged Gaussians không bị reset → SH freeze tiếp tục hoạt động trên Gaussians cũ.

### 9.3 Smoke test plan (no GPU)

1. Import test: `python -c "import utils.densify.lfcf; from scene.gaussian_model import GaussianModel"` → PASS
2. Mock densify call: tạo dummy GaussianModel với 1000 random Gaussians, call `densify_and_prune(use_lfcf=True, ...)` → no crash
3. Verify default OFF: với `opt.use_lfcf = False`, behavior identical Phase 8 FULL (grep code path)

### 9.4 Time budget

| Step | Time |
|---|---|
| Code port (utils/densify/lfcf.py + gaussian_model.py + arguments + train.py) | ~1.5 ngày |
| Smoke test + verify default OFF | ~2h |
| Multi-seed script + analyzer | ~3h |
| Round 1 server run (40 runs / 2 GPU) | ~4.5h |
| Analyze + decide | ~1h |
| Round 2 (if winner) | ~2h |

**Total max:** 2.5 ngày + 6.5h server.

---

## Section 10 — Decision tree

### 10.1 Round 1 verdict

Best Δ vs A0 (seed 42, N=8 paired):

```
≥ +0.20      → 🎯 BREAKTHROUGH — Round 2 multi-seed verify (seeds 137 + 9999)
+0.10..+0.20 → 🎯 WINNER — Round 2 verify
+0.05..+0.10 → 🟡 MARGINAL — λ sweep contingency:
               scaler_max ∈ {1.3, 1.5, 1.8, 2.0} × 2 scenes (horns + fortress)
               → If sweep finds Δ ≥ +0.10 → full Round 2
               → If sweep flat → REJECT
< +0.05      → ❌ REJECT — accept ceiling 21.16, paper writeup pivot
```

### 10.2 Honest decision threshold

Per existing memory `feedback_full_8scene_ablation.md`: single-seed 8-scene avg Δ trong band ±0.05 → noise. CẦN multi-seed cho commit.

Per existing memory `project_3dgs_variance_floor.md`: min detectable Δ ≈ ±0.10 dB với 3 seeds × 8 scenes paired.

→ Threshold +0.10 multi-seed = commit-worthy. +0.05 single-seed = MARGINAL band cần λ sweep verify.

### 10.3 Round 2 design

Best config từ Round 1 + A0, seeds 137 + 9999 × 8 scenes = 32 runs. Pooled N = 24 paired (Round 1 seed 42 + Round 2 seeds 137 + 9999).

### 10.4 Phase 13.1 — Tolerance sweep (CONTINGENT post-win)

**Trigger condition: Phase 13 Round 2 verify Δ ≥ +0.10 dB** (commit-worthy). KHÔNG kích hoạt nếu Phase 13 reject/marginal.

**Lý do:** Authors validate tolerance=1e-5 trên A100/V100 + Mip-NeRF360 + TaT. KHÔNG có evidence cho LLFF 3-view + sparse setting. CRSGaussian có atomicAdd variance ±1.3 dB single-scene (khác hardware regime với authors' test). Tolerance value có thể có sweet spot khác.

**Sweep design:**

| Config | tolerance | Mục đích |
|---|---|---|
| T-base | 1e-5 | Author default (verify) |
| T-tight | 5e-6 | Stricter signal/noise — đẩy về behavior strict comparison |
| T-loose | 5e-5 | Looser — nhiều enlarge hơn, ít split |
| T-very-loose | 1e-4 | Test boundary — atomicAdd variance scale |

Configs: 4 × seed 42 × 8 scenes = 32 runs ~3.5h (2 GPU parallel).

**Decision tree Phase 13.1:**

| Best tolerance Δ vs T-base | Action |
|---|---|
| ≥ +0.05 | 🎯 Adopt new tolerance, multi-seed verify (16 runs) |
| ±0.05 | 🟡 Keep T-base (TaT default robust) |
| Sweet spot rõ | Use sweep curve in paper appendix |

**Cleanup nếu sweep flat:** KHÔNG xóa data — log Round 1 ablation evidence cho paper appendix (defendable "we explored tolerance sensitivity").

**Tại sao gate sau Phase 13 win, không trước:**
- Round 1 baseline (tolerance=1e-5) phải win trước → confirm LFCF cơ chế work cho CRSGaussian
- Nếu Round 1 marginal (+0.05~+0.10) → uncertainty ở **scaler_max** thay vì tolerance (per Section 10.1 contingency)
- Tolerance sweep cost ~3.5h server time — chỉ đáng đầu tư khi base LFCF đã commit-worthy

---

## Section 11 — Probability honest estimate

### 11.1 Breakdown

```
P(Δ ≥ +0.20)         ≈ 20-25%   (breakthrough, paper headline)
P(Δ ≥ +0.10)         ≈ 25-30%   (commit-worthy, multi-seed verify)
P(Δ +0.05 ~ +0.10)   ≈ 25-30%   (marginal, λ sweep needed)
P(Δ < +0.05)         ≈ 25-35%   (reject)
```

**Net: 45-50% chance commit-worthy** (Δ ≥ +0.10 hoặc marginal-survive-λ-sweep).

### 11.2 Evidence supporting positive

- TaT regime gain **+0.17 ~ +0.22** (regime similar PSNR + forward-facing)
- Diffscale theoretically fixes Phase 12 thin-structure failure (mechanism mapped)
- Densify-axis CHƯA THỬ trong CRSGaussian — orthogonal với Phase 11 (loss) + Phase 12 (position)
- AbsGS infrastructure pre-built, dormant → free bonus axis
- **Tolerance mechanism mitigate FP non-determinism** (per README) — cùng root cause với CRSGaussian's atomicAdd variance ±1.3 dB single-scene. Bonus side-effect: TaT V100 +0.05 dB chỉ bằng tolerance fix alone (strict 21.67 → tolerance 21.72)

### 11.3 Risk factors

- **TaT ≠ LLFF 3-view**: TaT có 100+ training views, LLFF 3 — không tự động generalize
- **4 conflicts cần resolve đúng**: prune_points coverage, CRS inherit, opacity path, SH freeze
- **TaT hyperparams có thể không transfer**: scaler_max=1.5 tuned cho TaT scenes, scaler có thể cần adjust cho LLFF
- **Phase 8 FULL đã saturate**: 4 cơ chế chống floater sẵn (prune + SH freeze + opacity decay + post-warmup CRS) có thể left ít room cho LFCF
- **Statistical power**: single-seed Round 1 cao biến nếu Δ borderline — λ sweep contingency mandatory

### 11.4 KHÔNG over-claim

- Tài liệu này **không promise** breakthrough.
- 45-50% là **honest probability** dựa trên 9/9 fail trước + TaT evidence.
- Reject scenario (25-35%) là realistic — phải pre-commit accept ceiling pivot.

---

## Section 12 — Cleanup plan if reject

Per Rule 13:

### 12.1 Files xóa

```
utils/densify/lfcf.py                                (NEW file)
scripts/p13_lfcf_multiseed.sh                        (NEW script)
scripts/p13_lfcf_multiseed_analyze.py                (NEW script)
output/p13_lfcf/                                     (training outputs, GB)
```

### 12.2 Code revert — phân biệt 3 scenarios

**Scenario 1: Toàn bộ Phase 13 reject** (Δ_A1, Δ_A3, Δ_A4 all < +0.05) — full revert:

- `arguments/__init__.py` OptimizationParams — remove 10 LFCF flags + RE-comment `absdensify` line 225 (back to dormant)
- `scene/gaussian_model.py:154` — RE-comment `self.absdensify = args.absdensify`
- `scene/gaussian_model.py:155` — RE-assert `self.absdensify = False`
- `scene/gaussian_model.py` densify_and_prune — remove `use_lfcf` + `lfcf_opts` kwargs branches
- `train.py` — remove LFCF hook block
- `train.py:1053` — RE-comment `parser.add_argument("--absdensify", ...)`

**Scenario 2: LFCF reject NHƯNG A4 (AbsGS-only) win** (Δ_A4 ≥ +0.10, Δ_A1/A3 < +0.05) — partial revert:

- ❌ Remove LFCF code (giữ Scenario 1 list cho LFCF-specific):
  - Remove 10 LFCF flags (use_lfcf, lfcf_*) trong OptimizationParams
  - Remove LFCF hook block trong train.py
  - Remove `use_lfcf` branches trong densify_and_prune
  - Delete `utils/densify/lfcf.py`, scripts/p13_lfcf_*
- ✅ **KEEP AbsGS activation** (đây là persistent improvement):
  - GIỮ uncomment line 154 (`self.absdensify = args.absdensify`)
  - GIỮ uncomment line 1053 train.py (`parser.add_argument("--absdensify", ...)`)
  - Update Phase 8 FULL master config default → `--absdensify` ON (new ceiling)
  - Document trong decisions_log: "Phase 13 LFCF reject NHƯNG A4 AbsGS standalone +Δ ≥ +0.10 → AbsGS adopted as Phase 8.5 FULL recipe"

**Scenario 3: LFCF win (A1 hoặc A3)** — KHÔNG revert gì:

- Toàn bộ Phase 13 code → production
- Update Phase 8 FULL → Phase 13 FULL recipe (xem Section 8.5 Step 5)
- AbsGS uncomment giữ nguyên (đã active trong winning config)

### 12.3 Files GIỮ

```
logs/p13_lfcf/                                       (negative result reference cho paper)
docs/13_efa_gs_lfcf_design.md                        (this doc — paper "exhaustive ablation" section)
docs/04_decisions_log.md                             (Phase 13 verdict entry)
docs/03_task_queue.md                                (Phase 13 status)
```

### 12.4 Verify cleanup

```bash
grep -r "lfcf\|LFCF\|EFA-GS\|EFA_GS" CRSGaussian/ --exclude-dir=docs --exclude-dir=logs
```
→ Phải empty sau cleanup (chỉ còn references trong docs + logs).

---

## Section 13 — Mental preparation (CRITICAL)

### 13.1 Nếu Phase 13 REJECT

Đây là **last CRS-axis attempt trên Phase 8 FULL backbone**. CRS axis 9/9 + 1 = **10/10 failed** sẽ thành lịch sử project.

Paper narrative **PHẢI pivot**:

| Strategy CŨ (nếu Phase 13 win) | Strategy MỚI (nếu Phase 13 reject) |
|---|---|
| Ship CRSGaussian như **flagship method** (SOTA-targeting) | Ship như **methodology paper** (exhaustive ablation) |
| Headline: "CRSGaussian +X dB over baseline" | Headline: "Comprehensive evaluation of 10 mechanism classes for sparse-view 3DGS" |
| Contribution: Phase 8 FULL recipe + breakthrough | Contribution: Phase 8 FULL recipe + 10 negative results as defendable ablation |
| Gap vs ICO-GS (22.20): hide as "future work" | Gap vs ICO-GS: explicit acknowledgment, motivate ICO-GS follow-up |

### 13.2 10 mechanism classes (nếu Phase 13 reject)

| # | Class | Status |
|---|---|---|
| 1 | Loss reweighting (covisibility, Phase 11 Step 1) | 🟡 marginal |
| 2 | Perceptual loss (Phase 11 Step 2) | ❌ reject |
| 3 | Stack loss (Phase 11 Stack) | ❌ reject |
| 4 | Cross-view consistency (Phase 11 Step 4) | ❌ reject |
| 5 | Depth regularizer (Phase 11 Step 5) | ❌ reject |
| 6 | Position attraction (Phase 12 CRS-pull A1) | ❌ reject |
| 7 | Replace prune (Phase 12 CRS-pull A2) | ❌ reject |
| 8 | Position-only (Phase 12 CRS-pull A3) | ❌ reject |
| 9 | Foundation-model init (Phase 10A DUSt3R) | ❌ reject hard |
| 10 | LFCF densify (Phase 13, contingent) | ⏳ pending |

→ "Comprehensive negative ablation" là **defendable contribution** trong sparse-view 3DGS literature.

### 13.3 Commitment

**Sau Round 1 verdict, COMMIT decision tree thật sự — không tweak thêm.**

Nếu Δ_A1 = +0.08 (marginal) → λ sweep contingency → verdict cuối.

Nếu λ sweep flat → REJECT, KHÔNG cycle thêm hyperparam fishing. Mental energy spent better trên paper writeup.

### 13.4 Time floor sau Phase 13

Whatever verdict — paper writeup phải bắt đầu **trong vòng 1 tuần sau Phase 13 verdict**. Không có Phase 14 prophylactic (Mip-Splatting, Scaffold-GS, iter budget) trừ khi Phase 13 win mạnh (Δ ≥ +0.15) và stack potential rõ ràng.

---

## Appendix A — Key references

- **EFA-GS paper**: Wang et al., "Low-Frequency First: Eliminating Floating Artifacts in 3D Gaussian Splatting" (jcwang-gh/EFA-GS)
- **Mip-Splatting paper**: CVPR 2024 Best Student Paper — Yu et al.
- **AbsGS**: Absolute Gradient Densification — already infrastructure-ported into CRSGaussian (dormant)

## Appendix B — Hyperparams hoàn chỉnh (TaT defaults)

```python
# From EFA-GS/3DGS/scripts/run_tat.py + train.py defaults
init_scaling_multiplier_max = 1.5     # gentle enlarge ceiling
init_scaling_multiplier_min = 1.0     # bottom of depth-adaptive range
last_scaling_multiplier_max = 1.0     # decay target tại densify_until
pow                         = 1.0     # decay rate (linear)
splitting_ub                = 1.0     # split prob upper bound
splitting_lb                = 1.0     # initial; decays linear → 0 trong train()
interval_times              = 2       # LFCF mỗi 2 × densify_interval = 200 iter
tolerance                   = 1e-5    # FP-stability fix (NOT signal/noise hyperparam)
                                       # → Sweep contingent Phase 13.1 nếu win (Section 10.4)
diffscale                   = True    # volume-preserving isotropify

# CRSGaussian context-specific (giữ Phase 8 FULL):
densification_interval      = 100     # CRSGaussian sparse-view default
densify_from_iter           = 500     # Phase 8 FULL
densify_until_iter          = 5000    # Phase 8 FULL (NB: EFA-GS TaT dùng 15000)
T_warmup                    = 1000    # CRS warmup
tau_crs                     = 0.35    # CRS prune threshold
opacity_decay_factor        = 0.995   # CRSGaussian, không reset
```

**Nota bene về `densify_until_iter`**: CRSGaussian dùng 5000 (sparse-view tuned). EFA-GS TaT dùng 15000 (dense-view). Schedule mismatch có thể impact `training_percent_powered` decay curve — KHÔNG override, để TaT defaults adapt automatic qua `(densify_until - densify_from) / pow` formula.

---

---

## Section 14 — Round 1 RESULTS (2026-05-12, seed 42)

### 14.1 Per-scene PSNR

| scene | A0 | A1 | A2 | A3 | A4 |
|---|---|---|---|---|---|
| fern | 23.248 | 23.185 | 23.088 | 23.358 | 23.400 |
| flower | 20.923 | 20.947 | 20.642 | 21.021 | 20.917 |
| fortress | 23.979 | 24.126 | 24.041 | 24.056 | 23.687 |
| horns | 19.960 | 20.228 | 20.285 | **20.615** | 19.728 |
| leaves | 18.424 | 18.254 | 18.479 | 18.605 | 18.613 |
| orchids | 16.784 | 16.618 | 16.649 | 16.982 | 16.997 |
| room | 22.626 | 22.951 | 22.692 | 22.617 | 22.643 |
| trex | 23.429 | 23.185 | 23.341 | 23.392 | 23.640 |
| **8-avg** | **21.172** | **21.187** | **21.152** | **21.331** | **21.203** |

### 14.2 Paired Δ summary (N=8)

| Comparison | Δ mean | SEM | 95% CI | Verdict |
|---|---|---|---|---|
| Δ_A1 (LFCF core) | +0.0151 | 0.0754 | [−0.133, +0.163] | ❌ REJECT (<+0.05) |
| Δ_A2 (LFCF no diffscale) | −0.0196 | 0.0660 | [−0.149, +0.110] | ❌ REJECT |
| **Δ_A3 (LFCF + AbsGS)** | **+0.1590** | **0.0765** | **[+0.009, +0.309]** | **🎯 SIG WINNER** |
| Δ_A4 (AbsGS alone) | +0.0314 | 0.0707 | [−0.107, +0.170] | ❌ REJECT |

**A3 95% CI KHÔNG cross 0** — đầu tiên trong 10/10 attempts (Phase 11 6/6 + Phase 12 3/3 + Phase 13 A0/A1/A2/A4 4/4) có signal vượt noise floor.

### 14.3 Attribution metrics

| Metric | Formula | Value | Interpretation |
|---|---|---|---|
| LFCF full effect | Δ_A1 | +0.015 | LFCF alone neutral |
| Diffscale alone | Δ_A1 − Δ_A2 | +0.035 | Neutral (diffscale ≈ no-diffscale) |
| **AbsGS bonus on LFCF** | Δ_A3 − Δ_A1 | **+0.144** | **BIG positive** |
| AbsGS standalone | Δ_A4 | +0.031 | Neutral alone |
| **LFCF × AbsGS synergy** | Δ_A3 − (Δ_A1 + Δ_A4) | **+0.112** | **POSITIVE synergy** |

**Synergy 3.5× linear sum** (0.046 → 0.159) → mechanism reinforce nhau, không phải additive noise.

### 14.4 Per-scene complementarity pattern

| Scene type | Best config | Δ_A3 | Δ_A4 |
|---|---|---|---|
| Thin/complex (horns) | A3 ⭐ | +0.655 | −0.232 |
| Thin/complex (fortress) | A3 (A4 hại) | +0.078 | −0.292 |
| Thin (flower, leaves, orchids) | A3 | +0.098..+0.198 | mixed |
| Mid (fern) | A3 ≈ A4 | +0.110 | +0.151 |
| Simple planar (room) | neither | −0.009 | +0.018 |
| Simple planar (trex) | A4 | −0.038 | +0.211 |

**Hypothesis scene-complexity**:
- Thin/complex → LFCF tolerance + diffscale bảo vệ thin geometry → A3 wins
- Simple/planar → LFCF mechanism = overhead → A4 alone đủ
- → Round 2 multi-seed verify pattern này.

### 14.5 Risk factors cho Round 2

1. **Single-seed variance**: 95% CI lower bound [+0.009] gần 0 — Round 2 có thể demote A3 nếu seed 137/9999 mean drift xuống
2. **horns outlier**: Δ_A3 = +0.655 contribute 41% của 8-avg gain. Nếu Round 2 horns Δ_A3 drop về +0.10..+0.20 → confirm signal real. Nếu < +0.10 → seed-42 artifact
3. **A4 standalone neutral**: A3 = A1 + A4 + synergy formulation cần synergy stable across seeds, không phải artifact

### 14.6 Decision: Round 2 verify (per Section 10.3)

**Verdict per Section 10.1**: Δ_A3 = +0.159 ∈ WINNER band (+0.10..+0.20) → **Round 2 multi-seed verify**

**Round 2 deviation từ design doc** (approved by user):
- Original plan: best config (A3) + A0 × seeds {137, 9999} × 8 scenes = 16 runs ~2h
- **Modified**: A0 + **A3 + A4** × seeds {137, 9999} × 8 scenes = 48 runs ~3h
- Lý do thêm A4: complementarity data quan trọng cho paper (Section 14.4 pattern verify)
- +1h cost = đáng để có ablation data đầy đủ

**Round 2 commands** (đã chạy):

```bash
GPU=0 SEEDS_OVERRIDE="137" CONFIGS_OVERRIDE="A0 A3 A4" \
    SCENES_OVERRIDE="fern flower fortress horns" \
    bash scripts/p13_lfcf_multiseed.sh > logs/p13_lfcf/gpu0_seed137.log 2>&1 &
GPU=1 SEEDS_OVERRIDE="9999" CONFIGS_OVERRIDE="A0 A3 A4" \
    SCENES_OVERRIDE="leaves orchids room trex" \
    bash scripts/p13_lfcf_multiseed.sh > logs/p13_lfcf/gpu1_seed9999.log 2>&1 &
wait
python scripts/p13_lfcf_multiseed_analyze.py
```

### 14.7 Phase 13.1 gate (chờ Round 2 verdict)

- **Pooled N=24 Δ_A3 ≥ +0.10** → COMMIT-WORTHY → **Phase 13.1 tolerance sweep** (Section 10.4)
- **Pooled N=24 Δ_A3 < +0.10** → demote MARGINAL → λ scaler_max sweep contingency (Section 10.1)
- **Pooled N=24 Δ_A3 < +0.05** → REJECT (unlikely given current Δ=+0.159 SIG)

---

**END OF DESIGN DOC.**

**Current status (2026-05-12 post Round 1):** A3 (LFCF + AbsGS) SIG WINNER seed 42. Round 2 verify in progress trên server. Pending pooled N=24 verdict cho Phase 13.1 trigger decision.
