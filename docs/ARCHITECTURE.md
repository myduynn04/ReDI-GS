# CRSGaussian — Kiến trúc hệ thống hiện tại

> **Trạng thái**: 2026-05-17 — Phase 13 A3 COMMITTED (LFCF + AbsGS), Phase 13.2 bottleneck diagnostic.
> **Mục đích**: Tài liệu kiến trúc chi tiết toàn hệ thống — codebase, training pipeline, core mechanisms, config surface.
> **Đối tượng**: Session mới cần hiểu hệ thống nhanh; reference khi modify/debug.

---

## Section 1 — Tổng quan

### 1.1 Bài toán

Sparse-view 3D Gaussian Splatting trên LLFF 3-view benchmark. 2 vấn đề cốt lõi:
- **Floater**: Gaussian đặt sai vị trí 3D (densification không có position constraint)
- **Overdensification**: Quá nhiều Gaussian → overfit 3 train views (test gap ~12.9 dB)

### 1.2 Base + Lineage

```
3DGS (Kerbl SIGGRAPH 2023)
  └── CoR-GS (2-field co-regularization) ← BASE của CRSGaussian
        └── CRSGaussian (project này)
              ├── Phase 1-9: CRS module + depth prior + Phase 8 FULL recipe
              ├── Phase 13: EFA-GS LFCF + AbsGS port (frequency-axis)
              └── Phase 13.2: bottleneck diagnostic (current)
```

Folder `CoR-GS/` local = `CRSGaussian/`; trên server giữ tên `CoR-GS/` (git diff clean với upstream).

### 1.3 Contributions chính

1. **CRS (Confidence-Reliability Score)** — per-Gaussian dynamic signal = f(D_cycle, R_visible, S_stability), update mỗi 100 iter sau T_warmup
2. **CRS-guided pruning** — Option C: prune (CRS < τ AND isolated) OR (opacity < min)
3. **CRS-modulated SH freeze** — per-Gaussian zero SH grad khi CRS thấp (Phase 8c, biggest winner +0.189)
4. **LFCF + AbsGS densification** — Phase 13 frequency-axis (committed, Δ=+0.164 N=24)
5. **Single-model** thay 2-field CoR-GS (2× rẻ hơn)

### 1.4 Reference targets (LLFF 3-view PSNR)

| Method | PSNR | Note |
|--------|------|------|
| Phase 8 FULL (paper 1-sample) | 21.335 | Old baseline (lucky single-run) |
| Phase 8 FULL multi-seed | 21.16 | Fair N=24 (pre-Phase-13) |
| **Phase 13 A3 (LFCF+AbsGS) N=24** | **21.330** | 🏆 Committed recipe, multi-seed reproducible |
| DOC-GS | 21.38 | gap −0.05 |
| BinocularGS | 21.44 | gap −0.11 |
| ICO-GS SOTA | 22.20 | gap −0.87 |

### 1.5 Sơ đồ khối kiến trúc (high-level)

```
                          ╔══════════════════════════════╗
   INPUT                  ║   LLFF scene (8 scenes)      ║
                          ║   • images_8/  ~20-25 ảnh     ║
                          ║   • COLMAP sparse/0/*.bin     ║
                          ╚══════════════╤═══════════════╝
                                         │
                          ┌──────────────▼───────────────┐
   DATA SPLIT             │  3 train views (FIXED)        │
   (deterministic)        │  + test views + pseudo cams   │
                          └──────────────┬───────────────┘
                                         │
   ┌─────────────────────────────────────▼──────────────────────────────┐
   │  PRE-LOOP (1 lần)                                                   │
   │  ┌────────────────┐  ┌────────────────┐  ┌──────────────────────┐  │
   │  │ COLMAP sparse  │  │ DAV2 depth     │  │ Informed CRS₀ init   │  │
   │  │ → init ~3K     │  │ → WLS align    │  │ (reproj+depth+view)  │  │
   │  │   Gaussians    │  │   COLMAP       │  │ → _crs_score logit   │  │
   │  └────────────────┘  └────────────────┘  └──────────────────────┘  │
   └─────────────────────────────────────┬──────────────────────────────┘
                                         │
   ╔═════════════════════════════════════▼══════════════════════════════╗
   ║  TRAINING LOOP  (10000 iter)                                       ║
   ║                                                                    ║
   ║   ┌─────────┐   render    ┌──────────────┐                         ║
   ║   │ Gaussian│ ──────────▶ │ image, depth,│                         ║
   ║   │  params │   (CUDA     │ alpha, grad  │                         ║
   ║   │  + CRS  │  rasterizer)└──────┬───────┘                         ║
   ║   └────▲────┘                    │                                 ║
   ║        │              ┌──────────▼──────────┐                      ║
   ║        │              │ LOSS                 │                      ║
   ║        │              │ L_phot (L1+SSIM)     │  ← weight_map        ║
   ║        │              │ + 0.05·L_depth       │    (CRS_pix/covis)   ║
   ║        │              │ + coreg (nếu 2-field)│                      ║
   ║        │              └──────────┬───────────┘                      ║
   ║        │                         │ backward                         ║
   ║        │              ┌──────────▼───────────┐                      ║
   ║        │              │ CRS UPDATE (/100 it) │  D_cycle + R_visible ║
   ║        │              │ → CRS_i = sigmoid(   │  + S_stability       ║
   ║        │              │   w·[D,R,S])  EMA    │                      ║
   ║        │              └──────────┬───────────┘                      ║
   ║        │                         │                                  ║
   ║        │     ┌───────────────────▼────────────────────┐             ║
   ║        │     │ DENSIFY (500<it<5000, /100 it)          │             ║
   ║        │     │  ┌─────────────────────────────────┐    │             ║
   ║        │     │  │ is_lfcf_iter? (Phase 13)         │    │             ║
   ║        │     │  │  YES → LFCF: tolerance compare   │    │             ║
   ║        │     │  │    enlarge(diffscale)/split      │    │             ║
   ║        │     │  │  NO  → standard clone/split      │    │             ║
   ║        │     │  │        (+AbsGS nếu absdensify)   │    │             ║
   ║        │     │  └─────────────────────────────────┘    │             ║
   ║        │     │  PRUNE: (CRS<τ & isolated)|opacity<min  │             ║
   ║        │     └───────────────────┬────────────────────┘             ║
   ║        │                         │                                  ║
   ║        │              ┌──────────▼───────────┐                      ║
   ║        │              │ CRS-mod SH freeze    │  zero _features_rest ║
   ║        │              │ (CRS<τ → freeze SH)  │  .grad (Phase 8c)    ║
   ║        │              └──────────┬───────────┘                      ║
   ║        │                         │                                  ║
   ║        │              ┌──────────▼───────────┐                      ║
   ║        └──────────────┤ optimizer.step       │                      ║
   ║          update       │ + opacity_decay      │                      ║
   ║                       └──────────────────────┘                      ║
   ╚════════════════════════════════════╤═══════════════════════════════╝
                                         │
                          ┌──────────────▼───────────────┐
   OUTPUT                 │  Trained Gaussians (.ply)     │
                          │  → render test → PSNR/SSIM    │
                          │  Phase 13 A3 = 21.330 N=24    │
                          └───────────────────────────────┘
```

**Đọc sơ đồ**: data flow đi xuống. Trong training loop, Gaussian params được render → tính loss → CRS update đánh giá chất lượng từng Gaussian → densify (LFCF hoặc standard) sinh/xóa Gaussian → SH freeze khóa SH của Gaussian kém → optimizer cập nhật → vòng lặp. CRS là **signal trung tâm** điều phối cả pruning, SH freeze, và (optional) loss reweight.

**3 cơ chế quyết định Phase 13 recipe** (tô đậm trong loop):
1. **CRS update** (Phase 8 a/b/c) — đánh giá per-Gaussian reliability
2. **LFCF densify** (Phase 13) — tolerance-based enlarge/split + diffscale
3. **CRS-mod SH freeze** (Phase 8c) — biggest single winner (+0.189)

---

## Section 2 — Codebase structure

```
CoR-GS/  (= CRSGaussian local)
├── train.py                  ← orchestration spine (1095 lines)
├── render.py / metrics.py    ← eval pipeline
├── arguments/__init__.py     ← config surface (353 lines, 3 ParamGroups)
├── scene/
│   ├── __init__.py           ← Scene loader (train/test/pseudo cameras)
│   ├── gaussian_model.py      ← GaussianModel (1129 lines): params, densify, prune, CRS attrs
│   ├── dataset_readers.py     ← COLMAP/LLFF readers, n_views subsample
│   ├── cameras.py             ← Camera class (FoVx/FoVy convention)
│   └── colmap_loader.py       ← COLMAP binary readers
├── gaussian_renderer/
│   └── __init__.py           ← render() — rasterize + depth + alpha + dropout (242 lines)
├── utils/
│   ├── crs/
│   │   ├── crs_module.py      ← CRS spine (708 lines): D_i, R_visible, update_crs
│   │   ├── crs_init.py        ← Informed CRS₀ (3-signal: reproj+depth+view)
│   │   ├── d_cycle.py         ← D_cycle signal (no-DAV2 multi-view depth consistency)
│   │   ├── sh_freeze.py       ← Phase 8c CRS-modulated SH freeze
│   │   ├── sh_stability.py    ← Phase 8b S_stability EMA tracking
│   │   ├── crs_diagnostics.py ← DIAG 1-5 logging
│   │   └── tier_a_diag.py     ← Tier A formula diagnostics
│   ├── densify/
│   │   └── lfcf.py            ← Phase 13 LFCF (297 lines): tolerance+diffscale+depth-split
│   ├── depth/
│   │   ├── depth_model.py      ← DAV2 wrapper
│   │   ├── depth_alignment.py  ← WLS COLMAP alignment (closed-form)
│   │   └── depth_warping.py    ← forward-warp (covisibility)
│   ├── loss/
│   │   └── covisibility_depth.py ← Phase 11 Step 1 covisibility reweight
│   └── regularizer/
│       ├── dropansh.py / sh_dropout.py  ← DropAnSH
│       └── density_*.py        ← density-aware anchor sampling
├── scripts/
│   ├── p13_lfcf_multiseed.{sh,py}     ← Phase 13 ablation (committed)
│   ├── p13_2_spectrum_analysis.py     ← GT FFT analyzer
│   ├── p13_2_spectrum_diagnostic.py   ← render-vs-GT freq diagnostic
│   ├── p13_2_lambda_calibration.py    ← λ_HF calibration
│   └── p13_2_bottleneck_decompose.py  ← bottleneck attribution (current)
└── docs/
    ├── CLAUDE.md / 00*_session.md     ← session rules
    ├── 03_task_queue.md / 04_decisions_log.md
    ├── 13_efa_gs_lfcf_design.md       ← Phase 13 design (Section 1-20)
    └── ARCHITECTURE.md                ← (this file)
```

**Folders CHỈ ĐỌC** (không sửa): `FSGS/`, `DNGaussian/`, `LoopSparseGS/`, `DepthRegularizedGS/`, `SCGaussian/`, `EFA-GS/`, `mip-splatting/`, `submodules/`.

---

## Section 3 — Training pipeline (per-iteration sequence)

`training()` trong train.py. Mỗi iteration thực thi theo thứ tự sau:

```
┌─ PRE-LOOP (1 lần) ────────────────────────────────────────────────
│ 1. GaussianModel(args) + Scene(args) — load COLMAP, init từ sparse pts
│ 2. [use_depth_prior] precompute_depth_priors (DAV2) + align_depth_to_colmap (WLS)
│ 3. [use_coreliability_reweight] compute_covisibility_maps (forward-warp)
│ 4. [informed_crs_init] compute_informed_crs0 → overwrite gs0._crs_score
└────────────────────────────────────────────────────────────────────

┌─ PER ITERATION (1 → opt.iterations) ──────────────────────────────
│
│ A. SH degree up mỗi 500 iter (oneupSHdegree)
│ B. Pick random train camera (viewpoint_stack.pop random)
│ C. render() cho mỗi gs{i} → image, depth, alpha, viewspace_pts,
│    visibility_filter, radii, dropout_mask
│
│ D. LOSS computation:
│    ├─ weight_map (optional):
│    │   • Phase 7 use_loss_reweight → CRS_pix render cache
│    │   • Phase 11 Step 1 use_coreliability_reweight → cov_maps (OVERRIDE Phase 7)
│    ├─ L_phot = (1-λ_dssim)·L1 + λ_dssim·(1-SSIM)   [weighted nếu weight_map]
│    ├─ [coreg, gaussiansN=2] co-photometric pseudo loss
│    └─ [use_depth_prior] L_depth = 0.05·pearson_depth_loss(render_d, prior_d)
│
│ E. loss.backward()  (mỗi gs{i})
│
│ F. CRS UPDATE (mỗi crs_update_interval sau T_warmup):
│    update_crs(gaussians, cameras, aligned_depth, ...) →
│      D signal: D_cycle (use_d_cycle) hoặc D_DAV2
│      R signal: R_visible (use_r_visible, visibility-aware) hoặc skip
│      S signal: S_stability (use_sh_reliability, SH EMA variance)
│      → CRS_i = sigmoid(scale·(w_d·D + w_r·R + w_s·S − 0.5)), EMA update
│
│ G. DENSIFICATION (densify_from < iter < densify_until, mỗi densification_interval):
│    ├─ add_densification_stats (grad accum, dropout-aware mask reconstruct)
│    ├─ [Phase 13] build lfcf_opts nếu use_lfcf + is_lfcf_iter
│    └─ densify_and_prune(...):
│         IF is_lfcf_iter → LFCF path (compute_lfcf_decisions: enlarge/split)
│         ELSE → standard densify_and_clone + densify_and_split (+ AbsGS nếu absdensify)
│         + PRUNE: (CRS<τ_crs AND isolated) OR (opacity<min) OR big_screen
│
│ H. [use_rnrc] RNRC update (render-contribution coupling — post-densify)
│
│ I. [Phase 8c use_crs_modulated_sh_freeze] apply_crs_modulated_sh_freeze
│    (zero _features_rest.grad cho Gaussian CRS<τ_freeze) — TRƯỚC optimizer.step
│
│ J. optimizer.step() + zero_grad (mỗi gs{i})
│
│ K. [Phase 8b use_sh_reliability] update_sh_stability EMA (SAU optimizer.step)
│
│ L. [Phase 2c use_opacity_decay] opacity_decay(factor) mỗi iter sau densify_from
│
│ M. [DIAG freeze_sh_after] global freeze_sh (legacy, bypass nếu Phase 9)
│
│ N. update_learning_rate + opacity reset (mỗi opacity_reset_interval)
│ O. [coprune, gaussiansN=2] Open3D registration co-prune mỗi 500 iter
│
└────────────────────────────────────────────────────────────────────

┌─ POST-LOOP ────────────────────────────────────────────────────────
│ Timing summary + eval_history table (best test PSNR)
└────────────────────────────────────────────────────────────────────
```

**Key ordering invariants**:
- CRS update (F) TRƯỚC densify (G) — pruning dùng CRS vừa update
- SH freeze (I) SAU densify+RNRC (modify N), TRƯỚC optimizer.step (zero grad có effect)
- S_stability EMA (K) SAU optimizer.step (capture post-update SH state)
- opacity_decay (L) mỗi iter (continuous pressure, khác CRS prune sparse)

---

## Section 4 — Core mechanism: CRS module

### 4.1 CRS₀ initialization (informed, T5.5)

Gated `--informed_crs_init AND --use_depth_prior`. Compute trong `crs_init.py`:

```
COLMAP Gaussians (per point):
  q_reproj = 1 − clip(reproj_err / τ_r, 0, 1)        # SfM quality, τ_r=2.5
  q_depth  = 1 − clip(|d_DAV2 − d_COLMAP| / range, 0, 1)
  q_view   = (n_obs − 1) / max(N_train − 1, 1)        # multi-view support
  Q_i = w_reproj·q_reproj + w_depth·q_depth + w_view·q_view   (w auto-norm)
  ℓ₀ = γ·(Q_i − 0.5)   (γ=5.0 logit scale)
  CRS₀ = sigmoid(ℓ₀)   [lưu logit ℓ₀ vào _crs_score]

Densified children: CRS₀_child = clip(η·CRS_parent, 0, 0.5)   (η=0.7, conservative)
```

Default (flag OFF): `_crs_score = 0` → CRS₀ = sigmoid(0) = 0.5 neutral.

### 4.2 CRS dynamic update (`update_crs` in crs_module.py)

Gọi mỗi `crs_update_interval` (100) iter sau `T_warmup` (1000). 3-signal:

| Signal | Source | Flag | Mechanism |
|--------|--------|------|-----------|
| **D** (depth consistency) | D_cycle (no DAV2) hoặc D_DAV2 | `use_d_cycle` | Cycle-depth qua train views; render N depth maps, cache mỗi `d_cycle_update_freq` |
| **R** (reprojection consistency) | R_visible | `use_r_visible` | Chỉ aggregate views nơi Gaussian visible (gauss_z ≤ rendered_z × tolerance) — fix R contamination 36.5% |
| **S** (SH stability) | S_stability | `use_sh_reliability` | EMA variance của `_features_rest` → detect SH drift (memorize vs stable) |

```
crs_logit = scale · (w_d·D + w_r·R + w_s·S − 0.5)
CRS_i = sigmoid(crs_logit)   [range ~0.08-0.92]
_crs_score = EMA(_crs_score, crs_logit, decay=crs_ema_decay)
```

`w` auto-normalize khi signal tắt. Phase 9 `disable_r_signal` → D-only formula.

### 4.3 CRS-guided pruning (Option C, T4.2)

Trong `densify_and_prune` (gaussian_model.py), sau densify:

```
prune_mask = (opacity < prune_threshold)                    # legacy
           | (max_radii2D > max_screen_size)                # big screen
           | (scaling.max > 0.1·extent)                     # big world
IF iter > T_warmup AND use_crs_pruning:
  crs_low  = (get_crs < tau_crs)                            # τ_crs=0.35
  isolated = (knn_dist > tau_isolated · extent)             # τ_iso=0.1
  prune_mask |= (crs_low & isolated)                        # floater thật
prune_points(prune_mask, iter)
```

Lý do AND(crs_low, isolated): Gaussian đang học có CRS thấp nhưng gần surface (có neighbors) → KHÔNG prune nhầm. Floater thật đứng một mình.

### 4.4 CRS-modulated SH freeze (Phase 8c — biggest winner +0.189)

`apply_crs_modulated_sh_freeze` (sh_freeze.py), gọi mỗi iter sau densify, trước optimizer.step:

```
IF iter ≥ crs_freeze_start:
  freeze_mask = (get_crs < crs_freeze_tau)    # τ_freeze=0.5
  _features_rest.grad[freeze_mask] = 0        # zero SH grad cho low-CRS
```

→ Low-CRS Gaussian giữ nguyên SH coefficients (không học HF detail noise), chỉ update xyz/opacity/scale/rotation. Per-Gaussian thay global `freeze_sh_after`.

---

## Section 5 — Phase 13 LFCF + AbsGS densification (committed recipe)

### 5.1 Vị trí trong pipeline

LFCF **replace** standard `densify_and_clone/split` khi `is_lfcf_iter`. Gate trong train.py:

```python
is_lfcf_iter_now = opt.use_lfcf AND (iter % (lfcf_interval_times × densification_interval) == 0)
# lfcf_interval_times=2, densification_interval=100 → LFCF mỗi 200 iter
```

`densify_and_prune(..., is_lfcf_iter, lfcf_opts, cameras_for_lfcf)`:
- `is_lfcf_iter=True` → LFCF path (compute_lfcf_decisions)
- `is_lfcf_iter=False` → standard path (densify_and_clone + split, AbsGS nếu `absdensify`)

### 5.2 LFCF core logic (`compute_lfcf_decisions` in lfcf.py)

```
Step 1: selected = ‖grad‖ ≥ grad_threshold
Step 2: partition:
  new      = (¬prev_selected) & selected   → ENLARGE (chưa biết signal/noise)
  intersect = prev_selected & selected      → tolerance check:
Step 3: tolerance compare (intersect):
  decent = ‖grad‖ ≤ ‖prev_grad − tolerance‖   (grad ĐANG GIẢM = signal)
  decent     → SPLIT (modeling real detail)
  not decent → ENLARGE (grad stuck = HF noise)
Step 4: depth-aware multiplier:
  interval = compute_3D_interval(xyz, cameras)    # min projected depth
  interval_coef = normalize_interval(log, minmax) # [0,1], deep=high
  mult = interval_coef·scaler_min + (1−coef)·scaler_max   # deep→less enlarge
  log_mult = log(mult) · training_percent_powered          # decay over densify
Step 5: ENLARGE với diffscale (volume-preserving isotropify):
  sort scale ASC [s_min, s_mid, s_max]
  s_min ×mult^(+1), s_mid ×mult^(-1/3), s_max ×mult^(-2/3)
  → volume = mult^(1-1/3-2/3) = mult^0 PRESERVED, Gaussian "tròn hơn"
Step 6: SPLIT shrink (parent scale ÷, children inherit)
```

Caller (gaussian_model.densify_and_prune LFCF branch) dispatch:
- `set_attributes("scaling", enlarged_mask, changes)` — in-place .data modify (no optimizer sync, per EFA-GS)
- probabilistic split lottery: `prob = interval_coef·(ub−lb) + lb`, sample
- `_lfcf_split_children` → densification_postfix + prune parents

### 5.3 AbsGS (dormant infrastructure, activated Phase 13)

`absdensify` flag (arguments line 225, default False; auto-registered, KHÔNG add CLI duplicate trong train.py:1069). Khi ON:
- Selection criterion union: `(grad_norm ≥ τ) OR (grad_abs ≥ Q)` với `Q = quantile(grads_abs, 1-ratio)`
- `grads_abs` = norm của `viewspace_point_tensor.grad[:, 2:]` (channels asymmetric) — catch HF edge candidates standard norm miss
- Already infrastructure-complete trong gaussian_model.py densify_and_clone/split (`if self.absdensify:`)

### 5.4 Phase 13 FULL recipe (committed config)

```
Phase 8 FULL components:
  --use_depth_prior --informed_crs_init --use_crs_pruning
  --use_d_cycle --use_r_visible --use_sh_reliability
  --use_crs_modulated_sh_freeze --use_opacity_decay
+ Phase 13:
  --use_lfcf --lfcf_init_scaling_max 1.5 --lfcf_interval_times 2
  --lfcf_diffscale True --lfcf_tolerance 1e-5
  --absdensify
```

N=24 paired Δ vs Phase 8 FULL = **+0.164 dB** (95% CI [+0.101, +0.227], 7/8 wins + 1 neutral).

**Mechanism characterization** (Phase 13.2 diagnostic): A3 = **spatial alignment** (Gaussian đúng chỗ), KHÔNG phải spectral amplitude. 6/8 scenes A3 HF tệ hơn baseline nhưng PSNR vẫn cao.

---

## Section 6 — Depth prior subsystem

Gated `--use_depth_prior`. Pre-loop one-shot:

```
1. precompute_depth_priors(cameras, dav2_path, encoder=vitl)
   → DAV2 inverse-depth per train view
2. align_depth_to_colmap(depth_prior, cameras, source_path, n_views)
   → closed-form WLS: weight = 1/reproj_err, scale âm (DAV2 inverse)
   → aligned_depth_dict {cam.uid: (H,W) metric}, depth_range = median(far)−median(near)
```

Dùng cho: L_depth (Pearson), D signal (D_DAV2 hoặc validate D_cycle), informed CRS₀ q_depth, covisibility maps.

`depth_range` = normalize cho D_i, epsilon_depth (position constraint DISABLED — DAV2 noise gây −3dB).

---

## Section 7 — Config surface (3 ParamGroups)

### 7.1 ModelParams (`-` prefix = shorthand)

| Flag | Default | Vai trò |
|------|---------|---------|
| `n_views` | 0 | Sparse view count (3 cho LLFF 3-view) |
| `use_depth_prior` | False | DAV2 + COLMAP align (prerequisite nhiều feature) |
| `informed_crs_init` | False | CRS₀ geometry-informed (3-signal) |
| `crs_densify_inherit` | False | Child CRS₀ = clip(η·parent, 0, 0.5) |
| `use_opacity_decay` | False | Opacity ×factor mỗi iter (Binocular3DGS-style) |
| `use_coreliability_reweight` | False | Phase 11 Step 1 covisibility reweight |
| `tier_a_diag` | False | Formula diagnostics dump |

### 7.2 PipelineParams

| Flag | Default | Vai trò |
|------|---------|---------|
| `use_color` | True | SH color (vs geometry-only) |
| `use_dropout` | False | Track B dropout regularization |
| `use_dropansh` | False | DropAnSH anchor+SH dropout |
| `dropansh_density_method` | uniform | Anchor sampling: uniform/voxel/covariance/crs |

### 7.3 OptimizationParams (key subset)

| Flag | Default | Vai trò |
|------|---------|---------|
| `iterations` | 30000 | (LLFF override 10000) |
| `densify_from_iter` / `until` | 500 / 15000 | (LLFF override until 5000) |
| `densification_interval` | 100 | Densify cadence |
| `T_warmup` | 1000 | CRS active sau iter này |
| `tau_crs` / `tau_isolated` | 0.35 / 0.1 | CRS prune thresholds |
| `crs_ema_decay` | 0.9 | CRS EMA (LLFF dùng 0.3) |
| `use_crs_pruning` | False | Phase 4 CRS prune Option C |
| `use_d_cycle` | False | Phase 7 D_cycle signal |
| `use_r_visible` | False | Phase 8a visibility-aware R |
| `use_sh_reliability` | False | Phase 8b S_stability |
| `use_crs_modulated_sh_freeze` | False | Phase 8c per-Gaussian SH freeze |
| `absdensify` | False | AbsGS (Phase 13 activated) |
| `use_lfcf` | False | Phase 13 LFCF master switch |
| `lfcf_init_scaling_max` | 1.5 | LFCF enlarge ceiling |
| `lfcf_interval_times` | 2 | LFCF mỗi 2×densify_interval |
| `lfcf_diffscale` | True | Volume-preserving isotropify |
| `lfcf_tolerance` | 1e-5 | FP-stability grad compare |

**Master-switch pattern**: mọi feature default OFF → flag OFF = behavior baseline byte-identical (Rule 11). `str2bool` cho phép `--flag False` với default=True flags.

---

## Section 8 — Data flow (LLFF 3-view)

```
data/nerf_llff_data/<scene>/
├── images_8/          ← TẤT CẢ ~20-25 ảnh (r=8 downsample, ~378×504)
├── sparse/0/*.bin     ← COLMAP cameras/images/points3D
└── poses_bounds.npy

dataset_readers.py split (deterministic, NO random):
  test_idx  = {i : i % llffhold == 0}        # llffhold=8 → {0,8,16}
  train_pool = phần còn lại (~21 ảnh)
  train_3view = linspace(0, len(pool)-1, n_views=3).round()  # fixed

→ scene.getTrainCameras() = 3 fixed cameras
  scene.getTestCameras()  = test split (3-8 cams tùy scene)
  scene.getPseudoCameras() = interpolated (cho coreg/pseudo loss)
```

Convention chuẩn (RegNeRF/FSGS/CoR-GS/BinocularGS/ICO-GS đều dùng) → fair comparison. Multi-seed variance KHÔNG từ data split (fixed), mà từ random init + per-iter view pick (seed) + atomicAdd non-determinism (±1.3 dB single-scene).

---

## Section 9 — Module dependency map

```
train.py
 ├── scene.Scene / GaussianModel ──── scene.dataset_readers ── colmap_loader
 ├── gaussian_renderer.render ─────── submodules/diff-gaussian-rasterization-confidence
 │                                     (depth + alpha + dropout output)
 ├── utils.depth ── precompute_depth_priors (DAV2) + align_depth_to_colmap (WLS)
 ├── utils.crs.update_crs ─┬─ d_cycle (D signal)
 │                          ├─ R_visible (in crs_module)
 │                          └─ sh_stability (S signal)
 ├── utils.crs.sh_freeze ── apply_crs_modulated_sh_freeze (Phase 8c)
 ├── utils.crs.crs_init ─── compute_informed_crs0 (T5.5)
 ├── utils.densify.lfcf ─── compute_lfcf_decisions (Phase 13)
 │                          └── called by gaussian_model.densify_and_prune
 ├── utils.loss.covisibility_depth ── Phase 11 Step 1 reweight
 └── utils.crs.crs_diagnostics / tier_a_diag ── read-only logging

gaussian_model.py (state holder):
  params: _xyz, _features_dc, _features_rest, _opacity, _scaling, _rotation
  CRS attrs: _crs_score, confidence, spawn_iter, _rc_smooth, _crs_rnrc
  grad accum: xyz_gradient_accum(_abs/_abs_max), denom
  LFCF attrs: lff_xyz_grad_accum, lff_denom, prev_lff_xyz_grad,
              prev_selected_pts_mask_bool
  methods: densify_and_clone/split/prune, _lfcf_split_children,
           set_attributes, opacity_decay, freeze_sh/dc, prune_points
```

---

## Section 10 — Diagnostic tooling (Phase 13.2)

Post-hoc analysis trên A3 renders (KHÔNG train), trong `scripts/p13_2_*`:

| Script | Mục đích | Output |
|--------|----------|--------|
| `spectrum_analysis.py` | GT FFT spectrum per scene | k_50/80/90, σ_blur recommendations |
| `spectrum_diagnostic.py` | render-vs-GT rel_Δ per freq band | failure pattern (SPURIOUS/MISSING/NEAR_CEILING) |
| `lambda_calibration.py` | L_main / L_HF magnitude | R=0.631, calibrated λ levels |
| `bottleneck_decompose.py` | 12.88 dB gap → buckets | H1 irreducible / H3 geometry / H4 appearance / H5 detail |

**Diagnostic verdicts (2026-05-13/14)**:
- Universal HF deficit 8/8 scenes (render đạt 64% HF energy GT)
- A3 mechanism = spatial alignment, KHÔNG spectral amplitude
- DWTGS + standard FALA + HF-emphasis REJECTED (wrong sign / phase conflict với A3 spatial)
- Bottleneck decompose: pre-flight depth-alignment check (CRITICAL — detect A3-depth vs COLMAP-depth scale mismatch, robust median-ratio scale correction)

---

## Section 11 — Current status + ceiling

### 11.1 Phase history (verdict tổng hợp)

| Phase | Axis | Verdict |
|-------|------|---------|
| Phase 1-7 | Depth prior + CRS signal | +1.2 dB (DAV2 depth = info infusion) |
| Phase 8 | Formula redesign + SH path | BREAKTHROUGH +0.125 (SH freeze hero) |
| Phase 9 | Simplification | All simplifications HURT, recipe LOCKED |
| Phase 10A | Foundation init (DUSt3R) | FAIL hard, axis DEAD |
| Phase 11 (6/6) | Loss-axis | EXHAUSTED, all ≈ noise |
| Phase 12 (3/3) | Position-axis (CRS-pull) | EXHAUSTED |
| **Phase 13 A3** | **Densify-axis (LFCF+AbsGS)** | **COMMIT +0.164 N=24** 🏆 |
| Phase 13.1 | LFCF intensity sweep | A3 base optimal |
| Phase 13.2 | Frequency-axis loss + bottleneck | HF-emphasis REJECT; bottleneck diagnostic running |

→ **10/10 mechanism classes trên loss/position/CRS axis exhausted; Phase 13 densify-axis = first commit.**

### 11.2 Ceiling analysis

A3 = 21.330 (train ~34.2, test ~21.3, gap 12.88 dB). Diagnostic decompose 12.88 dB → buckets:
- H1 irreducible (3-view coverage limit) — accept ceiling nếu >50%
- H3 geometry (Gaussian mislocated) — dn-splatter normal prior nếu dominant
- H4 appearance/SH — SH-axis fix nếu chroma-dominated; near-optimal residual nếu uniform
- H5 detail — refuted (HF pilot)

Decision direction chờ bottleneck verdict (depth-alignment fix pending — robust median-ratio scale).

### 11.3 Constraints (memory-enforced)

- Multi-seed (3 seeds × 8 scenes paired) cho mọi ablation chính; min detectable Δ ±0.10
- 2 GPU parallel mọi ablation script
- Local Windows = code-only; training server Linux
- Planning session draft prompts, KHÔNG write production code
- Rule 13: reject hướng → cleanup (xóa script/flag/hook, giữ logs + docs)

---

## Appendix A — Key invariants (đừng vi phạm)

1. **Master-switch OFF = baseline byte-identical** — verify mọi feature mới
2. **CRS update TRƯỚC prune** — pruning dùng CRS fresh
3. **SH freeze SAU densify, TRƯỚC optimizer.step** — zero grad có effect
4. **S_stability EMA SAU optimizer.step** — capture post-update SH
5. **LFCF `set_attributes` KHÔNG sync optimizer** — intentional (EFA-GS pattern)
6. **prune_points cover MỌI per-Gaussian attr** — base 6 + grad accum + CRS (_crs_score, confidence, spawn_iter, _rc_smooth, _crs_rnrc) + LFCF (lff_*, prev_selected_bool); guard `.numel()>0`
7. **`_crs_score` lưu LOGIT** (sigmoid → CRS); KHÔNG dùng `confidence` attr cho CRS
8. **absdensify auto-register OptimizationParams** — KHÔNG add CLI duplicate (argparse conflict)
9. **LLFF train views FIXED** — multi-seed variance ≠ data split

## Appendix B — Verify-before-claim

Khi modify: chạy với flag OFF → metric MATCH baseline log. Smoke test (no GPU): import + mock numpy. KHÔNG run training local (no CUDA/data). Server-side: paste log về cho session phân tích.
