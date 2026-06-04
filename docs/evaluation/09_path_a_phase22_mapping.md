# Path A — Phase 22 → Nerfstudio mapping (DEFINITIVE)

> **Mục đích**: Tài liệu này thay thế phần B3/B4 trong [08_path_a_implementation_plan.md](08_path_a_implementation_plan.md). Mọi mapping ở đây dựa trên **đọc trực tiếp CRSGaussian source** (KHÔNG đoán, KHÔNG đọc CoR-GS base).
>
> **Bối cảnh**: Em đã sai khi đọc nhầm `CoR-GS/` (base) thay vì `CRSGaussian/` (fork chứa Phase 22 modifications) trong các iteration B2/B3 trước. Doc này CORRECT và chính xác.
>
> **Source of truth**: `CRSGaussian/docs/ARCHITECTURE.md` + `CRSGaussian/scripts/p20_ablation_dense_run.sh` + `CRSGaussian/scripts/p22_pilot_run.sh` + `CRSGaussian/train.py`.

---

## 1. Phase 22 = exact CLI from `p22_pilot_run.sh` + `p20_ablation_dense_run.sh`

Phase 22 pilot reuses `p20_ablation_dense_run.sh` với `CONFIG=trim_full`. Đây là **exact command line** được dùng để đạt PSNR project best 21.918:

### 1.1 PROTOCOL (fixed cho mọi config)

```
--eval -r 8 --n_views 3 --random_background
--iterations 10000
--densify_until_iter 5000              # NOT 15000 như em đoán
--densify_grad_threshold 0.0005        # NOT 0.0002 default
--gaussiansN 1                          # Single field (no CoR-GS 2-field)
--sample_pseudo_interval 1
--start_sample_pseudo 500               # NOT 2000 default
--test_iterations 10000
```

### 1.2 Module flags (A3-TRIM-FULL = 7 modules ON)

```
# ── CRS framework (4 modules — bỏ 3 verified-dead Phase 20) ──
M_DEPTHCFG:  --use_depth_prior
             --dav2_path ../Depth-Anything-V2
             --crs_ema_decay 0.3
             --crs_update_interval 100

M_DCYCLE:    --use_d_cycle
             --d_cycle_warmup 1000
             --d_cycle_sigma 5.0
             --d_cycle_update_freq 100

M_SHFREEZE:  --use_crs_modulated_sh_freeze
             --crs_freeze_start 1000
             --crs_freeze_tau 0.5

M_SHREL:     --use_sh_reliability
             --sh_stability_warmup 1000
             --sh_stability_ema_beta 0.95
             --crs_w_s 0.33

# ── Independent pillars (3 modules) ──
M_DROP:      --use_dropansh
             --dropansh_pa 0.02
             --dropansh_psh 0.2

M_OPACITY:   --use_opacity_decay
             --opacity_decay_factor 0.999    # 0.999 for 10k iter (0.995 = default cho 30k)

M_EFA:       --use_lfcf
             --lfcf_init_scaling_max 1.5
             --lfcf_init_scaling_min 1.0
             --lfcf_last_scaling_max 1.0
             --lfcf_pow 1.0
             --lfcf_splitting_ub 1.0
             --lfcf_interval_times 2
             --lfcf_tolerance 1e-5
             --lfcf_diffscale True
             --absdensify
```

### 1.3 Trim 3 modules dead trên dense init (Phase 20 verified):

```
KHÔNG dùng (Phase 24 N=24 stack HURT −0.796):
  --informed_crs_init
  --use_crs_pruning
  --use_r_visible
```

### 1.4 + RoMa v1 dense init (Phase 22 unique):

```
Sinh từ scripts/p22_romav1_preprocess.py:
  data/nerf_llff_data/<scene>/3_views/dense/fused.ply.romav1
  → ~24000 points (vs ~3000 từ COLMAP MVS)
```

### 1.5 Tóm tắt: 8 modules = 7 ON flags + RoMa init

```
1. RoMa v1 dense init                    (data pre-processing)
2. Depth prior + Pearson loss            (use_depth_prior)
3. D_cycle (CRS depth signal)            (use_d_cycle)
4. CRS-modulated SH freeze               (use_crs_modulated_sh_freeze)
5. SH stability EMA                      (use_sh_reliability)
6. DropAnSH                              (use_dropansh)
7. Opacity decay                         (use_opacity_decay)
8. LFCF + AbsGS densifier                (use_lfcf + absdensify)
```

---

## 2. Phase 22 training loop — EXACT step-by-step (ARCHITECTURE.md Section 3)

Theo `CRSGaussian/train.py`, ordering INVARIANTS (Appendix A):

```
PRE-LOOP (1 lần):
  1. GaussianModel + Scene load (RoMa PLY init)
  2. [use_depth_prior] precompute DAV2 depth + align to COLMAP (WLS)
  3. [NO informed_crs_init Phase 22] _crs_score init = 0 → CRS = sigmoid(0) = 0.5

PER ITERATION (1 → 10000):
  A. iter % 500 == 0: oneupSHdegree()
  B. Pick random train camera (viewpoint_stack)
  C. render() → image, depth, alpha, viewspace_pts, visibility_filter, radii
     (CoR-GS rasterizer, NOT gsplat)
  D. LOSS:
     L_phot = (1 - lambda_dssim) * L1 + lambda_dssim * (1 - SSIM)
            = 0.8 * L1 + 0.2 * (1 - SSIM)
     [use_depth_prior] L_depth = 0.05 * pearson_depth_loss(render_d, prior_d)
     loss = L_phot + L_depth
  E. loss.backward()
  F. CRS UPDATE (mỗi crs_update_interval=100 sau T_warmup=1000):
     update_crs(gaussians, cameras, aligned_depth, ...)
       D_cycle signal: render N depth, compute cycle-depth consistency
       S_stability signal: EMA variance _features_rest
       (R_visible NOT used Phase 22 — trimmed)
     CRS_i = sigmoid(scale * (w_d*D + w_s*S - 0.5)), EMA update with decay=0.3
  G. DENSIFICATION (500 < iter < 5000, mỗi 100 iter):
     - add_densification_stats (with dropout-aware mask if DropAnSH)
     - IF iter % 200 == 0 AND use_lfcf: LFCF path (enlarge/split with diffscale)
     - ELSE: standard densify_and_prune (+ AbsGS if absdensify)
     - PRUNE: opacity < 0.005 OR max_radii2D > screen OR scale > 0.1*extent
     - (NO CRS pruning Phase 22 — trimmed)
  H. CRS-MODULATED SH FREEZE (sau iter 1000):
     apply_crs_modulated_sh_freeze(gauss, iter, freeze_start=1000, tau_freeze=0.5)
       → zero _features_rest.grad cho Gaussian có CRS < 0.5
     **MUST CALL TRƯỚC optimizer.step()** (Phase 8c invariant)
  I. optimizer.step() + zero_grad
  J. S_STABILITY EMA (sau optimizer.step, mỗi crs_update_interval=100):
     update_sh_stability(gauss, beta=0.95)
       → EMA capture post-update SH state
  K. OPACITY DECAY (mỗi iter sau densify_from=500):
     gaussians.opacity_decay(factor=0.999)
       → sigmoid(opacity) *= 0.999 → continuous pressure
  L. update_learning_rate(iter) (position lr scheduler)
  M. OPACITY RESET (conditional):
     IF (iter - 500 - 1) % 3000 == 0 AND iter > 500:
       reset_opacity()
     → Resets at iter 501, 3501, 6501, 9501 (4 resets in 10k training)
```

### 2.1 Key timing invariants (KHÔNG được vi phạm)

1. **CRS update (F) TRƯỚC densify (G)** — pruning dùng CRS vừa update
2. **SH freeze (H) SAU densify (G), TRƯỚC optimizer.step (I)** — zero grad mới có effect
3. **S_stability EMA (J) SAU optimizer.step (I)** — capture post-update SH state
4. **Opacity decay (K) mỗi iter sau densify_from** — continuous pressure
5. **Opacity reset (M) ở iter SPECIFIC** (501, 3501, 6501, 9501) — KHÔNG phải mỗi 3000

---

## 3. Method signatures cần wrap (CRSGaussian source — EXACT)

### 3.1 GaussianModel methods (`scene/gaussian_model.py`)

| Method | Signature | Phase 22 caller |
|--------|-----------|-----------------|
| `oneupSHdegree()` | `(self)` | iter % 500 == 0 |
| `create_from_pcd(pcd, spatial_lr_scale)` | `(self, BasicPointCloud, float)` | populate_modules |
| `training_setup(training_args)` | `(self, opt_namespace)` | populate_modules |
| `update_learning_rate(iteration)` | `(self, int)` | every iter |
| `opacity_decay(factor=0.995)` | `(self, float)` | every iter sau densify_from, factor=0.999 |
| `reset_opacity()` | `(self)` | iter ở 501, 3501, 6501, 9501 |
| `add_densification_stats(viewspace_pts, update_filter)` | `(self, Tensor[N,3], Tensor[N])` | iter < densify_until |
| `densify_and_prune(max_grad, min_opacity, extent, max_screen_size, iter, **kwargs)` | thấy section 3.2 | iter > densify_from, mỗi 100 iter |
| `freeze_sh()` / `freeze_dc()` | `(self)` | Optional global freeze (NOT Phase 22) |

### 3.2 `densify_and_prune` FULL signature (CRSGaussian fork)

```python
def densify_and_prune(
    self,
    max_grad,                 # 0.0005 Phase 22
    min_opacity,              # 0.005 default
    extent,                   # scene.cameras_extent
    max_screen_size,          # None Phase 22 (line 657-658)
    iter,                     # current iteration
    cameras=None,             # train cams (cho LFCF + position constraint)
    aligned_depth_dict=None,  # cho LFCF depth interval
    depth_range=None,         # normalize depth
    T_warmup=1000,            # CRS active
    tau_crs=0.35,             # CRS prune threshold
    tau_isolated=0.1,         # KNN isolation threshold
    crs_prune_dict=None,      # CRS prune dict (use_crs_pruning gating — trimmed Phase 22)
    eta=0.0,                  # Child CRS₀ = clip(eta*parent, 0, 0.5) — trimmed Phase 22
    is_lfcf_iter=False,       # bool — apply LFCF path
    lfcf_opts=None,           # dict — LFCF hyperparams
    cameras_for_lfcf=None,    # train cams for LFCF
)
# Phase 22 plug-in chỉ cần pass 5 positional + 4 kwargs (is_lfcf_iter, lfcf_opts,
# cameras_for_lfcf, cameras nếu use_lfcf). crs_prune_dict + eta giữ default vì
# Phase 22 trim use_crs_pruning + crs_densify_inherit.
```

### 3.3 Utils functions (CRSGaussian custom modules)

| Module | Function | Usage |
|--------|----------|-------|
| `utils.crs.crs_module` | `update_crs(...)` | F per iter |
| `utils.crs.sh_freeze` | `apply_crs_modulated_sh_freeze(gauss, iter, freeze_start, tau_freeze)` | H per iter sau 1000 |
| `utils.crs.sh_stability` | `update_sh_stability(gauss, beta=0.95)` | J per crs_update_interval sau optimizer.step |
| `utils.densify.lfcf` | `calculate_training_percent_powered(iter, from, until, pow, percent_lb)` | Build lfcf_opts G |
| `utils.depth.depth_alignment` | `align_depth_to_colmap(depth_dict, cams, source_path, n_views)` | PRE-LOOP |
| `utils.depth.depth_model` | `precompute_depth_priors(cams, dav2_path, encoder)` | PRE-LOOP |
| `utils.loss_utils` | `pearson_depth_loss(rendered, prior)` | D per iter |

### 3.4 LFCF cadence (Phase 13 detail)

```python
# Per iter (G section):
is_lfcf_iter_now = False
lfcf_opts = None
if opt.use_lfcf AND iter < opt.densify_until_iter:
    is_lfcf_iter_now = (iter % (lfcf_interval_times * densification_interval) == 0)
    # = (iter % (2 * 100) == 0) = (iter % 200 == 0)
    if is_lfcf_iter_now:
        from utils.densify.lfcf import calculate_training_percent_powered
        percent_lb = (
            math.log(lfcf_last_scaling_max) / math.log(lfcf_init_scaling_max)
            if lfcf_init_scaling_max != 1.0 else 1.0
        )
        tpp = calculate_training_percent_powered(
            iter, densify_from, densify_until, lfcf_pow, percent_lb,
        )
        splitting_lb_now = 1.0 - (iter - densify_from) / max(densify_until - densify_from, 1)
        lfcf_opts = {
            'scaling_multiplier_max': 1.5,
            'scaling_multiplier_min': 1.0,
            'training_percent_powered': tpp,
            'splitting_ub': 1.0,
            'splitting_lb': splitting_lb_now,
            'tolerance': 1e-5,
            'diffscale': True,
        }
```

Then pass to `densify_and_prune(..., is_lfcf_iter=is_lfcf_iter_now, lfcf_opts=lfcf_opts, cameras_for_lfcf=cameras)`.

---

## 4. Em đã sai gì trong B2/B3 implementation cũ

### 4.1 Critical bugs (cần sửa hoàn toàn)

| Bug | Sai gì | Đúng |
|-----|--------|------|
| **Opacity decay** | Em viết logit math thủ công | Phải call `gaussians.opacity_decay(factor=0.999)` method (đã có sẵn trong CRSGaussian GaussianModel line 251) |
| **Opacity decay condition** | `step >= opacity_decay_start_iter` | Đúng: `step > densify_from_iter` (line 814) |
| **Reset opacity timing** | `step % 3000 == 0` | Đúng: `(step - 500 - 1) % 3000 == 0 AND step > 500` → resets ở 501, 3501, 6501, 9501 |
| **Densify size_threshold** | `20 if step > opacity_reset_interval else None` | Đúng: `None` always (Phase 22 line 657-658 comment out) |
| **CRS update** | KHÔNG implement | Phase 22 REQUIRED (D_cycle + S_stability + EMA) — em deferred to "B3b" |
| **SH freeze** | KHÔNG implement | Phase 22 REQUIRED — Phase 8c +0.189 biggest winner |
| **densify_grad_threshold** | 0.0002 (gsplat default) | Đúng: 0.0005 (Phase 22) |
| **densify_until_iter** | 15000 | Đúng: 5000 (Phase 22) |
| **LFCF** | KHÔNG implement | Phase 22 REQUIRED — needs lfcf_opts build per iter |
| **Config defaults missing** | use_d_cycle / crs_freeze_start / sh_stability_warmup / crs_w_s không có trong config dataclass | Phase 22 enables 4 CRS-axis flags với hyperparams cụ thể — config dataclass phải có đủ. Xem §5.3 cho full list |

### 4.2 Architectural decisions cần re-confirm

| Decision cũ | Vẫn đúng? |
|-------------|-----------|
| Use CoR-GS `diff_gaussian_rasterization` (not gsplat) | ✅ Đúng — Phase 22 dùng renderer này |
| CrsGaussianModel subclass nerfstudio Model (not SplatfactoModel) | ✅ Đúng |
| `get_param_groups()` return empty → CoR-GS optimizer riêng | ✅ Đúng |
| DropAnSH inline trong get_outputs (snapshot opacity + features_rest) | ✅ Đúng — Phase 22 pattern |
| Camera adapter ns Cameras → CoR-GS MiniCam | ✅ Đúng |

---

## 5. Plan implementation rewrite

### 5.1 Files cần REWRITE (5 callback modules + model)

```
crsgaussian_plugin/
├── corgs_imports.py                  ✅ Giữ (sys.path setup)
├── corgs_dataparser.py               ✅ Giữ (load RoMa PLY + inject metadata)
├── camera_adapter.py                 ✅ Giữ (ns → MiniCam)
├── crsgaussian_config.py             ✏️ Update (add ALL Phase 22 hyperparams)
├── crsgaussian_model.py              ✏️ MAJOR rewrite (full Phase 22 loop)
├── depth_loss.py                     ✅ Giữ (Pearson)
├── dropansh.py                       ✅ Giữ (anchor + SH dropout)
└── callbacks/
    ├── corgs_optimizer.py            ✏️ Rewrite (use update_learning_rate)
    ├── densify.py                    ✏️ MAJOR rewrite (LFCF + kwargs + size=None)
    ├── opacity_decay.py              ✏️ Rewrite (USE method, separate reset)
    ├── opacity_reset.py              🆕 NEW (Phase 22 exact timing)
    ├── crs_update.py                 🆕 NEW (D_cycle + S_stability + EMA)
    ├── sh_freeze.py                  🆕 NEW (Phase 8c — biggest winner)
    └── sh_stability.py               🆕 NEW (EMA tracking post-optimizer)
```

### 5.2 Callback order (CRITICAL — match Phase 22 invariants)

```
Nerfstudio Trainer cycle:
  loss.backward()                              ← grads
  ns optimizer.step (no-op, empty groups)

  callbacks AFTER_TRAIN_ITERATION (em define order via priority):
    1. crs_update                              ← F (mỗi 100 iter sau 1000)
    2. densify                                 ← G (mỗi 100 iter, 500 < iter < 5000)
    3. sh_freeze                               ← H (zero grads cho Gaussian CRS low)
    4. corgs_optimizer_step                    ← I (apply grads + zero_grad)
    5. sh_stability                            ← J (EMA capture post-update)
    6. opacity_decay                           ← K (×0.999 mỗi iter sau 500)
    7. update_learning_rate                    ← L (in corgs_optimizer)
    8. opacity_reset                           ← M (conditional 501/3501/6501/9501)
```

### 5.3 Config dataclass — add ALL Phase 22 hyperparams

```python
@dataclass
class CrsGaussianModelConfig(ModelConfig):
    # ── Core training (Phase 22 PROTOCOL) ──
    iterations: int = 10000
    resolution: int = 8
    n_views: int = 3
    random_background: bool = True
    sh_degree: int = 3

    # ── Densify ──
    use_densify: bool = True
    densify_from_iter: int = 500
    densify_until_iter: int = 5000        # NOT 15000
    densify_grad_threshold: float = 0.0005  # NOT 0.0002
    densification_interval: int = 100
    prune_threshold: float = 0.005

    # ── Opacity reset ──
    opacity_reset_interval: int = 3000
    start_sample_pseudo: int = 500         # cho reset timing offset

    # ── M_DEPTHCFG ──
    use_depth_prior: bool = True
    depth_loss_weight: float = 0.05
    crs_update_interval: int = 100
    crs_ema_decay: float = 0.3
    T_warmup: int = 1000

    # ── M_DCYCLE ──
    use_d_cycle: bool = True
    d_cycle_warmup: int = 1000
    d_cycle_sigma: float = 5.0
    d_cycle_update_freq: int = 100

    # ── M_SHFREEZE ──
    use_crs_modulated_sh_freeze: bool = True
    crs_freeze_start: int = 1000
    crs_freeze_tau: float = 0.5

    # ── M_SHREL ──
    use_sh_reliability: bool = True
    sh_stability_warmup: int = 1000
    sh_stability_ema_beta: float = 0.95
    crs_w_s: float = 0.33

    # ── M_DROP ──
    use_dropansh: bool = True
    dropansh_pa: float = 0.02
    dropansh_psh: float = 0.2

    # ── M_OPACITY ──
    use_opacity_decay: bool = True
    opacity_decay_factor: float = 0.999    # 0.999 for 10k iter

    # ── M_EFA (LFCF + AbsGS) ──
    use_lfcf: bool = True
    lfcf_init_scaling_max: float = 1.5
    lfcf_init_scaling_min: float = 1.0
    lfcf_last_scaling_max: float = 1.0
    lfcf_pow: float = 1.0
    lfcf_splitting_ub: float = 1.0
    lfcf_interval_times: int = 2
    lfcf_tolerance: float = 1e-5
    lfcf_diffscale: bool = True
    absdensify: bool = True

    # ── CoR-GS optimizer (Phase 22 default) ──
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    percent_dense: float = 0.01
    lambda_dssim: float = 0.2

    # ── Tau (default — KHÔNG dùng vì use_crs_pruning=False Phase 22) ──
    tau_crs: float = 0.35
    tau_isolated: float = 0.1
```

---

## 6. Acceptance criteria

### 6.1 Smoke test fern 10k (single scene verify)

| Metric | Target | Source |
|--------|--------|--------|
| Train PSNR | ≥ 25 dB | Phase 22 standalone fern ~26 |
| Eval PSNR | ≥ 21 dB | Phase 22 standalone fern 23.84 (cho phép -2.8 buffer adapter) |
| N_gauss final | 80k-200k | Densify ACTIVE |
| Reset opacity calls | 4 (iter 501, 3501, 6501, 9501) | Phase 22 timing |
| LFCF iter count | 23 (iter 600, 800, ..., 5000 in densify range mỗi 200) | use_lfcf=True |
| CRS update count | ~90 (iter 1000 → 10000 mỗi 100) | use_d_cycle=True |
| SH freeze calls | **exact 9000** (iter 1001 → 10000) | use_crs_modulated_sh_freeze=True. Verified `sh_freeze.py:39-40` skip nếu `iter <= freeze_start=1000` → applied 9000 lần |

### 6.2 Full benchmark 4 scene

| Scene | Phase 22 standalone | Plug-in target (≥95%) |
|-------|---------------------|----------------------|
| fern | 23.84 | ≥ 22.6 |
| horns | 21.08 | ≥ 20.0 |
| fortress | 25.57 | ≥ 24.3 |
| flower | 21.41 | ≥ 20.3 |
| **Mean** | **22.97** | **≥ 21.8** |

Compare vs Splatfacto baseline (~16.10) → ≥ **+5.7 dB headline** cho defense.

---

## 7. Status implementation hiện tại (em đã sai gì)

| Component | Status | Action |
|-----------|--------|--------|
| corgs_imports.py | ✅ OK | Giữ |
| corgs_dataparser.py | ✅ OK | Giữ |
| camera_adapter.py | ✅ OK | Giữ |
| crsgaussian_config.py | ❌ Sai params | REWRITE với Phase 22 exact values |
| crsgaussian_model.py | ⚠️ Partial | Rewrite ordering + remove em's opacity logit math |
| depth_loss.py | ✅ OK | Giữ |
| dropansh.py | ✅ OK | Giữ |
| callbacks/corgs_optimizer.py | ✅ OK | Giữ |
| callbacks/densify.py | ⚠️ Missing kwargs | Rewrite — add LFCF kwargs + size=None |
| callbacks/opacity_decay.py | ❌ Sai math | Rewrite — call `gaussians.opacity_decay()` method |
| callbacks/opacity_reset.py | ❌ Missing | NEW — Phase 22 exact timing |
| callbacks/crs_update.py | ❌ Missing | NEW — wrap utils.crs.crs_module.update_crs |
| callbacks/sh_freeze.py | ❌ Missing | NEW — wrap utils.crs.sh_freeze.apply_crs_modulated_sh_freeze |
| callbacks/sh_stability.py | ❌ Missing | NEW — wrap utils.crs.sh_stability.update_sh_stability |

---

## 8. Câu hỏi quan trọng cần anh confirm trước rewrite

### Q1: PSNR target — em chấp nhận drop bao nhiêu vì adapter overhead?

Phase 22 standalone fern = 23.84. Plug-in chấp nhận:
- **(A) Strict** ≥ 22.6 (95%) — chặt chẽ, có thể fail khi optimizer interaction khác
- **(B) Relaxed** ≥ 20.0 (84%) — cho phép buffer adapter overhead
- **(C) Just beat Splatfacto** ≥ 19.5 — minimum cho defense

### Q2: Skip CRS-axis modules (D_cycle + SH freeze + S_stability)?

Per Phase 23 memory: CRS-axis WASH trên dense init (RoMa v1 backbone). Em có thể:
- **(A) Implement đầy đủ** — match Phase 22 standalone 100% (5-7 ngày)
- **(B) Skip CRS-axis** — chỉ implement 3 cross-backbone-stable pillars: depth+CRS via depth-only, DropAnSH, Opacity decay + RoMa + LFCF + AbsGS. PSNR có thể tương đương vì CRS-axis wash trên dense. (2-3 ngày)

### Q3: Implementation strategy — full rewrite vs incremental fix?

- **(A) Full rewrite** — delete cũ, viết lại từ scratch theo doc này (chính xác nhất)
- **(B) Incremental fix** — sửa từng bug em đã liệt kê (nhanh hơn nhưng risk miss something)

---

## 9. Liên quan

- [ARCHITECTURE.md](../ARCHITECTURE.md) — kiến trúc CRSGaussian đầy đủ
- [08_path_a_implementation_plan.md](08_path_a_implementation_plan.md) — plan cũ (deprecated B3+B4 sections)
- `scripts/p22_pilot_run.sh` + `scripts/p20_ablation_dense_run.sh` — exact Phase 22 CLI
- Memory `project_phase22_roma_v1_pilot.md` — PSNR results
- Memory `project_phase23_ablation_v1.md` — 4 contributions framing
- Memory `project_phase24_trim_verify_v1.md` — A3-TRIM 8-module recipe

---

**Doc này chỉ là PLAN — em đợi anh confirm Q1+Q2+Q3 trước khi rewrite code.**

---

## 10. RESULTS (2026-06-04) — ✅ B3 PASS

### 10.1 Eval PSNR fern 10k single seed

```json
{
  "experiment_name": "fern",
  "method_name": "crsgaussian",
  "results": {
    "psnr": 25.084308624267578,
    "num_rays_per_sec": 2869569.5,
    "fps": 15.132386207580566
  }
}
```

| Method | PSNR fern | Note |
|--------|-----------|------|
| Splatfacto vanilla | ~16-17 | Nerfstudio baseline |
| Splatfacto + RoMa init (Phase A1) | 19.08 | Just dense init contribution |
| Phase 22 standalone N=24 | 23.84 | Multi-seed mean (reference target) |
| **Plug-in CrsGaussian (single seed)** | **25.08** | **+1.24 vs Phase 22 standalone** |

**Caveat**: Single-seed single-scene. 3DGS atomicAdd variance ±1.3 dB → có thể là lucky run. Multi-seed × 4-scene paired = next phase (B4 benchmark).

### 10.2 Checkpoint save/load journey (Pipeline.load_state_dict path)

3 bugs critical em đã fix qua workflow PORT-not-redesign:

1. **extra_state native PyTorch mechanism KHÔNG fire qua Pipeline** (base_pipeline.py:100-125 wraps Model.load_state_dict in strict=True+fallback try/except). Fix: override Model.load_state_dict manually extract `_extra_state` key before super (Splatfacto pattern splatfacto.py:343-356).

2. **`confidence` attribute KHÔNG được capture() save** (gaussian_model.py:169-184 only saves 13 fields). Sau restore _xyz resize N=24k→47k nhưng confidence vẫn N_init. Renderer line 120 `torch.ones_like(pc.confidence)` shape mismatch → CUDA illegal memory access. Fix: re-init `gaussians.confidence = torch.ones((N, 1), device="cuda")` sau restore.

3. **torch.load(map_location="cpu") → CPU tensors → CUDA kernel access CPU mem → illegal access**. Fix: `.detach().cuda()` per tuple element + wrap nn.Parameter cho 6 per-Gaussian indices (1-6) trong tuple để Adam optimizer (training_setup line 370) accept leaf Parameter.

### 10.3 Files final

**9 files plug-in folder**:
```
crsgaussian_plugin/
├── __init__.py
├── crsgaussian_config.py        # Phase 22 hyperparams verbatim
├── crsgaussian_model.py         # Main Model — 600 LOC
├── camera_adapter.py            # ns Cameras → CoR-GS Camera/MiniCam
├── corgs_dataparser.py          # downscale_factor=8 + RoMa PLY load
├── corgs_imports.py             # sys.path setup
├── depth_loss.py                # Pearson depth loss
├── dropansh.py                  # Legacy from B1 (renderer built-in dùng cho B3)
├── callbacks/
│   ├── corgs_optimizer.py       # optimizer.step + update_lr
│   ├── densify.py               # LFCF + AbsGS + size_threshold=None
│   ├── opacity_decay.py         # call gaussians.opacity_decay() method
│   ├── opacity_reset.py         # Phase 22 timing (501, 3501, 6501, 9501)
│   ├── crs_update.py            # D_cycle + S_stability EMA
│   ├── sh_freeze.py             # Phase 8c (+0.189 winner)
│   └── sh_stability.py          # EMA post-optimizer
└── scripts/
    └── benchmark_phase22_fern_10k.sh
```

### 10.4 Next steps (B4 multi-scene benchmark)

- Run 4-scene 10k benchmark (fern + horns + fortress + flower) parallel 2 GPU
- Multi-seed (3 seeds × 4 scenes paired N=12) cho defense-grade evidence
- Compare vs Phase 22 standalone Phase 22 v1 N=24
- Update memory `project_phase22_roma_v1_pilot.md` với plug-in results
- Update [07_nerfstudio_defense_framing.md](07_nerfstudio_defense_framing.md) với actual numbers

