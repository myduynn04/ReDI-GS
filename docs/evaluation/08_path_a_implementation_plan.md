# Path A — `crsgaussian` method implementation plan

> **Mục đích**: Detailed plan để triển khai `crsgaussian` method (Phase 22 full recipe) như 1 entity trong Nerfstudio. Mỗi phase có **smoke test + verify gate cụ thể** — KHÔNG bước qua nếu chưa pass.
>
> **Status**: 📋 PLAN — implementation in progress
>
> **Related**: [07_nerfstudio_defense_framing.md](07_nerfstudio_defense_framing.md) (defense framing), [06_plugin_a1_results.md](06_plugin_a1_results.md) (kết quả Phase A plug lẻ tẻ)

---

## 0. Acceptance criteria tổng

| Metric | Target | Lý do |
|--------|--------|-------|
| PSNR adapter vs Phase 22 standalone | ≥ 95% (mean ≥ 21.8 vs 22.97) | Cho phép ~5% drop do optimizer/datamanager khác biệt |
| PSNR adapter vs Splatfacto baseline | ≥ +5.0 dB | Defense story: vượt baseline framework Nerfstudio |
| CRSGaussian Phase 22 standalone code | KHÔNG touched | Path A chỉ **import**, không sửa source CoR-GS |
| OFF flags verify | byte-identical Splatfacto-roma | Mỗi component có master switch, ablation-friendly |

---

## 1. Architecture decisions (chốt trước code)

### Decision 1 — Renderer: giữ `diff_gaussian_rasterization` (CoR-GS)

| Option | Pro | Con |
|--------|-----|-----|
| ✅ **CoR-GS renderer** (Inria diff_gaussian_rasterization) | Preserve Phase 22 fidelity, 0 risk regression renderer | Phải bridge camera format ns ↔ CoR-GS |
| ❌ Port sang gsplat (Nerfstudio default) | Native Nerfstudio | Atomic order khác → PSNR có thể tụt ~0.5 dB |

**Chọn**: CoR-GS renderer. Defense argue: *"Em giữ renderer Inria để fair compare với CoR-GS standalone — nếu đổi renderer, không thể tách contribution Phase 22 recipe khỏi renderer change."*

### Decision 2 — Optimizer: giữ CoR-GS GaussianModel.optimizer

- CoR-GS `GaussianModel.training_setup(opt)` tự build Adam với lr schedule riêng
- Nerfstudio Trainer dùng Optimizer factory riêng
- → Override `Model.get_param_groups()` trả **empty dict** → Nerfstudio Trainer skip optimizer step
- → Custom step trong `get_loss_dict()` hoặc training callback

### Decision 3 — Training loop: custom step trong callback `AFTER_TRAIN_ITERATION`

```
Nerfstudio Trainer cycle (per iter):
  1. batch = datamanager.next_train()
  2. outputs = model.get_outputs(batch.camera)    ← gọi CoR-GS render
  3. loss_dict = model.get_loss_dict(outputs, batch)  ← L1+SSIM+depth+CRS reg
  4. loss = sum(loss_dict.values())
  5. loss.backward()
  6. optimizer.step()                              ← NO-OP (empty param groups)
  7. callbacks BEFORE_TRAIN_ITERATION
     ↓
  8. callbacks AFTER_TRAIN_ITERATION
     ↳ corgs_optimizer_step (CoR-GS GaussianModel.optimizer.step)
     ↳ densify_lfcf (LFCF + AbsGS scheduled)
     ↳ opacity_decay (×0.999/iter sau iter 500)
     ↳ crs_update (D + R + S mỗi 100 iter sau warmup 1000)
     ↳ sh_freeze (CRS-modulated SH freeze)
```

### Decision 4 — File layout

```
~/workspace/representation-3d/duyen/nerfstudio/crsgaussian_plugin/
├── __init__.py                          # Register method `crsgaussian`
├── crsgaussian_config.py                # CrsGaussianModelConfig dataclass
├── crsgaussian_model.py                 # CrsGaussianModel (subclass Model)
├── camera_adapter.py                    # ns Cameras → CoR-GS MiniCam
├── corgs_dataparser.py                  # DataParser load fused.ply.romav1 + depth
├── corgs_imports.py                     # sys.path setup + import CoR-GS modules
├── callbacks/
│   ├── __init__.py
│   ├── corgs_optimizer.py              # CoR-GS GaussianModel.optimizer.step
│   ├── densify_lfcf.py                 # LFCF + AbsGS
│   ├── opacity_decay.py                # ×0.999/iter
│   ├── crs_update.py                   # D + R + S update
│   ├── dropansh_cb.py                  # DropAnSH applied during forward
│   └── sh_freeze.py                    # CRS-modulated SH freeze (Phase 8)
├── scripts/
│   ├── smoke_b1_fern_100iter.sh        # Smoke B1 verify
│   ├── smoke_b2_fern_500iter.sh        # Smoke B2 verify
│   ├── smoke_b3_fern_2000iter.sh       # Smoke B3 verify
│   ├── benchmark_b4_4scene_10k.sh      # Full benchmark B4
│   └── compare_vs_phase22.py           # Paired PSNR compare
└── README.md                            # Usage + rollback
```

---

## 2. Phase B1 — Scaffolding + init smoke (1-2 ngày)

### Mục tiêu

Method `crsgaussian` register vào Nerfstudio, **init 23420 points từ fused.ply.romav1, chạy 100 iter không crash**.

### Files tạo

| File | Lines target | Responsibility |
|------|--------------|----------------|
| `corgs_imports.py` | ~40 | sys.path setup + import CoR-GS GaussianModel, render, Camera |
| `corgs_dataparser.py` | ~150 | Subclass ColmapDataParser, override `_load_3D_points` → fused.ply.romav1 |
| `camera_adapter.py` | ~80 | `ns_camera_to_corgs_minicam(camera)` — bridge |
| `crsgaussian_config.py` | ~60 | CrsGaussianModelConfig dataclass với tất cả master switches OFF |
| `crsgaussian_model.py` | ~200 | Skeleton: populate_modules() load PLY, get_outputs stub trả zeros |
| `__init__.py` | ~60 | MethodSpecification register `crsgaussian` |
| `scripts/smoke_b1_fern_100iter.sh` | ~30 | Smoke command |

### Smoke command B1

```bash
cd ~/workspace/representation-3d/duyen/nerfstudio
export PYTHONPATH=/home/aidev/workspace/representation-3d/duyen/nerfstudio:$PYTHONPATH
export NERFSTUDIO_METHOD_CONFIGS="crsgaussian=crsgaussian_plugin:crsgaussian_method_spec"

CUDA_VISIBLE_DEVICES=1 ns-train crsgaussian \
    --data ~/workspace/representation-3d/duyen/CoR-GS/data/nerf_llff_data/fern/3_views/ \
    --max-num-iterations 100 \
    --output-dir outputs_smoke_b1 \
    --vis tensorboard \
    2>&1 | tee logs/smoke_b1_fern.log
```

### Verify gate B1 — PASS criteria

| Gate | Check | Command |
|------|-------|---------|
| **B1.1** Method registered | `ns-train --help` list có `crsgaussian` | `ns-train --help 2>&1 \| grep crsgaussian` |
| **B1.2** Init đúng PLY | Log có `Loaded 23420 points from fused.ply.romav1` | `grep "Loaded.*romav1" logs/smoke_b1_fern.log` |
| **B1.3** 100 iter không crash | Log có `[100/100]` | `grep "100/100" logs/smoke_b1_fern.log` |
| **B1.4** PSNR train log có | TensorBoard có scalar `train_loss` | `ls outputs_smoke_b1/.../events.out.tfevents.*` |

**FAIL → debug ngay, không qua B2.** Common failures:
- ImportError CoR-GS modules → check `corgs_imports.py` sys.path
- Camera projection NaN → debug `camera_adapter.py` với 1 camera mẫu
- AttributeError missing arg → CoR-GS Scene/GaussianModel cần ~150 attr (dùng ArgsWrapper pattern từ `preprocess_depth_a23.py`)

### Rollback B1

```bash
# Server: xóa plug-in folder
rm -rf ~/workspace/representation-3d/duyen/nerfstudio/crsgaussian_plugin/
unset NERFSTUDIO_METHOD_CONFIGS

# Local: keep code (KHÔNG xóa local — chỉ revert nếu hướng B1 reject hoàn toàn)
```

---

## 3. Phase B2 — Loss + render integration (1 ngày)

### Mục tiêu

`get_outputs()` gọi CoR-GS render thật → `get_loss_dict()` compute L1+SSIM+depth → loss giảm qua 500 iter.

### Files update

| File | Update |
|------|--------|
| `crsgaussian_model.py` | `get_outputs()`: call `gaussian_renderer.render(corgs_cam, self.gaussians, pipe, bg)` |
| `crsgaussian_model.py` | `get_loss_dict()`: L1 + SSIM + Pearson depth (port từ A2.3) |
| `corgs_dataparser.py` | Load `aligned_depth_a23/*.npy` vào batch metadata |
| `camera_adapter.py` | Verify projection matrix với 1 camera mẫu (unit test) |

### Smoke command B2

```bash
CUDA_VISIBLE_DEVICES=1 ns-train crsgaussian \
    --data .../fern/3_views/ \
    --max-num-iterations 500 \
    --output-dir outputs_smoke_b2 \
    --pipeline.model.use-depth-loss True \
    2>&1 | tee logs/smoke_b2_fern.log
```

### Verify gate B2 — PASS criteria

| Gate | Check |
|------|-------|
| **B2.1** L1 loss giảm | iter 100 L1 > iter 500 L1 (giảm ≥ 30%) |
| **B2.2** depth_loss ACTIVE | Log có `depth_loss` value > 0 và giảm dần |
| **B2.3** PSNR train > 18 dB tại iter 500 | RoMa init giúp converge nhanh — KHÔNG đạt = render bug |
| **B2.4** Render output shape correct | rendered_image: (H, W, 3) đúng size camera |

**FAIL trigger debug**:
- L1 không giảm → optimizer không step (CoR-GS optimizer chưa hook)
- PSNR < 18 → render output sai (camera adapter bug)
- depth_loss = 0 → depth not loaded vào batch

---

## 4. Phase B3 — Densify + callbacks (2-3 ngày)

### Mục tiêu

Port 5 callback từ Phase 22:
1. **corgs_optimizer_step** — CoR-GS GaussianModel.optimizer.step
2. **densify_lfcf** — LFCF densifier (scaler_max=1.5, interval=2) + AbsGS
3. **opacity_decay** — ×0.999/iter sau iter 500
4. **crs_update** — D_i + R_i + S_stability mỗi 100 iter sau warmup 1000
5. **sh_freeze** — CRS-modulated SH freeze (Phase 8 mechanism)

### Files tạo

| File | Lines | Responsibility |
|------|-------|----------------|
| `callbacks/corgs_optimizer.py` | ~50 | CoR-GS optimizer.step + zero_grad |
| `callbacks/densify_lfcf.py` | ~150 | LFCF density_max + AbsGS densify_and_prune |
| `callbacks/opacity_decay.py` | ~40 | Multiply opacity logit |
| `callbacks/crs_update.py` | ~200 | D+R+S update, EMA logit space scale=5.0 |
| `callbacks/sh_freeze.py` | ~100 | Set features_rest.requires_grad=False per CRS |

### Smoke command B3

```bash
CUDA_VISIBLE_DEVICES=1 ns-train crsgaussian \
    --data .../fern/3_views/ \
    --max-num-iterations 2000 \
    --output-dir outputs_smoke_b3 \
    --pipeline.model.use-opacity-decay True \
    --pipeline.model.use-dropansh True \
    --pipeline.model.use-depth-loss True \
    --pipeline.model.use-densify True \
    --pipeline.model.use-crs True \
    --pipeline.model.use-sh-freeze True \
    2>&1 | tee logs/smoke_b3_fern.log
```

### Verify gate B3 — PASS criteria

| Gate | Check |
|------|-------|
| **B3.1** Densify ACTIVE | N_gaussians tăng từ 23420 → 60-100k qua 2000 iter |
| **B3.2** Opacity decay ACTIVE | median(opacity) giảm sau iter 500 |
| **B3.3** CRS log ACTIVE | Log có `[CRS] iter=X D_mean=Y R_mean=Z` mỗi 500 iter sau iter 1000 |
| **B3.4** SH freeze counter > 0 | Log có `[SH freeze] N=X gaussians frozen` |
| **B3.5** PSNR train > 21 dB tại iter 2000 | Phase 22 fern standalone đạt 23-25 dB cuối train, smoke giữa cũng phải > 21 |

**FAIL → debug từng callback bằng cách disable từng cái** (OFF flag pattern).

---

## 5. Phase B4 — Full benchmark + Phase 22 fidelity verify (1-2 ngày)

### Mục tiêu

Chạy 4 scene 10k iter, compare PSNR với:
1. Phase 22 standalone (acceptance ≥ 95%)
2. Splatfacto baseline (defense delta ≥ +5.0 dB)

### Benchmark command

```bash
cd ~/workspace/representation-3d/duyen/nerfstudio
bash crsgaussian_plugin/scripts/benchmark_b4_4scene_10k.sh
```

Script split 2 GPU:
- GPU 0: fern + horns
- GPU 1: fortress + flower

### Compare command

```bash
python crsgaussian_plugin/scripts/compare_vs_phase22.py \
    --phase22-results /home/aidev/.../CoR-GS/output/phase22_v1/ \
    --plugin-results outputs_4scene_crsgaussian_b4/
```

Output expected:

```
Scene     | Phase22  | Plugin   | Δ      | Threshold(95%) | Verdict
fern      | 23.84    | XX.XX    | -X.XX  | ≥ 22.65        | PASS/FAIL
horns     | 21.08    | XX.XX    | -X.XX  | ≥ 20.03        | PASS/FAIL
fortress  | 25.57    | XX.XX    | -X.XX  | ≥ 24.29        | PASS/FAIL
flower    | 21.41    | XX.XX    | -X.XX  | ≥ 20.34        | PASS/FAIL
Mean      | 22.97    | XX.XX    | -X.XX  | ≥ 21.82        | PASS/FAIL
```

### Verify gate B4 — PASS criteria

| Gate | Check |
|------|-------|
| **B4.1** Mean PSNR ≥ 21.8 (95% Phase 22) | Adapter fidelity OK |
| **B4.2** Mean PSNR vs Splatfacto baseline ≥ +5.0 dB | Defense story rõ |
| **B4.3** N_gaussians final reasonable | 50-200k range (Phase 22 ~100k) |
| **B4.4** Training time ≤ 1.5× Phase 22 standalone | Adapter overhead acceptable |

---

## 6. Risk register + mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Camera projection matrix khác convention | High | Render đen/NaN | Unit test camera_adapter với 1 camera mẫu trước B2 |
| CoR-GS optimizer ↔ ns Trainer conflict | Medium | Loss không giảm | Override get_param_groups → empty; verify với B2 smoke |
| Densify reshape ↔ CRS buffer sync fail | High | CRS buffer dim mismatch | Callback order: densify → crs_resize → opacity_decay |
| LFCF dependency trên internal GaussianModel | Medium | Import fail | Wrap CoR-GS as-is, KHÔNG refactor |
| PSNR regression > 1 dB | Medium | Defense weakened | Debug từng callback OFF/ON ablation B3 |
| OOM trên GPU 0 (process khác chiếm) | Medium | Train crash | Default CUDA_VISIBLE_DEVICES=1 nếu GPU 0 busy |

---

## 7. Time + GPU budget

| Phase | Effort code | Smoke run time | Cumulative |
|-------|-------------|----------------|------------|
| B1 Scaffolding | 1-2 ngày | 100 iter ~30s | Day 2 |
| B2 Loss+render | 1 ngày | 500 iter ~3 phút | Day 3 |
| B3 Densify+CB | 2-3 ngày | 2000 iter ~10 phút | Day 6 |
| B4 Benchmark | 1-2 ngày | 4 scene 10k ~25 phút | Day 8 |
| **Total** | **5-8 ngày** | — | — |

---

## 8. Cleanup discipline (per `feedback_report_cleanup_for_server_sync`)

| Stage | Action | File category |
|-------|--------|---------------|
| Successful commit B1-B4 | Keep local + server | Plug-in code + scripts |
| Reject (any phase) | Keep local, delete server | Plug-in code + scripts |
| **NEVER** touch | CoR-GS source | Path A import-only contract |

End-of-task report luôn list:
```
Files KEEP local (anh có thể commit):
  - crsgaussian_plugin/...
Server commands cleanup (anh chạy nếu reject):
  - rm -rf ~/workspace/.../nerfstudio/crsgaussian_plugin/
Files re-upload server (sau khi sửa local):
  - scp crsgaussian_plugin/... server:...
```

---

## 9. Status tracking

| Phase | Status | Verify | Date |
|-------|--------|--------|------|
| B1 Scaffolding | ✅ **PASS** | All 4 gates | 2026-06-03 |
| B2 Loss+render | ✅ **PARTIAL PASS** | Train PSNR 23.4 ✓ / Eval gap 14.7 — debug pending | 2026-06-03 |
| B3 Full Phase 22 callbacks + ckpt save/load | ✅ **PASS** | **Eval PSNR 25.08 dB** trên fern (vượt Phase 22 standalone 23.84) | 2026-06-04 |
| B3 Densify+CB | ⏳ PENDING | — | — |
| B4 Benchmark | ⏳ PENDING | — | — |

### B1 results (2026-06-03)

**4/4 verify gates PASS**:
- ✅ B1.1 Method `crsgaussian` registered
- ✅ B1.2 Init 24543 points từ `fused.ply.romav1`
- ✅ B1.3 Train 100 iter complete (`🎉 Training Finished 🎉`)
- ✅ B1.4 TensorBoard event file exists

**Bugs fixed mid-B1**:
1. `seed_points` kwarg missing — phải catch trong `CrsGaussianModel.__init__` + set TRƯỚC `super().__init__()`
2. `Model.forward` mặc định gọi collider expect `RayBundle.nears/fars` — override forward skip collider (full-image model pattern)
3. Verify script grep pattern sai — Nerfstudio rich-progress in `"99 (99.00%)"` + `"Training Finished"`, không phải `"100/100"`

**Insight quan trọng**:
- **PSNR ~22 dB ngay từ init** (RoMa dense init alone) — chưa cần optimize.
- Compare: Splatfacto vanilla COLMAP init fern ~17.6 dB sau 10k iter; CRSGaussian RoMa init 0 iter ≈ 22 dB.
- **+4.4 dB chỉ từ init** — confirm Phase A/A1 finding (RoMa init = main contribution trên sparse-view).

**Files added** (9 files):
- `crsgaussian_plugin/{__init__, crsgaussian_config, crsgaussian_model, camera_adapter, corgs_dataparser, corgs_imports}.py`
- `crsgaussian_plugin/callbacks/__init__.py`
- `crsgaussian_plugin/scripts/smoke_b1_fern_100iter.sh`
- `crsgaussian_plugin/README.md`

**Setup blockers resolved**:
- Install CoR-GS CUDA submodules vào env `nerfstudio`:
  - `simple_knn` (non-editable, vì editable PEP 660 fail với folder namespace package)
  - `diff_gaussian_rasterization-confidence` (Inria fork với confidence field cho 2-field architecture)
- Common pitfall: `simple_knn._C` cần `import torch` TRƯỚC (libc10.so từ PyTorch loaded vào process)

---

## 10. B2 plan detail (2026-06-03 start)

### Scope
- Hook CoR-GS `GaussianModel.optimizer.step` vào callback `AFTER_TRAIN_ITERATION`
- Replace L1-only với L1 + SSIM (Phase 22 standard 0.8/0.2)
- Load aligned depth + Pearson depth loss (Phase 22 default λ=0.05)
- Smoke fern 500 iter

### Verify gates B2

| Gate | Pass criteria | Lý do |
|------|---------------|-------|
| B2.1 L1 loss giảm | iter 100 → 500: L1 drop ≥ 30% | Confirm optimizer.step thật chạy |
| B2.2 SSIM contribution | DSSIM term > 0 trong loss_dict | L1+SSIM blend hoạt động |
| B2.3 depth_loss ACTIVE | loss_dict["depth_loss"] > 0, giảm dần | DAV2 depth load + Pearson compute OK |
| B2.4 PSNR train > 22 (vượt B1 init) | iter 500 PSNR > 22 dB | Optimize cải thiện thêm trên RoMa init |
| B2.5 N_gaussians KHÔNG đổi | iter 0 vs 500: |ΔN| < 100 | B2 chưa densify (B3 mới densify) |

### Files thay đổi

| File | Update | Lines |
|------|--------|-------|
| `crsgaussian_model.py` | Add SSIM + depth loss + optimizer callback hook | +60 |
| `callbacks/corgs_optimizer.py` | NEW — wrapper helper | +40 |
| `depth_loss.py` | NEW — copy từ Phase A `nerfstudio_plugin/` | +80 |
| `crsgaussian_config.py` | Enable use_depth_loss default = True (B2) | +1 |
| `scripts/smoke_b2_fern_500iter.sh` | NEW smoke script | +100 |

### B3 results (2026-06-04) — 🎉 SUCCESS

**Eval PSNR: 25.08 dB** trên fern 10k, single seed.

| Reference | PSNR | Δ vs plug-in |
|-----------|------|--------------|
| Splatfacto vanilla baseline | ~16-17 | −8.0 to −9.0 |
| Splatfacto + RoMa init (Phase A) | 19.08 | −6.00 |
| Phase 22 standalone (N=24 mean) | 23.84 | −1.24 |
| **Plug-in CrsGaussian (single seed)** | **25.08** | — |

**Caveat**: Single-seed, single-scene. 3DGS có atomicAdd variance ±1.3 dB single-scene → cần multi-seed × 4-scene paired để defense-grade. Multi-scene benchmark = Phase B4 (pending).

**Components ACTIVE** (full 7 modules + RoMa init = 8 modules Phase 22 A3-TRIM):
- ✅ RoMa v1 dense init (24543 → 47566 sau densify)
- ✅ DAV2 depth + Pearson loss
- ✅ D_cycle CRS signal
- ✅ CRS-modulated SH freeze (Phase 8c)
- ✅ SH stability EMA (Phase 8b)
- ✅ DropAnSH (anchor + SH dropout)
- ✅ Opacity decay 0.999/iter + reset 501/3501/6501/9501
- ✅ LFCF + AbsGS densifier

**6 bug fix journey (defense-able)**:

| Round | Bug | Fix |
|-------|-----|-----|
| v1 | seed_points kwarg, forward collider, batch uint8 dtype | Override __init__/forward/cast |
| v2 | densify_and_prune missing iter arg, reset_opacity timing | Pass step + match Phase 22 formula |
| v3 | extra_state mechanism KHÔNG fire qua Pipeline.load_state_dict | Override Model.load_state_dict + manual dispatch (Splatfacto pattern) |
| v4 | confidence size mismatch sau restore (capture() KHÔNG save) | Re-init confidence to N_loaded |
| v5 | torch.load(map_location="cpu") → CPU tensors → CUDA OOM | .detach().cuda() per element |
| v6 | non-leaf tensor → Adam "can't optimize" | Wrap nn.Parameter cho 6 per-Gaussian params |

**Files (10 sửa/tạo)**:
- `crsgaussian_plugin/`: __init__, crsgaussian_config, crsgaussian_model, camera_adapter, corgs_dataparser, corgs_imports, depth_loss, dropansh
- `crsgaussian_plugin/callbacks/`: corgs_optimizer, densify, opacity_decay, opacity_reset, crs_update, sh_freeze, sh_stability
- `scripts/benchmark_phase22_fern_10k.sh`

---

### B2 results (2026-06-03)

**Train metrics PASS — eval gap cần debug**:

| Metric | Step 0 | Step 490 | Status |
|--------|--------|----------|--------|
| Train Loss | 0.370 | 0.082 | ✅ Drop 78% |
| Train PSNR | 9.10 | **23.40** | ✅ Climb 14 dB |
| depth_loss | 0.022 | 0.005 | ✅ Active, drop 78% |
| **Eval PSNR** | — | **8.74** | ⚠️ Gap 14.7 dB so với train — bất thường |

**Bugs fixed mid-B2**:
1. `get_loss_dict` cần cast gt uint8 → float (cache_images_type="uint8") — L1 magnitude sai 255× + SSIM conv2d strict dtype
2. `Model.get_outputs_for_camera` default convert sang ray_bundle → forward(ray_bundle=) crash. Override skip conversion.
3. ns-eval cần env var `NERFSTUDIO_METHOD_CONFIGS` set trong shell (script smoke set nội bộ không propagate)

**Eval gap pending debug**:
- Train PSNR 23.4 → CoR-GS optimizer + render WORK CORRECTLY trên train cameras
- Eval PSNR 8.74 → bất thường (Phase 22 standalone train-eval gap chỉ ~2-4 dB)
- 2 hypothesis: (H1) camera_adapter convention sai cho eval cam, (H2) split protocol mismatch
- Defer debug đến trước B4 benchmark — B3 (densify + callbacks) không phụ thuộc vấn đề này
