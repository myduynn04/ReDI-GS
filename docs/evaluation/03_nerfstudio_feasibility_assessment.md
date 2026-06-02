# Nerfstudio + splatfacto — Feasibility Assessment

> Đánh giá tính khả thi của 2 hướng dùng [Nerfstudio splatfacto](https://github.com/nerfstudio-project/nerfstudio) +
> [gsplat](https://github.com/nerfstudio-project/gsplat) làm so sánh open-source production-grade cho defense:
> - **Hướng A — Plug-in**: Nerfstudio làm framework, CRSGaussian là custom method registered qua plugin API
> - **Hướng B — Standalone comparison**: splatfacto vs CRSGaussian, cùng input, cùng eval
>
> Code đọc tại `d:\Dowload\Paper\3D representation\code\nerfstudio\` — không build, không run.
> Cập nhật: 2026-06-01.

---

## 1. Why this matters — context production thực tế (đã verify độc lập)

Khác Brush (open-source app cross-platform) — Nerfstudio + gsplat là **infrastructure production thật** của ngành 3DGS:

| Evidence | Nguồn |
|----------|-------|
| NVIDIA chính thức tích hợp **3DGUT** vào gsplat (04/2025) | [NVIDIA Developer Blog](https://developer.nvidia.com/blog/revolutionizing-neural-reconstruction-and-rendering-in-gsplat-with-3dgut/) |
| **Bentley Systems** (enterprise software construction/infrastructure) tích hợp 3DGS vào iTwin Capture + Cesium ion | [Wearfits industry adoption](https://wearfits.com/articles/gaussian-splatting-twins-industry-adoption) |
| **G-SHARP** = commercial real-time surgical scene modeling app, gsplat làm rasterization backbone (Holoscan SDK) | [gsplat arXiv paper](https://arxiv.org/abs/2409.06765) |
| **AWS Physical AI** infrastructure cho gsplat pipeline (media, retail, manufacturing) | [AWS Physical AI Blog](https://aws.amazon.com/blogs/physical-ai/3d-gaussian-splatting-performant-3d-scene-reconstruction-at-scale/) |
| **XGRIDS LCC Cloud** = commercial SLAM+3DGS service $800/year trên Nerfstudio | [The Future 3D](https://www.thefuture3d.com/software/nerfstudio/) |
| **Meta Project Aria** smart glasses export pipeline → Nerfstudio | [The Future 3D](https://www.thefuture3d.com/software/nerfstudio/) |
| **AMD ROCm** chính thức port gsplat | [AMD ROCm docs](https://rocm.docs.amd.com/projects/gsplat/en/docs-25.11/what-is-gsplat.html) |
| **NHM Vienna** dùng splatfacto restore heritage (Maria Theresia gem bouquet) | [The Future 3D](https://www.thefuture3d.com/software/nerfstudio/) |

→ **Defense story**: plug CRSGaussian vào Nerfstudio = đóng góp module cho infrastructure đang được NVIDIA, Bentley, AWS, Meta, AMD, G-SHARP, XGRIDS dùng. Không phải "research toy".

⚠️ **Caveat (transparent)**: Claim *"gsplat đang vận hành phần lớn triển khai 3DGS doanh nghiệp"* — số "phần lớn" KHÔNG có nguồn cụ thể, NHƯNG evidence trên đủ để bảo "production-tier với đối tác lớn".

---

## 2. Nerfstudio + splatfacto architecture summary

### Tech stack
- **Ngôn ngữ**: Python 3.8+
- **ML framework**: PyTorch (cùng stack CRSGaussian)
- **Rasterizer**: gsplat library (CUDA-accelerated, có hook MCMCStrategy / DefaultStrategy)
- **Build**: `pip install nerfstudio` — không cần compile từ source

### Plugin framework chính thức
[nerfstudio/plugins/registry.py](../../nerfstudio/nerfstudio/plugins/registry.py) — 2 cách register custom method:
1. **Entry point** trong `pyproject.toml` (`nerfstudio.method_configs` group)
2. **Environment variable** `NERFSTUDIO_METHOD_CONFIGS=<name>=<module>:<config>`

Mỗi method là `MethodSpecification(config=TrainerConfig, description=str)` — có sẵn class `SplatfactoModelConfig` để extend.

→ **Anh có thể tạo `crsgaussian` method registered** mà KHÔNG cần fork Nerfstudio. User cài `pip install nerfstudio` + `pip install crsgaussian-nerfstudio-plugin` rồi chạy `ns-train crsgaussian`.

### Splatfacto key API
[nerfstudio/models/splatfacto.py](../../nerfstudio/nerfstudio/models/splatfacto.py):

| Hook | Line | Vai trò | CRSGaussian inject ở đây không |
|------|------|---------|--------------------------------|
| `__init__(seed_points)` | 180 | Init từ tuple `(positions, colors)` — **CỰC DỄ inject RoMa init** | ✅ |
| `populate_modules()` | 189 | Init gauss_params từ seed_points hoặc random | ✅ |
| `get_outputs(camera)` | 485 | Forward pass — render | ✅ |
| `get_loss_dict(outputs, batch)` | 652 | Loss dict — **dễ add custom loss term** | ✅ |
| `get_param_groups()` | 420 | Optimizer param management | ✅ |
| `step_cb(optimizers, step)` | 407 | Per-step callback — **dễ inject opacity decay, CRS update** | ✅ |

### Default config tham chiếu (so với CRSGaussian)
| Param | Splatfacto default | CRSGaussian (Phase 22) |
|-------|--------------------|--------------------------|
| `warmup_length` (refinement off) | 500 | 500 (`densify_from_iter`) |
| `refine_every` (densify interval) | 100 | 100 |
| `densify_grad_thresh` | 0.0008 | 0.0005 |
| `cull_alpha_thresh` | 0.1 | 0.005 |
| `reset_alpha_every` | 30 (× refine_every = mỗi 3000 iter) | dùng `opacity_reset_interval` của CoR-GS |
| `use_absgrad` | **True (default)** ⚠️ | True (Phase 13 contribution) |
| `sh_degree_interval` | 1000 | 500 (CoR-GS) |
| `stop_split_at` | 15000 | `densify_until_iter` = 5000 |
| Iterations total | 30000 | 10000 |
| `strategy` | "default" / "mcmc" (gsplat) | custom (CoR-GS densify + LFCF + AbsGS) |

⚠️ **AbsGS đã có sẵn trong splatfacto** (`use_absgrad: bool = True` mặc định). Đây là 1 trong contribution Phase 13 của CRSGaussian → **không thể claim AbsGS là novel** khi compare với splatfacto. Phải re-frame contribution.

---

## 3. Hướng A — Plug-in: feasibility từng module CRSGaussian

### Bảng feasibility (chi tiết hơn Brush nhiều vì CÙNG Python/PyTorch)

| Module CRSGaussian | Plug-in Splatfacto được? | Cách | Effort |
|---------------------|--------------------------|------|--------|
| **RoMa v1 dense init** | ✅✅✅ Cực dễ | Custom DataParser xuất `seed_points=(xyz, rgb)` → `SplatfactoModel.__init__()` nhận trực tiếp | 🟢 0.5 ngày |
| **DAV2 depth loss** | ✅✅ Dễ | Subclass `SplatfactoModel`, override `get_loss_dict()`, thêm `L_depth` term | 🟢 1 ngày |
| **Opacity decay** | ✅✅ Dễ | Override `step_cb()`, multiply opacity each step | 🟢 0.5 ngày |
| **CRS score per-Gaussian** | ✅ Trung bình | Subclass Model, thêm `_crs_score` attribute trong `gauss_params`, update trong `step_cb()` | 🟡 2-3 ngày |
| **AbsGS densify** | ⚠️ **ĐÃ CÓ SẴN** | `use_absgrad=True` đã default → 0 effort | ✅ Free (nhưng mất novelty) |
| **DropAnSH** | ✅ Trung bình | Override `get_outputs()` mask Gaussians trước rasterize | 🟡 2 ngày |
| **SH freeze CRS-modulated** | ✅ Trung bình | Override `get_param_groups()` + per-step grad zero | 🟡 2 ngày |
| **D_cycle** | ✅ Trung bình | Custom loss render N views compute cycle | 🟡 2-3 ngày |
| **LFCF densifier** | 🟡 Khó nhất | DefaultStrategy là gsplat external. Phải tạo `LfcfStrategy(DefaultStrategy)` subclass, override `step_post_backward()`. **Hoặc fork gsplat** | 🟡 3-5 ngày |

### So sánh effort với Brush
| Module | Brush | Nerfstudio |
|--------|-------|------------|
| RoMa init | ✅ Khả thi qua COLMAP file (~1 ngày) | ✅✅ Native API (~0.5 ngày) |
| Tất cả module khác | ❌ Rust mismatch | ✅ Cùng Python, hầu hết dễ-trung bình |

→ **Nerfstudio plug-in mạnh hơn Brush rất nhiều** vì cùng stack.

### Cách register `crsgaussian` method (template)
```python
# crsgaussian_nerfstudio_plugin/__init__.py
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.configs.method_configs import TrainerConfig
from .model import CRSGaussianModelConfig

crsgaussian_method = MethodSpecification(
    config=TrainerConfig(
        method_name="crsgaussian",
        pipeline=...,  # custom Pipeline với DAV2 depth + RoMa init
        ...
    ),
    description="CRSGaussian sparse-view 3DGS with CRS + LFCF + RoMa init"
)
```

```toml
# pyproject.toml
[project.entry-points."nerfstudio.method_configs"]
crsgaussian = "crsgaussian_nerfstudio_plugin:crsgaussian_method"
```

→ User chạy `ns-train crsgaussian --data <path>` y hệt `ns-train splatfacto`. Defense effect: hội đồng thấy thuật toán **gọi được bằng lệnh chuẩn của hệ sinh thái production**.

---

## 4. Hướng B — Standalone comparison setup

### Setup fair
- **Cùng input**: COLMAP format hoặc Nerfstudio format. LLFF 3-view của anh đã có sẵn COLMAP `sparse/0/` — splatfacto nhận trực tiếp.
- **Cùng iteration budget**: splatfacto default 30k, CRSGaussian 10k → set `--max-num-iterations 10000` cho splatfacto fair.
- **Cùng eval protocol**: dùng metric implementation chuẩn (PSNR/SSIM/LPIPS từ torchmetrics) — splatfacto và CRSGaussian đều dùng torchmetrics → fair.
- **AbsGS toggle**: splatfacto default `use_absgrad=True`. Để compare apple-to-apple với A0 baseline CRSGaussian (no AbsGS), set `--use-absgrad False`.

### Build trên server
```bash
# Trên Linux server (env riêng tránh conflict corgs)
conda create -n nerfstudio python=3.10 -y
conda activate nerfstudio
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install nerfstudio  # tự cài gsplat dependency
ns-install-cli  # tab completion
```

### Run splatfacto trên LLFF 3-view
```bash
# LLFF có sẵn COLMAP — dùng dataparser colmap
ns-train splatfacto \
    --data <path>/data/nerf_llff_data/fern/3_views/ \
    --max-num-iterations 10000 \
    --pipeline.model.use-absgrad False \
    colmap \
    --eval-mode interval --eval-interval 8
```

### Eval fair với CRSGaussian
splatfacto output `.ply` chuẩn 3DGS → load qua CRSGaussian `render.py` (hoặc dùng `ns-render` của Nerfstudio) → compute PSNR cùng metric implementation.

---

## 5. Cost / effort

| Hướng | Setup | Run | Total |
|-------|-------|-----|-------|
| **A — Plug-in (full plugin)** | Build plugin scaffold + register 6-8 module | 8 scene × 1 config × 3 seed = 24 run | ~7-10 ngày |
| **A — Plug-in (minimal: RoMa init only)** | Custom DataParser xuất seed_points | 8 scene × 2 config × 3 seed = 48 run | ~2 ngày |
| **B — Standalone compare** | Install Nerfstudio | 8 scene × 1 config × 3 seed = 24 run | ~2-3 ngày |
| **A-minimal + B combined** | — | 48+24 = 72 run | ~4-5 ngày |

GPU yêu cầu: Nerfstudio dùng CUDA → server NVIDIA OK. Có thể song song GPU0+GPU1.

---

## 6. Recommend

### Đi 3 step combined cho defense impact tối đa

**Step 1 — Standalone compare (Hướng B, ~2-3 ngày)**
- Chạy splatfacto vanilla 8-scene LLFF 3-view (cùng 10k iter)
- So với CRSGaussian Phase 22 RoMa v1
- Bảng PSNR: splatfacto X dB vs CRSGaussian 21.92 dB
- Câu chuyện: "Em vượt industry default Nerfstudio splatfacto X dB"

**Step 2 — Plug-in RoMa init (Hướng A-minimal, ~2 ngày)**
- Custom DataParser xuất `seed_points` từ RoMa v1 `fused.ply`
- splatfacto + RoMa init vs splatfacto + COLMAP init
- Bảng: Δ PSNR khi swap init
- Câu chuyện: "Module RoMa init của em là drop-in upgrade cho splatfacto, +Y dB"

**Step 3 — Full plug-in (Hướng A, optional, ~7-10 ngày nếu thời gian)**
- Tạo `crsgaussian-nerfstudio-plugin` package
- Plug đủ 6-8 module
- User cài `pip install` rồi chạy `ns-train crsgaussian`
- Câu chuyện: "Toàn bộ CRSGaussian recipe có thể chạy trong ecosystem Nerfstudio production-grade"

### Trade-off với Brush
| Khía cạnh | Brush | Nerfstudio |
|------------|-------|------------|
| Plug được nhiều module | 🟡 Chỉ init | ✅ Hầu hết |
| Stack tương thích | ❌ Rust | ✅ Python |
| Production thực tế evidence | 🟡 App users | ✅ Enterprise (Bentley/AWS/G-SHARP) + NVIDIA partner |
| Defense story rõ | 🟡 OK | ✅✅ Mạnh hơn |

→ **Nerfstudio nên là backbone evaluation chính**, Brush + OpenSplat là supplement.

---

## 7. ⚠️ Warning quan trọng cho contribution claim

### AbsGS overlap
splatfacto default `use_absgrad=True`. Phase 13 CRSGaussian commit `LFCF + AbsGS` (Δ +0.164 SIG vs A0).
- Nếu compare với splatfacto vanilla: AbsGS không phải contribution mới.
- Phải tách: **LFCF alone** vs splatfacto-with-absgrad → so sánh fair contribution mới.

### Strategy gsplat
LFCF densifier không thể plug-in trivial vì DefaultStrategy nằm trong gsplat library (không phải splatfacto). Có 2 lựa chọn:
1. Tạo `LfcfStrategy(DefaultStrategy)` subclass — khả thi nhưng phải hiểu rõ gsplat internal
2. Bypass gsplat strategy hoàn toàn, dùng custom densification trong CRSGaussian splatfacto subclass — đơn giản hơn

→ Đề xuất cách 2 cho effort thấp.

### Iteration budget mismatch
splatfacto default 30k iter, CRSGaussian 10k. Set fair: `--max-num-iterations 10000`. Cần verify splatfacto ở 10k có đạt baseline kỳ vọng hay không (vì recipe của họ tune cho 30k).

---

## 8. Việc cần làm tiếp

- [ ] Server: install Nerfstudio env riêng (`conda create -n nerfstudio`)
- [ ] Smoke test splatfacto trên 1 scene LLFF fern 3-view, 10k iter
- [ ] Verify số PSNR splatfacto trên 8 scene LLFF 3-view (baseline cho compare)
- [ ] (Hướng A-minimal) Viết custom DataParser xuất `seed_points` từ RoMa `fused.ply`
- [ ] Compare splatfacto-COLMAP vs splatfacto-RoMa-init
- [ ] Document kết quả vào `docs/evaluation/04_nerfstudio_comparison_results.md`

---

## 9. Liên quan
- [docs/evaluation/01_real_world_comparison_plan.md](01_real_world_comparison_plan.md) — overall evaluation plan
- [docs/evaluation/02_brush_feasibility_assessment.md](02_brush_feasibility_assessment.md) — Brush evaluation (parallel)
- Nerfstudio plugin framework: `nerfstudio/plugins/registry.py` + `types.py`
- splatfacto code: `nerfstudio/models/splatfacto.py`
- CRSGaussian Phase 22 RoMa v1: memory `project_phase22_roma_v1_pilot.md`
- CRSGaussian Phase 13 LFCF+AbsGS: memory `project_phase13_round1_winner.md` (lưu ý AbsGS overlap)
