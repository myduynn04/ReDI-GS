# Plug-in CRSGaussian → Nerfstudio — Design Document

> Thiết kế chi tiết cho việc plug module CRSGaussian vào Nerfstudio framework như external plugin (không sửa Nerfstudio source).
>
> **Mục tiêu defense**: chứng minh module CRSGaussian là **drop-in upgrade** cho Nerfstudio production (NVIDIA / Bentley / AWS / Adobe ecosystem) — không phải research toy độc lập.
>
> **Trạng thái** (cập nhật 2026-06-02 evening):
> - ✅ Architecture chốt: **Option 1** — Bổ trợ splatfacto (subclass + override hook)
> - ✅ **A1 (RoMa init) DONE**: +1.523 dB mean, 4/4 scene positive
> - ✅ **A2.1 (opacity decay) DONE**: WASH alone (đúng Phase 23 synergy-only pattern)
> - ✅ **A2.2 (DropAnSH) DONE**: WASH cumulative
> - 🟡 **A2.3 (DAV2 depth loss) smoke PASS**, cascade pending
> - 🟡 **A2.4 (CRS score D-only) smoke PASS**, cascade pending
> - ⏳ **A3 (5 module deferred)**: SH freeze, LFCF, D_cycle, S_stability, R_i — pip-installable post-defense
>
> **Coverage Phase 22 recipe**: 4 FULL + 1 PARTIAL = **5/10 component (50%)**
>
> Chi tiết kết quả: [06_plugin_a1_results.md](06_plugin_a1_results.md)
>
> **Verified Nerfstudio**: 1.1.5 + gsplat 1.4.0 (Apache 2.0).

---

## 1. Defense narrative — vì sao plug-in

### Position trước plug-in
CRSGaussian = research code fork CoR-GS, standalone. PSNR 21.92 trên LLFF 3-view. KHÓ chứng minh "ứng dụng thực tế".

### Position SAU plug-in
Module em (CRS score / RoMa init / DAV2 depth / opacity decay / DropAnSH / SH freeze / D_cycle / LFCF) registered vào Nerfstudio CLI:

```bash
# Trước: chỉ có vanilla splatfacto
ns-train splatfacto --data ...

# Sau khi plug-in: thêm method `splatfacto-roma`, `splatfacto-crsg`
ns-train splatfacto-roma --data ...     # vanilla + RoMa init
ns-train splatfacto-crsg --data ...     # vanilla + RoMa + depth + CRS + opacity decay + ...
```

→ Nerfstudio = framework NVIDIA / Bentley / AWS production đang dùng. Plug-in vào đây = **drop-in upgrade cho pipeline production thật**.

---

## 2. Architecture overview

### Cách Nerfstudio register custom method (chính thức)
Reference: [nerfstudio/plugins/registry.py:34-79](../../nerfstudio/nerfstudio/plugins/registry.py)

2 cách official:

#### Cách A — Environment variable (recommend cho develop)
```bash
export NERFSTUDIO_METHOD_CONFIGS="splatfacto-roma=crsgaussian_plugin:roma_method_spec"
export PYTHONPATH=/home/aidev/workspace/representation-3d/duyen/nerfstudio:$PYTHONPATH
ns-train splatfacto-roma --data ...
```
→ KHÔNG cần `pip install`, KHÔNG cần `setup.py`. Code chỉ là folder Python thường.

#### Cách B — Pip entry point (production, sau khi develop xong)
```toml
# pyproject.toml
[project.entry-points."nerfstudio.method_configs"]
splatfacto-roma = "crsgaussian_plugin:roma_method_spec"
```
```bash
pip install -e .
ns-train --help  # tự thấy splatfacto-roma
```

→ Develop xài cách A, ship release xài cách B.

### Folder layout

```
/home/aidev/workspace/representation-3d/duyen/nerfstudio/    ← working dir
├── outputs_4scene_10k/                  baseline splatfacto vanilla
├── outputs_4scene_roma/                 (mới) splatfacto-roma
├── logs/
├── run_4scene_10k.sh                    baseline script
├── run_4scene_roma.sh                   (mới) plug-in script
└── crsgaussian_plugin/                  ← code plug-in (5-tier verified)
    ├── __init__.py                      register MethodSpecification
    ├── README.md                        setup + run commands
    ├── roma_dataparser.py               A1: custom DataParser load RoMa PLY
    ├── crsg_model.py                    A2+: subclass SplatfactoModel (depth, opacity decay, CRS)
    ├── losses.py                        A2+: DAV2 depth loss + CRS-weighted loss
    ├── strategy.py                      A3: LFCF densifier (subclass DefaultStrategy)
    └── verify_plugin.sh                 5-tier test script
```

### KHÔNG touch Nerfstudio source

```
/home/aidev/miniconda3/envs/nerfstudio/lib/.../site-packages/nerfstudio/    ← READ-ONLY
```

Verify Tier 4 sẽ confirm 0 file Nerfstudio bị modify.

---

## 3. Scope — 3 sub-options

### A1 — Minimal (RoMa init only) ⭐ recommend bắt đầu
- **Effort**: 1 ngày code + 0.5 ngày test
- **Module plug**: 1 module (RoMa v1 dense init thay COLMAP)
- **Mechanism**: Custom DataParser xuất `seed_points=(xyz, rgb)` từ `dense/fused.ply.romav1`
- **Method name**: `splatfacto-roma`
- **Expected Δ PSNR**: +0.5 đến +0.6 dB (Phase 22 RoMa v1 = +0.584 SIG project best)
- **Defense story**: *"Module RoMa init em là drop-in upgrade cho splatfacto, +X dB no extra training time"*

### A2 — Sequential cascade (RoMa + depth + opacity decay + CRS)
- **Effort**: 3-5 ngày
- **Module plug**: 5 module (build từng bước, mỗi step ablation)
- **Method names**:
  - `splatfacto-roma` (A1 base)
  - `splatfacto-roma-depth` = + DAV2 depth loss
  - `splatfacto-roma-depth-opdecay` = + opacity decay
  - `splatfacto-crsg` = + CRS modulated SH freeze
- **Defense story**: *"Mỗi module em đóng góp +X dB. Total stacked +Y dB."*

### A3 — Full pip-installable package
- **Effort**: 1-2 tuần
- **Module plug**: tất cả 8 module + LFCF densifier (subclass gsplat Strategy)
- **Method name**: `crsgaussian` (production-ready)
- **Output**: `pip install crsgaussian-nerfstudio` (có thể publish PyPI)
- **Defense story**: *"Em ship được pip package"* — high impact nhưng heavy

### Recommend
**A1 trước (1 ngày verify pattern)** → nếu pass 5-tier criteria → scale lên A2.

---

## 4. 5-Tier Verification Criteria

> Tiêu chí PHẢI pass để gọi là "plug-in đúng + đủ". Mỗi tier có pass/fail rõ ràng.

### Tier 1 — Code chạy đúng kỹ thuật (smoke tests)

| # | Criteria | Pass condition |
|---|----------|----------------|
| 1.1 | Nerfstudio detect plugin | `ns-train --help \| grep splatfacto-roma` thấy method |
| 1.2 | Run end-to-end | Train fern 1000 iter không exception |
| 1.3 | Eval ra số hợp lệ | PSNR > 10 dB (not NaN) |
| 1.4 | Output PLY load được | Drag SuperSplat render được |

### Tier 2 — Module ACTIVE đúng nguồn (mechanism verify) ⭐⭐⭐ CRITICAL

| # | Criteria | Pass condition |
|---|----------|----------------|
| 2.1 | **OFF flag = vanilla byte-identical** | `use_roma_init=False` → PSNR khớp `splatfacto` vanilla ±0.10 dB |
| 2.2 | **ON → N_gauss khớp RoMa** | `len(means)` iter 0 ≈ 17K (Phase 22), KHÔNG ~vài trăm (COLMAP) |
| 2.3 | **Init points match `fused.ply.romav1`** | Sample 5 points đầu vs `load_ply` độc lập → match ±1e-4 |
| 2.4 | **Missing PLY raise error** | Rename `fused.ply.romav1` → run → FileNotFoundError, không silent fallback |

→ **Tier 2 là critical**: chứng minh module thực sự được dùng, không phải code chạy nhưng vô tác dụng.

### Tier 3 — Kết quả khớp Phase 22 expectation

| # | Criteria | Pass condition |
|---|----------|----------------|
| 3.1 | **Per-scene Δ trend match Phase 22** | horns Δ lớn nhất, trex marginal — same direction |
| 3.2 | **Mean Δ SIG** (multi-seed paired N=12) | 95% CI excludes 0 |
| 3.3 | **Sign nhất quán cross-seed** | 3/3 seeds cùng dấu |

### Tier 4 — Tách biệt với Nerfstudio source

| # | Criteria | Pass condition |
|---|----------|----------------|
| 4.1 | Không sửa site-packages Nerfstudio | `find site-packages/nerfstudio -newer plugin/` empty |
| 4.2 | Plugin 100% trong folder riêng | `crsgaussian_plugin/` self-contained |
| 4.3 | Tag `[CRSGaussian]` mọi code | `grep -r "\[CRSGaussian\]" crsgaussian_plugin/` cover các function |
| 4.4 | Reversible | `rm -rf crsgaussian_plugin/` + unset env → `ns-train` lại như cũ |

### Tier 5 — Reproducibility (defense-grade)

| # | Criteria | Pass condition |
|---|----------|----------------|
| 5.1 | Cùng seed → cùng PSNR | Run fern seed 42 hai lần ±0.05 dB |
| 5.2 | Version locked | `requirements.txt` với nerfstudio==1.1.5, gsplat==1.4.0 |
| 5.3 | README plugin folder | Setup + run commands |
| 5.4 | Run command 1-line copy-paste | Trong README |

---

## 5. Implementation plan A1 (RoMa init plug-in)

### Phase A1.1 — Đọc preprocess Phase 22 (1 giờ)

Đọc `scripts/p22_romav1_preprocess.py` trên server để hiểu:
- Format `fused.ply.romav1` (RGB + XYZ? Có normal không?)
- Số points typical (~17K theo memory)
- Coordinate system (cùng COLMAP convention không)

### Phase A1.2 — Viết `crsgaussian_plugin/roma_dataparser.py` (3-4 giờ)

```python
# ============================================================
# [CRSGaussian Plug-in A1] RoMa v1 dense init for Nerfstudio
# File: crsgaussian_plugin/roma_dataparser.py
# Mục đích: Subclass ColmapDataParser → load fused.ply.romav1 thay
#           vì COLMAP points3D.bin
# ============================================================
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type
import numpy as np
from plyfile import PlyData
import torch

from nerfstudio.data.dataparsers.colmap_dataparser import (
    ColmapDataParser, ColmapDataParserConfig
)

@dataclass
class RomaDataParserConfig(ColmapDataParserConfig):
    """[CRSGaussian Plug-in A1] DataParser dùng RoMa v1 dense init."""
    _target: Type = field(default_factory=lambda: RomaDataParser)
    use_roma_init: bool = True   # master switch — verify Tier 2.1
    roma_ply_relpath: str = "dense/fused.ply.romav1"

class RomaDataParser(ColmapDataParser):
    config: RomaDataParserConfig

    def _load_3D_points(self, colmap_path, transform_matrix, scale_factor):
        # ── [CRSGaussian Plug-in A1] Override init points source ──
        # Nếu use_roma_init=True → load fused.ply.romav1
        # Nếu False → fallback parent class (đọc points3D.bin COLMAP)
        if not self.config.use_roma_init:
            return super()._load_3D_points(colmap_path, transform_matrix, scale_factor)

        ply_path = self.config.data / self.config.roma_ply_relpath
        if not ply_path.is_file():
            raise FileNotFoundError(
                f"[CRSGaussian Plug-in A1] RoMa PLY not found: {ply_path}\n"
                f"  Verify Phase 22 preprocess: scripts/p22_romav1_preprocess.py"
            )

        # Load PLY (cùng schema COLMAP: x,y,z + r,g,b uint8)
        plydata = PlyData.read(str(ply_path))
        vertex = plydata['vertex']
        positions = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)
        colors = np.stack([vertex['red'], vertex['green'], vertex['blue']], axis=-1)

        # Apply same transform as parent class (orientation + scale)
        positions_t = (transform_matrix[:3, :3] @ positions.T).T + transform_matrix[:3, 3:].T
        positions_t = positions_t * scale_factor

        print(f"[CRSGaussian Plug-in A1] Loaded {len(positions_t)} points from {ply_path.name}")

        return {
            "points3D_xyz": torch.from_numpy(positions_t.astype(np.float32)),
            "points3D_rgb": torch.from_numpy(colors.astype(np.uint8)),
        }
```

### Phase A1.3 — Viết `crsgaussian_plugin/__init__.py` (1 giờ)

```python
# ============================================================
# [CRSGaussian Plug-in A1] Method registration
# File: crsgaussian_plugin/__init__.py
# Register `splatfacto-roma` vào Nerfstudio CLI
# ============================================================
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.configs.method_configs import method_configs

# Copy config splatfacto vanilla làm base, swap dataparser
splatfacto_base = method_configs["splatfacto"]

roma_method_spec = MethodSpecification(
    config=TrainerConfig(
        method_name="splatfacto-roma",
        steps_per_eval_image=splatfacto_base.steps_per_eval_image,
        steps_per_eval_batch=splatfacto_base.steps_per_eval_batch,
        steps_per_save=splatfacto_base.steps_per_save,
        steps_per_eval_all_images=splatfacto_base.steps_per_eval_all_images,
        max_num_iterations=splatfacto_base.max_num_iterations,
        mixed_precision=splatfacto_base.mixed_precision,
        pipeline=splatfacto_base.pipeline.copy(),  # deep copy
        optimizers=splatfacto_base.optimizers,
        viewer=splatfacto_base.viewer,
        vis=splatfacto_base.vis,
    ),
    description="[CRSGaussian Plug-in A1] splatfacto + RoMa v1 dense init",
)

# Swap dataparser sang RomaDataParserConfig (giữ field khác)
from .roma_dataparser import RomaDataParserConfig
_old_dp = roma_method_spec.config.pipeline.datamanager.dataparser
roma_method_spec.config.pipeline.datamanager.dataparser = RomaDataParserConfig(
    colmap_path=_old_dp.colmap_path,
    images_path=_old_dp.images_path,
    eval_mode=_old_dp.eval_mode,
    eval_interval=_old_dp.eval_interval,
    # ... copy field khác
)

# Optional: tắt AbsGS để fair với CRSGaussian A0 baseline
roma_method_spec.config.pipeline.model.use_absgrad = False
```

### Phase A1.4 — Verify script (`crsgaussian_plugin/verify_plugin.sh`, 2 giờ)

Bash script tự động chạy 5-tier:

```bash
#!/bin/bash
# Verify plug-in A1 — chạy 5 tier
set +e   # continue khi 1 tier fail

PLUGIN_DIR=/home/aidev/workspace/representation-3d/duyen/nerfstudio/crsgaussian_plugin
export NERFSTUDIO_METHOD_CONFIGS="splatfacto-roma=crsgaussian_plugin:roma_method_spec"
export PYTHONPATH=/home/aidev/workspace/representation-3d/duyen/nerfstudio:$PYTHONPATH

echo "=== TIER 1 — Smoke tests ==="
ns-train --help 2>&1 | grep splatfacto-roma && echo "✓ 1.1 detect" || echo "✗ 1.1 FAIL"

# Train fern 1000 iter test
TEST_OUT=outputs/_plugin_verify
rm -rf $TEST_OUT && mkdir -p $TEST_OUT
ns-train splatfacto-roma \
    --data /home/aidev/workspace/representation-3d/duyen/CoR-GS/data/nerf_llff_data/fern/3_views/ \
    --output-dir $TEST_OUT \
    --max-num-iterations 1000 \
    --pipeline.model.use-absgrad False \
    colmap --colmap-path triangulated --images-path images \
    --eval-mode interval --eval-interval 8 > /tmp/verify_train.log 2>&1

grep "Training Finished" /tmp/verify_train.log && echo "✓ 1.2 run e2e" || echo "✗ 1.2 FAIL"

RUN=$(ls -td $TEST_OUT/*/splatfacto-roma/*/ | head -1)
ns-eval --load-config $RUN/config.yml --output-path $RUN/eval.json
PSNR=$(python3 -c "import json; print(json.load(open('$RUN/eval.json'))['results']['psnr'])")
echo "  PSNR=$PSNR"

# ... (tier 2, 3, 4, 5 mỗi tier 1 block)
```

### Phase A1.5 — Run A/B compare 4 scene (4-6 giờ)

```bash
# Đã có splatfacto vanilla 4 scene
# Now run splatfacto-roma 4 scene
./run_4scene_roma.sh  # tương tự run_4scene_10k.sh, đổi method
```

Bảng so sánh Δ PSNR.

### Phase A1.6 — Document kết quả (1 giờ)

File `docs/evaluation/06_plugin_a1_results.md` với:
- 5-tier checklist pass/fail
- Bảng PSNR 4 scene paired
- Insights + next step (A2 hay không)

---

## 6. Risk + mitigation

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Format `fused.ply.romav1` khác chuẩn PLY 3DGS | 🟡 Medium | Đọc preprocess code Phase 22 trước khi viết DataParser |
| Coordinate system mismatch (RoMa world ≠ COLMAP world) | 🟡 Medium | Verify Tier 2.3 — sample point match |
| Splatfacto eval differ với CoR-GS render.py | 🟢 Low | Dùng `ns-eval` cho cả 2 method, fair |
| Tier 2.1 OFF byte-identical fail (side-effect bug) | 🟡 Medium | Subclass cẩn thận, không touch attribute khác |
| Tier 3.1 trend không match Phase 22 | 🔴 High impact | Indication mechanism khác → debug deeper |
| GPU 0 share với process khác | 🟡 Medium | Switch GPU 1, hoặc đợi GPU 0 free |

---

## 7. Compare với CRSGaussian standalone

| Aspect | CRSGaussian standalone (Phase 22) | CRSGaussian → Nerfstudio plug-in |
|--------|-----------------------------------|-----------------------------------|
| Run command | `python train.py --use_depth_prior --use_d_cycle ...` | `ns-train splatfacto-roma --data ...` |
| Code base | Fork CoR-GS | Plug vào Nerfstudio production |
| User audience | Researcher có CoR-GS env | Mọi user Nerfstudio |
| PSNR LLFF 3-view | 21.92 (verified) | Có thể khác (splatfacto framework khác) |
| Defense story | "Em làm research độc lập" | **"Module em drop-in cho production"** |
| Reproducibility | Cần clone CRSGaussian repo | `pip install` + 1 lệnh |

→ Plug-in **bổ trợ**, không thay thế CRSGaussian standalone. Cả 2 cùng tồn tại trong defense.

---

## 8. Checklist trước khi user confirm

User cần xác nhận:

- [ ] Đồng ý đi **Hướng A external plugin** (không sửa Nerfstudio source)
- [ ] Đồng ý bắt đầu **A1 minimal** (RoMa init only) trước
- [ ] Đồng ý **5-tier verification criteria** trước khi gọi plug-in xong
- [ ] OK với folder `crsgaussian_plugin/` trong `nerfstudio/` working dir
- [ ] OK với method name `splatfacto-roma`
- [ ] Có thể đợi 1-2 ngày code + test trước khi compare A/B

User confirm → tôi bắt tay theo Phase A1.1 → A1.6 sequential.

---

## 9. Liên quan

- [01_real_world_comparison_plan.md](01_real_world_comparison_plan.md) — overall evaluation plan
- [03_nerfstudio_feasibility_assessment.md](03_nerfstudio_feasibility_assessment.md) — feasibility analysis
- [04_nerfstudio_setup_and_run.md](04_nerfstudio_setup_and_run.md) — setup + run baseline 4 scene
- (sau A1 xong) `06_plugin_a1_results.md` — verify checklist + bảng PSNR
- CRSGaussian Phase 22: memory `project_phase22_roma_v1_pilot.md`
- Nerfstudio plugin framework: [nerfstudio/plugins/registry.py](../../nerfstudio/nerfstudio/plugins/registry.py)
