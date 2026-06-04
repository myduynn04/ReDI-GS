# CRSGaussian Path A plug-in — `crsgaussian` method cho Nerfstudio

> **Mục đích**: Plug nguyên khối Phase 22 recipe (10 component) thành 1 method độc lập `crsgaussian` trong Nerfstudio, parallel với Splatfacto/Nerfacto.
>
> **Status**: 🔨 B1 SCAFFOLDING (init + 100 iter smoke pending verify)

## Khác Phase A plug-in (`nerfstudio_plugin/splatfacto-roma*`)

| | Phase A (`splatfacto-roma*`) | Path A (`crsgaussian`) |
|---|------------------------------|------------------------|
| Subclass | SplatfactoModel | Model (direct) |
| Renderer | gsplat (Nerfstudio default) | diff_gaussian_rasterization (Inria/CoR-GS) |
| GaussianModel | splatfacto's | CoR-GS GaussianModel |
| Methods registered | 10 (5 ON + 5 OFF) | 1 (`crsgaussian`) |
| Coverage Phase 22 | 5/10 (wash trên CRS-axis) | 10/10 (full Phase 22) |
| Target PSNR | ~17.6 (4 scene mean) | ~21.8+ (95% Phase 22 standalone 22.97) |

## Setup server

```bash
# 1. Upload plug-in folder
scp -r crsgaussian_plugin/ aidev@server:~/workspace/representation-3d/duyen/nerfstudio/

# 2. Verify CoR-GS source exists
ssh aidev@server "ls ~/workspace/representation-3d/duyen/CoR-GS/scene/gaussian_model.py"

# 3. Run B1 smoke
ssh aidev@server "cd ~/workspace/representation-3d/duyen/nerfstudio && \
    bash crsgaussian_plugin/scripts/smoke_b1_fern_100iter.sh"
```

## Phase status

| Phase | Status | Verify | Files |
|-------|--------|--------|-------|
| B1 Scaffolding | 🔨 PENDING smoke | 100 iter no crash | corgs_imports, corgs_dataparser, camera_adapter, crsgaussian_config/model, __init__ |
| B2 Loss + render | ⏳ TODO | 500 iter, PSNR > 18 | get_outputs real, get_loss_dict L1+SSIM+depth |
| B3 Callbacks | ⏳ TODO | 2000 iter, PSNR > 21 | 5 callbacks (optimizer, densify, decay, crs, sh_freeze) |
| B4 Benchmark | ⏳ TODO | 4 scene 10k, mean ≥ 21.8 | benchmark script + compare_vs_phase22 |

## Rollback

```bash
# Server
ssh aidev@server "rm -rf ~/workspace/representation-3d/duyen/nerfstudio/crsgaussian_plugin/"

# Local: KEEP — do not delete
```

## Liên quan

- [docs/evaluation/08_path_a_implementation_plan.md](../docs/evaluation/08_path_a_implementation_plan.md) — full plan
- [docs/evaluation/07_nerfstudio_defense_framing.md](../docs/evaluation/07_nerfstudio_defense_framing.md) — defense framing
- Phase A plug-in: [nerfstudio_plugin/](../nerfstudio_plugin/)
