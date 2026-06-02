# CRSGaussian Plug-in A1 — RoMa v1 init for Nerfstudio splatfacto

> External plugin: plug RoMa v1 dense init (Phase 22 contribution) vào Nerfstudio splatfacto pipeline.
>
> KHÔNG sửa Nerfstudio source. Tắt plugin → splatfacto vanilla y nguyên.

## Files

| File | Mục đích |
|------|----------|
| `__init__.py` | Register method `splatfacto-roma` qua `MethodSpecification` |
| `roma_dataparser.py` | Subclass `ColmapDataParser` — override `_load_3D_points()` đọc `fused.ply.romav1` |
| `verify_plugin.sh` | 5-tier verification script |
| `README.md` | File này |

## Pre-requisites

1. Nerfstudio env đã setup (xem `docs/evaluation/04_nerfstudio_setup_and_run.md`)
2. Phase 22 RoMa preprocess đã chạy cho các scene cần test:
   ```bash
   conda activate roma_v1
   SCENE=fern python scripts/p22_romav1_preprocess.py
   # output: data/nerf_llff_data/fern/3_views/dense/fused.ply.romav1
   ```
3. `pip install plyfile` (đã có sẵn nếu nerfstudio cài đủ)

## Setup trên server (1 lần)

```bash
# 1. Upload folder crsgaussian_plugin/ lên server
#    Target: /home/aidev/workspace/representation-3d/duyen/nerfstudio/crsgaussian_plugin/

# 2. Activate env nerfstudio
conda activate nerfstudio

# 3. Verify plyfile có sẵn (nerfstudio đã cài)
python -c "import plyfile; print('plyfile OK')"
# Nếu thiếu: pip install plyfile

# 4. Set env vars (persist qua activate.d cho convenience)
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/crsgaussian_plugin.sh << 'EOF'
export PYTHONPATH=/home/aidev/workspace/representation-3d/duyen/nerfstudio:$PYTHONPATH
export NERFSTUDIO_METHOD_CONFIGS="splatfacto-roma=crsgaussian_plugin:roma_method_spec"
EOF

# 5. Reload env
conda deactivate
conda activate nerfstudio

# 6. Verify method registered
ns-train --help 2>&1 | grep splatfacto-roma
# → Should see: splatfacto-roma   ...
```

## Run

```bash
cd /home/aidev/workspace/representation-3d/duyen/nerfstudio

# Test 1 scene (LLFF defaults: colmap_path=triangulated, images=images, eval_interval=8 ĐÃ preset trong plugin)
CUDA_VISIBLE_DEVICES=0 ns-train splatfacto-roma \
    --data /home/aidev/workspace/representation-3d/duyen/CoR-GS/data/nerf_llff_data/fern/3_views/ \
    --output-dir outputs_4scene_roma \
    --experiment-name fern \
    --max-num-iterations 10000 \
    --steps-per-eval-all-images 10000 \
    --vis tensorboard
```

**Lưu ý**: KHÔNG dùng `colmap` subcommand như vanilla splatfacto. Plug-in đã preset RomaDataParserConfig với LLFF layout (`triangulated/`, `images/`, eval interval=8). Nếu gõ `colmap` subcommand → tyro override → plug-in bị bypass.

Override field cá biệt qua dotted path:
```bash
# Vd: tắt RoMa init để test Tier 2.1 byte-identical vanilla
--pipeline.datamanager.dataparser.use-roma-init False
```

## Verify 5-tier

```bash
bash crsgaussian_plugin/verify_plugin.sh
```

Output: PASS/FAIL từng tier. Cần 100% Tier 1 + 2 + 4 pass trước khi run full 4 scene.

## Architecture

```
[Nerfstudio splatfacto vanilla]
   ↓
   ColmapDataParser._load_3D_points()
        → reads points3D.bin (~vài trăm điểm sparse)

[CRSGaussian Plug-in A1 — splatfacto-roma]
   ↓
   RomaDataParser._load_3D_points()
        if use_roma_init=True:
            → reads dense/fused.ply.romav1 (~17K điểm dense, Phase 22)
        else:
            → fallback parent (byte-identical với vanilla)
```

## Tier 2.1 — OFF flag byte-identical (manual test)

Verify rằng `use_roma_init=False` cho kết quả khớp `splatfacto` vanilla:

```bash
# Run với flag OFF (override config)
ns-train splatfacto-roma \
    --pipeline.datamanager.dataparser.use-roma-init False \
    --data .../fern/3_views/ \
    --max-num-iterations 10000 \
    ... (other args)

# So PSNR với `splatfacto` vanilla (đã có ở outputs_4scene_10k/fern/)
# Pass nếu Δ < 0.10 dB
```

## Rollback

Plugin reversible 100%:
```bash
# Tắt plugin (chỉ trong current shell)
unset NERFSTUDIO_METHOD_CONFIGS
unset PYTHONPATH  # hoặc remove plugin path

# Hoặc remove persistent:
rm $CONDA_PREFIX/etc/conda/activate.d/crsgaussian_plugin.sh

# Hoặc xóa hoàn toàn
rm -rf /home/aidev/workspace/representation-3d/duyen/nerfstudio/crsgaussian_plugin
```

→ Nerfstudio quay về vanilla y nguyên, `splatfacto-roma` biến mất khỏi CLI.

## Verified config

| Component | Version |
|-----------|---------|
| Nerfstudio | 1.1.5 |
| gsplat | 1.4.0 |
| PyTorch | 2.5.1+cu124 |
| CUDA toolkit | 12.4 (system /usr/local/cuda) |
| Python | 3.10 |

## Liên quan

- Design doc: `docs/evaluation/05_plugin_design.md`
- Nerfstudio setup: `docs/evaluation/04_nerfstudio_setup_and_run.md`
- Phase 22 RoMa preprocess: `scripts/p22_romav1_preprocess.py`
- Phase 22 results: memory `project_phase22_roma_v1_pilot.md`
