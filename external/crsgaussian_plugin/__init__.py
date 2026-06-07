# ============================================================
# [CRSGaussian Path A B1] Register method `crsgaussian` vào Nerfstudio
# File: crsgaussian_plugin/__init__.py  (KEEP LOCAL — upload server)
#
# Mục đích: Method `crsgaussian` = full Phase 22 recipe wrapped trong Nerfstudio.
#
# Khác `splatfacto-*` plug-in (Phase A) ở chỗ:
#   - KHÔNG subclass SplatfactoModel — subclass Model trực tiếp
#   - Dùng CoR-GS GaussianModel + render (KHÔNG gsplat)
#   - 1 method duy nhất, không phải 10 method sub-variant
#
# Activate qua env var (develop):
#   export NERFSTUDIO_METHOD_CONFIGS="crsgaussian=crsgaussian_plugin:crsgaussian_method_spec"
#   export PYTHONPATH=/home/aidev/workspace/representation-3d/duyen/nerfstudio:$PYTHONPATH
#   ns-train crsgaussian --data <colmap-folder>
#
# Phase B1 default: tất cả master switch OFF (smoke verify init only)
# Phase B2+: enable từng switch (use_depth_loss, use_opacity_decay, etc.)
# ============================================================
"""[CRSGaussian Path A] Register `crsgaussian` method vào Nerfstudio CLI."""

from pathlib import Path

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanagerConfig,
)
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from .corgs_dataparser import CrsGaussianDataParserConfig
from .crsgaussian_config import CrsGaussianModelConfig


def _build_crsgaussian_config():
    """[Path A] Build TrainerConfig cho method `crsgaussian`.

    Phase 22 default values + B1 master switches OFF (smoke init only).
    """

    # Lazy import để tránh circular
    from .crsgaussian_model import CrsGaussianModel

    model_config = CrsGaussianModelConfig(
        _target=CrsGaussianModel,
        # B1 defaults: all master switches OFF
        # (anh override qua CLI: --pipeline.model.use-depth-loss True)
    )

    dataparser_config = CrsGaussianDataParserConfig(
        # [B3 split-fix 2026-06-04] Full data folder convention (fern/, NOT fern/3_views/):
        #   fern/sparse/0/      → COLMAP 24 cams (replace `triangulated/` of 3_views/)
        #   fern/images_8/      → ns auto-load via downscale_factor=8
        #   fern/3_views/dense/fused.ply.romav1  → RoMa init
        #   fern/3_views/aligned_depth_a23/      → DAV2 aligned depth
        # Phase 22 split protocol port (dataset_readers.py:356-366):
        #   Step 1: eval_interval=8 → test=3 cams, train_pool=21 cams
        #   Step 2: n_views_phase22=3 → linspace subsample → 3 train cams
        colmap_path=Path("sparse/0"),
        images_path=Path("images"),
        eval_mode="interval",
        eval_interval=8,
        n_views_phase22=3,
        llffhold=8,
        downscale_factor=1,  # [B3 split-fix v2] in-memory resize qua camera_res_scale_factor
        use_roma_init=True,
        roma_ply_relpath="3_views/dense/fused.ply.romav1",
        aligned_depth_relpath="3_views/aligned_depth_a23",
        load_3D_points=True,
    )

    return TrainerConfig(
        method_name="crsgaussian",
        steps_per_eval_image=500,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=10000,
        max_num_iterations=10000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                dataparser=dataparser_config,
                cache_images_type="uint8",
                # [B3 split-fix v2 2026-06-04] In-memory resize 1/8 thay vì pre-gen folder.
                # ns InputDataset.scale_factor (base_dataset.py:88-91) resize ảnh
                # qua PIL.resize(BILINEAR) khi load. Cameras width/height tự rescale
                # qua line 59 self.cameras.rescale_output_resolution(scale_factor).
                # = Phase 22 standalone behavior (in-memory resize tại Camera __init__).
                camera_res_scale_factor=0.125,  # 1/8 — match Phase 22 args.resolution=8
            ),
            model=model_config,
        ),
        # Empty optimizers dict — CrsGaussianModel.get_param_groups() returns {} →
        # nerfstudio Trainer no-op cho optimizer step. CoR-GS GaussianModel.optimizer
        # tự handle qua callback (B3+).
        optimizers={
            # Dummy optimizer cho compat — không thực sự dùng
            "_dummy": {
                "optimizer": AdamOptimizerConfig(lr=1e-3),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="tensorboard",
    )


crsgaussian_method_spec = MethodSpecification(
    config=_build_crsgaussian_config(),
    description="[CRSGaussian Path A] Full Phase 22 recipe wrapped in Nerfstudio (RoMa init + LFCF + AbsGS + DropAnSH + DAV2 depth + CRS + SH freeze + opacity decay)",
)


# ════════════════════════════════════════════════════════
# [B5 Defense Compare 2026-06-04 evening] `splatfacto-sparse` method spec
# Mục đích: Fair compare baseline — Splatfacto model + plug-in Phase 22 split
#          (3 train + 3 test) + RoMa init.
# Khác với `splatfacto` default ns:
#   - ns default: random split full cams (~17 train cho fern)
#   - splatfacto-sparse: Phase 22 3-cam protocol
# Use case: Viewer demo so sánh visual quality 3-cam:
#   - crsgaussian (full Phase 22 8 modules) vs splatfacto-sparse (vanilla baseline)
# ════════════════════════════════════════════════════════

def _build_splatfacto_sparse_config():
    """[B5] Splatfacto vanilla + Phase 22 3-view sparse split + RoMa init."""

    from nerfstudio.models.splatfacto import SplatfactoModelConfig
    from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig

    dataparser_config = CrsGaussianDataParserConfig(
        colmap_path=Path("sparse/0"),
        images_path=Path("images"),
        eval_mode="interval",
        eval_interval=8,
        n_views_phase22=3,
        llffhold=8,
        downscale_factor=1,
        # [B5 v2 FIX 2026-06-05] use_roma_init=False — RoMa = anh's Phase 22
        # contribution, baseline phải dùng COLMAP sparse init thật cho fair compare.
        use_roma_init=False,
        roma_ply_relpath="3_views/dense/fused.ply.romav1",
        aligned_depth_relpath="3_views/aligned_depth_a23",
        load_3D_points=True,
    )

    return TrainerConfig(
        method_name="splatfacto-sparse",
        steps_per_eval_image=500,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=10000,
        max_num_iterations=10000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                dataparser=dataparser_config,
                cache_images_type="uint8",
                camera_res_scale_factor=0.125,  # Phase 22 res=1/8
            ),
            model=SplatfactoModelConfig(),
        ),
        # Splatfacto-specific optimizers (ported từ ns method_configs.py:607-643)
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1.6e-6, max_steps=10000),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-7, max_steps=10000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
            "bilateral_grid": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=10000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="tensorboard",
    )


splatfacto_sparse_method_spec = MethodSpecification(
    config=_build_splatfacto_sparse_config(),
    description="[B5 Defense] Splatfacto vanilla baseline trên Phase 22 3-view sparse split + RoMa init (fair compare với crsgaussian)",
)


# ════════════════════════════════════════════════════════
# [B5 Defense Compare] `splatfacto-17` method spec
# Mục đích: Splatfacto vanilla với 17-cam ns default split + COLMAP init.
# Khác `splatfacto` default ns: dùng ColmapDataParser (KHÔNG NerfstudioDataParser)
# vì LLFF fern là COLMAP format, không có transforms.json.
# ════════════════════════════════════════════════════════

def _build_splatfacto_17_config():
    """[B5] Splatfacto vanilla + ColmapDataParser default (17 train + 3 test cho fern)."""

    from nerfstudio.models.splatfacto import SplatfactoModelConfig
    from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig
    from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig

    return TrainerConfig(
        method_name="splatfacto-17",
        steps_per_eval_image=500,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=10000,
        max_num_iterations=10000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                dataparser=ColmapDataParserConfig(
                    colmap_path=Path("sparse/0"),
                    images_path=Path("images"),
                    downscale_factor=1,
                    eval_mode="interval",
                    eval_interval=8,
                    load_3D_points=True,
                ),
                cache_images_type="uint8",
                camera_res_scale_factor=0.125,  # match plug-in resolution
            ),
            model=SplatfactoModelConfig(),
        ),
        # Splatfacto-specific optimizers (ported từ ns method_configs.py:607-643)
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1.6e-6, max_steps=10000),
            },
            "features_dc": {"optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15), "scheduler": None},
            "features_rest": {"optimizer": AdamOptimizerConfig(lr=0.0025/20, eps=1e-15), "scheduler": None},
            "opacities": {"optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15), "scheduler": None},
            "scales": {"optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15), "scheduler": None},
            "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-7, max_steps=10000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
            "bilateral_grid": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=10000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="tensorboard",
    )


splatfacto_17_method_spec = MethodSpecification(
    config=_build_splatfacto_17_config(),
    description="[B5 Defense] Splatfacto vanilla baseline COLMAP 17-cam ns default (data abundance compare)",
)


# ════════════════════════════════════════════════════════
# [B5 v3 2026-06-05] `splatfacto-sparse-noabs` method spec
# Mục đích: Splatfacto vanilla NHƯNG disable AbsGS (use_absgrad=False).
# Lý do: ns Splatfacto mặc định BẬT AbsGS (splatfacto.py:107) — đó là cải tiến từ
#        paper AbsGS 2024, KHÔNG phải vanilla 3DGS Inria gốc.
# So sánh "pure 3DGS vanilla" cần disable AbsGS để fair với anh's recipe Inria.
# ════════════════════════════════════════════════════════

def _build_splatfacto_sparse_noabs_config():
    """[B5 v3] Splatfacto vanilla + 3-cam + COLMAP init + use_absgrad=False (pure 3DGS-style)."""

    from nerfstudio.models.splatfacto import SplatfactoModelConfig
    from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig

    dataparser_config = CrsGaussianDataParserConfig(
        colmap_path=Path("sparse/0"),
        images_path=Path("images"),
        eval_mode="interval",
        eval_interval=8,
        n_views_phase22=3,
        llffhold=8,
        downscale_factor=1,
        use_roma_init=False,  # COLMAP init (KHÔNG RoMa)
        roma_ply_relpath="3_views/dense/fused.ply.romav1",
        aligned_depth_relpath="3_views/aligned_depth_a23",
        load_3D_points=True,
    )

    return TrainerConfig(
        method_name="splatfacto-sparse-noabs",
        steps_per_eval_image=500,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=10000,
        max_num_iterations=10000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                dataparser=dataparser_config,
                cache_images_type="uint8",
                camera_res_scale_factor=0.125,
            ),
            # [v3 KEY] use_absgrad=False — disable AbsGS (KHÔNG vanilla, AbsGS là paper 2024)
            model=SplatfactoModelConfig(use_absgrad=False),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1.6e-6, max_steps=10000),
            },
            "features_dc": {"optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15), "scheduler": None},
            "features_rest": {"optimizer": AdamOptimizerConfig(lr=0.0025/20, eps=1e-15), "scheduler": None},
            "opacities": {"optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15), "scheduler": None},
            "scales": {"optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15), "scheduler": None},
            "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-7, max_steps=10000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
            "bilateral_grid": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=10000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="tensorboard",
    )


splatfacto_sparse_noabs_method_spec = MethodSpecification(
    config=_build_splatfacto_sparse_noabs_config(),
    description="[B5 v3] Splatfacto 3-cam + COLMAP init + use_absgrad=False (pure 3DGS-style, no AbsGS)",
)


__all__ = [
    "crsgaussian_method_spec",
    "splatfacto_sparse_method_spec",
    "splatfacto_17_method_spec",
    "splatfacto_sparse_noabs_method_spec",
]
