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
        # LLFF CoR-GS layout defaults
        colmap_path=Path("triangulated"),
        images_path=Path("images"),
        eval_mode="interval",
        eval_interval=8,
        # Path A specific
        use_roma_init=True,
        roma_ply_relpath="dense/fused.ply.romav1",
        aligned_depth_relpath="aligned_depth_a23",
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


__all__ = ["crsgaussian_method_spec"]
