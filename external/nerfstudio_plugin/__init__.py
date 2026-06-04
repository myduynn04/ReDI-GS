# ============================================================
# [CRSGaussian Plug-in A1] Method registration cho Nerfstudio
# File: crsgaussian_plugin/__init__.py  (KEEP LOCAL — upload server)
#
# Register method `splatfacto-roma` qua MethodSpecification.
# Activate qua env var (develop) HOẶC entry point (production, sau):
#
#   # Develop — env var
#   export NERFSTUDIO_METHOD_CONFIGS="splatfacto-roma=crsgaussian_plugin:roma_method_spec"
#   export PYTHONPATH=/home/aidev/workspace/representation-3d/duyen/nerfstudio:$PYTHONPATH
#   ns-train splatfacto-roma --data ...
#
#   # Production — pyproject.toml entry_points (sau A1 verify)
#   [project.entry-points."nerfstudio.method_configs"]
#   splatfacto-roma = "crsgaussian_plugin:roma_method_spec"
#
# Khác splatfacto vanilla CHỈ ở 2 chỗ:
#   1. DataParser: ColmapDataParserConfig → RomaDataParserConfig (RoMa PLY init)
#   2. use_absgrad=False (fair với CRSGaussian A0 baseline, không có AbsGS)
# ============================================================
"""[CRSGaussian Plug-in A1] Register splatfacto-roma method vào Nerfstudio CLI."""

import copy
from pathlib import Path

from nerfstudio.configs.method_configs import method_configs
from nerfstudio.plugins.types import MethodSpecification

from .roma_dataparser import RomaDataParserConfig


def _build_roma_config():
    """Build TrainerConfig cho splatfacto-roma từ splatfacto vanilla base.

    KHÁC splatfacto vanilla 2 chỗ:
      1. Dataparser: NerfstudioDataParserConfig → RomaDataParserConfig
         (LLFF defaults: colmap_path=triangulated, images=images, eval_interval=8)
      2. use_absgrad = False (fair với CRSGaussian A0)
    """
    # Deep copy splatfacto vanilla
    base = copy.deepcopy(method_configs["splatfacto"])
    base.method_name = "splatfacto-roma"

    # Set dataparser = RomaDataParserConfig với LLFF defaults preset
    # User KHÔNG cần gõ `colmap` subcommand nữa, default đã đúng cho LLFF CoR-GS layout
    new_dp = RomaDataParserConfig(
        # ── LLFF CoR-GS layout defaults ──
        colmap_path=Path("triangulated"),
        images_path=Path("images"),
        eval_mode="interval",
        eval_interval=8,
        # ── Plug-in A1 fields ──
        use_roma_init=True,
        roma_ply_relpath="dense/fused.ply.romav1",
        # ── Khác để default của ColmapDataParserConfig ──
        load_3D_points=True,
    )
    base.pipeline.datamanager.dataparser = new_dp

    # Tắt AbsGS để fair compare với CRSGaussian A0 baseline
    base.pipeline.model.use_absgrad = False

    return base


roma_method_spec = MethodSpecification(
    config=_build_roma_config(),
    description="[CRSGaussian Plug-in A1] splatfacto + RoMa v1 dense init (Phase 22 contribution)",
)


def _build_roma_off_config():
    """[CRSGaussian Plug-in A1 — Tier 2.1] OFF flag config.

    Method `splatfacto-roma-off` — RomaDataParser fallback COLMAP points3D.bin.
    Dùng để verify Tier 2.1: PSNR khớp splatfacto vanilla (với colmap subcommand) ±0.10 dB.
    """
    config = _build_roma_config()
    config.method_name = "splatfacto-roma-off"
    config.pipeline.datamanager.dataparser.use_roma_init = False
    return config


roma_off_method_spec = MethodSpecification(
    config=_build_roma_off_config(),
    description="[CRSGaussian Plug-in A1 — Test 2.1] OFF flag: fallback COLMAP (verify byte-identical vanilla)",
)


# ============================================================
# [CRSGaussian Plug-in A2.1] splatfacto-roma + opacity decay
# ============================================================
# NOTE: import CrsgSplatfactoModelConfig LAZY (inside function body) để tránh
# circular import — tyro re-discover plugin lúc import nerfstudio splatfacto.

def _make_crsg_model_config(base_model, **overrides):
    """Helper: build CrsgSplatfactoModelConfig từ vanilla SplatfactoModelConfig.

    Preserve field inherit + cho overrides (vd use_opacity_decay, use_dropansh).
    """
    from .crsg_model import CrsgSplatfactoModelConfig  # LAZY import
    return CrsgSplatfactoModelConfig(
        warmup_length=base_model.warmup_length,
        refine_every=base_model.refine_every,
        resolution_schedule=base_model.resolution_schedule,
        background_color=base_model.background_color,
        num_downscales=base_model.num_downscales,
        cull_alpha_thresh=base_model.cull_alpha_thresh,
        cull_scale_thresh=base_model.cull_scale_thresh,
        reset_alpha_every=base_model.reset_alpha_every,
        densify_grad_thresh=base_model.densify_grad_thresh,
        use_absgrad=base_model.use_absgrad,
        densify_size_thresh=base_model.densify_size_thresh,
        n_split_samples=base_model.n_split_samples,
        sh_degree_interval=base_model.sh_degree_interval,
        cull_screen_size=base_model.cull_screen_size,
        split_screen_size=base_model.split_screen_size,
        stop_screen_size_at=base_model.stop_screen_size_at,
        random_init=base_model.random_init,
        ssim_lambda=base_model.ssim_lambda,
        stop_split_at=base_model.stop_split_at,
        sh_degree=base_model.sh_degree,
        **overrides,
    )


def _build_roma_opdecay_config():
    """[A2.1] A1 + opacity decay only (DropAnSH OFF)."""
    base = _build_roma_config()
    base.method_name = "splatfacto-roma-opdecay"
    base.pipeline.model = _make_crsg_model_config(
        base.pipeline.model,
        use_opacity_decay=True,
        opacity_decay_factor=0.999,
        opacity_decay_start_iter=500,
        use_dropansh=False,   # A2.1 isolate opacity decay
    )
    return base


roma_opdecay_method_spec = MethodSpecification(
    config=_build_roma_opdecay_config(),
    description="[A2.1] splatfacto + RoMa init + opacity decay",
)


def _build_roma_opdecay_off_config():
    """[A2.1 Tier 2.1] opacity decay OFF (DropAnSH cũng OFF) — byte-identical splatfacto-roma."""
    config = _build_roma_opdecay_config()
    config.method_name = "splatfacto-roma-opdecay-off"
    config.pipeline.model.use_opacity_decay = False
    return config


roma_opdecay_off_method_spec = MethodSpecification(
    config=_build_roma_opdecay_off_config(),
    description="[A2.1 Test 2.1] opacity decay OFF",
)


# ============================================================
# [CRSGaussian Plug-in A2.2] splatfacto-roma + opacity decay + DropAnSH
# ============================================================

def _build_roma_opdecay_dropansh_config():
    """[A2.2] A2.1 + DropAnSH (anchor + SH degree dropout, Phase 22 contribution #3)."""
    base = _build_roma_config()
    base.method_name = "splatfacto-roma-opdecay-dropansh"
    base.pipeline.model = _make_crsg_model_config(
        base.pipeline.model,
        use_opacity_decay=True,
        opacity_decay_factor=0.999,
        opacity_decay_start_iter=500,
        use_dropansh=True,
        dropansh_pa_max=0.02,
        dropansh_k=10,
        dropansh_psh=0.2,
        dropansh_total_iter=10000,
        dropansh_schedule=(2000, 4000, 6000),
    )
    return base


roma_opdecay_dropansh_method_spec = MethodSpecification(
    config=_build_roma_opdecay_dropansh_config(),
    description="[A2.2] splatfacto + RoMa init + opacity decay + DropAnSH (Phase 22 cascade)",
)


def _build_roma_opdecay_dropansh_off_config():
    """[A2.2 Tier 2.1] DropAnSH OFF — verify byte-identical splatfacto-roma-opdecay."""
    config = _build_roma_opdecay_dropansh_config()
    config.method_name = "splatfacto-roma-opdecay-dropansh-off"
    config.pipeline.model.use_dropansh = False
    return config


roma_opdecay_dropansh_off_method_spec = MethodSpecification(
    config=_build_roma_opdecay_dropansh_off_config(),
    description="[A2.2 Test 2.1] DropAnSH OFF",
)


# ============================================================
# [CRSGaussian Plug-in A2.3] splatfacto + RoMa + DAV2 depth loss
# (Clean ablation: SKIP opdecay + dropansh vì cả 2 wash trên splatfacto)
# ============================================================

def _build_roma_depth_config():
    """[A2.3] A1 (RoMa init) + DAV2 depth loss only.

    Skip opdecay + dropansh để clean ablation: chỉ verify depth loss alone effect.
    Phase 23 depth+CRS cascade = biggest contribution (−0.86).
    """
    base = _build_roma_config()
    base.method_name = "splatfacto-roma-depth"
    base.pipeline.model = _make_crsg_model_config(
        base.pipeline.model,
        use_opacity_decay=False,    # SKIP A2.1
        use_dropansh=False,          # SKIP A2.2
        use_depth_loss=True,
        depth_loss_weight=0.05,
        # Force depth output trong training (cần để compute depth_loss)
        output_depth_during_training=True,
    )
    return base


roma_depth_method_spec = MethodSpecification(
    config=_build_roma_depth_config(),
    description="[A2.3] splatfacto + RoMa init + DAV2 depth loss",
)


def _build_roma_depth_off_config():
    """[A2.3 Tier 2.1] depth loss OFF — verify byte-identical splatfacto-roma."""
    config = _build_roma_depth_config()
    config.method_name = "splatfacto-roma-depth-off"
    config.pipeline.model.use_depth_loss = False
    return config


roma_depth_off_method_spec = MethodSpecification(
    config=_build_roma_depth_off_config(),
    description="[A2.3 Test 2.1] depth loss OFF",
)


# ============================================================
# [CRSGaussian Plug-in A2.4] splatfacto + RoMa + depth + CRS module
# A2.4-minimal: CRS score (D-only) + log diagnostic. SH freeze deferred A3.
# ============================================================

def _build_roma_depth_crsg_config():
    """[A2.4] A2.3 + CRS score per-Gaussian computation.

    CRS score = sigmoid(scale × (D_i - 0.5)) updated EMA mỗi 100 iter.
    A2.4-minimal: D-only (skip R for simplicity).
    """
    base = _build_roma_config()
    base.method_name = "splatfacto-roma-depth-crsg"
    base.pipeline.model = _make_crsg_model_config(
        base.pipeline.model,
        use_opacity_decay=False,
        use_dropansh=False,
        use_depth_loss=True,
        depth_loss_weight=0.05,
        output_depth_during_training=True,
        # A2.4 — CRS module
        use_crs=True,
        crs_update_interval=100,
        crs_update_warmup=1000,
        crs_ema_decay=0.9,
        crs_logit_scale=5.0,
    )
    return base


roma_depth_crsg_method_spec = MethodSpecification(
    config=_build_roma_depth_crsg_config(),
    description="[A2.4] splatfacto + RoMa + depth + CRS module (Phase 22 cascade final)",
)


def _build_roma_depth_crsg_off_config():
    """[A2.4 Tier 2.1] CRS OFF — verify byte-identical splatfacto-roma-depth."""
    config = _build_roma_depth_crsg_config()
    config.method_name = "splatfacto-roma-depth-crsg-off"
    config.pipeline.model.use_crs = False
    return config


roma_depth_crsg_off_method_spec = MethodSpecification(
    config=_build_roma_depth_crsg_off_config(),
    description="[A2.4 Test 2.1] CRS OFF",
)
