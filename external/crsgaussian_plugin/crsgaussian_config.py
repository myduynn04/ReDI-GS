# ============================================================
# [CRSGaussian Path A B3] CrsGaussianModelConfig — full Phase 22 hyperparams
# File: crsgaussian_plugin/crsgaussian_config.py  (KEEP LOCAL — upload server)
#
# Source of truth: scripts/p22_pilot_run.sh → scripts/p20_ablation_dense_run.sh
# CONFIG=trim_full (Phase 22 A3-TRIM 8-module recipe).
#
# All flag defaults verified against:
#   - CRSGaussian/arguments/__init__.py
#   - CRSGaussian/scripts/p20_ablation_dense_run.sh lines 50-64
# ============================================================
"""[Path A B3] Config dataclass cho method `crsgaussian` — full Phase 22."""

from dataclasses import dataclass, field
from typing import Type

from nerfstudio.models.base_model import ModelConfig


@dataclass
class CrsGaussianModelConfig(ModelConfig):
    """[Path A B3] Phase 22 A3-TRIM full recipe (8 modules)."""

    _target: Type = field(default_factory=lambda: None)  # lazy from __init__.py

    # ════════════════════════════════════════════════════════
    # Core training PROTOCOL (p20_ablation_dense_run.sh:50-52)
    # ════════════════════════════════════════════════════════
    iterations: int = 10000
    resolution: int = 8
    n_views: int = 3
    random_background: bool = True
    sh_degree: int = 3
    white_background: bool = False

    # ── Densify master switch + schedule (Phase 22 different from defaults!) ──
    use_densify: bool = True                # Master switch — enable densify callback
    densify_from_iter: int = 500
    densify_until_iter: int = 5000          # Phase 22: 5000 (NOT 15000)
    densify_grad_threshold: float = 0.0005  # Phase 22: 0.0005 (NOT 0.0002)
    densification_interval: int = 100
    prune_threshold: float = 0.005

    # ── Opacity reset timing (CRSGaussian train.py:846) ──
    opacity_reset_interval: int = 3000
    start_sample_pseudo: int = 500          # Phase 22: 500 → reset at 501/3501/6501/9501

    # ════════════════════════════════════════════════════════
    # M_DEPTHCFG — depth prior + CRS framework cadence
    # ════════════════════════════════════════════════════════
    use_depth_prior: bool = True
    depth_loss_weight: float = 0.05         # Phase 3 default
    crs_update_interval: int = 100
    crs_ema_decay: float = 0.3              # Phase 22: 0.3 (NOT 0.9 default)
    T_warmup: int = 1000

    # ════════════════════════════════════════════════════════
    # M_DCYCLE — depth cycle signal (Phase 7+)
    # ════════════════════════════════════════════════════════
    use_d_cycle: bool = True
    d_cycle_warmup: int = 1000
    d_cycle_sigma: float = 5.0
    d_cycle_update_freq: int = 100

    # ── CRS formula weights (default w_d=0.5, w_r=0.5) ──
    crs_w1: float = 0.5    # w_d
    crs_w2: float = 0.5    # w_r (won't be used if disable_r_signal)
    crs_scale: float = 5.0
    disable_r_signal: bool = False  # Phase 9 D-only; Phase 22 leaves R weight=0.5 but R compute skipped via use_r_visible=False

    # ════════════════════════════════════════════════════════
    # M_SHFREEZE — Phase 8c CRS-modulated SH freeze (biggest +0.189)
    # ════════════════════════════════════════════════════════
    use_crs_modulated_sh_freeze: bool = True
    crs_freeze_start: int = 1000
    crs_freeze_tau: float = 0.5

    # ════════════════════════════════════════════════════════
    # M_SHREL — Phase 8b S_stability EMA
    # ════════════════════════════════════════════════════════
    use_sh_reliability: bool = True
    sh_stability_warmup: int = 1000
    sh_stability_ema_beta: float = 0.95
    crs_w_s: float = 0.33

    # ════════════════════════════════════════════════════════
    # M_DROP — DropAnSH (cross-backbone-stable pillar)
    # ════════════════════════════════════════════════════════
    use_dropansh: bool = True
    dropansh_pa: float = 0.02               # anchor sample rate
    dropansh_psh: float = 0.2               # per-Gaussian SH dropout probability
    dropansh_k: int = 10
    dropansh_total_iter: int = 10000        # for linear ramp
    dropansh_schedule_0: int = 2000
    dropansh_schedule_1: int = 4000
    dropansh_schedule_2: int = 6000

    # ════════════════════════════════════════════════════════
    # M_OPACITY — opacity decay (cross-backbone-stable pillar)
    # ════════════════════════════════════════════════════════
    use_opacity_decay: bool = True
    opacity_decay_factor: float = 0.999     # Phase 22: 0.999 for 10k iter

    # ════════════════════════════════════════════════════════
    # M_EFA — LFCF + AbsGS densifier (Phase 13 +0.164 winner)
    # ════════════════════════════════════════════════════════
    use_lfcf: bool = True
    lfcf_init_scaling_max: float = 1.5
    lfcf_init_scaling_min: float = 1.0
    lfcf_last_scaling_max: float = 1.0
    lfcf_pow: float = 1.0
    lfcf_splitting_ub: float = 1.0
    lfcf_interval_times: int = 2            # LFCF mỗi 2*densification_interval = 200 iter
    lfcf_tolerance: float = 1e-5
    lfcf_diffscale: bool = True
    absdensify: bool = True

    # ════════════════════════════════════════════════════════
    # CoR-GS optimizer hyperparams (training_setup)
    # ════════════════════════════════════════════════════════
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

    # ════════════════════════════════════════════════════════
    # CRS pruning thresholds (DEFAULT values — Phase 22 use_crs_pruning=False)
    # Pass to densify_and_prune for completeness (will not trigger pruning logic)
    # ════════════════════════════════════════════════════════
    tau_crs: float = 0.35
    tau_isolated: float = 0.1

    # ── SH increment cadence (CoR-GS train.py:120) ──
    sh_increment_interval: int = 500


__all__ = ["CrsGaussianModelConfig"]
