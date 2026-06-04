# ============================================================
# [CRSGaussian Path A B3 REWRITE] Densify callback — Phase 22 full LFCF support
# File: crsgaussian_plugin/callbacks/densify.py  (KEEP LOCAL)
#
# Source verbatim from CRSGaussian train.py:626-732. Two-step:
#   1. ALWAYS (when iter < densify_until): add_densification_stats
#   2. CONDITIONAL (when iter > densify_from AND iter % densification_interval == 0):
#      - build lfcf_opts if use_lfcf AND is_lfcf_iter (iter % (lfcf_interval_times * densification_interval) == 0)
#      - densify_and_prune(... full kwargs including LFCF)
#
# densify_and_prune signature (CRSGaussian/scene/gaussian_model.py:854-858):
#   def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, iter,
#                         cameras=None, aligned_depth_dict=None, depth_range=None,
#                         T_warmup=1000, tau_crs=0.35, tau_isolated=0.1,
#                         crs_prune_dict=None, eta=0.0,
#                         is_lfcf_iter=False, lfcf_opts=None, cameras_for_lfcf=None)
#
# Phase 22 size_threshold = None (train.py:657-658).
# ============================================================
"""[Path A B3] Full Phase 22 densify callback with LFCF + AbsGS."""

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..crsgaussian_model import CrsGaussianModel


def _build_lfcf_opts(model: "CrsGaussianModel", step: int):
    """Build lfcf_opts dict per CRSGaussian train.py:678-708."""
    cfg = model.config
    if cfg.lfcf_init_scaling_max != 1.0:
        percent_lb = math.log(cfg.lfcf_last_scaling_max) / math.log(cfg.lfcf_init_scaling_max)
    else:
        percent_lb = 1.0

    from utils.densify.lfcf import calculate_training_percent_powered

    tpp = calculate_training_percent_powered(
        step, cfg.densify_from_iter, cfg.densify_until_iter,
        cfg.lfcf_pow, percent_lb,
    )
    splitting_lb_now = 1.0 - (step - cfg.densify_from_iter) / max(
        cfg.densify_until_iter - cfg.densify_from_iter, 1
    )
    return {
        'scaling_multiplier_max': cfg.lfcf_init_scaling_max,
        'scaling_multiplier_min': cfg.lfcf_init_scaling_min,
        'training_percent_powered': tpp,
        'splitting_ub': cfg.lfcf_splitting_ub,
        'splitting_lb': splitting_lb_now,
        'tolerance': cfg.lfcf_tolerance,
        'diffscale': cfg.lfcf_diffscale,
    }


def make_densify_step_fn(model: "CrsGaussianModel"):
    """Phase 22 densify callback — match train.py:626-732 logic.

    Reads from model: _last_viewspace_pts, _last_visibility, _last_radii,
                      _cameras_extent, _train_corgs_cams (cho LFCF)
    """

    def _step_fn(step: int):
        if not model.config.use_densify:
            return
        if step >= model.config.densify_until_iter:
            return

        gaussians = model.gaussians

        # ── Pre-check: render state ──
        if (
            getattr(model, "_last_visibility", None) is None
            or getattr(model, "_last_radii", None) is None
            or getattr(model, "_last_viewspace_pts", None) is None
        ):
            return

        visibility = model._last_visibility
        radii = model._last_radii
        viewspace_pts = model._last_viewspace_pts

        # ── Step 1: Update max_radii2D + add_densification_stats (mỗi iter trong range) ──
        try:
            gaussians.max_radii2D[visibility] = torch.max(
                gaussians.max_radii2D[visibility], radii[visibility]
            )
            gaussians.add_densification_stats(viewspace_pts, visibility)
        except Exception as e:
            print(f"[Path A B3 densify] stats update ERROR: {e}")
            return

        # ── Step 2: Densify + prune (mỗi densification_interval, iter > densify_from) ──
        if step <= model.config.densify_from_iter:
            return
        if step % model.config.densification_interval != 0:
            return

        # ── Phase 22: size_threshold = None ALWAYS (train.py:657) ──
        size_threshold = None

        # ── LFCF: build lfcf_opts nếu is_lfcf_iter ──
        is_lfcf_iter_now = False
        lfcf_opts = None
        cameras_for_lfcf = None
        if model.config.use_lfcf:
            lfcf_cadence = model.config.lfcf_interval_times * model.config.densification_interval
            is_lfcf_iter_now = (step % lfcf_cadence == 0)
            if is_lfcf_iter_now:
                lfcf_opts = _build_lfcf_opts(model, step)
                cameras_for_lfcf = model._train_corgs_cams

        try:
            # Full Phase 22 signature — match train.py:713-727
            gaussians.densify_and_prune(
                model.config.densify_grad_threshold,    # max_grad = 0.0005
                model.config.prune_threshold,           # min_opacity = 0.005
                model._cameras_extent,                  # extent
                size_threshold,                         # = None (Phase 22)
                step,                                   # iter
                cameras=None,                           # position_constraint not used Phase 22
                aligned_depth_dict=None,
                depth_range=None,
                T_warmup=model.config.T_warmup,
                tau_crs=model.config.tau_crs,
                tau_isolated=model.config.tau_isolated,
                # crs_prune_dict=None, eta=0.0 (defaults — Phase 22 trim)
                is_lfcf_iter=is_lfcf_iter_now,
                lfcf_opts=lfcf_opts,
                cameras_for_lfcf=cameras_for_lfcf,
            )
        except Exception as e:
            print(f"[Path A B3 densify] densify_and_prune ERROR at iter {step}: {e}")
            import traceback; traceback.print_exc()
            return

        # Clear render state (N thay đổi sau densify)
        model._last_viewspace_pts = None
        model._last_visibility = None
        model._last_radii = None

        # Periodic log
        if step % 500 == 0:
            n_after = gaussians._xyz.shape[0]
            print(f"[Path A B3 densify] step={step}: N_gauss={n_after} (lfcf={is_lfcf_iter_now})")


    return _step_fn


__all__ = ["make_densify_step_fn"]
