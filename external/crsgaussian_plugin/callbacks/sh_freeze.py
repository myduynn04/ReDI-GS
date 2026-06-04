# ============================================================
# [CRSGaussian Path A B3 NEW] CRS-modulated SH freeze (Phase 8c)
# File: crsgaussian_plugin/callbacks/sh_freeze.py  (KEEP LOCAL)
#
# Source verbatim from CRSGaussian train.py:776-784:
#   if opt.use_crs_modulated_sh_freeze:
#       apply_crs_modulated_sh_freeze(
#           gaussians, iter=iteration,
#           freeze_start=opt.crs_freeze_start, tau_freeze=opt.crs_freeze_tau,
#       )
#
# CRITICAL: phải gọi TRƯỚC optimizer.step() để zero grad có effect.
# Function signature verified in CRSGaussian/utils/crs/sh_freeze.py:15.
# Phase 8c +0.189 dB — biggest single winner trong recipe.
# ============================================================
"""[Path A B3] CRS-modulated SH freeze — Phase 8c (+0.189 dB winner)."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..crsgaussian_model import CrsGaussianModel


def make_sh_freeze_step_fn(model: "CrsGaussianModel"):
    """Closure callback function — zero _features_rest.grad cho Gaussian CRS thấp."""

    def _step_fn(step: int):
        if not model.config.use_crs_modulated_sh_freeze:
            return
        # sh_freeze.py:39-40 internal skip nếu iter <= freeze_start
        # → calls iter 1001-10000 = exact 9000 iter
        from utils.crs.sh_freeze import apply_crs_modulated_sh_freeze

        apply_crs_modulated_sh_freeze(
            model.gaussians,
            iter=step,
            freeze_start=model.config.crs_freeze_start,
            tau_freeze=model.config.crs_freeze_tau,
        )

    return _step_fn


__all__ = ["make_sh_freeze_step_fn"]
