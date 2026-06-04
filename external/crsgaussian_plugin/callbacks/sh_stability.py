# ============================================================
# [CRSGaussian Path A B3 NEW] SH stability EMA (Phase 8b)
# File: crsgaussian_plugin/callbacks/sh_stability.py  (KEEP LOCAL)
#
# Source verbatim from CRSGaussian train.py:797-805:
#   if (opt.use_sh_reliability
#           AND iteration > opt.sh_stability_warmup
#           AND iteration % opt.crs_update_interval == 0):
#       update_sh_stability(gaussians, beta=opt.sh_stability_ema_beta)
#
# Gọi SAU optimizer.step() để capture POST-update SH state.
# Signature verified in CRSGaussian/utils/crs/sh_stability.py:15.
# ============================================================
"""[Path A B3] SH stability EMA — Phase 8b, called AFTER optimizer.step."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..crsgaussian_model import CrsGaussianModel


def make_sh_stability_step_fn(model: "CrsGaussianModel"):
    """Closure callback function — EMA mean/variance of _features_rest."""

    def _step_fn(step: int):
        if not model.config.use_sh_reliability:
            return
        if step <= model.config.sh_stability_warmup:
            return
        if step % model.config.crs_update_interval != 0:
            return
        from utils.crs.sh_stability import update_sh_stability

        update_sh_stability(model.gaussians, beta=model.config.sh_stability_ema_beta)

    return _step_fn


__all__ = ["make_sh_stability_step_fn"]
