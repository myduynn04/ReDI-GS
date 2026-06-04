# ============================================================
# [CRSGaussian Path A B3 rewrite] Opacity decay — call CRSGaussian method
# File: crsgaussian_plugin/callbacks/opacity_decay.py  (KEEP LOCAL)
#
# Source verbatim from CRSGaussian train.py:813-816:
#   if (dataset.use_opacity_decay
#           and iteration > opt.densify_from_iter):
#       gaussians.opacity_decay(factor=dataset.opacity_decay_factor)
#
# Method exists in CRSGaussian/scene/gaussian_model.py:251 — em CALL nó,
# KHÔNG reimplement logit math (em đã sai trước đó).
# ============================================================
"""[Path A B3] Opacity decay callback — wrap gaussians.opacity_decay()."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..crsgaussian_model import CrsGaussianModel


def make_opacity_decay_step_fn(model: "CrsGaussianModel"):
    """Closure callback function.

    Phase 22 condition: iteration > densify_from_iter (per train.py:814).
    """

    def _step_fn(step: int):
        if not model.config.use_opacity_decay:
            return
        if step <= model.config.densify_from_iter:
            return
        # Call CRSGaussian method (gaussian_model.py:251)
        model.gaussians.opacity_decay(factor=model.config.opacity_decay_factor)

    return _step_fn


__all__ = ["make_opacity_decay_step_fn"]
