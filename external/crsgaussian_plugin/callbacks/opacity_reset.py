# ============================================================
# [CRSGaussian Path A B3 NEW] Opacity reset callback
# File: crsgaussian_plugin/callbacks/opacity_reset.py  (KEEP LOCAL)
#
# Source verbatim from CRSGaussian train.py:846-850:
#   if (iteration - args.start_sample_pseudo - 1) % opt.opacity_reset_interval == 0 and \
#           iteration > args.start_sample_pseudo:
#       gaussians.reset_opacity()
#
# Phase 22: start_sample_pseudo=500, opacity_reset_interval=3000
#   → reset at iter 501, 3501, 6501, 9501 (4 resets in 10k training)
# ============================================================
"""[Path A B3] Opacity reset callback — Phase 22 exact timing."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..crsgaussian_model import CrsGaussianModel


def make_opacity_reset_step_fn(model: "CrsGaussianModel"):
    """Closure callback function — Phase 22 reset timing."""

    def _step_fn(step: int):
        # Phase 22 condition (train.py:846-847):
        #   (iter - start_sample_pseudo - 1) % opacity_reset_interval == 0
        #   AND iter > start_sample_pseudo
        if step <= model.config.start_sample_pseudo:
            return
        offset = step - model.config.start_sample_pseudo - 1
        if offset % model.config.opacity_reset_interval != 0:
            return
        print(f"[Path A B3] reset_opacity at iter {step}")
        model.gaussians.reset_opacity()

    return _step_fn


__all__ = ["make_opacity_reset_step_fn"]
