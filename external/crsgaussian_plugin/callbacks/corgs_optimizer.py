# ============================================================
# [CRSGaussian Path A B3 rewrite] CoR-GS optimizer step + LR update
# File: crsgaussian_plugin/callbacks/corgs_optimizer.py  (KEEP LOCAL)
#
# Source verbatim from CRSGaussian train.py:786-790 + 845:
#   if iteration < opt.iterations:
#       gaussians.optimizer.step()
#       gaussians.optimizer.zero_grad(set_to_none=True)
#   gaussians.update_learning_rate(iteration)
#
# Note: oneupSHdegree (train.py:120) moved to BEFORE_TRAIN_ITERATION callback
# (model._sh_increment_step) — KHÔNG handle ở đây.
# ============================================================
"""[Path A B3] CoR-GS optimizer step callback (apply grads + LR scheduler)."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..crsgaussian_model import CrsGaussianModel


def make_corgs_optimizer_step_fn(model: "CrsGaussianModel"):
    """Closure callback function.

    Called AFTER_TRAIN_ITERATION, AFTER densify + SH freeze (Phase 22 order I).
    SH stability EMA callback runs AFTER this (Phase 22 order J).
    """

    def _step_fn(step: int):
        gaussians = model.gaussians
        # 1. Optimizer step (skip last iter — train.py:787)
        if step < model.config.iterations:
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
        # 2. Update LR scheduler (train.py:845, mỗi iter)
        gaussians.update_learning_rate(step)

    return _step_fn


__all__ = ["make_corgs_optimizer_step_fn"]
