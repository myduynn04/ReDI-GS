# [CRSGaussian] Regularizer module — features anti-overfit
from utils.regularizer.pseudo_photo import compute_pseudo_photo_loss
from utils.regularizer.pseudo_depth import compute_pseudo_depth_loss

__all__ = [
    "compute_pseudo_photo_loss",
    "compute_pseudo_depth_loss",
]
