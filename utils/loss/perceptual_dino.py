# ============================================================
# [CRSGaussian Phase 11 Step 2] Same-view perceptual loss via DINOv2.
# File: utils/loss/perceptual_dino.py (NEW)
# Mục đích:
#   - Render image @ training cam → DINO patch features (F_render).
#   - GT image @ same cam → DINO patch features (F_GT, từ cache).
#   - Distance F_render vs F_GT → perceptual loss (LPIPS-style với
#     DINOv2 backbone — self-supervised, không phụ thuộc VGG ImageNet
#     supervised features).
#   - NO warping → same-view, đơn giản, gradient flow trực tiếp về
#     rendered image (model frozen, requires_grad đã off).
#
# Optional: weight per-patch bằng CRS_pix downsampled (Hp, Wp) khi
#           dataset.perceptual_dino_crs_weight=True.
# ============================================================

import torch
import torch.nn.functional as F


def perceptual_loss_dino(image_render, cam_uid, dino_cache, dino,
                          mode='cosine', crs_pix=None):
    """Same-view perceptual loss DINOv2.

    Args:
        image_render: (3, H, W) tensor render output (gradient enabled).
        cam_uid: int — viewpoint cam id để lookup GT features.
        dino_cache: FeatureCache — chứa GT patch features per cam.
        dino: DINOWrapper.
        mode: 'cosine' (1 - cos_sim) | 'l1' (mean L1).
        crs_pix: optional (H, W) ∈ [0, 1] — pixel-level CRS map. Khi có,
                 downsample về (Hp, Wp) qua adaptive_avg_pool2d và weight
                 per-patch loss. Pixel reliable hơn → loss vùng đó được
                 emphasize.

    Returns:
        scalar tensor (gradient-enabled). Trả 0 (no-op) nếu cache miss.
    """
    F_GT = dino_cache.get(cam_uid)
    if F_GT is None:
        # Cache miss → no-op, tránh crash. Trả tensor 0 trên cùng device.
        return torch.zeros((), device=image_render.device)

    # Extract render features (gradient-enabled — model params frozen,
    # nhưng activations qua model giữ grad flow ngược về image_render).
    F_render = dino.extract_patch_features(image_render)   # (Np, D)

    # Distance per-patch (Np,) — không reduce ở đây để chấp nhận weight sau.
    if mode == 'cosine':
        sim = F.cosine_similarity(F_render, F_GT, dim=-1)  # (Np,) ∈ [-1, 1]
        per_patch = 1.0 - sim                              # (Np,) ≥ 0
    elif mode == 'l1':
        per_patch = F.l1_loss(F_render, F_GT, reduction='none').mean(dim=-1)
    else:
        raise ValueError(f"[Phase 11 Step 2] mode invalid: {mode}")

    if crs_pix is None:
        return per_patch.mean()

    # CRS-weighted: downsample crs_pix (H, W) về (Hp, Wp), weight per-patch.
    Hp, Wp = dino_cache.grid_shape
    crs_down = F.adaptive_avg_pool2d(
        crs_pix.unsqueeze(0).unsqueeze(0), output_size=(Hp, Wp),
    ).flatten()  # (Np,)
    weighted_sum = (per_patch * crs_down).sum()
    return weighted_sum / (crs_down.sum() + 1e-6)
