# ============================================================
# [CRSGaussian Phase 11] DINOv2 wrapper — shared module cho Steps 2/3/4.
# File: utils/feature/dino_wrapper.py
# Mục đích:
#   - Load DINOv2 ViT-S/14 (88 MB ckpt download 1 lần qua timm).
#   - Extract patch features (1369 patches × 384 dim cho input 518×518).
#   - FeatureCache: pre-compute GT features per training camera trước
#     training loop → tránh re-compute mỗi iter.
#
# Memory: 1369 × 384 × 4 bytes = ~2 MB / camera. 3 LLFF cams = ~6 MB.
# Compute: ~10-20 ms / image trên GPU (model frozen, eval mode).
#
# Convention:
#   - Image input: (3, H, W) ∈ [0, 1] tensor trên cuda.
#   - Resize → (518, 518) (DINOv2 native, multiple of 14).
#   - ImageNet normalize trước forward.
#   - Patch grid (Hp, Wp) = (37, 37) sau resize.
#
# Loader: dùng timm thay torch.hub vì DINOv2 main repo dùng PEP 604
# syntax (`float | None`) cần Python 3.10+; corgs env Python 3.8 fail.
# timm 1.0.26+ host cùng official Meta DINOv2 weights, Python 3.8 OK.
# ============================================================

import torch
import torch.nn.functional as F


# ImageNet normalization constants — DINOv2 trained với chuẩn này.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


# Map model name cũ (torch.hub style) → timm model id.
# Giữ API public DINOWrapper(model_name='dinov2_vits14') không đổi để
# train.py / Step 2/3/4 callers không cần sửa.
_TIMM_NAME_MAP = {
    'dinov2_vits14': 'vit_small_patch14_dinov2.lvd142m',
    'dinov2_vitb14': 'vit_base_patch14_dinov2.lvd142m',
    'dinov2_vitl14': 'vit_large_patch14_dinov2.lvd142m',
    'dinov2_vitg14': 'vit_giant_patch14_dinov2.lvd142m',
}


class DINOWrapper:
    """DINOv2 ViT-S/14 wrapper qua timm. Load 1 lần, reuse cho mọi extract call."""

    def __init__(self, model_name='dinov2_vits14', device='cuda', input_size=518):
        import timm   # lazy import — chỉ load khi feature flag bật.
        # Translate torch.hub-style name → timm. Cho phép truyền thẳng timm id.
        timm_id = _TIMM_NAME_MAP.get(model_name, model_name)
        # dynamic_img_size=True cho phép input shape khác native 518 (vẫn
        # patch-aligned). Pretrained tải ckpt chính chủ Meta lần đầu, cache
        # ở ~/.cache/huggingface/hub/.
        self.model = timm.create_model(
            timm_id, pretrained=True, dynamic_img_size=True,
        ).to(device).eval()
        # Freeze model — Phase 11 dùng feature-extractor only, KHÔNG fine-tune.
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.device = device
        self.patch_size = 14   # DINOv2 ViT-S/14
        self.input_size = input_size  # 518 = 14 × 37

        # Buffers cho normalize (broadcast khi forward)
        self.mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std  = torch.tensor(_IMAGENET_STD,  device=device).view(1, 3, 1, 1)

    def get_patch_grid_shape(self, H=None, W=None):
        """(Hp, Wp) cho input đã resize về (input_size, input_size)."""
        H = H or self.input_size
        W = W or self.input_size
        return H // self.patch_size, W // self.patch_size

    def _preprocess(self, image):
        """(3, H, W) ∈ [0, 1] → (1, 3, input_size, input_size) normalized."""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        # Bilinear resize → multiple of 14.
        image = F.interpolate(
            image, size=(self.input_size, self.input_size),
            mode='bilinear', align_corners=False,
        )
        return (image - self.mean) / self.std

    def extract_patch_features(self, image):
        """Extract patch tokens. (3, H, W) → (Np, D).

        KHÔNG dùng @torch.no_grad() vì Step 2 perceptual loss cần grad
        flow ngược về image_render. Model parameters đã frozen
        (requires_grad_=False), nên không tốn memory cho activations grad
        của model — chỉ cho input image.

        timm forward_features trả tensor (1, 1+Np, D) với token đầu là CLS;
        drop CLS qua [:, 1:] để giữ shape (Np, D) tương đương torch.hub
        output dict['x_norm_patchtokens'] cũ.
        """
        x = self._preprocess(image)
        feats = self.model.forward_features(x)   # (1, 1+Np, D) — incl. CLS
        return feats[:, 1:].squeeze(0)            # (Np, D) — drop CLS


class FeatureCache:
    """Pre-compute + lưu GT patch features cho mỗi training camera.

    Computed 1 lần ở train init; immutable sau đó. Tất cả features detached
    (no grad), chỉ serve as supervision target.
    """

    def __init__(self, dino_wrapper, train_cameras):
        self.cache = {}
        # grid_shape inferred sau lần extract đầu tiên.
        self.grid_shape = dino_wrapper.get_patch_grid_shape()  # (37, 37)
        self.feature_dim = None

        with torch.no_grad():
            for cam in train_cameras:
                if not hasattr(cam, 'original_image'):
                    continue   # skip pseudo cams
                feat = dino_wrapper.extract_patch_features(cam.original_image)
                self.cache[cam.uid] = feat.detach()
                if self.feature_dim is None:
                    self.feature_dim = feat.shape[-1]

    def get(self, cam_uid):
        """Return (Np, D) tensor hoặc None nếu cam_uid không có cache."""
        return self.cache.get(cam_uid)

    def __len__(self):
        return len(self.cache)
