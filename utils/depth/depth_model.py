# ============================================================
# [CRSGaussian] Task: T0.2 — depth_model.py
# File: CRSGaussian/utils/depth/depth_model.py  (TẠO MỚI)
# Mục đích: Wrapper DepthAnything V2 — precompute dense depth
#           prior cho tất cả training cameras. Gọi 1 lần trước
#           training loop, xong xóa model khỏi GPU.
# Được gọi từ: train.py, trước training loop
# ============================================================

import sys
import os
import cv2
import torch
import numpy as np


def precompute_depth_priors(cameras, dav2_repo_path, checkpoint_path=None,
                            encoder='vitl', input_size=518):
    """Chạy DepthAnything V2 trên tất cả training cameras.

    Args:
        cameras: list of Camera objects (có .original_image (3,H,W) float [0,1] cuda,
                 .uid, .image_width, .image_height)
        dav2_repo_path: path đến thư mục Depth-Anything-V2 repo
                        (chứa depth_anything_v2/ package)
        checkpoint_path: path đến .pth file. Nếu None, tự tìm trong
                         dav2_repo_path/checkpoints/depth_anything_v2_{encoder}.pth
        encoder: 'vits' | 'vitb' | 'vitl'
        input_size: resolution cho DAV2 inference (default 518)

    Returns:
        depth_prior_dict: {cam.uid: torch.Tensor (H, W) float32 trên CPU}
            Relative depth — giá trị lớn = xa hơn, scale arbitrary.
            Cần align với COLMAP trước khi dùng (xem depth_alignment.py).
    """
    # -- Import DAV2 từ repo path --
    # Thêm repo path vào sys.path tạm thời để import được
    # depth_anything_v2.dpt.DepthAnythingV2
    if dav2_repo_path not in sys.path:
        sys.path.insert(0, dav2_repo_path)

    from depth_anything_v2.dpt import DepthAnythingV2

    # -- Model config theo encoder --
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64,
                 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128,
                 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256,
                 'out_channels': [256, 512, 1024, 1024]},
    }
    assert encoder in model_configs, f"encoder must be one of {list(model_configs.keys())}"

    # -- Load model --
    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            dav2_repo_path, 'checkpoints', f'depth_anything_v2_{encoder}.pth')
    assert os.path.exists(checkpoint_path), \
        f"Checkpoint not found: {checkpoint_path}"

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model = model.to('cuda').eval()
    print(f"[CRSGaussian] DepthAnything V2 ({encoder}) loaded from {checkpoint_path}")

    # -- Inference trên từng camera --
    depth_prior_dict = {}

    for cam in cameras:
        # cam.original_image: (3, H, W) float [0,1] trên cuda
        # DAV2 infer_image cần: (H, W, 3) uint8 BGR numpy
        img_tensor = cam.original_image.detach().cpu()          # (3, H, W)
        img_np = img_tensor.permute(1, 2, 0).numpy()           # (H, W, 3) RGB [0,1]
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)  # RGB uint8
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)      # BGR uint8

        # infer_image trả về numpy (H, W) float32, relative depth
        depth_np = model.infer_image(img_bgr, input_size)

        depth_prior_dict[cam.uid] = torch.from_numpy(depth_np).float()  # CPU tensor

    print(f"[CRSGaussian] Depth priors computed for {len(depth_prior_dict)} cameras")

    # -- Giải phóng VRAM --
    del model
    torch.cuda.empty_cache()

    return depth_prior_dict
