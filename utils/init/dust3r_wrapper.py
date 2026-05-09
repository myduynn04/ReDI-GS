# ============================================================
# [CRSGaussian Phase 10A] DUSt3R wrapper for dense initialization
# File: utils/init/dust3r_wrapper.py (NEW)
# Mục đích: Load cached DUSt3R point cloud (precomputed bởi
#           scripts/precompute_dust3r.py), filter theo confidence,
#           subsample, merge/replace với COLMAP PC.
#
# QUAN TRỌNG: Wrapper này KHÔNG chạy DUSt3R inference. Inference
#             nặng (load 1.7GB checkpoint, ViT forward, global align)
#             thực hiện 1-lần ở pre-compute step. Wrapper chỉ
#             thao tác numpy arrays đã cache → load nhanh, không
#             ảnh hưởng training time.
#
# Cache format: <cache_dir>/<scene_name>.npz với 3 arrays:
#   points     (N, 3) float32 — world-space XYZ (COLMAP frame)
#   colors     (N, 3) float32 — RGB in [0, 1]
#   confidence (N,)   float32 — DUSt3R log-confidence
# ============================================================

import os
import numpy as np


def load_dust3r_cache(cache_dir, scene_name):
    """Load precomputed DUSt3R PC từ <cache_dir>/<scene_name>.npz.

    Returns:
        points     (N, 3) float32
        colors     (N, 3) float32 in [0, 1]
        confidence (N,)   float32
    """
    cache_path = os.path.join(cache_dir, f"{scene_name}.npz")
    if not os.path.isfile(cache_path):
        raise FileNotFoundError(
            f"[Phase 10A] DUSt3R cache không tồn tại: {cache_path}\n"
            f"  Chạy scripts/precompute_dust3r.py trước.")
    data = np.load(cache_path)
    pts = data["points"].astype(np.float32)
    cols = data["colors"].astype(np.float32)
    conf = data["confidence"].astype(np.float32)
    # Sanity: stripped clip colors về [0,1] (DUSt3R imgs đã normalize sẵn).
    cols = np.clip(cols, 0.0, 1.0)
    return pts, cols, conf


def filter_by_confidence(points, colors, conf, threshold):
    """Loại bỏ points có conf < threshold.

    DUSt3R log-confidence: typical values [1.0, 5.0]. threshold=1.5
    là default phù hợp cho LLFF scenes (giữ ~70-90% points)."""
    mask = conf >= threshold
    return points[mask], colors[mask], conf[mask]


def subsample_to_max(points, colors, max_points, mode="confidence", conf=None):
    """Giảm số points xuống <= max_points.

    Mode:
        "confidence" (default): giữ top-K theo conf (cần truyền conf).
        "random":               sub-sample uniform.

    Trả về (points, colors). Nếu len(points) <= max_points, return as-is.
    """
    n = len(points)
    if n <= max_points:
        return points, colors

    if mode == "confidence" and conf is not None:
        # argpartition + slice — nhanh hơn argsort full khi N lớn.
        top_idx = np.argpartition(-conf, max_points)[:max_points]
        return points[top_idx], colors[top_idx]
    else:
        rng = np.random.default_rng(seed=0)
        idx = rng.choice(n, size=max_points, replace=False)
        return points[idx], colors[idx]


def merge_with_colmap(colmap_points, colmap_colors,
                      dust3r_points, dust3r_colors,
                      dedupe_radius):
    """Concat COLMAP + DUSt3R; loại DUSt3R points trùng với COLMAP
    trong bán kính dedupe_radius (đơn vị scene metric).

    Triết lý: COLMAP points có triangulation chính xác (multi-view
    SfM verified) → ưu tiên giữ. DUSt3R fill-in vùng COLMAP thưa
    (textureless surfaces, ngoài keypoint regions). Tránh duplicate
    để không inflate count vô ích.

    KDTree.query(k=1) cho mỗi DUSt3R point → distance tới COLMAP
    nearest neighbor; nếu > dedupe_radius thì giữ.
    """
    from scipy.spatial import cKDTree
    if len(colmap_points) == 0:
        return dust3r_points, dust3r_colors
    if len(dust3r_points) == 0:
        return colmap_points, colmap_colors

    tree = cKDTree(colmap_points)
    dists, _ = tree.query(dust3r_points, k=1)
    keep_mask = dists > dedupe_radius
    kept_pts = dust3r_points[keep_mask]
    kept_cols = dust3r_colors[keep_mask]

    merged_pts = np.concatenate([colmap_points, kept_pts], axis=0)
    merged_cols = np.concatenate([colmap_colors, kept_cols], axis=0)
    return merged_pts.astype(np.float32), merged_cols.astype(np.float32)
