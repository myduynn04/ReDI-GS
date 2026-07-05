"""Sinh camera trajectory mượt cho orbit GIF của Tab 1 và Tab 2.

Cách tiếp cận:
- Lấy tất cả LLFF camera (train + test) của 1 scene.
- Sắp xếp theo tên ảnh (image_XX.png) để đảm bảo trật tự dọc trajectory
  gốc của LLFF (camera đi vòng scene theo thứ tự chụp).
- Interpolate 60 frame giữa các camera liên tiếp bằng linear blend
  cho position và slerp cho rotation.
- Trả về list các MiniCam để feed vào renderer.

Không sinh camera đi ngược lại các góc chưa quan sát (không rotate 360
xung quanh scene). Ưu tiên an toàn: các frame nội suy đều nằm trong
phân phối pose training/test.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from scene.cameras import MiniCam


# ---------------------------------------------------------------------------
# Slerp cho quaternion
# ---------------------------------------------------------------------------


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix 3x3 sang quaternion (w, x, y, z)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) sang rotation matrix 3x3."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation giữa 2 quaternion (w,x,y,z)."""
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta_0 = float(np.arccos(dot))
    theta = theta_0 * t
    sin_theta_0 = float(np.sin(theta_0))
    sin_theta = float(np.sin(theta))
    s0 = float(np.cos(theta)) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q0 + s1 * q1


# ---------------------------------------------------------------------------
# Interpolate 1 camera qua Camera-based fields của 3DGS
# ---------------------------------------------------------------------------


def _get_R_t_from_camera(cam) -> tuple[np.ndarray, np.ndarray]:
    """Lấy R (3x3) và t (3,) numpy từ Camera object của 3DGS.

    Camera.R và Camera.T được lưu ở CPU numpy. Convention COLMAP:
    camera-to-world sau khi apply.
    """
    R = np.asarray(cam.R, dtype=np.float64)
    T = np.asarray(cam.T, dtype=np.float64).reshape(3)
    return R, T


def _interpolate_pose(cam_a, cam_b, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Nội suy R (slerp) và t (lerp) giữa 2 Camera object của 3DGS."""
    Ra, Ta = _get_R_t_from_camera(cam_a)
    Rb, Tb = _get_R_t_from_camera(cam_b)
    qa = _rotmat_to_quat(Ra)
    qb = _rotmat_to_quat(Rb)
    q = _slerp(qa, qb, t)
    R = _quat_to_rotmat(q)
    T = (1.0 - t) * Ta + t * Tb
    return R, T


# ---------------------------------------------------------------------------
# Public API — sinh MiniCam list cho orbit
# ---------------------------------------------------------------------------


def _build_minicam_from_pose(R: np.ndarray, T: np.ndarray, ref_cam) -> MiniCam:
    """Tạo MiniCam từ pose (R, T) đã nội suy, mượn intrinsic của ref_cam."""
    from utils.graphics_utils import getWorld2View2, getProjectionMatrix

    znear = getattr(ref_cam, "znear", 0.01)
    zfar = getattr(ref_cam, "zfar", 100.0)
    trans = getattr(ref_cam, "trans", np.array([0.0, 0.0, 0.0]))
    scale = getattr(ref_cam, "scale", 1.0)

    w2v = torch.tensor(
        getWorld2View2(R.astype(np.float32), T.astype(np.float32), trans, scale)
    ).transpose(0, 1).cuda()
    proj = getProjectionMatrix(
        znear=znear, zfar=zfar, fovX=ref_cam.FoVx, fovY=ref_cam.FoVy
    ).transpose(0, 1).cuda()
    full = (w2v.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)

    return MiniCam(
        width=int(ref_cam.image_width),
        height=int(ref_cam.image_height),
        fovy=ref_cam.FoVy,
        fovx=ref_cam.FoVx,
        znear=znear,
        zfar=zfar,
        world_view_transform=w2v,
        full_proj_transform=full,
    )


def build_orbit_trajectory(cameras: List, n_frames: int = 60) -> List[MiniCam]:
    """Nội suy `n_frames` MiniCam mượt qua danh sách cameras đã sort.

    Args:
        cameras: list Camera object (train + test), đã sort theo thứ tự
            LLFF trajectory (tên ảnh tăng dần).
        n_frames: tổng số frame trong GIF cuối cùng. Mặc định 60 -> 2s @30fps.

    Returns:
        list `n_frames` MiniCam, sẵn sàng feed vào render_utils.render_view.

    Phase 2.5 fix (2026-07-05) — GLOBAL PARAMETERIZATION:
        Bản cũ chia n_frames cho từng segment bằng integer division
        (`frames_per_seg = n_frames // n_seg`) gây 2 bug:
        1. Duplicate frame tại biên (t=1 seg_k = t=0 seg_{k+1} = cùng cam),
           tạo cảm giác GIF "pause" tại mỗi camera → giật giật.
        2. Frame dư dồn hết vào segment cuối (leftover = n_frames % n_seg),
           làm đoạn cuối mượt bất thường so với các đoạn trước.

        Bản mới compute global `u ∈ [0, n_seg]` cho từng frame, phân phối
        đều toàn trajectory, không duplicate biên.
    """
    if not cameras:
        raise ValueError("Empty cameras list")
    if len(cameras) == 1:
        # Chỉ 1 camera, trả về n_frames bản sao MiniCam của camera đó.
        R, T = _get_R_t_from_camera(cameras[0])
        cam0 = _build_minicam_from_pose(R, T, cameras[0])
        return [cam0 for _ in range(n_frames)]

    n_seg = len(cameras) - 1
    trajectory: List[MiniCam] = []
    for f in range(n_frames):
        # Global param u ∈ [0, n_seg] chia đều toàn trajectory.
        u = f / max(n_frames - 1, 1) * n_seg
        # Clamp seg vào [0, n_seg-1] để f cuối cùng (u = n_seg) không tràn index.
        seg = min(int(u), n_seg - 1)
        t = u - seg   # local t ∈ [0, 1], không lặp biên
        cam_a = cameras[seg]
        cam_b = cameras[seg + 1]
        R, T = _interpolate_pose(cam_a, cam_b, t)
        trajectory.append(_build_minicam_from_pose(R, T, cam_a))
    return trajectory


def _camera_center_world(cam) -> np.ndarray:
    """Vị trí camera trong world frame (3,) — pattern từ getNerfppNorm."""
    from utils.graphics_utils import getWorld2View2

    R = np.asarray(cam.R, dtype=np.float64)
    T = np.asarray(cam.T, dtype=np.float64).reshape(3)
    W2C = getWorld2View2(R, T)             # world → camera
    C2W = np.linalg.inv(W2C)               # camera → world
    return C2W[:3, 3]                      # cột tịnh tiến = camera center


def sort_llff_cameras(cameras: List) -> List:
    """Sort cameras theo PCA principal axis để sweep monotonic không zigzag.

    Phase 2.5b fix (2026-07-05): thay sort theo `image_name` (thứ tự chụp
    hand-held serpentine) bằng sort theo axis biến thiên nhiều nhất trong
    không gian. GIF orbit sẽ sweep smooth từ 1 đầu scene đến đầu kia.

    Quy trình:
      1. Compute vị trí camera trong world (via getWorld2View2 + inv).
      2. Center hoá + SVD → PC1 = hướng biến thiên nhất.
      3. Ensure direction dương X (L→R) để consistent qua các scene.
      4. Project positions lên PC1 + argsort.
    """
    if len(cameras) < 2:
        return list(cameras)

    positions = np.array([_camera_center_world(c) for c in cameras])  # (N, 3)
    centered = positions - positions.mean(axis=0)
    # SVD stable hơn np.linalg.eig cho covariance ill-conditioned.
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    axis = Vt[0]                            # principal direction (3,)

    # Đảm bảo direction từ Trái sang Phải: PC1[0] dương.
    # LLFF world frame typical: X = left-right, ta pin direction dương X.
    if axis[0] < 0:
        axis = -axis

    projections = centered @ axis           # scalar mỗi camera
    order = np.argsort(projections)
    return [cameras[i] for i in order]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    from demo.render_utils import load_llff_scene

    parser = argparse.ArgumentParser(description="Smoke test trajectory_utils.")
    parser.add_argument("--source", required=True,
                        help="LLFF scene dir")
    parser.add_argument("--n-frames", type=int, default=60)
    args = parser.parse_args()

    scene = load_llff_scene(args.source, n_views=3, resolution=8)
    all_cams = list(scene.getTrainCameras()) + list(scene.getTestCameras())
    all_cams = sort_llff_cameras(all_cams)
    print(f"[demo] {len(all_cams)} cameras total ({args.source})")

    traj = build_orbit_trajectory(all_cams, n_frames=args.n_frames)
    print(f"[demo] built trajectory with {len(traj)} MiniCam frames")
    print(f"[demo] first cam WVT[:2,:3] =\n{traj[0].world_view_transform[:2, :3].cpu().numpy()}")
