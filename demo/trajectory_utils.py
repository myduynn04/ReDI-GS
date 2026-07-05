"""Sinh camera trajectory mượt cho orbit GIF của Tab 1 và Tab 2.

Cách tiếp cận (Phase 2.5c — ellipse analytic, MẶC ĐỊNH):
- KHÔNG nội suy qua từng camera thô nữa. Camera LLFF là hand-held nên cả
  vị trí lẫn hướng đều có noise; mọi path đi QUA từng waypoint (dù sort
  thông minh đến đâu) đều kế thừa noise đó → giật.
- Thay vào đó chỉ dùng THỐNG KÊ của cụm pose: mean, mặt phẳng PCA
  (PC1-PC2), spread percentile 5-95, và focus point (giao least-squares
  của các optical axis). Sinh path ellipse giải tích trong mặt phẳng đó,
  rotation look-at về focus point.
- Bán trục ellipse ≤ 0.9 × spread thật → mọi frame nằm TRONG phân phối
  pose train/test, không lộ vùng chưa quan sát (giữ ràng buộc an toàn cũ).
- θ chạy trọn [0, 2π) không lặp endpoint → GIF autoplay loop LIỀN MẠCH.

Legacy (mode="sweep"): sort PCA + nội suy slerp/lerp global-param qua
waypoint — giữ lại để A/B so sánh.

Trả về list các MiniCam để feed vào renderer.
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


def build_orbit_trajectory(cameras: List, n_frames: int = 60,
                           mode: str = "ellipse",
                           radius_scale: float = 0.9) -> List[MiniCam]:
    """Sinh `n_frames` MiniCam cho orbit GIF.

    Args:
        cameras: list Camera object (train + test). Với mode="ellipse"
            KHÔNG cần sort trước (path chỉ dùng thống kê cụm pose, bất biến
            với thứ tự). Với mode="sweep" cần sort trước (sort_llff_cameras).
        n_frames: tổng số frame trong GIF cuối cùng. Mặc định 60 -> 2s @30fps.
        mode: "ellipse" (mặc định, Phase 2.5c) hoặc "sweep" (legacy 2.5b)
            để A/B so sánh.
        radius_scale: hệ số co bán trục ellipse so với spread thật (0.9 =
            path nằm gọn trong 90% phân phối pose).

    Returns:
        list `n_frames` MiniCam, sẵn sàng feed vào render_utils.render_view.

    Lịch sử fix:
    Phase 2.5 (2026-07-05) — GLOBAL PARAMETERIZATION:
        Bản cũ chia n_frames cho từng segment bằng integer division
        (`frames_per_seg = n_frames // n_seg`) gây 2 bug:
        1. Duplicate frame tại biên (t=1 seg_k = t=0 seg_{k+1} = cùng cam),
           tạo cảm giác GIF "pause" tại mỗi camera → giật giật.
        2. Frame dư dồn hết vào segment cuối (leftover = n_frames % n_seg),
           làm đoạn cuối mượt bất thường so với các đoạn trước.
        Bản mới compute global `u ∈ [0, n_seg]` cho từng frame.

    Phase 2.5b (2026-07-05) — PCA SORT:
        Sort theo image_name (thứ tự chụp serpentine) → zigzag L/R.
        Fix: sort theo projection lên PC1. Nhưng chỉ monotonic được 1 trục;
        Y/Z và rotation của waypoint thô vẫn zigzag → GIF giật lên xuống.

    Phase 2.5c (2026-07-05) — ELLIPSE ANALYTIC (root fix):
        Bỏ hẳn nội suy qua waypoint thô. Path = ellipse trong mặt phẳng
        PCA của cụm camera, rotation = look-at về focus point ước lượng.
        Zero jitter cả position lẫn rotation + GIF loop liền mạch.
    """
    if not cameras:
        raise ValueError("Empty cameras list")
    if len(cameras) == 1:
        # Chỉ 1 camera, trả về n_frames bản sao MiniCam của camera đó.
        R, T = _get_R_t_from_camera(cameras[0])
        cam0 = _build_minicam_from_pose(R, T, cameras[0])
        return [cam0 for _ in range(n_frames)]
    if mode == "ellipse" and len(cameras) >= 3:
        return _build_ellipse_trajectory(cameras, n_frames, radius_scale)
    return _build_sweep_trajectory(cameras, n_frames)


def _build_sweep_trajectory(cameras: List, n_frames: int) -> List[MiniCam]:
    """(Legacy Phase 2.5b) Nội suy global-param qua waypoint đã sort.

    Giữ lại để A/B với ellipse. Kế thừa noise Y/Z + rotation của hand-held
    capture, và GIF loop bị "nhảy" khi quay từ frame cuối về frame 0.
    """
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
# Phase 2.5c — Ellipse trajectory: path giải tích fit từ PHÂN PHỐI pose,
# không nội suy qua từng waypoint thô (root fix cho jitter Y/Z + rotation)
# ---------------------------------------------------------------------------


def _camera_axes(cam) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(center, forward, up) của camera trong world frame.

    Convention 3DGS/COLMAP: cam.R là C2W rotation; camera frame X right /
    Y down / Z forward → forward_world = R[:, 2], up_world = -R[:, 1].
    """
    R = np.asarray(cam.R, dtype=np.float64)
    C = _camera_center_world(cam)
    f = R[:, 2]
    u = -R[:, 1]
    return C, f / np.linalg.norm(f), u / np.linalg.norm(u)


def _estimate_focus_point(centers: np.ndarray,
                          forwards: np.ndarray) -> np.ndarray:
    """Giao điểm least-squares của các optical axis = điểm scene được nhắm.

    Giải Σ (I - f fᵀ)(p - C) = 0. LLFF capture nhắm subject giữa scene nên
    hệ thường well-posed. Guard: nghiệm không hữu hạn / ở sau lưng / quá xa
    → fallback điểm phía trước mean pose (rotation gần constant, vẫn hợp lệ).
    """
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for C, f in zip(centers, forwards):
        M = np.eye(3) - np.outer(f, f)
        A += M
        b += M @ C
    C_mean = centers.mean(axis=0)
    f_mean = forwards.mean(axis=0)
    f_mean = f_mean / np.linalg.norm(f_mean)
    spread = max(float(np.linalg.norm(centers - C_mean, axis=1).max()), 1e-6)
    try:
        p = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return C_mean + f_mean * 8.0 * spread
    depth = float((p - C_mean) @ f_mean)
    if (not np.all(np.isfinite(p))) or depth < 1.5 * spread \
            or depth > 60.0 * spread:
        return C_mean + f_mean * 8.0 * spread
    return p


def _lookat_pose(C: np.ndarray, target: np.ndarray,
                 up_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pose camera tại C nhìn về target, ảnh upright theo up_ref.

    Trả về (R, T) đúng convention 3DGS: R = C2W rotation,
    T = W2C translation — feed thẳng vào _build_minicam_from_pose.

    Lưu ý convention COLMAP (Y down, Z forward, right-handed):
    right_world = forward × up (KHÔNG phải up × forward như OpenGL).
    Đã verify numeric: orthonormal, det=1, upright, center roundtrip
    qua getWorld2View2, điểm phía trên project về -y ảnh.
    """
    z = target - C
    z = z / np.linalg.norm(z)
    x = np.cross(z, up_ref)
    n = np.linalg.norm(x)
    if n < 1e-8:   # nhìn thẳng theo up_ref — không xảy ra với forward-facing
        x = np.cross(z, np.array([1.0, 0.0, 0.0]))
        n = np.linalg.norm(x)
    x = x / n
    y = np.cross(z, x)                 # down = forward × right
    R = np.stack([x, y, z], axis=1)    # cột = right / down / forward (C2W)
    T = -R.T @ C
    return R, T


def _build_ellipse_trajectory(cameras: List, n_frames: int,
                              radius_scale: float = 0.9) -> List[MiniCam]:
    """Path ellipse analytic trong mặt phẳng PC1-PC2 của cụm camera.

    Vì sao thay thế nội suy waypoint: camera LLFF hand-held → vị trí lẫn
    hướng đều có noise; path đi QUA từng điểm thô kế thừa noise đó. Ở đây
    chỉ dùng thống kê cụm pose rồi sinh path giải tích:
      - Position: ellipse trong best-fit plane, bán trục = radius_scale ×
        spread percentile 5-95 → mọi frame nằm TRONG phân phối pose
        train/test (giữ ràng buộc an toàn cũ).
      - Rotation: look-at về focus point → biến thiên trơn theo θ.
      - θ trọn [0, 2π) không lặp endpoint → GIF loop liền mạch.
    """
    axes = [_camera_axes(c) for c in cameras]
    centers = np.array([a[0] for a in axes])
    forwards = np.array([a[1] for a in axes])
    ups = np.array([a[2] for a in axes])

    up_ref = ups.mean(axis=0)
    up_ref = up_ref / np.linalg.norm(up_ref)

    # Best-fit plane của cụm camera (PC1 = trục ngang chính, PC2 = phụ).
    mean = centers.mean(axis=0)
    X = centers - mean
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    pc1, pc2 = Vt[0], Vt[1]
    if pc1[0] < 0:                     # pin L→R, consistent với 2.5b
        pc1 = -pc1
    if pc2 @ up_ref < 0:               # pin PC2 hướng "lên"
        pc2 = -pc2

    # Bán trục từ percentile 5-95 (loại outlier), co lại theo radius_scale.
    s1, s2 = X @ pc1, X @ pc2
    lo1, hi1 = np.percentile(s1, [5, 95])
    lo2, hi2 = np.percentile(s2, [5, 95])
    center = mean + 0.5 * (lo1 + hi1) * pc1 + 0.5 * (lo2 + hi2) * pc2
    a = radius_scale * 0.5 * (hi1 - lo1)
    b = radius_scale * 0.5 * (hi2 - lo2)
    b = max(b, 0.05 * a)               # tránh degenerate về đoạn thẳng

    focus = _estimate_focus_point(centers, forwards)
    ref_cam = cameras[0]               # LLFF: intrinsic giống nhau mọi cam

    # θ bắt đầu -π/2 (đáy ellipse) → frame đầu chuyển động +PC1 (L→R).
    thetas = -np.pi / 2 + np.linspace(0.0, 2.0 * np.pi, n_frames,
                                      endpoint=False)
    trajectory: List[MiniCam] = []
    for th in thetas:
        C = center + a * np.cos(th) * pc1 + b * np.sin(th) * pc2
        R, T = _lookat_pose(C, focus, up_ref)
        trajectory.append(_build_minicam_from_pose(R, T, ref_cam))
    return trajectory


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
    print(f"[demo] {len(all_cams)} cameras total ({args.source})")

    # Mode ellipse (mặc định) — không cần sort.
    traj = build_orbit_trajectory(all_cams, n_frames=args.n_frames)
    print(f"[demo] ellipse: {len(traj)} MiniCam frames")
    print(f"[demo] first cam WVT[:2,:3] =\n"
          f"{traj[0].world_view_transform[:2, :3].cpu().numpy()}")

    # Legacy sweep để A/B — cần sort trước.
    traj_sweep = build_orbit_trajectory(sort_llff_cameras(all_cams),
                                        n_frames=args.n_frames, mode="sweep")
    print(f"[demo] sweep (legacy): {len(traj_sweep)} MiniCam frames")