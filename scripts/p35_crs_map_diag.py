# ============================================================
# [CRSGaussian P35] Step 1 — 3-signal shootout diagnostic
# File: scripts/p35_crs_map_diag.py  (TẠO MỚI, standalone, read-only)
# Doc:  docs/35_paper_novelty_research.md  Section 12
#
# Mục đích: đo bằng số xem trong 3 ứng viên gate
#   (1) CRS cũ  = sigmoid(_crs_score)          — floater detector (D+R)
#   (2) G       = viewspace-grad struggle      — under-fit/blur detector
#   (3) CRS_att = CRS_old × (1 − G)            — kết hợp v1 (dạng nhân)
# ứng viên nào splat lên 2D thực sự tương quan với vùng render tệ
# (error map trên TEST views). Gate cho toàn bộ paradigm attention.
#
# KHÔNG train, KHÔNG sửa checkpoint, KHÔNG đụng production code.
# Renderer dùng nguyên bản qua cổng override_color + disable_dropout=True.
#
# Chạy trên server (env corgs), từ repo root CoR-GS:
#   conda activate corgs
#   # 1 GPU đủ (~20 phút). Nếu muốn split 2 GPU theo quy tắc:
#   CUDA_VISIBLE_DEVICES=0 python scripts/p35_crs_map_diag.py \
#       --scenes fern flower fortress horns &
#   CUDA_VISIBLE_DEVICES=1 python scripts/p35_crs_map_diag.py \
#       --scenes leaves orchids room trex &
#   wait
#   cat output/p35_crs_map_diag/summary.csv
# ============================================================

import argparse
import csv
import glob
import math
import os
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

# Cho phép chạy `python scripts/p35_crs_map_diag.py` từ repo root
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim


# ------------------------------------------------------------
# Scene args tối thiểu — copy pattern verified từ demo/render_utils.py
# (_build_scene_args): đủ mọi field mà Scene → create_from_pcd đọc
# qua đường LLFF inference-only. KHÔNG import demo module để script
# tự đứng độc lập (demo có thể đổi).
# ------------------------------------------------------------
def _build_scene_args(source_path, model_path, n_views=3, resolution=8):
    return Namespace(
        source_path=source_path,
        model_path=model_path,
        images="images",
        resolution=resolution,
        white_background=False,
        data_device="cuda",
        eval=True,
        n_views=n_views,
        rand_pcd=False,
        sh_degree=3,
        absdensify=False,
        train_bg=False,
        use_color=True,
    )


# Pipe tối thiểu — 4 field render() truy cập trực tiếp; mọi dropout
# field khác đọc qua getattr(default) và bị disable_dropout=True bỏ qua.
_PIPE = Namespace(
    convert_SHs_python=False,
    compute_cov3D_python=False,
    debug=False,
    use_confidence=False,
)

_BG = None  # set sau khi có cuda


# ------------------------------------------------------------
# Checkpoint load — manual unpack thay vì gaussians.restore()
# Lý do: restore() gọi training_setup(training_args) → cần optimizer
# args + tạo Adam state — thừa cho diagnostic read-only.
# Tuple order verified tại gaussian_model.py:196-213 (13 field).
# ------------------------------------------------------------
def load_checkpoint_into(gaussians, ckpt_path):
    # weights_only=True chặn unpickle object tùy ý (an toàn hơn).
    # Torch cũ (<1.13) không có param này → fallback plain load
    # (checkpoint tự train trên server, trusted source).
    try:
        raw = torch.load(ckpt_path, map_location="cuda", weights_only=True)
    except Exception:
        # TypeError (torch cũ không có param) hoặc UnpicklingError
        # (tuple chứa type ngoài whitelist) → plain load, trusted file.
        raw = torch.load(ckpt_path, map_location="cuda")
    # train.py lưu (capture(), iteration) — handle cả 2 format
    if isinstance(raw, (tuple, list)) and len(raw) == 2:
        model_args, it = raw
    else:
        model_args, it = raw, -1

    if len(model_args) == 13:
        (gaussians.active_sh_degree,
         gaussians._xyz,
         gaussians._features_dc,
         gaussians._features_rest,
         gaussians._scaling,
         gaussians._rotation,
         gaussians._opacity,
         gaussians.max_radii2D,
         _grad_accum,          # đóng băng từ lúc densify dừng — KHÔNG dùng
         _denom,
         _opt_dict,            # bỏ qua optimizer state
         gaussians.spatial_lr_scale,
         gaussians._crs_score) = model_args
    else:
        raise RuntimeError(
            f"Checkpoint {ckpt_path} có {len(model_args)} field (kỳ vọng 13 "
            f"với _crs_score). Checkpoint quá cũ, không có CRS.")
    n = gaussians._xyz.shape[0]
    # ── FIX confidence-rasterizer size mismatch ──
    # Scene init tạo confidence size N_init (create_from_pcd:378); checkpoint
    # có N lớn hơn sau densify. Rasterizer backward nhân grad × confidence
    # → crash nếu không resize. (Cùng bẫy Path A B3 "confidence re-init".)
    gaussians.confidence = torch.ones_like(gaussians._opacity, device="cuda")
    print(f"    loaded {n} gaussians, iter={it}, "
          f"CRS mean={torch.sigmoid(gaussians._crs_score).mean().item():.3f}")
    return n


# ------------------------------------------------------------
# G signal — tính TƯƠI viewspace gradient per-Gaussian.
# Lý do tính tươi: xyz_gradient_accum trong checkpoint đóng băng từ
# lúc densification dừng (~iter 3000), không phản ánh trạng thái cuối.
# Cách: 3 train view × (render → loss photometric → backward, KHÔNG
# optimizer step, model không đổi) → đọc screenspace_points.grad.
# Norm 2 chiều đầu (x,y screen) — đúng convention densification 3DGS.
# ------------------------------------------------------------
def compute_fresh_G(gaussians, train_cams, lambda_dssim=0.2):
    N = gaussians._xyz.shape[0]
    accum = torch.zeros(N, device="cuda")
    count = torch.zeros(N, device="cuda")
    for cam in train_cams:
        pkg = render(cam, gaussians, _PIPE, _BG, disable_dropout=True)
        image = pkg["render"]
        gt = cam.original_image.cuda()
        loss = (1.0 - lambda_dssim) * l1_loss(image, gt) \
            + lambda_dssim * (1.0 - ssim(image, gt))
        loss.backward()
        vsp = pkg["viewspace_points"]
        if vsp.grad is not None:
            g = vsp.grad[:, :2].norm(dim=-1)          # (N,)
            vis = pkg["visibility_filter"]
            accum[vis] += g[vis]
            count[vis] += 1.0
        # zero mọi grad dính vào params để lần backward sau sạch
        gaussians_zero_grads(gaussians)
    G = accum / count.clamp(min=1.0)
    # Percentile-normalize: robust với outlier, G_norm ∈ [0,1]
    pos = G[G > 0]
    if pos.numel() > 100:
        p99 = torch.quantile(pos, 0.99)
        G = (G / p99.clamp(min=1e-12)).clamp(0.0, 1.0)
    return G.detach()


def gaussians_zero_grads(gaussians):
    for t in (gaussians._xyz, gaussians._features_dc, gaussians._features_rest,
              gaussians._scaling, gaussians._rotation, gaussians._opacity):
        if t.grad is not None:
            t.grad = None


# ------------------------------------------------------------
# Splat 1 tín hiệu scalar per-Gaussian ra map 2D qua override_color.
# Alpha map trả sẵn trong output dict của renderer → normalize
# raw / clamp(alpha, 0.05) để tránh false-low ở vùng ít Gaussian.
# ------------------------------------------------------------
@torch.no_grad()
def splat_signal(gaussians, cam, signal_n1):
    """signal_n1: (N,) hoặc (N,1) tensor ∈ [0,1] → return (map HW, alpha HW)."""
    sig = signal_n1.reshape(-1, 1).expand(-1, 3).contiguous().float()
    pkg = render(cam, gaussians, _PIPE, _BG,
                 override_color=sig, disable_dropout=True)
    raw = pkg["render"][0]                      # (H,W) — 3 kênh giống nhau
    alpha = pkg["alpha"]
    alpha = alpha[0] if alpha.dim() == 3 else alpha
    m = raw / alpha.clamp(min=0.05)
    return m.clamp(0.0, 1.0), alpha


@torch.no_grad()
def render_rgb(gaussians, cam):
    pkg = render(cam, gaussians, _PIPE, _BG, disable_dropout=True)
    return pkg["render"].clamp(0.0, 1.0)        # (3,H,W)


# ------------------------------------------------------------
# Sobel magnitude — conv 3×3, output liên tục (không threshold).
# Chọn Sobel vs Laplacian/Canny: differentiable + rẻ + smooth
# (doc 35 Section 11.1). Ở diagnostic này chỉ dùng forward.
# ------------------------------------------------------------
_KX = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                   dtype=torch.float32).view(1, 1, 3, 3) / 4.0
_KY = _KX.transpose(2, 3).contiguous()


@torch.no_grad()
def sobel_mag(img_3hw):
    gray = img_3hw.mean(dim=0, keepdim=True)[None]      # (1,1,H,W)
    kx, ky = _KX.to(gray.device), _KY.to(gray.device)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return (gx ** 2 + gy ** 2).sqrt()[0, 0]             # (H,W)


# ------------------------------------------------------------
# Spearman rank correlation — tự implement (không phụ thuộc scipy).
# Ties xử lý thô bằng argsort kép — đủ cho diagnostic.
# ------------------------------------------------------------
def spearman(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.size < 100:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = math.sqrt((ra ** 2).sum() * (rb ** 2).sum()) + 1e-12
    return float((ra * rb).sum() / denom)


# ------------------------------------------------------------
# Panel visualize — 2 hàng × 4 cột mỗi view
# ------------------------------------------------------------
def save_panel(out_png, gt, rgb, err, alpha, maps):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    items = [
        ("GT", gt.permute(1, 2, 0).cpu().numpy(), None),
        ("Render", rgb.permute(1, 2, 0).cpu().numpy(), None),
        ("Error |render-GT|", err.cpu().numpy(), "inferno"),
        ("Alpha", alpha.cpu().numpy(), "gray"),
        ("CRS map (old)", maps["crs"].cpu().numpy(), "viridis"),
        ("G map (struggle)", maps["g"].cpu().numpy(), "viridis"),
        ("CRS_att map", maps["att"].cpu().numpy(), "viridis"),
        ("(1-CRS_att) x err", ((1 - maps["att"]) * err).cpu().numpy(), "inferno"),
    ]
    for ax, (title, img, cmap) in zip(axes.flat, items):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------
# Main per-scene
# ------------------------------------------------------------
def run_scene(scene_name, args, csv_rows):
    print(f"\n=== {scene_name} ===")
    src = os.path.join(args.data_root, scene_name)
    ckpt_dir = os.path.join(args.ckpt_root, args.ckpt_pattern.format(scene=scene_name))
    cands = sorted(glob.glob(os.path.join(ckpt_dir, "chkpnt*.pth")))
    if not cands:
        print(f"  !! KHÔNG thấy chkpnt*.pth trong {ckpt_dir} — skip. "
              f"(ply không chứa CRS, cần .pth)")
        return
    ckpt_path = cands[-1]
    print(f"  ckpt: {ckpt_path}")

    tmpdir = tempfile.mkdtemp(prefix="p35_")
    sargs = _build_scene_args(src, tmpdir, args.n_views, args.resolution)
    gaussians = GaussianModel(sargs)
    scene = Scene(sargs, gaussians, load_iteration=None, shuffle=False,
                  resolution_scales=[1.0])
    load_checkpoint_into(gaussians, ckpt_path)

    train_cams = scene.getTrainCameras()
    test_cams = scene.getTestCameras()
    print(f"  cams: {len(train_cams)} train / {len(test_cams)} test")

    # ── 3 ứng viên signal per-Gaussian ──
    crs_old = torch.sigmoid(gaussians._crs_score).reshape(-1).detach()
    G = compute_fresh_G(gaussians, train_cams)
    crs_att = (crs_old * (1.0 - G)).clamp(0.0, 1.0)
    print(f"  signal stats: CRS mean={crs_old.mean():.3f} std={crs_old.std():.3f}"
          f" | G mean={G.mean():.3f} std={G.std():.3f}")

    out_dir = os.path.join(args.out_dir, scene_name)
    os.makedirs(out_dir, exist_ok=True)

    # ── Chấm trên TEST views (train views mù vì overfit ~34dB) ──
    # + vẽ panel cho cả vài train view để đối chiếu trực quan
    agg = {k: {"rho_all": [], "rho_edge": [], "std": []}
           for k in ("crs", "g", "att")}

    eval_views = [("test", c) for c in test_cams[: args.max_test_views]] \
        + [("train", c) for c in train_cams]

    for tag, cam in eval_views:
        gt = cam.original_image.cuda().clamp(0.0, 1.0)
        rgb = render_rgb(gaussians, cam)
        err = (rgb - gt).abs().mean(dim=0)                       # (H,W)

        m_crs, alpha = splat_signal(gaussians, cam, crs_old)
        m_g, _ = splat_signal(gaussians, cam, G)
        m_att, _ = splat_signal(gaussians, cam, crs_att)
        maps = {"crs": m_crs, "g": m_g, "att": m_att}

        save_panel(os.path.join(out_dir, f"panel_{tag}_{cam.image_name}.png"),
                   gt, rgb, err, alpha, maps)

        if tag != "test":
            continue  # metric chỉ tính trên test views

        valid = (alpha > 0.5)
        edge = (sobel_mag(gt) > args.edge_thresh) & valid
        err_np = err[valid].cpu().numpy()
        err_edge_np = err[edge].cpu().numpy()
        for key, m in maps.items():
            inv = (1.0 - m)
            agg[key]["rho_all"].append(spearman(inv[valid].cpu().numpy(), err_np))
            agg[key]["rho_edge"].append(
                spearman(inv[edge].cpu().numpy(), err_edge_np))
            agg[key]["std"].append(float(m[valid].std().cpu()))

    for key in ("crs", "g", "att"):
        row = {
            "scene": scene_name,
            "signal": key,
            "rho_all": float(np.nanmean(agg[key]["rho_all"])),
            "rho_edge": float(np.nanmean(agg[key]["rho_edge"])),
            "map_std": float(np.nanmean(agg[key]["std"])),
        }
        csv_rows.append(row)
        print(f"  [{key:>3}] rho_all={row['rho_all']:+.3f} "
              f"rho_edge={row['rho_edge']:+.3f} std={row['map_std']:.3f}")

    # Giải phóng VRAM giữa các scene
    del gaussians, scene
    torch.cuda.empty_cache()


def main():
    global _BG
    ap = argparse.ArgumentParser(
        description="[CRSGaussian P35] 3-signal shootout diagnostic")
    ap.add_argument("--data_root", default="data/nerf_llff_data")
    ap.add_argument("--ckpt_root", default="output/p28_crs_boost/tau65")
    ap.add_argument("--ckpt_pattern", default="A3_seed42_{scene}")
    ap.add_argument("--out_dir", default="output/p35_crs_map_diag")
    ap.add_argument("--scenes", nargs="+",
                    default=["fern", "flower", "fortress", "horns",
                             "leaves", "orchids", "room", "trex"])
    ap.add_argument("--n_views", type=int, default=3)
    ap.add_argument("--resolution", type=int, default=8)
    ap.add_argument("--max_test_views", type=int, default=8,
                    help="số test view dùng để chấm metric (đủ thống kê, đỡ chậm)")
    ap.add_argument("--edge_thresh", type=float, default=0.05,
                    help="ngưỡng sobel(GT) cho edge mask")
    args = ap.parse_args()

    _BG = torch.tensor([0.0, 0.0, 0.0], device="cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    csv_rows = []
    for s in args.scenes:
        try:
            run_scene(s, args, csv_rows)
        except Exception as e:
            print(f"  !! {s} FAILED: {e}")
            import traceback
            traceback.print_exc()

    # summary.csv — append-safe khi chạy split 2 GPU (mỗi process ghi
    # file riêng theo scene đầu tiên, tránh race)
    suffix = args.scenes[0] if len(args.scenes) < 8 else "all"
    csv_path = os.path.join(args.out_dir, f"summary_{suffix}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scene", "signal", "rho_all",
                                          "rho_edge", "map_std"])
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\nSaved {csv_path}")

    # ── Console verdict theo gate doc 35 Section 12 ──
    print("\n===== GATE SUMMARY (doc 35 §12) =====")
    for key, label in (("crs", "CRS old (D+R)"), ("g", "G struggle"),
                       ("att", "CRS_att = CRS×(1-G)")):
        rows = [r for r in csv_rows if r["signal"] == key]
        if not rows:
            continue
        n_pass = sum(1 for r in rows if r["rho_all"] >= 0.25)
        mean_rho = float(np.nanmean([r["rho_all"] for r in rows]))
        mean_std = float(np.nanmean([r["map_std"] for r in rows]))
        g1 = "PASS" if mean_std >= 0.08 else "FAIL"
        g2 = f"{n_pass}/{len(rows)} scenes rho>=0.25"
        print(f"  {label:<22} G1(std={mean_std:.3f}):{g1} | "
              f"G2: {g2} (mean rho={mean_rho:+.3f})")
    print("Gate G3: xem panel horns/trex bằng mắt + rho_edge trong CSV.")


if __name__ == "__main__":
    main()
