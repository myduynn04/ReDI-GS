# ============================================================
# [CRSGaussian P35b] Step 2a — Blur detector offline validation
# File: scripts/p35b_blur_diag.py  (TẠO MỚI, standalone, read-only)
# Doc:  docs/35_paper_novelty_research.md  Section 13.1
#
# Trả lời 2 câu TRƯỚC khi đụng train.py:
#   Câu A — deficit có SỐNG SÓT tại train views không (overfit ~30dB
#           có thể đã giết signal: render khớp GT → deficit ≈ 0)?
#   Câu B — deficit V1 (nhân) hay V2 (hiệu relu) tốt hơn?
#
# Công thức (chuẩn hóa CHUNG mẫu số p95 của GT — quan trọng):
#   s_gt = sobel(GT) / p95(sobel(GT));  s_r = sobel(render) / p95(sobel(GT))
#   V1: deficit = s_gt × (1 − s_r)
#   V2: deficit = relu(s_gt − s_r)
#   blur_map = (1 − CRS_att_map) × deficit
#
# Tái dùng helpers từ p35_crs_map_diag.py (load ckpt, splat, sobel...).
# KHÔNG train, KHÔNG sửa production code.
#
# Chạy trên server (env corgs, repo root CoR-GS):
#   mkdir -p logs/p35b
#   CUDA_VISIBLE_DEVICES=0 python scripts/p35b_blur_diag.py \
#       --scenes fern flower fortress horns > logs/p35b/gpu0.log 2>&1 &
#   CUDA_VISIBLE_DEVICES=1 python scripts/p35b_blur_diag.py \
#       --scenes leaves orchids room trex > logs/p35b/gpu1.log 2>&1 &
#   wait
#   cat output/p35b_blur_diag/summary_*.csv
# ============================================================

import argparse
import csv
import glob
import os
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SCRIPTS = str(Path(__file__).resolve().parent)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scene import Scene
from scene.gaussian_model import GaussianModel

# Tái dùng helpers đã verified từ p35 (import module, không chạy main
# vì p35 gated bởi __main__)
import p35_crs_map_diag as p35


# ------------------------------------------------------------
# Deficit computation — cả 2 variant, chuẩn hóa chung mẫu số
# ------------------------------------------------------------
@torch.no_grad()
def compute_deficits(gt_3hw, render_3hw):
    """Return (s_gt, s_r, deficit_v1, deficit_v2) — all (H,W) in [0,1]."""
    s_gt_raw = p35.sobel_mag(gt_3hw)
    s_r_raw = p35.sobel_mag(render_3hw)
    # Mẫu số CHUNG = p95 của sobel(GT). Nếu normalize riêng từng ảnh,
    # render mờ toàn cục sẽ bị rescale thành "có cạnh bình thường"
    # → deficit sai. Cùng mẫu số giữ nghĩa "so với GT".
    denom = torch.quantile(s_gt_raw.reshape(-1), 0.95).clamp(min=1e-6)
    s_gt = (s_gt_raw / denom).clamp(0.0, 1.0)
    s_r = (s_r_raw / denom).clamp(0.0, 1.0)
    deficit_v1 = s_gt * (1.0 - s_r)
    deficit_v2 = torch.relu(s_gt - s_r)
    return s_gt, s_r, deficit_v1, deficit_v2


# ------------------------------------------------------------
# Panel — 2 hàng × 5 cột
# ------------------------------------------------------------
def save_panel(out_png, gt, rgb, err, s_gt, s_r, d1, d2, b1, b2, att_map):
    fig, axes = plt.subplots(2, 5, figsize=(20, 7))
    items = [
        ("GT", gt.permute(1, 2, 0).cpu().numpy(), None),
        ("Render", rgb.permute(1, 2, 0).cpu().numpy(), None),
        ("sobel(GT)", s_gt.cpu().numpy(), "gray"),
        ("sobel(render)", s_r.cpu().numpy(), "gray"),
        ("Error", err.cpu().numpy(), "inferno"),
        ("deficit V1 = s_gt*(1-s_r)", d1.cpu().numpy(), "inferno"),
        ("deficit V2 = relu(s_gt-s_r)", d2.cpu().numpy(), "inferno"),
        ("CRS_att map", att_map.cpu().numpy(), "viridis"),
        ("blur V1 = (1-att)*d1", b1.cpu().numpy(), "inferno"),
        ("blur V2 = (1-att)*d2", b2.cpu().numpy(), "inferno"),
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
    ckpt_dir = os.path.join(args.ckpt_root,
                            args.ckpt_pattern.format(scene=scene_name))
    cands = sorted(glob.glob(os.path.join(ckpt_dir, "chkpnt*.pth")))
    if not cands:
        print(f"  !! KHÔNG thấy chkpnt*.pth trong {ckpt_dir} — skip.")
        return
    print(f"  ckpt: {cands[-1]}")

    tmpdir = tempfile.mkdtemp(prefix="p35b_")
    sargs = p35._build_scene_args(src, tmpdir, args.n_views, args.resolution)
    gaussians = GaussianModel(sargs)
    scene = Scene(sargs, gaussians, load_iteration=None, shuffle=False,
                  resolution_scales=[1.0])
    p35.load_checkpoint_into(gaussians, cands[-1])

    train_cams = scene.getTrainCameras()
    test_cams = scene.getTestCameras()

    # CRS_att per-Gaussian (winner Step 1 — doc 35 §12.1)
    crs_old = torch.sigmoid(gaussians._crs_score).reshape(-1).detach()
    G = p35.compute_fresh_G(gaussians, train_cams)
    crs_att = (crs_old * (1.0 - G)).clamp(0.0, 1.0)

    out_dir = os.path.join(args.out_dir, scene_name)
    os.makedirs(out_dir, exist_ok=True)

    # Câu A đo trên TRAIN views (nơi attention loss sẽ chạy thật).
    # Test views chỉ để đối chiếu deficit ở nơi lỗi lộ rõ.
    stats = {("train", "v1"): [], ("train", "v2"): [],
             ("test", "v1"): [], ("test", "v2"): []}

    eval_views = [("train", c) for c in train_cams] \
        + [("test", c) for c in test_cams[: args.max_test_views]]

    for tag, cam in eval_views:
        gt = cam.original_image.cuda().clamp(0.0, 1.0)
        rgb = p35.render_rgb(gaussians, cam)
        err = (rgb - gt).abs().mean(dim=0)

        s_gt, s_r, d1, d2 = compute_deficits(gt, rgb)
        att_map, alpha = p35.splat_signal(gaussians, cam, crs_att)
        b1 = (1.0 - att_map) * d1
        b2 = (1.0 - att_map) * d2

        save_panel(os.path.join(out_dir, f"panel_{tag}_{cam.image_name}.png"),
                   gt, rgb, err, s_gt, s_r, d1, d2, b1, b2, att_map)

        valid = (alpha > 0.5)
        edge = (s_gt > args.edge_thresh) & valid
        edge_frac = float(edge.float().mean().cpu())  # % pixel là edge
        err_np = err[valid].cpu().numpy()
        for ver, d, b in (("v1", d1, b1), ("v2", d2, b2)):
            # Deficit mass concentration: % tổng deficit nằm trong edge mask
            # so với % diện tích edge → ratio ≥ 2 = tập trung đúng chỗ
            total = float(d[valid].sum().cpu()) + 1e-9
            in_edge = float(d[edge].sum().cpu())
            conc = (in_edge / total) / max(edge_frac, 1e-6)
            stats[(tag, ver)].append({
                "deficit_mean": float(d[valid].mean().cpu()),
                "deficit_p99": float(torch.quantile(
                    d[valid].reshape(-1), 0.99).cpu()),
                "edge_conc": conc,
                "rho_blur_err": p35.spearman(
                    b[valid].cpu().numpy(), err_np),
            })

    for (tag, ver), rows in stats.items():
        if not rows:
            continue
        row = {
            "scene": scene_name, "views": tag, "variant": ver,
            "deficit_mean": float(np.mean([r["deficit_mean"] for r in rows])),
            "deficit_p99": float(np.mean([r["deficit_p99"] for r in rows])),
            "edge_conc": float(np.mean([r["edge_conc"] for r in rows])),
            "rho_blur_err": float(np.nanmean(
                [r["rho_blur_err"] for r in rows])),
        }
        csv_rows.append(row)
        print(f"  [{tag:>5}/{ver}] deficit_mean={row['deficit_mean']:.4f} "
              f"p99={row['deficit_p99']:.3f} edge_conc={row['edge_conc']:.2f}x "
              f"rho(blur,err)={row['rho_blur_err']:+.3f}")

    del gaussians, scene
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(
        description="[CRSGaussian P35b] Blur detector offline validation")
    ap.add_argument("--data_root", default="data/nerf_llff_data")
    ap.add_argument("--ckpt_root", default="output/p28_crs_boost/tau65")
    ap.add_argument("--ckpt_pattern", default="A3_seed42_{scene}")
    ap.add_argument("--out_dir", default="output/p35b_blur_diag")
    ap.add_argument("--scenes", nargs="+",
                    default=["fern", "flower", "fortress", "horns",
                             "leaves", "orchids", "room", "trex"])
    ap.add_argument("--n_views", type=int, default=3)
    ap.add_argument("--resolution", type=int, default=8)
    ap.add_argument("--max_test_views", type=int, default=4)
    ap.add_argument("--edge_thresh", type=float, default=0.2,
                    help="ngưỡng s_gt (đã normalize p95) cho edge mask")
    args = ap.parse_args()

    p35._BG = torch.tensor([0.0, 0.0, 0.0], device="cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    csv_rows = []
    for s in args.scenes:
        try:
            run_scene(s, args, csv_rows)
        except Exception as e:
            print(f"  !! {s} FAILED: {e}")
            import traceback
            traceback.print_exc()

    suffix = args.scenes[0] if len(args.scenes) < 8 else "all"
    csv_path = os.path.join(args.out_dir, f"summary_{suffix}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "scene", "views", "variant", "deficit_mean", "deficit_p99",
            "edge_conc", "rho_blur_err"])
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\nSaved {csv_path}")

    # ── Gate 2a verdict (doc 35 §13.1) ──
    print("\n===== GATE 2a SUMMARY =====")
    for ver in ("v1", "v2"):
        tr = [r for r in csv_rows if r["views"] == "train"
              and r["variant"] == ver]
        if not tr:
            continue
        dm = float(np.mean([r["deficit_mean"] for r in tr]))
        ec = float(np.mean([r["edge_conc"] for r in tr]))
        rho = float(np.nanmean([r["rho_blur_err"] for r in tr]))
        alive = "ALIVE" if dm > 0.02 else "DEAD (overfit killed signal)"
        conc = "PASS" if ec >= 2.0 else "FAIL"
        print(f"  [{ver}] train-view deficit: mean={dm:.4f} → {alive} | "
              f"edge_conc={ec:.2f}x → {conc} | rho(blur,err)={rho:+.3f}")
    print("Câu B: variant nào rho + edge_conc cao hơn → chọn cho Phase 2b.")
    print("Nếu cả 2 DEAD → detector đổi thiết kế (doc 35 §13.1 fallback).")


if __name__ == "__main__":
    main()
