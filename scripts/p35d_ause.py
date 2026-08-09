# ============================================================
# [CRSGaussian P35d] AUSE — Area Under Sparsification Error
# File: scripts/p35d_ause.py  (TẠO MỚI, standalone, read-only)
# Doc:  docs/35_paper_novelty_research.md §14
#
# Mục đích: chuyển kết quả detector (Spearman ρ=0.30, §12.1) sang
# AUSE — metric CHUẨN của subfield uncertainty NVS (WarpRF 2506.22433,
# OUGS 2511.09397, View-Dependent Unc 3DGS 2504.07370 đều báo AUSE).
# Có AUSE mới so trực tiếp được với số họ publish; ρ thì không.
#
# AUSE = diện tích giữa 2 đường sparsification:
#   (a) bỏ dần pixel theo UNCERTAINTY DỰ ĐOÁN (1 − CRS map)
#   (b) bỏ dần pixel theo ERROR THẬT (oracle — tốt nhất có thể)
# Thấp = tốt. 0 = xếp hạng pixel chính xác như oracle.
#
# Kèm baseline RANDOM để con số tự diễn giải được, không cần chờ
# đọc paper ngoài: ours ≈ random → signal vô dụng.
#
# Đo trên TEST views (train views overfit ~30dB, error ≈ 0 → vô nghĩa).
#
# Chạy trên server (env corgs, repo root):
#   mkdir -p logs/p35d
#   CUDA_VISIBLE_DEVICES=0 python scripts/p35d_ause.py \
#       --scenes fern flower fortress horns > logs/p35d/gpu0.log 2>&1 &
#   CUDA_VISIBLE_DEVICES=1 python scripts/p35d_ause.py \
#       --scenes leaves orchids room trex > logs/p35d/gpu1.log 2>&1 &
#   wait
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

from scene import Scene
from scene.gaussian_model import GaussianModel

# Tái dùng helper đã verified từ p35 (load ckpt, splat, compute G)
import p35_crs_map_diag as p35


# ============================================================
# [CRSGaussian P35d] compute_ause
# Mục đích: AUSE cho 1 cặp (uncertainty, error) đã flatten.
# Được gọi từ: run_scene, mỗi test view × mỗi ứng viên signal.
# ============================================================
@torch.no_grad()
def compute_ause(unc, err, n_steps=100):
    """AUSE + 2 đường sparsification.

    Args:
        unc: (M,) uncertainty DỰ ĐOÁN — cao = nghi ngờ nhiều.
        err: (M,) error THẬT |render − GT|.
        n_steps: số điểm lấy mẫu trên trục "% pixel bị bỏ".

    Returns:
        (ause_ours, ause_random, curve_ours, curve_oracle) — ause thấp = tốt.

    Chuẩn hoá mỗi đường theo giá trị tại 0% để scene có magnitude error
    khác nhau vẫn so được (convention chung của các paper AUSE).
    """
    M = unc.numel()
    if M < 1000:
        return float("nan"), float("nan"), None, None

    total = err.sum()
    fracs = torch.linspace(0.0, 0.99, n_steps, device=err.device)
    ks = (fracs * M).long().clamp(max=M - 1)

    def _curve(order_idx):
        # err sắp theo thứ tự bỏ; cumsum-exclusive = tổng error của k
        # pixel ĐÃ BỎ → phần còn lại = total − đó, chia số pixel còn lại.
        e = err[order_idx]
        cs = torch.cumsum(e, 0)
        cse = torch.cat([torch.zeros(1, device=e.device), cs[:-1]])
        remain_sum = total - cse[ks]
        remain_cnt = (M - ks).float().clamp(min=1.0)
        c = remain_sum / remain_cnt
        return c / c[0].clamp(min=1e-12)      # normalize về 1.0 tại 0%

    c_ours = _curve(torch.argsort(unc, descending=True))   # bỏ chỗ nghi nhất trước
    c_oracle = _curve(torch.argsort(err, descending=True))  # oracle: bỏ chỗ tệ thật
    # Random baseline — mốc "signal vô dụng" để con số tự diễn giải
    g = torch.Generator(device="cpu").manual_seed(0)
    c_rand = _curve(torch.randperm(M, generator=g).to(err.device))

    # torch mới đổi tên trapz → trapezoid; giữ cả 2 cho chắc
    _trapz = getattr(torch, "trapezoid", None) or torch.trapz
    ause_ours = float(_trapz(c_ours - c_oracle, fracs))
    ause_rand = float(_trapz(c_rand - c_oracle, fracs))
    return ause_ours, ause_rand, c_ours.cpu().numpy(), c_oracle.cpu().numpy()


def run_scene(scene_name, args, rows):
    print(f"\n=== {scene_name} ===")
    src = os.path.join(args.data_root, scene_name)
    ckpt_dir = os.path.join(args.ckpt_root,
                            args.ckpt_pattern.format(scene=scene_name))
    cands = sorted(glob.glob(os.path.join(ckpt_dir, "chkpnt*.pth")))
    if not cands:
        print(f"  !! khong thay chkpnt*.pth trong {ckpt_dir} — skip")
        return

    tmpdir = tempfile.mkdtemp(prefix="p35d_")
    sargs = p35._build_scene_args(src, tmpdir, args.n_views, args.resolution)
    gaussians = GaussianModel(sargs)
    scene = Scene(sargs, gaussians, load_iteration=None, shuffle=False,
                  resolution_scales=[1.0])
    p35.load_checkpoint_into(gaussians, cands[-1])

    train_cams = scene.getTrainCameras()
    test_cams = scene.getTestCameras()

    # 3 ứng viên signal — giống Step 1 (§12.1) để kết quả so được
    crs = torch.sigmoid(gaussians._crs_score).reshape(-1).detach()
    G = p35.compute_fresh_G(gaussians, train_cams)
    crs_att = (crs * (1.0 - G)).clamp(0.0, 1.0)
    SIGNALS = {"crs": crs, "att": crs_att}

    acc = {k: {"ours": [], "rand": []} for k in SIGNALS}
    for cam in test_cams[: args.max_test_views]:
        gt = cam.original_image.cuda().clamp(0.0, 1.0)
        rgb = p35.render_rgb(gaussians, cam)
        err_map = (rgb - gt).abs().mean(dim=0)               # (H,W)

        for key, sig in SIGNALS.items():
            m, alpha = p35.splat_signal(gaussians, cam, sig)
            valid = alpha > 0.5           # bỏ vùng trống, không kết luận gì
            if valid.sum() < 1000:
                continue
            # uncertainty = 1 − confidence
            unc = (1.0 - m)[valid].reshape(-1)
            err = err_map[valid].reshape(-1)
            a_ours, a_rand, _, _ = compute_ause(unc, err)
            if not np.isnan(a_ours):
                acc[key]["ours"].append(a_ours)
                acc[key]["rand"].append(a_rand)

    for key in SIGNALS:
        if not acc[key]["ours"]:
            continue
        ao = float(np.mean(acc[key]["ours"]))
        ar = float(np.mean(acc[key]["rand"]))
        # Gain = giảm được bao nhiêu % AUSE so với đoán mò
        gain = (ar - ao) / max(ar, 1e-9) * 100.0
        rows.append({"scene": scene_name, "signal": key,
                     "ause": ao, "ause_random": ar, "gain_pct": gain})
        print(f"  [{key:>3}] AUSE={ao:.4f} | random={ar:.4f} | "
              f"tot hon random {gain:+.1f}%")

    del gaussians, scene
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(
        description="[CRSGaussian P35d] AUSE cho CRS uncertainty")
    ap.add_argument("--data_root", default="data/nerf_llff_data")
    ap.add_argument("--ckpt_root", default="output/p28_crs_boost/tau65")
    ap.add_argument("--ckpt_pattern", default="A3_seed42_{scene}")
    ap.add_argument("--out_dir", default="output/p35d_ause")
    ap.add_argument("--scenes", nargs="+",
                    default=["fern", "flower", "fortress", "horns",
                             "leaves", "orchids", "room", "trex"])
    ap.add_argument("--n_views", type=int, default=3)
    ap.add_argument("--resolution", type=int, default=8)
    ap.add_argument("--max_test_views", type=int, default=8)
    args = ap.parse_args()

    p35._BG = torch.tensor([0.0, 0.0, 0.0], device="cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for s in args.scenes:
        try:
            run_scene(s, args, rows)
        except Exception as e:
            print(f"  !! {s} FAILED: {e}")
            import traceback
            traceback.print_exc()

    suffix = args.scenes[0] if len(args.scenes) < 8 else "all"
    csv_path = os.path.join(args.out_dir, f"ause_{suffix}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scene", "signal", "ause",
                                          "ause_random", "gain_pct"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {csv_path}")

    print("\n===== AUSE SUMMARY (thap = tot) =====")
    for key, label in (("crs", "CRS alone"), ("att", "CRS_att = CRS x (1-G)")):
        rs = [r for r in rows if r["signal"] == key]
        if not rs:
            continue
        ao = float(np.mean([r["ause"] for r in rs]))
        ar = float(np.mean([r["ause_random"] for r in rs]))
        gain = (ar - ao) / max(ar, 1e-9) * 100.0
        print(f"  {label:<22} AUSE={ao:.4f} | random={ar:.4f} | "
              f"tot hon random {gain:+.1f}% (n={len(rs)} scene)")
    print("\n  Doc ket qua:")
    print("   - gain < ~15%  -> signal yeu, huong uncertainty NEN DUNG")
    print("   - gain 15-40%  -> co gia tri, can so voi WarpRF/OUGS moi ket luan")
    print("   - gain > 40%   -> manh, dang theo duoi truc uncertainty")
    print("  (Nguong la uoc luong cua Claude, CHUA verify voi so publish —")
    print("   phai doc WarpRF 2506.22433 / OUGS 2511.09397 de co moc that.)")


if __name__ == "__main__":
    main()
