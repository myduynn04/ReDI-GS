#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 13.2.3 pre-flight] Validate weighted L1 supervision direction.
# File: scripts/p13_2_weighted_preflight.py  (TẠO MỚI)
# Mục đích: 2 cheap post-hoc tests TRƯỚC khi commit 1-2 ngày code
#           weighted L1 (covisibility × frequency cross-product).
#   Test A — Noise direction: variance(error|mono-covis) vs (error|multi-covis)
#   Test B — Dynamic range : Shannon H(covis) + Pearson(covis, freq)
# KHÔNG training. CHỈ post-hoc trên existing A3 renders.
#
# ── 2 CORRECTNESS FIXES vs original spec (bắt buộc, nếu không → false verdict) ──
#   FIX 1 (CRITICAL): train/test split sort theo IMAGE NAME, KHÔNG theo COLMAP id.
#     CRSGaussian dataset_readers.py:353 sort key=image_name. COLMAP id gán
#     tùy ý lúc reconstruct → sort-by-id ≠ sort-by-name → split sai → covis
#     tính cho sai cặp → Test A/B vô nghĩa.
#   FIX 2 (METHODOLOGICAL): validity mask — pixel KHÔNG có COLMAP keypoint
#     support (weight_dense ~ 0) bị LOẠI khỏi analysis, KHÔNG đếm là mono-covis.
#     Nếu không: textureless region (low HF, low error, no feature) flood
#     "mono" bucket → var_mono giảm giả → Test A PASS giả.
# ============================================================
"""[CRSGaussian Phase 13.2.3 pre-flight] Validate weighted L1 supervision direction.

Test A: Noise direction — variance(error|mono-covis) vs variance(error|multi-covis)
        Hypothesis bị test: up-weight mono-covis = amplify noise (anti-pattern
        robust regression). Nếu var_mono >> var_multi → REJECT up-weight formula.

Test B: Dynamic range — Shannon entropy H(covis), Pearson(covis, freq)
        Hypothesis bị test: LLFF 3-view → covis distribution degenerate
        (>80% pixels covis=max) → no signal traction. Plus covis×freq
        collinear → joint product redundant.

Inputs:
  output/p13_lfcf/A3_seed42_<scene>/test/ours_10000/{renders,gt}/*.png
  data/nerf_llff_data/<scene>/sparse/0/{cameras,images}.bin  (COLMAP)

Outputs:
  logs/p13_2_preflight/SUMMARY.txt
  logs/p13_2_preflight/<scene>_diagnostic.png

Run:
    python scripts/p13_2_weighted_preflight.py
    SCENES="orchids horns trex" python scripts/p13_2_weighted_preflight.py
"""

import os
import sys
import glob
import statistics

import numpy as np
from PIL import Image as PILImage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary


# ── Config ──
DATA_ROOT = os.environ.get("DATA_ROOT", "data/nerf_llff_data")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "output/p13_lfcf")
SEED = os.environ.get("SEED", "42")
ITERATION = int(os.environ.get("ITERATION", "10000"))
SCENES = os.environ.get(
    "SCENES", "fern flower fortress horns leaves orchids room trex").split()
PLOT_DIR = os.environ.get("PLOT_DIR", "logs/p13_2_preflight")
LLFFHOLD = 8
N_VIEWS_TRAIN = 3

# Decision thresholds
NOISE_RATIO_REJECT = 2.0      # Test A: var_mono/var_multi > 2.0 → REJECT
NOISE_RATIO_PASS = 1.5        # < 1.5 → PASS
ENTROPY_MIN = 0.5             # Test B: H(covis) < 0.5 bit → degenerate
CORRELATION_MAX = 0.6         # Test B: |Pearson(covis,freq)| > 0.6 → redundant

# FIX 2: validity — only analyze pixels with real COLMAP keypoint support.
# weight_dense (Gaussian-splat coverage) phải vượt ngưỡng này mới tính.
COVIS_SUPPORT_MIN = 0.05


# ── Utilities ──
def load_image_lum(path):
    img = np.asarray(PILImage.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def _stem(name):
    """Strip path + extension — khớp dataset_readers.py:217
    image_name = os.path.basename(extr.name).split('.')[0].
    Byte-exact mirror để sort order + render-PNG name join chính xác.
    """
    return os.path.basename(name).split(".")[0]


def get_train_test_camera_ids(images):
    """[FIX 1] Split theo IMAGE NAME (stripped) — khớp dataset_readers.py:353.

    dataset_readers.py:
        :217 image_name = os.path.basename(extr.name).split('.')[0]
        :353 cam_infos = sorted(cam_infos, key=lambda x: x.image_name)
        :356 train = [idx%8 != 0];  test = [idx%8 == 0]
        :363 n_views: idx_sub = round(linspace(0, len(train)-1, n_views))

    COLMAP image id gán tùy ý lúc reconstruct → KHÔNG dùng sorted(keys()).
    Sort theo _stem(.name) (basename, bỏ ext) để byte-exact mirror.
    """
    sorted_ids = sorted(images.keys(), key=lambda k: _stem(images[k].name))
    test_ids, train_ids = [], []
    for i, img_id in enumerate(sorted_ids):
        if i % LLFFHOLD == 0:
            test_ids.append(img_id)
        else:
            train_ids.append(img_id)
    # Subsample train → N_VIEWS_TRAIN (khớp dataset_readers.py:363-365)
    if len(train_ids) > N_VIEWS_TRAIN:
        idx_sub = np.linspace(0, len(train_ids) - 1, N_VIEWS_TRAIN)
        idx_sub = [round(i) for i in idx_sub]
        train_ids = [train_ids[i] for i in idx_sub]
    return train_ids, test_ids


def compute_covis_sparse(test_img_id, train_img_ids, images):
    """Per-keypoint covis count cho test view, dùng COLMAP image tracks.

    Mỗi test keypoint có point3D_id (-1 nếu no match). Đếm bao nhiêu train
    view cũng thấy point3D_id đó. KHÔNG cần MASt3R/points3D.bin.

    Returns: covis_count (M,), pixel_xy (M, 2)  — M = số keypoint có 3D match.
    """
    test_image = images[test_img_id]
    test_xys = test_image.xys                 # (N_kp, 2)
    test_pids = test_image.point3D_ids        # (N_kp,)

    train_point_sets = []
    for tid in train_img_ids:
        tr = images[tid]
        train_point_sets.append(
            set(tr.point3D_ids[tr.point3D_ids > 0].tolist()))

    covis_count, pixel_xy = [], []
    for kp_idx, p3d_id in enumerate(test_pids):
        if p3d_id < 0:
            continue
        covis = sum(1 for ts in train_point_sets if p3d_id in ts)
        covis_count.append(covis)
        pixel_xy.append(test_xys[kp_idx])
    return np.array(covis_count), np.array(pixel_xy)


def rasterize_covis(covis_sparse, pixel_xy, H, W, sigma=15.0):
    """Sparse covis → dense via Gaussian splat. Returns (covis_dense, weight_dense).

    [FIX 2] weight_dense trả về để build validity mask. Pixel weight thấp
    = không có keypoint support → loại khỏi analysis (KHÔNG đếm mono).
    """
    covis_dense = np.zeros((H, W), dtype=np.float32)
    weight_dense = np.zeros((H, W), dtype=np.float32)
    if len(pixel_xy) == 0:
        return covis_dense, weight_dense

    ks = int(3 * sigma)
    for i in range(len(covis_sparse)):
        cx, cy = int(round(pixel_xy[i][0])), int(round(pixel_xy[i][1]))
        if not (0 <= cx < W and 0 <= cy < H):
            continue
        x0, x1 = max(0, cx - ks), min(W, cx + ks + 1)
        y0, y1 = max(0, cy - ks), min(H, cy + ks + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        wk = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        covis_dense[y0:y1, x0:x1] += wk * covis_sparse[i]
        weight_dense[y0:y1, x0:x1] += wk

    valid = weight_dense > 1e-6
    covis_dense[valid] /= weight_dense[valid]
    return covis_dense, weight_dense


def laplacian_magnitude(img):
    """|∇²I| HF proxy. img: (H, W)."""
    K = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    pad = np.pad(img, 1, mode='edge')
    lap = np.zeros_like(img)
    for dy in range(3):
        for dx in range(3):
            lap += K[dy, dx] * pad[dy:dy + img.shape[0], dx:dx + img.shape[1]]
    return np.abs(lap)


def shannon_entropy(values, n_bins=10):
    hist, _ = np.histogram(values, bins=n_bins)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


# ── Per-scene analysis ──
def analyze_scene(scene):
    print(f"\n──── {scene} ────")
    sparse_dir = os.path.join(DATA_ROOT, scene, "sparse", "0")
    try:
        _ = read_intrinsics_binary(os.path.join(sparse_dir, "cameras.bin"))
        images = read_extrinsics_binary(os.path.join(sparse_dir, "images.bin"))
    except Exception as e:
        print(f"  ERR loading COLMAP: {e}")
        return None

    train_ids, test_ids = get_train_test_camera_ids(images)
    train_names = [images[i].name for i in train_ids]
    print(f"  Train views (by name): {train_names}")
    print(f"  N test: {len(test_ids)}")

    test_dir = os.path.join(OUTPUT_ROOT, f"A3_seed{SEED}_{scene}",
                            "test", f"ours_{ITERATION}")
    render_paths = sorted(glob.glob(os.path.join(test_dir, "renders", "*.png")))
    if not render_paths:
        print(f"  ERR no renders at {test_dir}")
        return None

    # [FIX 3 — Caveat 1 resolved] render.py:48-49 lưu file = view.image_name
    # + '.png' (stripped basename, KHÔNG phải {idx:05d}.png). Pair render ↔
    # COLMAP-test BY NAME tường minh, bỏ positional zip hoàn toàn → loại
    # giả định pairing (cùng class lỗi sort-by-id nếu sai).
    gt_dir = os.path.join(test_dir, "gt")
    test_name_to_id = {_stem(images[i].name): i for i in test_ids}

    all_covis, all_freq, all_error = [], [], []
    total_px, valid_px = 0, 0
    matched, unmatched = 0, 0

    for rp in render_paths:
        stem = _stem(os.path.basename(rp))
        cam_id = test_name_to_id.get(stem)
        if cam_id is None:
            unmatched += 1
            continue
        gp = os.path.join(gt_dir, stem + ".png")
        if not os.path.isfile(gp):
            unmatched += 1
            continue
        matched += 1
        render_lum = load_image_lum(rp)
        gt_lum = load_image_lum(gp)
        H, W = gt_lum.shape

        covis_sparse, pixel_xy = compute_covis_sparse(
            cam_id, train_ids, images)
        if len(covis_sparse) == 0:
            print(f"    {stem}: no covis data, skip")
            continue

        covis_dense, weight_dense = rasterize_covis(
            covis_sparse, pixel_xy, H, W, sigma=15.0)
        hf_map = laplacian_magnitude(gt_lum)
        err_map = np.abs(render_lum - gt_lum)

        # [FIX 2] validity mask — chỉ pixel có COLMAP keypoint support thật.
        # Pixel weight thấp = textureless / no feature → KHÔNG phải mono-view,
        # loại khỏi Test A/B để tránh confound.
        valid_mask = weight_dense > COVIS_SUPPORT_MIN
        total_px += H * W
        valid_px += int(valid_mask.sum())

        cv = covis_dense[valid_mask]
        fv = hf_map[valid_mask]
        ev = err_map[valid_mask]

        # Subsample 10% để giảm memory
        if len(cv) > 2000:
            sub = np.random.choice(len(cv), len(cv) // 10, replace=False)
            cv, fv, ev = cv[sub], fv[sub], ev[sub]

        all_covis.append(cv)
        all_freq.append(fv)
        all_error.append(ev)
        print(f"    {stem}: {H}×{W}  covis[{covis_dense.min():.2f},"
              f"{covis_dense.max():.2f}]  valid={valid_mask.mean()*100:.1f}%")

    print(f"  Pairing: matched={matched} unmatched={unmatched}"
          f"  (unmatched>0 → render PNG name không khớp COLMAP test split)")
    if not all_covis:
        return None

    covis_arr = np.concatenate(all_covis)
    freq_arr = np.concatenate(all_freq)
    error_arr = np.concatenate(all_error)
    coverage = valid_px / total_px if total_px else 0.0

    # ── Test A ──
    mono_mask = covis_arr < 1.0
    multi_mask = covis_arr >= 2.0
    if mono_mask.sum() < 100 or multi_mask.sum() < 100:
        print(f"  WARN few pixels (mono={mono_mask.sum()} "
              f"multi={multi_mask.sum()}) — Test A unreliable")
        return None

    var_mono = float(error_arr[mono_mask].var())
    var_multi = float(error_arr[multi_mask].var())
    var_ratio = var_mono / var_multi if var_multi > 1e-8 else float('inf')

    # ── Test B ──
    h_covis = shannon_entropy(covis_arr, n_bins=10)
    h_freq = shannon_entropy(freq_arr, n_bins=10)
    pearson = float(np.corrcoef(covis_arr, freq_arr)[0, 1])

    print(f"  Test A: var_mono={var_mono:.6f} var_multi={var_multi:.6f} "
          f"ratio={var_ratio:.2f}")
    print(f"  Test B: H(covis)={h_covis:.3f} bit  "
          f"Pearson(covis,freq)={pearson:.3f}  coverage={coverage*100:.1f}%")

    # Plot
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(covis_arr, bins=20)
    ax[0].set_title(f"{scene} covis (H={h_covis:.2f}b, cov={coverage*100:.0f}%)")
    ax[0].set_xlabel("covis count")
    ax[1].hist(freq_arr, bins=50)
    ax[1].set_yscale('log')
    ax[1].set_title(f"{scene} |∇²I|")
    ax[1].set_xlabel("HF magnitude")
    sub = np.random.choice(len(covis_arr), min(5000, len(covis_arr)), replace=False)
    ax[2].scatter(covis_arr[sub], error_arr[sub], s=1, alpha=0.3)
    ax[2].set_title(f"{scene} error vs covis (r={pearson:.2f})")
    ax[2].set_xlabel("covis")
    ax[2].set_ylabel("|render-GT|")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{scene}_diagnostic.png"), dpi=80)
    plt.close()

    return {
        'scene': scene, 'var_mono': var_mono, 'var_multi': var_multi,
        'var_ratio': var_ratio, 'h_covis': h_covis, 'h_freq': h_freq,
        'pearson': pearson, 'coverage': coverage,
        'n_mono': int(mono_mask.sum()), 'n_multi': int(multi_mask.sum()),
    }


def main():
    print("=== Phase 13.2.3 Weighted Supervision Pre-flight ===")
    print(f"SCENES={SCENES}  SEED={SEED}  ITERATION={ITERATION}")
    print(f"FIX 1: split sort-by-NAME (match dataset_readers.py:353)")
    print(f"FIX 2: validity mask COVIS_SUPPORT_MIN={COVIS_SUPPORT_MIN}")
    print(f"Output: {PLOT_DIR}")
    os.makedirs(PLOT_DIR, exist_ok=True)

    results = []
    for sc in SCENES:
        r = analyze_scene(sc)
        if r is not None:
            results.append(r)
    if not results:
        print("\nNO scenes analyzed.")
        sys.exit(1)

    # ── Test A table ──
    print(f"\n=== Test A: Noise direction (var_mono / var_multi) ===")
    print(f"{'scene':<10} {'var_mono':>10} {'var_multi':>10} {'ratio':>7} "
          f"{'cover%':>7} {'verdict':>9}")
    print("-" * 60)
    a_pass = 0
    for r in results:
        v = ("REJECT" if r['var_ratio'] > NOISE_RATIO_REJECT
             else ("PASS" if r['var_ratio'] < NOISE_RATIO_PASS else "MARGIN"))
        if v == "PASS":
            a_pass += 1
        print(f"{r['scene']:<10} {r['var_mono']:>10.6f} {r['var_multi']:>10.6f} "
              f"{r['var_ratio']:>7.2f} {r['coverage']*100:>6.1f}% {v:>9}")
    avg_ratio = statistics.fmean(r['var_ratio'] for r in results)
    print(f"\n  Aggregate var ratio = {avg_ratio:.2f}")

    # ── Test B table ──
    print(f"\n=== Test B: Dynamic range + collinearity ===")
    print(f"{'scene':<10} {'H(covis)':>9} {'H(freq)':>9} {'Pearson':>9} "
          f"{'verdict':>9}")
    print("-" * 55)
    b_pass = 0
    for r in results:
        h_ok = r['h_covis'] >= ENTROPY_MIN
        c_ok = abs(r['pearson']) <= CORRELATION_MAX
        v = "PASS" if (h_ok and c_ok) else "REJECT"
        if v == "PASS":
            b_pass += 1
        print(f"{r['scene']:<10} {r['h_covis']:>9.3f} {r['h_freq']:>9.3f} "
              f"{r['pearson']:>9.3f} {v:>9}")
    avg_h = statistics.fmean(r['h_covis'] for r in results)
    avg_p = statistics.fmean(abs(r['pearson']) for r in results)
    avg_cov = statistics.fmean(r['coverage'] for r in results)
    print(f"\n  Aggregate H(covis)={avg_h:.3f}b  |Pearson|={avg_p:.3f}  "
          f"coverage={avg_cov*100:.1f}%")

    # ── Final verdict ──
    print(f"\n=== FINAL VERDICT ===")
    print(f"  Test A PASS: {a_pass}/{len(results)}  (gate ≥75%)")
    print(f"  Test B PASS: {b_pass}/{len(results)}  (gate ≥75%)")
    if avg_cov < 0.10:
        print(f"  ⚠️  COLMAP coverage {avg_cov*100:.1f}% < 10% — covis estimate")
        print(f"      sparse/unreliable; treat verdict as low-confidence")

    ta = a_pass >= 0.75 * len(results)
    tb = b_pass >= 0.75 * len(results)
    if ta and tb:
        print(f"\n  ✅ BOTH PASS — mechanism direction OK")
        print(f"  → Proceed lightweight pilot V1-V4 (1 day)")
    elif ta and not tb:
        print(f"\n  ⚠️  A PASS, B FAIL (degenerate signal)")
        print(f"  → Direction OK but no traction 3-view → skip pilot, Tier 3")
    elif not ta and tb:
        print(f"\n  ⚠️  A FAIL (noise dir), B PASS")
        print(f"  → Sign error — try DOWN-weight mono-covis (inverse formula)")
    else:
        print(f"\n  ❌ BOTH FAIL — direction fundamentally flawed")
        print(f"  → REJECT weighted supervision; pivot Tier 3 or accept ceiling")

    # ── Honest scope caveats — verdict KHÔNG được over-claim ──
    print(f"\n=== Verdict caveats (đọc kèm, KHÔNG tuyệt đối hóa) ===")
    print(f"  C2 — Scope: chỉ test necessary conditions (noise dir + dynamic")
    print(f"       range/collinearity). Lỗ hổng #4 (3DGS đã implicit reweight")
    print(f"       qua densification density) CHƯA test. 'BOTH PASS' = điều")
    print(f"       kiện cần đạt, KHÔNG phải 'mechanism validated'.")
    print(f"  C3 — Test A đo var(error): cao ở mono có thể do (a) noisy")
    print(f"       supervision (REJECT đúng) HOẶC (b) mono region high-bias")
    print(f"       không noisy (false-REJECT). REJECT nên đọc kèm caveat này.")
    print(f"  C1 — Resolved: render↔COLMAP join BY NAME (render.py:48 lưu")
    print(f"       view.image_name+'.png'). Check 'unmatched=0' mỗi scene;")
    print(f"       unmatched>0 → split/name mismatch, verdict scene đó invalid.")


if __name__ == "__main__":
    main()
