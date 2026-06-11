# Baseline 05 — DNGaussian (CVPR 2024)

> Li et al., "DNGaussian: Optimizing Sparse-View 3D Gaussian Radiance Fields with Global-Local Depth Normalization", CVPR 2024.
> [Paper](https://arxiv.org/abs/2403.06912) · base 3DGS + global-local depth normalization (MiDaS DPT depth).
> Protocol: **-r 8 ✓ · 6k iter (NHANH) ✓ · n_sparse 3 ✓** · init **rand_pcd** (sparse, KHÔNG đụng fused.ply). Có 3/6/9 trong paper.

**Trạng thái (2026-06-10): ✅ DONE — reproduced AVG PSNR 19.091 (paper 19.12, Δ −0.029, khớp gần hoàn hảo). 8/8 scene. FPS 305.1 (NHANH NHẤT nhóm).**

---

## 1. Method
Base 3DGS, **6k iter**, init random từ sparse COLMAP (`--rand_pcd`), + **global-local depth normalization** dùng
**MiDaS DPT_Hybrid** monocular depth (precompute `depth_maps/`). Rasterizer = **ashawkey diff-gaussian-rasterization** (depth, KHÔNG confidence). Không cần gridencoder cho LLFF (train_llff.py không import).

Paper LLFF 3-view: cite từ paper DNGaussian (có 3/6/9). README repo (MVS init, 2 random tests best): 19.942.

---

## 2. Setup — khác các baseline trước
| Thành phần | Chi tiết |
|---|---|
| Env | official torch 1.12/cu113 → **hướng B cu121** (như fsgs) |
| Rasterizer | **ashawkey** `diff-gaussian-rasterization` (clone github, KHÔNG có sẵn — submodules/ trống) + simple-knn |
| **Depth** | **MiDaS DPT_Hybrid** (đã cache từ FSGS `~/.cache/torch/hub`) → `dpt/get_depth_map_for_llff_dtu.py` ghi `<scene>/depth_maps/depth_*.png` |
| Init | `--rand_pcd` (random từ `sparse/0/points3D.bin`) → ghi `points3D_random.ply` vào sparse |
| **Ghi vào data** | depth_maps + points3D_random → **dùng DATA TREE RIÊNG** (copy sparse + symlink images/images_8) để KHÔNG đụng Phase 22 |
| Resolution/iter | -r 8, 6k iter |

---

## 3. Env setup (server) — RECIPE ĐÚNG (2026-06-10)

> **ROOT CAUSE "build hell" lần trước** (đã verify code): `gridencoder/setup.py` + `shencoder/setup.py`
> hard-code `-std=c++14` (line 8 nvcc + line 13 cxx). **PyTorch 2.x headers cần C++17** → fail.
> Fix = sửa 4 dòng `c++14→c++17`. **TUYỆT ĐỐI KHÔNG downgrade torch** — downgrade 2.4→2.1.2 chính
> là thứ đẻ ra cascade numpy2/pkg_resources/setuptools/opencv lần trước (self-inflicted).
>
> **Đính chính claim cũ "không cần gridencoder cho LLFF" = SAI.** train_llff.py → render() →
> `pc.neural_renderer()` (gaussian_renderer/__init__.py:24) → GridRenderer dùng `hashgrid`(gridencoder)
> + `sphere_harmonics`(shencoder). **Cả 2 ext bắt buộc**, nằm thẳng trên render path.

```bash
ROOT=/home/aidev/workspace/representation-3d/duyen
# B1. Env — torch 2.4.1+cu121 (KHÔNG downgrade), CUDA_HOME=/usr/local/cuda (nvcc 12.4)
conda create -n dngaussian python=3.10 -y
conda activate dngaussian
export CUDA_HOME=/usr/local/cuda
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install ninja plyfile tqdm timm torchmetrics open3d "opencv-python<4.10" imageio matplotlib scipy scikit-image lpips

# B2. FIX ROOT CAUSE — c++14 → c++17 (gridencoder + shencoder, mỗi file 2 dòng)
cd $ROOT/DNGaussian
sed -i "s/-std=c++14/-std=c++17/g" gridencoder/setup.py shencoder/setup.py
grep -n "std=c++" gridencoder/setup.py shencoder/setup.py   # verify → phải thấy c++17

# B3. Clone rasterizer (ashawkey, depth-capable) + simple-knn vào submodules/ (đang trống)
git clone https://github.com/ashawkey/diff-gaussian-rasterization.git --recursive submodules/diff-gaussian-rasterization
git clone https://gitlab.inria.fr/bkerbl/simple-knn.git submodules/simple-knn

# B4. Build 4 CUDA ext — --no-build-isolation (PEP517 isolation thiếu torch)
pip install --no-build-isolation ./submodules/diff-gaussian-rasterization
pip install --no-build-isolation ./submodules/simple-knn
pip install --no-build-isolation ./gridencoder
pip install --no-build-isolation ./shencoder

# B5. Verify cả 4 ext
python -c "import torch,diff_gaussian_rasterization,simple_knn,gridencoder,shencoder; print('BUILD OK',torch.__version__)"

# B6. PATCH depth uint16 — utils/general_utils.py:23 PILtoTorch (DPT ghi uint16 PNG,
#     torch.from_numpy không nhận uint16). Thêm .astype('float32'):
sed -i "s/torch.from_numpy(np.array(resized_image_PIL))/torch.from_numpy(np.array(resized_image_PIL).astype('float32'))/" utils/general_utils.py
grep -n "astype('float32')" utils/general_utils.py   # verify
```

---

## 4. Diagnostic
```bash
ROOT=/home/aidev/workspace/representation-3d/duyen
ls -d $ROOT/DNGaussian && echo "repo OK"
ls $ROOT/DNGaussian/dpt/get_depth_map_for_llff_dtu.py
ls ~/.cache/torch/hub/checkpoints/dpt_hybrid_384.pt && echo "DPT cached (từ FSGS)" || echo "DPT cần tải"
conda env list | grep -i dngaussian || echo "env chưa có"
```

---

## 5. Pipeline (`run_dngaussian_llff.sh`)
- `prepare`: data tree riêng (copy sparse + symlink images/images_8) + **DPT depth gen** → depth_maps trong tree.
- per scene: `train_llff.py -s tree/<scene> -m $OUT/<scene> -r 8 --eval --n_sparse 3 --rand_pcd --iterations 6000 ...` → render → unified eval.
- Output: `DNGaussian/output/LLFF_ablation/<scene>/` · Log: `CoR-GS/logs/ablation/dngaussian/`.

---

## 6. Kết quả (reproduced 2026-06-10, unified eval corgs, -r8, 6k iter, rand_pcd)
| Scene | PSNR | SSIM | SSIM_sk | LPIPS | AVGE | N_gauss | Train(s) |
|-------|------|------|---------|-------|------|---------|----------|
| fern | 20.505 | 0.6536 | 0.6444 | 0.2879 | 0.1217 | 39,667 | 169* |
| flower | 18.197 | 0.5275 | 0.5181 | 0.3317 | 0.1652 | 39,413 | 427 |
| fortress | 20.855 | 0.4331 | 0.3900 | 0.3075 | 0.1286 | 39,148 | 467 |
| horns | 19.245 | 0.6388 | 0.6251 | 0.3415 | 0.1384 | 34,844 | 452 |
| leaves | 16.821 | 0.5669 | 0.5811 | 0.2711 | 0.1582 | 94,906 | 629 |
| orchids | 15.500 | 0.4335 | 0.4347 | 0.3407 | 0.2036 | 55,390 | 586 |
| room | 20.326 | 0.7668 | 0.7472 | 0.2723 | 0.1108 | 26,149 | 588 |
| trex | 21.278 | 0.7516 | 0.7463 | 0.2058 | 0.0937 | 45,636 | 617 |
| **Avg** | **19.091** | **0.597** | **0.586** | **0.295** | **0.140** | **46,894** | **~492** |

\* fern train_s = từ smoke run (GPU rảnh hơn) → thấp bất thường; 7 scene full ~427-629s.

**Reproduction vs paper:** PSNR 19.091 vs paper **19.12** (Δ **−0.029**) · SSIM 0.597 vs 0.591 (+0.006) ·
LPIPS 0.295 vs 0.294 (+0.001) → **khớp gần hoàn hảo** trong sai số non-determinism.

vs ours **21.918** (Δ **+2.827** ⭐) · Binocular 21.356 · FSGS 20.407 · CoR-GS 20.110 → **DNGaussian = thấp nhất nhóm reproduced**.

**Điểm cho paper:**
- **Compact nhất nhóm**: N_gauss avg ~47k (rand_pcd + neural hashgrid renderer) vs FSGS 282k / Binocular 107k / ours 95k → ít Gaussian nhất NHƯNG PSNR cũng thấp nhất.
- orchids 15.50 / leaves 16.82 kéo avg xuống (scene khó cho rand_pcd init, không có MVS dense).
- **Khác class kiến trúc**: DNGaussian thay SH-color + opacity bằng hashgrid-MLP (neural_renderer) → render KHÔNG thuần rasterization. Caption figure nên ghi chú.
- **FPS = 305.1 — NHANH NHẤT nhóm** (vs FSGS 281.2 / Binocular 266.6 / ours 213.2-fair), per-scene: room 353.6 (N=26k) … leaves 257.3 (N=95k). ⚠️ Dự đoán ban đầu "MLP → chậm" SAI: với DNGaussian, rasterization cost ∝ N_gauss áp đảo overhead MLP → ít Gaussian nhất ⇒ nhanh nhất (FPS tỉ lệ nghịch N_gauss **trong cùng rasterizer**). LƯU Ý: quy luật này KHÔNG xuyên-method — ours 95k < FSGS 282k/Binocular 107k nhưng vẫn chậm hơn do chi phí rasterizer-confidence (mỗi rasterizer có phí cố định riêng). Bench unified (warmup50+timed300, restore chkpnt + neural_renderer, inference=True). Script `scripts/ablation/bench_fps_dngaussian.sh`, json `CoR-GS/output/ablation/dngaussian/`.

---

## 7. Caveats
- Init rand_pcd (sparse/random) = native DNGaussian → fair. Paper note: "metrics unstable, especially PSNR" (non-deterministic).
- 6k iter = nhanh nhất nhóm.
- DPT depth = external prior (giống ta dùng DAV2/RoMa) → ghi rõ.
- Có đủ 3/6/9 trong paper → dùng cho bảng 6/9.
