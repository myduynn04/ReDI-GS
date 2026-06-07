# Baseline 01 — Binocular3DGS (NeurIPS 2024)

> **Han et al., "Binocular-Guided 3D Gaussian Splatting with View Consistency for Sparse View Synthesis", NeurIPS 2024.**
> [Paper](https://arxiv.org/abs/2410.18822) · [Project](https://hanl2010.github.io/Binocular3DGS/) · repo: `graphdeco/3DGS` base
> Mục tiêu: chạy lại full method trên LLFF 3-view ở **protocol của CRSGaussian (-r 8, hold-out test)** để có số fair so với 21.89.

**Trạng thái (2026-06-07): 🔧 SETUP — còn 1 blocker (build rasterizer trong env `binocular3dgs`).**

---

## 1. Method tóm tắt (cho paper)

Build trên 3DGS gốc, **30k iter**, 3 thành phần:

1. **Dense init bằng PDCNet+** — dense matching giữa các view → `cv2.triangulatePoints` → SSIM-based densification (1000 iter sampling) → `<scene>_keypoints_to_3d.ply`. Thay COLMAP-sparse init (~vài K điểm) bằng dense (~chục K điểm).
2. **Binocular Stereo Consistency Loss** — tịnh tiến camera một đoạn (`cam_trans_dist=0.4`), warp ảnh render theo disparity, L1 + smooth loss. **Chỉ bật sau iter `shift_cam_start=20000`** (self-supervised, single-view stereo — KHÔNG cross-view wide-baseline).
3. **Opacity Decay** — nhân opacity ×`0.995` mỗi iter sau densify_from (default ON), và `densify_until_iter = iterations` (extend densify cả quá trình).

**Số paper LLFF 3-view: PSNR 21.44** (ở resolution gốc của repo = `--resolution 2`).

> Quan hệ với CRSGaussian: opacity-decay đã được "mượn" (Phase 20 attribution). Dense-init PDCNet+
> đã thử ở Phase 18 (Δ+0.27 nhưng C3 fail thin-structure) rồi pivot RoMa v1 (Phase 22, hiện là best).
> Binocular stereo (cross-view) thuộc lớp đã loại "structurally dead trên 3-view wide-baseline".

---

## 2. State trên server (verified 2026-06-07)

| Mục | Trạng thái |
|---|---|
| Repo (sibling clone) | `/home/aidev/workspace/representation-3d/duyen/Binocular3DGS` ✅ |
| PDCNet+ weight | `submodules/dense_matcher/pre_trained_models/PDCNet_plus_megadepth.pth.tar` (+`.pth`) ✅ |
| Submodule `diff-gaussian-rasterization` (ashawkey, depth-enabled) | checked out ✅, **chưa build** ❌ |
| Submodule `simple-knn` | source ✅, **chưa build** ❌ |
| `triangulate.py` | EDITED (có 4 bug-fix + P18b diag, gated OFF) ✅ |
| Env `binocular3dgs` | có (đủ PDCNet+/GOCor/cupy) ✅ — **thiếu 2 CUDA ext** |
| Data | `duyen/CoR-GS/data/nerf_llff_data/` (COLMAP `sparse/0/*.bin` + `images/`) ✅ |

> ⚠️ **Phase 18 chỉ chạy `triangulate.py`** (lấy .ply nhét vào CRSGaussian). `train.py`/`render.py`/`metrics.py`
> của Binocular3DGS **chưa từng chạy** → phần GS training là MỚI, chưa kiểm chứng.

---

## 3. Quyết định fair-comparison (QUAN TRỌNG cho paper)

| Vấn đề | Repo gốc | CRSGaussian | Quyết định cho ablation |
|---|---|---|---|
| **Resolution** | `--resolution 2` (~2016px nếu trên `images/` full-res) | **`-r 8`** (~504px) | **Chạy ở `-r 8`** để khớp test images. Số sẽ khác paper 21.44. Ghi cả hai. |
| **Iterations** | 30k | 10k | Chạy **30k như paper** (đó là method của họ). Ghi rõ "cross-budget: 3× compute". |
| **Random bg** | không | `--random_background` | Giữ default của Binocular3DGS (không thêm) — đây là method của họ. |
| **P18b env flags** | — | — | **KHÔNG set** `P18B_CYCLIC_TAU` / `P18B_DIAGNOSTIC_ONLY` → pipeline byte-identical upstream. |
| **Seed** | không set | multi-seed | Binocular3DGS non-deterministic single-run (như đa số baseline). Chạy 1 lần; nếu cần, lặp để ước SEM. |

> **Lý do resolution = -r 8**: PSNR/SSIM/LPIPS phụ thuộc mạnh vào độ phân giải test. CRSGaussian báo
> 21.89 ở -r 8. Để cột "reproduced" có thể đặt cạnh 21.89 thì baseline phải eval cùng test images (-r 8).
> Số paper 21.44 (r2) chỉ để tham chiếu, ghi ở cột riêng.
>
> 🔲 **Chờ user chốt**: nếu muốn so ở r2 thay vì r8, đổi `resolution` trong run script + ghi lại đây.

---

## 4. Setup — build 2 CUDA extension (BLOCKER)

Chạy trên server, env `binocular3dgs`. **Env thực tế: torch 2.4.1+cu121** (KHÔNG phải cu118 như
requirements.txt) → cần nvcc 12.x. Chạy LẺ từng lệnh để dễ debug:

```bash
conda activate binocular3dgs

# B1 — verify toolchain
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available())"
which nvcc && nvcc --version            # cần 12.x
echo "CUDA_HOME=$CUDA_HOME"
# nếu thiếu nvcc / lệch bản:
#   export CUDA_HOME=/usr/local/cuda-12.1 ; export PATH=$CUDA_HOME/bin:$PATH

# B2 — build TỪNG cái, BẮT BUỘC --no-build-isolation
cd /home/aidev/workspace/representation-3d/duyen/Binocular3DGS
pip install --no-build-isolation ./submodules/diff-gaussian-rasterization
pip install --no-build-isolation ./submodules/simple-knn

# B3 — verify
python -c "import diff_gaussian_rasterization, simple_knn; print('BUILD OK')"
```

**Lỗi đã gặp + fix**:
- `ModuleNotFoundError: No module named 'torch'` khi build (trong `/tmp/pip-build-env-*`):
  pip PEP517 dựng build-env CÔ LẬP không có torch, mà `setup.py` import torch.
  → **Fix: `--no-build-isolation`** (dùng torch của env hiện tại). Đây là lỗi kinh điển rasterizer 3DGS.
- Build cần `nvcc` 12.x khớp torch cu121. Lỗi compiler → set `CUDA_HOME=/usr/local/cuda-12.x`.
- Import name `diff_gaussian_rasterization` trùng rasterizer CRSGaussian (`-confidence`) NHƯNG env
  `binocular3dgs` isolated → không xung đột. Bản đúng có `rendered_depth`+`rendered_alpha` ([train.py:105-106](../../../Binocular3DGS/train.py#L105)).

---

## 5. Sửa run script

`script/run_llff.py` có placeholder cần sửa:

```python
# DÒNG 8:  data_base_path='DATA_DIR'
data_base_path='/home/aidev/workspace/representation-3d/duyen/CoR-GS/data/nerf_llff_data'

# DÒNG 11: resolution = 2
resolution = 8        # khớp protocol CRSGaussian (-r 8). Đổi về 2 nếu muốn số paper-native.
```

`run_llff.py` tự dò GPU rảnh qua GPUtil (`mem_threshold=0.2`) và phân job 8 scene lên các GPU → tự dùng cả 2 GPU. Mỗi scene chạy tuần tự: `triangulate → train → render → metrics`.

---

## 6. Lệnh chạy

> **Cách nhanh nhất**: dùng script `scripts/ablation/run_binocular3dgs_llff.sh` — đã gói build +
> triangulate + train + render + unified-eval + gom kết quả + đo N_gauss/FPS/VRAM/time, 2-GPU song song.
> Upload lên server (chạy từ đâu cũng được, script tự `cd` tới `$REPO`):
> ```bash
> conda activate binocular3dgs
> bash run_binocular3dgs_llff.sh build    # build 2 CUDA ext (1 lần)
> bash run_binocular3dgs_llff.sh smoke    # thử fern trước
> bash run_binocular3dgs_llff.sh full     # 8 scene, 2-GPU
> bash run_binocular3dgs_llff.sh agg      # in lại bảng kết quả
> # đổi resolution:  RES=2 bash run_binocular3dgs_llff.sh full
> ```
> Sửa biến `REPO` / `DATA` / `ENV_NAME` ở đầu script nếu path khác.
>
> ⚠️ **KHÔNG dùng `read_eval_result.py`** của repo: nó tìm dir `{scene}_3views_{suffix}` (luôn thừa
> dấu `_`) → không khớp `{scene}_3views` mà train tạo → ra rỗng. Script trên có bộ gom riêng đọc thẳng
> `results.json`. Cấu trúc: `results.json = {"ours_30000": {"PSNR":_, "SSIM":_, "LPIPS":_}}`.

Phần dưới là các lệnh thủ công tương đương (nếu muốn chạy tay từng bước).

### 6a. Smoke test 1 scene (fern) trước khi chạy full

```bash
cd /home/aidev/workspace/representation-3d/duyen/Binocular3DGS
conda activate binocular3dgs
DATA=/home/aidev/workspace/representation-3d/duyen/CoR-GS/data/nerf_llff_data
RES=8

# (1) dense init
cd submodules/dense_matcher
CUDA_VISIBLE_DEVICES=0 python triangulate.py \
    --data_path $DATA/fern \
    --output_path $(pwd)/../../keypoints_to_3d/LLFF \
    --resolution $RES --dataset_name LLFF --n_views 3
cd ../..
ls -la keypoints_to_3d/LLFF/fern_keypoints_to_3d.ply   # phải tồn tại

# (2) train 30k
CUDA_VISIBLE_DEVICES=0 python train.py -s $DATA/fern -m output/LLFF_ablation/fern_3views \
    --n_views 3 --dataset_name LLFF --resolution $RES --eval

# (3) render + (4) metrics
CUDA_VISIBLE_DEVICES=0 python render.py -m output/LLFF_ablation/fern_3views \
    --n_views 3 --skip_train --resolution $RES --eval --dataset_name LLFF
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/LLFF_ablation/fern_3views
```

→ Kỳ vọng fern PSNR cao nhất trong 8 scene (tham chiếu Phase 22 v1: fern 23.84 @ r8 10k). Nếu hợp lý → chạy full.

### 6b. Full 8-scene (sau khi smoke OK)

```bash
cd /home/aidev/workspace/representation-3d/duyen/Binocular3DGS
conda activate binocular3dgs
# (đã sửa data_base_path + resolution=8 trong run_llff.py)
python script/run_llff.py 2>&1 | tee logs_binocular3dgs_llff_r8.log
```

### 6c. Đọc kết quả tổng hợp

```bash
python read_eval_result.py   # (kiểm tra script này gom PSNR/SSIM/LPIPS từ output/LLFF/*/results.json)
```

---

## 7. Kết quả (điền sau khi chạy)

> Quality metrics tính bằng **evaluator chung** `CoR-GS/metrics.py` (env `corgs`) trên renders của
> Binocular3DGS — KHÔNG dùng `metrics.py` riêng của repo (xem `00_index.md` §1.2).
> Script tự in bảng này khi chạy `agg`.

**Config chạy**: resolution = `__`, iter = 30k, render-env `binocular3dgs`, eval-env `corgs`, ngày `____`, GPU `____`.

| Scene | PSNR ↑ | SSIM ↑ | SSIM_sk ↑ | LPIPS ↓ | AVGE ↓ | N_gauss | FPS ↑ | MB ↓ | VRAM ↓ | train(s) |
|-------|--------|--------|-----------|---------|--------|---------|-------|------|--------|----------|
| fern | | | | | | | | | | |
| flower | | | | | | | | | | |
| fortress | | | | | | | | | | |
| horns | | | | | | | | | | |
| leaves | | | | | | | | | | |
| orchids | | | | | | | | | | |
| room | | | | | | | | | | |
| trex | | | | | | | | | | |
| **Avg** | | | | | | | | | | |

**So với CRSGaussian** (cùng -r 8, hold-out test, cùng evaluator):

| Method | Iter | PSNR | SSIM | LPIPS | AVGE | N_gauss | FPS | Δ PSNR | Compute |
|--------|------|------|------|-------|------|---------|-----|--------|---------|
| Binocular3DGS (reproduced, -r8) | 30k | _ | _ | _ | _ | _ | _ | — | 1× |
| Binocular3DGS (paper, r2) | 30k | 21.44 | — | — | — | — | — | — | native |
| **CRSGaussian (ours)** | 10k | **21.89** | _ | _ | _ | _ | _ | **+_** | 3× ít hơn |

---

## 8. Caveats cho paper (ghi để dùng khi viết)

- **Cross-budget**: Binocular3DGS 30k vs ours 10k → nếu ta ≥ họ thì "match/beat SOTA ở 3× ít compute" (narrative doc 24_3).
- **Cross-resolution**: số paper 21.44 ở r2; reproduced của ta ở r8 — KHÔNG trộn 2 cột.
- **Non-deterministic**: Binocular3DGS không set seed → single-run có atomicAdd noise (±1.3 dB single-scene). Avg 8-scene ổn định hơn (±0.46). Nếu cần CI, lặp ≥3 lần.
- **Data khác nguồn**: ta dùng COLMAP của CoR-GS, không phải LLFF "processed" mà Binocular3DGS phân phối → sparse model có thể khác → 1 nguồn sai khác nữa so với 21.44 paper. Reproduced-ở-protocol-ta mới là số fair để đặt cạnh ours.
- **P18b edits OFF**: pipeline byte-identical upstream khi không set env `P18B_*`.

---

## 9. Provenance / repro

- Repo: `duyen/Binocular3DGS` (sibling clone), commit: `____` (chạy `git -C ... rev-parse HEAD`).
- `triangulate.py` đã sửa 4 bug repo gốc (device-mismatch, SIMPLE_RADIAL intrinsics, images numpy↔tensor, OOM load-all-GPU) + thêm P18b diag (gated OFF). Xem decisions_log [2026-05-19] Phase 18.
- Output: `Binocular3DGS/output/LLFF_ablation/<scene>_3views/` (point_cloud + results.json + speed.json + renders).
- Log + timings: `CoR-GS/logs/ablation/binocular/` (`binocular_llff_r8_<mode>_<ts>.log` + `timings_llff_r8.csv`).
- Dense init cache: `Binocular3DGS/keypoints_to_3d/LLFF/<scene>_keypoints_to_3d.ply`.

---

## 10. Checklist trước khi báo "done"

- [ ] Build rasterizer + simple-knn OK (`import` pass trong env `binocular3dgs`)
- [ ] Sửa `data_base_path` + `resolution` trong `run_llff.py`
- [ ] Smoke fern: .ply sinh ra + PSNR hợp lý
- [ ] Full 8-scene xong, `read_eval_result.py` ra bảng
- [ ] Điền Mục 7 (per-scene + avg + N_gauss + time)
- [ ] Ghi commit hash + config vào Mục 9
- [ ] Cập nhật dòng Binocular3DGS trong `00_index.md` (status ✅ + PSNR reproduced)
