# Ablation / Baseline Reproduction — Index & Chuẩn Evaluation Chung

> **Mục đích**: Chạy lại các baseline 3DGS sparse-view **trên cùng một protocol + cùng một
> evaluator** để có bảng so sánh fair cho paper.
> **Cấu trúc**: docs `.md` ở `docs/ablation/NN_<method>.md` · scripts `.sh` ở `scripts/ablation/run_<method>_*.sh`.
>
> **Nguyên tắc vàng**: KHÔNG so số paper gốc trực tiếp nếu protocol khác (resolution/iter/data).
> Luôn ghi 2 cột: (a) **reproduced** ở protocol của ta, (b) **paper** + protocol gốc.

---

## 1. CHUẨN EVALUATION CHUNG (bắt buộc cho mọi baseline)

### 1.1 Protocol chạy (LLFF 3-view)

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| Dataset | LLFF 8 scenes | `duyen/CoR-GS/data/nerf_llff_data/` |
| Train split | `idx % 8 != 0`, rồi `np.linspace(0, n_train-1, 3).round()` | universal toàn dòng |
| Test split | `idx % 8 == 0` (hold-out) | llffhold=8 |
| **Resolution** | **`-r 8`** | chuẩn áp đảo (FSGS/DNGaussian/CoR-GS/LoopSparseGS/NexusGS). Baseline default khác (Binocular/Co-Adapt r2) PHẢI ép về -r 8 |
| n_views | 3 | |
| Iterations | **theo native của từng method** | ghi rõ; so cross-budget với ours 10k |

### 1.2 Evaluator chung — MỘT script cho tất cả

⚠️ **KHÔNG dùng `metrics.py` riêng của từng repo** — chúng KHÁC nhau (Binocular3DGS thiếu SSIM_sk
+ AVGE; LPIPS net khác nhau ở vài repo). Thay vào đó:

> **Render bằng repo baseline → đánh giá bằng `CoR-GS/metrics.py` (= codebase CRSGaussian, env `corgs`).**

Lý do dùng được cho MỌI repo: mọi repo 3DGS xuất renders ra cùng cấu trúc
`<model_path>/test/ours_<iter>/{renders,gt}/*.png`, và `CoR-GS/metrics.py` đọc đúng cấu trúc đó.
→ PSNR/SSIM/SSIM_sk/LPIPS/AVGE tính bằng **cùng một đoạn code, cùng trọng số LPIPS, cùng skimage**.

Lệnh: `cd duyen/CoR-GS && conda run -n corgs python metrics.py -s <data/scene> -m <baseline_model_path>`

### 1.3 Bộ metric report — BẮT BUỘC chạy đủ cho MỌI baseline

> ⚠️ **Mọi baseline (Binocular3DGS, FSGS, DNGaussian, ...) PHẢI report đủ 11 metric dưới đây.**
> Script `run_<method>_llff.sh` của mỗi bài đều phải sinh ra cả 2 nhóm. Thiếu metric nào → chưa "done".

**Nhóm 1 — Chất lượng** (6, từ evaluator chung `corgs/metrics.py`):
| # | Metric | Hướng | Nguồn |
|---|---|:---:|---|
| 1 | PSNR | ↑ | `utils.image_utils.psnr` |
| 2 | SSIM (GS) | ↑ | `utils.loss_utils.ssim` (3DGS) — bộ chính cả dòng dùng |
| 3 | SSIM_sk | ↑ | `skimage.structural_similarity` — DNGaussian/CoR-GS report thêm |
| 4 | LPIPS | ↓ | `lpipsPyTorch`, **net='vgg'** (universal) |
| 5 | AVGE | ↓ | average-error geom-mean `[√(1−SSIM), 10^(−PSNR/10), LPIPS]` (`utils.image_utils.avge`) |
| 6 | AVGE_sk | ↓ | AVGE dùng SSIM_sk |

**Nhóm 2 — Hiệu quả / chi phí** (5, ta tự đo — KHÔNG repo nào đo sẵn; "always measure compute cost"):
| # | Metric | Hướng | Cách đo |
|---|---|:---:|---|
| 7 | **#Gaussians** | — | `get_xyz.shape[0]` lúc render (`bench_render_speed.py`) |
| 8 | Train time (s/scene) | ↓ | wall-clock quanh `train.py` |
| 9 | FPS (infer) | ↑ | warmup 20 + timed 5×N test views, `cuda.synchronize` @ -r 8 |
| 10 | Model size (MB) | ↓ | size `point_cloud.ply` |
| 11 | Peak VRAM render (MB) | ↓ | `torch.cuda.max_memory_allocated` quanh render loop |

> (ms/frame = nghịch đảo FPS, có trong `speed.json`.)

**Optional (chỉ khi cần / khi repo hỗ trợ — KHÔNG bắt buộc):**
- Depth error — LLFF không có GT depth chuẩn → bỏ qua. Chỉ làm nếu chuyển sang DTU (có GT).
- Peak VRAM lúc **train** — cần sửa `train.py` từng repo (invasive) → để optional.

---

## 2. Khảo sát metric các repo (verified từ code, 2026-06-07)

| Repo | LLFF? | PSNR | SSIM(GS) | SSIM_sk | LPIPS | AVGE | -r LLFF | iter | N_gauss log | FPS |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CRSGaussian (ours, = CoR-GS)** | ✓ | ✓ | ✓ | ✓ | vgg | ✓ | 8 | 10k | ✓ | (ns-eval) |
| Binocular3DGS | ✓ | ✓ | ✓ | ✗ | vgg | ✗ | **2**→ép 8 | 30k | ✗ | ✗ |
| FSGS | ✓ | ✓ | ✓ | ✗ | vgg | ✗ | 8 | 10k | ✗ | ✗ |
| DNGaussian | ✓ | ✓ | ✓ | ✓ | vgg | ✗ | 8 | 6k | ✗ | ✗ |
| CoR-GS | ✓ | ✓ | ✓ | ✓ | vgg | ✓ | 8 | 10k | ✓ | ✗ |
| DepthRegularizedGS | ✓ | ✓ | ✓ | ✗ | vgg | ✗ | 1(def) | 30k | ✗ | ✗ |
| Co-Adaptation-of-3DGS | ✓ | ✓ | ✓ | ✗ | vgg | ✗ | **2** | 30k | ✓ | ✗ |
| LoopSparseGS | ✓ | ✓ | ✓ | ✗ | vgg | ✗ | 8 | multi-loop | ✗ | ✗ |
| NexusGS | ✓ | ✓ | ✓ | ✗ | vgg | ✗ | 8 | 30k | ✗ | ✗ |
| SCGaussian | ✓ | ✓(mask) | ✓ | ✗ | vgg | ✓ | full | 2k | ✗ | ✗ |
| EFA-GS | ✗ | ✓ | ✓ | ✗ | vgg | ✗ | — | 30k | — | ✗ |
| FreGS / mip-splatting | ✗ | ✓ | ✓ | ✗ | vgg | ✗ | — | 30k | — | ✗ |
| GDAGS | ✗ | ✓ | ✓ | ✗ | vgg | ✗ | — | 30k | — | ✗ |

**Kết luận khảo sát:**
- Core PSNR/SSIM(GS)/LPIPS-vgg = **đồng thuận 100%** → bộ tối thiểu.
- SSIM_sk + AVGE = subset report (CoR-GS/DNGaussian/SCGaussian) → ta report cả để khớp họ.
- **FPS/model-size: KHÔNG repo nào đo** → ta thêm như bonus efficiency, không phải để so paper.
- LLFF chỉ có ở 9/13 repo. EFA-GS/FreGS/mip-splatting/GDAGS = không LLFF (Mip360/T&T/Blender) → **không phải baseline LLFF 3-view**, chỉ tham khảo cơ chế.
- Khác biệt resolution (Binocular/Co-Adapt r2, DepthReg r1, SCGaussian full) = **lý do bắt buộc dùng evaluator chung + ép -r 8**.

---

## 3. Bảng tổng baseline (điền dần)

| # | Method | Venue | LLFF iter | PSNR paper | PSNR reproduced (-r8, unified) | Status | Doc |
|---|--------|-------|------|-----------|-------------------------------|--------|-----|
| 01 | Binocular3DGS | NeurIPS'24 | 30k | 21.44 (r2) | _pending_ | 🔧 build rasterizer | [01_binocular3dgs.md](01_binocular3dgs.md) |
| — | FSGS | ICLR'24 | 10k | 20.31* | — | ⬜ chưa | — |
| — | DNGaussian | CVPR'24 | 6k | 19.94 (MVS) | — | ⬜ chưa | — |
| — | CoR-GS | ECCV'24 | 10k | 20.11* | — | ⬜ chưa | — |
| — | DepthRegularizedGS | CVPRW'24 | 30k | — | — | ⬜ chưa | — |
| — | Co-Adaptation | — | 30k | 20.20 | — | ⬜ chưa | — |
| — | LoopSparseGS | — | multi | — | — | ⬜ chưa | — |
| — | NexusGS | CVPR'25 | 30k | — | — | ⬜ chưa | — |
| — | SCGaussian | NeurIPS'24 | 2k | — | — | ⬜ chưa | — |

`*` = số ours đã cite ở fair-compare 10k bucket (CLAUDE.md). Legend: 🔧 setup · ▶️ running · ✅ done · ❌ blocked · ⬜ chưa làm

---

## 4. Server context (verified 2026-06-07)

- Root: `/home/aidev/workspace/representation-3d/duyen/`
- Data chung: `duyen/CoR-GS/data/nerf_llff_data/` (`images/` full + `images_4` + `images_8`)
- Envs: `corgs` (CRSGaussian + **evaluator chung**), `binocular3dgs` (PDCNet+/GS), `romav2` (RoMa)
- Mỗi baseline: render ở env riêng → **eval đồng nhất qua `corgs`**.
- Quy ước local↔server: xem `docs/00b_execution_session.md`.

---

## 5. Quy trình chuẩn cho 1 baseline mới (template)

1. Verify state: repo clone? weight? submodule build? env? data path? (script chẩn đoán)
2. Tạo `NN_<method>.md` từ template `01_binocular3dgs.md`.
3. Viết `run_<method>_llff.sh`: render pass (env baseline, -r 8, 2-GPU) + bench FPS.
4. **unified_eval** qua `corgs/metrics.py` → results.json đủ bộ.
5. Aggregate → bảng per-scene + AVG (PSNR/SSIM/SSIM_sk/LPIPS/AVGE + N_gauss/FPS/MB/time).
6. Điền doc Mục kết quả + cập nhật bảng Mục 3 ở đây + note caveats (cross-budget/res).
