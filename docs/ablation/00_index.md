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

**Nhóm 2 — Hiệu quả / chi phí** (3 chính + 2 optional, ta tự đo):
| # | Metric | Hướng | Cách đo |
|---|---|:---:|---|
| 7 | **#Gaussians** | — | `get_xyz.shape[0]` lúc render (proxy cho model size; MB ≈ N_gauss × 236 bytes) |
| 8 | Train time (s/scene) | ↓ | wall-clock quanh `train.py` |
| 9 | FPS (infer) | ↑ | warmup 50 + timed 300 renders, `cuda.synchronize` @ -r 8 |

**Optional (chỉ khi cần — KHÔNG bắt buộc):**
| # | Metric | Hướng | Cách đo |
|---|---|:---:|---|
| 10 | Model size (MB) | ↓ | size `point_cloud.ply` — **redundant với N_gauss** (correlation ~1.0). Chỉ report nếu reviewer hỏi deployment size |
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
- **FPS/model-size/train-time: KHÔNG repo nào đo trong code** (verified: arg `fps` của FSGS/CoR-GS/NexusGS/Binocular = framerate xuất **video mp4** `VideoWriter`, KHÔNG phải infer speed). → FPS infer ta **tự đo** (`bench_render_speed.py`: warmup+timed+`cuda.synchronize`; `ms_per_frame`=1000/FPS=thời gian infer/ảnh). Đo CẢ train-time + infer-FPS = superset.
- **FPS@-r8 chỉ so NỘI BỘ** giữa các reproduction -r8 của ta — KHÔNG đặt cạnh FPS paper (khác res + GPU).
- LLFF chỉ có ở 9/13 repo. EFA-GS/FreGS/mip-splatting/GDAGS = không LLFF (Mip360/T&T/Blender) → **không phải baseline LLFF 3-view**, chỉ tham khảo cơ chế.
- Khác biệt resolution (Binocular/Co-Adapt r2, DepthReg r1, SCGaussian full) = **lý do bắt buộc dùng evaluator chung + ép -r 8**.

---

## 3. BẢNG MASTER — kết quả các mô hình (LLFF, -r8, evaluator chung)

> **3-view = REPRODUCED** trên server (unified eval `corgs/metrics.py`).
> **6-view & 9-view = LẤY TỪ PAPER** (không tự chạy — quyết định 2026-06-07).
> Legend: ✅ done · ▶️ running · 🔧 setup · ❌ blocked · ⬜ chưa làm · `P`=số paper

### 3.1 — 3-view (reproduced, đủ metric)

| Method | Venue | Iter | PSNR ↑ | SSIM ↑ | SSIM_sk ↑ | LPIPS ↓ | AVGE ↓ | N_gauss | FPS ↑ | Train s/scene ↓ | Status |
|--------|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 3DGS (vanilla) | SIGGRAPH'23 | — | — | — | — | — | — | — | — | — | ⬜ |
| FSGS | ICLR'24 | 10k | — | — | — | — | — | — | — | — | ⬜ |
| CoR-GS | ECCV'24 | 10k | — | — | — | — | — | — | — | — | ⬜ |
| NexusGS | CVPR'25 | 30k | — | — | — | — | — | — | — | — | ⬜ |
| Binocular3DGS | NeurIPS'24 | 30k | 21.356 | 0.744 | 0.742 | 0.171 | 0.092 | 106,620 | 266.6 | 1483 (~25m) | ✅ |
| **CRSGaussian (ours)** | — | 10k | **21.918** | **0.769** | **0.767** | **0.158** | **0.084** | **94,917** | 175.4 | **676 (~11m)** | ⭐ |

> **Ours**: 21.918 = Phase 22 pilot **N=24** (3-seed mean). Commit/defense = **21.89 ± 0.10** (mean 4× N=24, Phase 25_2). Chi tiết per-scene: [ours_crsgaussian.md](ours_crsgaussian.md).
> **FPS = unified protocol** (warmup50+300timed, -r8, cùng GPU → SO THẲNG được). Binocular 266.6 vs ours 175.4: Binocular render nhanh hơn nhưng **cả hai real-time** (≫30 FPS); ours chậm hơn/frame do rasterizer-confidence.
> **vs Binocular3DGS**: ours thắng MỌI metric chất lượng (PSNR **+0.562**, SSIM/SSIM_sk/LPIPS/AVGE) + **ít Gaussian** (95k<107k) + **train ~2.2× nhanh** (676 vs 1483s) ở **3× ít iter** — **infer là trục DUY NHẤT Binocular nhỉnh hơn**.

### 3.2 — 6-view (từ paper)

| Method | PSNR `P` | SSIM `P` | LPIPS `P` | Nguồn |
|--------|:---:|:---:|:---:|---|
| 3DGS (vanilla) | _ | _ | _ | paper |
| FSGS | _ | _ | _ | paper |
| CoR-GS | _ | _ | _ | paper |
| NexusGS | _ | _ | _ | paper |
| Binocular3DGS | _ | _ | _ | paper Table |
| **CRSGaussian (ours)** | _ | _ | _ | ours |

### 3.3 — 9-view (từ paper)

| Method | PSNR `P` | SSIM `P` | LPIPS `P` | Nguồn |
|--------|:---:|:---:|:---:|---|
| 3DGS (vanilla) | _ | _ | _ | paper |
| FSGS | _ | _ | _ | paper |
| CoR-GS | _ | _ | _ | paper |
| NexusGS | _ | _ | _ | paper |
| Binocular3DGS | _ | _ | _ | paper Table |
| **CRSGaussian (ours)** | _ | _ | _ | ours |

### Paper-reported 3-view (tham chiếu, KHÔNG trộn với cột reproduced)

| Method | PSNR paper | Protocol gốc | Ghi chú |
|--------|-----------|--------------|---------|
| Binocular3DGS | 21.44 | r2, 30k | ✅ reproduced 21.356 (−0.084) → [01](01_binocular3dgs.md) |
| FSGS | 20.31 | 10k bucket | ours đã cite |
| CoR-GS | 20.11 | 10k bucket | ours đã cite |
| NexusGS | _điền_ | 30k | — |
| 3DGS (vanilla) | ~thấp (floater) | — | lower-bound |

> Baselines khác đã khảo sát (DNGaussian/DepthReg/Co-Adapt/LoopSparse/SCGaussian) — xem §2, chạy sau nếu cần.

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
