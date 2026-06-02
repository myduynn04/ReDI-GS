# Brush — Feasibility Assessment

> Đánh giá tính khả thi của 2 hướng dùng [Brush](https://github.com/ArthurBrussee/brush) (Arthur Brussee, Rust + WebGPU)
> làm so sánh open-source cho defense:
> - **Hướng A — Plug-in**: Brush làm engine, CRSGaussian module phụ trợ (vd init)
> - **Hướng B — Standalone comparison**: Brush vs CRSGaussian, cùng input, cùng eval
>
> Code đọc tại `d:\Dowload\Paper\3D representation\code\brush\` — không build, không run.

---

## 1. Brush architecture summary

### Tech stack (key facts ảnh hưởng plug-in)
- **Ngôn ngữ**: Rust 1.88+
- **ML framework**: [Burn](https://github.com/tracel-ai/burn) (Rust-native ML, KHÔNG phải PyTorch)
- **GPU**: WebGPU compatible — chạy NVIDIA/AMD/Intel/Apple/mobile, **không cần CUDA**
- **Build**: `cargo run --release` từ workspace root → binary đơn không phụ thuộc

### Data flow
```
Input COLMAP folder
  ├── sparse/0/cameras.bin    → camera intrinsics (Pinhole/Radial/OpenCV/Fisheye)
  ├── sparse/0/images.bin     → camera poses + image names
  └── sparse/0/points3D.bin   → init point cloud (POSITIONS + COLORS only)
                              → Brush dùng làm SplatData init
        ↓
Brush training (Burn ML, default 30k iter, refine_every=200)
        ↓
Output .ply (chuẩn 3DGS) / .compressed.ply
```

Source: [crates/brush-dataset/src/formats/colmap.rs:250-311](../../brush/crates/brush-dataset/src/formats/colmap.rs)

### Init fallback
Nếu `points3D.bin` không tồn tại → [splat_init.rs:54-128](../../brush/crates/brush-train/src/splat_init.rs):
- `create_random_splats()` sample 10000 điểm random inside camera frustums
- Log-uniform depth, random color, random rotation

### Training config (CLI flags)
File [crates/brush-train/src/config.rs](../../brush/crates/brush-train/src/config.rs):
- `--total-train-iters 30000` (default)
- `--refine-every 200` (densify interval)
- `--growth-grad-threshold 0.0025`
- `--growth-stop-iter 15000`
- `--opac-decay 0.004`
- `--ssim-weight 0.2`
- `--lr-mean 2e-5`
- `--max-splats 10M`

Standard 3DGS pipeline, KHÔNG có sparse-view specific feature (no depth prior, no CRS, no LFCF).

---

## 2. Hướng A — Plug-in: khả thi từng module CRSGaussian

### Bảng feasibility từng module

| Module CRSGaussian | Plug-in Brush được? | Cách | Effort |
|---------------------|---------------------|------|--------|
| **RoMa v1 dense init** (Phase 22 — project best +0.584) | ✅ **Khả thi cao** | Convert `fused.ply` → COLMAP `points3D.bin` format, ghi đè input của Brush | 🟢 1 ngày |
| **Opacity decay** | 🟡 Một phần | Brush đã có `--opac-decay 0.004` mặc định, anh tune match Phase 22 (0.999/iter) | 🟢 0.5 ngày |
| **DAV2 depth loss** | ❌ Không khả thi | Loss function trong Rust + Burn, phải port toàn bộ pipeline | 🔴 ≥1 tuần |
| **CRS score per-Gaussian** | ❌ Không khả thi | Cần modify Rust struct Splats + training loop | 🔴 ≥1 tuần |
| **LFCF densifier** | ❌ Không khả thi | Densification trong `train.rs` Rust, phải fork | 🔴 ≥1 tuần |
| **AbsGS densify** | ❌ Không khả thi | Tương tự LFCF, modify gradient flow trong Rust | 🔴 ≥1 tuần |
| **DropAnSH** | ❌ Không khả thi | SH update trong Rust, không thấy hook | 🔴 ≥1 tuần |
| **SH freeze CRS-modulated** | ❌ Không khả thi | Cần modify SH gradient logic Rust | 🔴 ≥1 tuần |
| **D_cycle** | ❌ Không khả thi | Cần render N camera depth từ Burn, không có hook | 🔴 ≥1 tuần |

### Kết luận Hướng A
**Chỉ 1 module (RoMa v1 init) plug được vào Brush** qua đường file `points3D.bin`. Plus opacity decay tune nhẹ.

**KHÔNG plug được** 7+ module khác — vì khác stack ngôn ngữ (Rust vs Python), khác ML framework (Burn vs PyTorch).

### Story defense Hướng A (nếu chọn)
> *"Em demo RoMa v1 dense init (Phase 22 contribution chính) là drop-in upgrade cho pipeline open-source production-grade Brush. Cùng LLFF 3-view, Brush + COLMAP init (sparse) vs Brush + RoMa init (em): Δ PSNR = X dB."*

→ Story này **chỉ chứng minh được 1/8 module** của CRSGaussian. Hội đồng có thể hỏi: *"Còn 7 module khác sao không plug?"* — anh phải trả lời "Brush dùng Rust, port code mất quá lâu".

---

## 3. Hướng B — Standalone comparison setup

### Setup chính xác để fair
- **Cùng input**: COLMAP folder LLFF 3-view (anh đã có ở `data/nerf_llff_data/<scene>/3_views/`)
- **Cùng iteration budget**: 10k (CRSGaussian dùng 10k, Brush default 30k → giảm Brush xuống 10k qua `--total-train-iters 10000`)
- **Cùng eval protocol**: render test views, compute PSNR/SSIM/LPIPS theo CoR-GS pipeline
- **Cùng eval split**: Brush có `--eval-split-every` flag, set giống CRSGaussian (mỗi 8 view 1 test)

### Build Brush trên server (~30 phút)
```bash
# Trên Linux server
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
rustup default 1.88.0  # hoặc latest stable
cd ~/workspace/representation-3d/duyen/
git clone https://github.com/ArthurBrussee/brush.git
cd brush
cargo build --release  # ~15-20 phút compile
./target/release/brush --help
```

### Run Brush trên LLFF 3-view
```bash
# Per-scene, COLMAP format đã có sẵn ở 3_views/sparse/0/
./target/release/brush \
    <path>/data/nerf_llff_data/fern/3_views/ \
    --total-train-iters 10000 \
    --eval-split-every 8 \
    --max-resolution 1008  # match -r 8 LLFF protocol
```

### Eval fair với CRSGaussian
Brush output `.ply` chuẩn → load qua CRSGaussian `render.py` để tính PSNR với cùng metric implementation
(tránh sai lệch do khác lpips library, khác mask logic).

### Kết luận Hướng B
✅ **Rất khả thi**. ~2-3 ngày setup + run 8 scene × 3 seed = 24 run. Brush nhanh hơn Inria gốc claim
(README "generally faster than gsplat") → run thời gian chấp nhận được.

### Story defense Hướng B (nếu chọn)
> *"Brush là open-source 3DGS app phổ biến nhất hiện nay (Rust, WebGPU, cross-platform, có cả app Android + browser). Em compare CRSGaussian với Brush trên LLFF 3-view, cùng COLMAP input, cùng 10k iter, cùng eval. Brush PSNR X dB, CRSGaussian 21.92 dB. Δ = Y dB. Reason: Brush implement vanilla 3DGS, không sparse-view aware; CRSGaussian có 8 module sparse-specific."*

→ Story này clean, fair, reproducible. **Defense-friendly nhất** trong 2 hướng.

---

## 4. Cost / effort

| Hướng | Setup | Run | Total |
|-------|-------|-----|-------|
| **A — Plug-in (RoMa init only)** | Build Brush + convert PLY→bin script | 8 scene × 2 config × 3 seed = 48 run | ~3-4 ngày |
| **B — Standalone compare** | Build Brush | 8 scene × 1 config × 3 seed = 24 run | ~2-3 ngày |
| **A + B combined** | — | 72 run | ~5-6 ngày |

GPU yêu cầu: Brush dùng WebGPU → server NVIDIA OK. Có thể chạy song song GPU0 + GPU1 (theo memory `feedback_use_both_gpus`).

---

## 5. Recommend

### Đi Hướng B trước (Standalone comparison)
Lý do:
1. **Defense impact mạnh hơn**: chứng minh full CRSGaussian recipe vượt full Brush, không phải chỉ 1 module.
2. **Effort thấp hơn**: 2-3 ngày vs 3-4 ngày
3. **Risk thấp hơn**: không cần convert format, không cần test compatibility
4. **Reproducible 100%**: cả 2 đều open-source, hội đồng verify lại được
5. **Câu chuyện "cùng open-source app, em hơn"** rõ ràng hơn câu chuyện plug-in

### Hướng A chỉ thêm khi có thời gian
Sau khi xong Hướng B (đã có bảng compare), Hướng A là **bonus story phụ** chứng minh module RoMa init alone đã đủ tốt để plug vào pipeline thứ ba. Nhưng KHÔNG nên dùng làm chính vì:
- Chỉ chứng minh 1/8 contribution
- Hội đồng có thể bắt bẻ "tại sao chỉ plug 1 module"

### Plan defense kết hợp với [01_real_world_comparison_plan.md](01_real_world_comparison_plan.md)
- **Production closed app** (Polycam, Luma, Scaniverse): same-input qualitative compare từ ảnh điện thoại anh chụp
- **Production open-source app** (Brush): standalone PSNR compare trên LLFF (Hướng B)
- **Production open-source pipeline** (OpenSplat, Nerfstudio): plug-in RoMa init (Hướng A áp dụng)
- **Research baseline** (3DGS Inria, CoR-GS, FSGS): academic bảng số

→ 4 tier compare, mỗi tier 1 evidence loại khác nhau.

---

## 6. Khoảng cần làm tiếp

- [ ] Build Brush trên server (~30 phút)
- [ ] Verify Brush chạy được LLFF fern 3-view (1 scene smoke test, ~10 phút)
- [ ] Eval LLFF fern Brush vs CRSGaussian Phase 22 single-scene để check format compat
- [ ] (Nếu PASS) full 8 scene × 3 seed Brush vs CRSGaussian
- [ ] Convert RoMa `fused.ply` → COLMAP `points3D.bin` script (cho Hướng A nếu làm)
- [ ] Document kết quả vào `docs/evaluation/03_brush_comparison_results.md`

---

## 7. Liên quan
- [docs/evaluation/01_real_world_comparison_plan.md](01_real_world_comparison_plan.md) — overall evaluation plan
- Brush README: `d:/Dowload/Paper/3D representation/code/brush/README.md`
- Brush COLMAP loader: `crates/brush-dataset/src/formats/colmap.rs`
- Brush splat init: `crates/brush-train/src/splat_init.rs`
- Brush training config: `crates/brush-train/src/config.rs`
- CRSGaussian Phase 22 RoMa v1: `scripts/p22_*` + memory `project_phase22_roma_v1_pilot.md`
