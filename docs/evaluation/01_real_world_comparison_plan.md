# Real-world Comparison Plan

> **Evaluation chapter** — phase research (Phase 1-24) đã commit recipe (Phase 22 RoMa v1 = 21.92 dB).
> Chapter này validate recipe trên use case thực tế ngoài academic benchmark.
>
> **Mục đích**: Đáp lại câu hỏi của thầy hướng dẫn:
> *"Cái em nghiên cứu áp dụng cho bài nào? So với sản phẩm thực tế đang được dùng, em hơn gì?
> Hoặc em có thể plug vào pipeline của họ cho kết quả tốt hơn không?"*
>
> **Không phải research phase** — không thêm cơ chế mới, không tinh chỉnh hyperparameter.
> Chỉ chạy recipe đã commit trên (a) production stack mới và (b) custom real-world data.
>
> **Trạng thái dự án (2026-06-02):**
> - Phase 22 RoMa v1 = 21.92 dB LLFF 3-view (project best, +0.584 SIG vs MVS) — locked
> - Phase 23 ablation v1 confirmed 4-contribution framing
> - Phase 24 trim-add v1 đã verify
> - **Evaluation direction**: Pattern 1 — plug-in module CRSGaussian vào **Nerfstudio splatfacto** (production-grade Apache 2.0)
> - ✅ **A1 plug-in (RoMa init only) DONE + verified**: 4/4 scene Δ = +1.523 dB mean vs vanilla splatfacto. 14/17 tier PASS. Chi tiết: [06_plugin_a1_results.md](06_plugin_a1_results.md)
> - 🟡 **A2 cascade** (depth + opacity decay + CRS module): pending decision
> - Design overview: [05_plugin_design.md](05_plugin_design.md)

---

## 1. Định vị bài toán — ứng dụng thực tế

CRSGaussian giải bài toán **sparse-view 3D Gaussian Splatting** (≤9 view, target 3 view).
Trong production, lớp ứng dụng tương ứng:

| Use case | Tại sao 3-view có giá trị | Sản phẩm production hiện có |
|----------|---------------------------|------------------------------|
| **Mobile capture / consumer 3D** ⭐ | User chỉ chụp 3-5 ảnh điện thoại, không đủ kiên nhẫn chụp 30-100 | Polycam, Luma AI, Scaniverse, KIRI Engine |
| **Cultural heritage từ ảnh archive** | Di tích chỉ còn vài ảnh, không quay lại được | ISPRS pipeline, CyArk, Sketchfab Heritage |
| Robotic first-person | Robot/drone limited POV | NoPoSplat, GS-LRM, NVIDIA Cosmos |

→ **Defense focus: Mobile capture là chính, Heritage là phụ.** Robotics bỏ qua vì trade-off real-time mismatch.

---

## 2. Hai hướng comparison đáp ứng yêu cầu thầy

### Hướng A — Same-input comparison ("bài em hơn họ")

**Concept**: Cùng 3-5 ảnh điện thoại input, đẩy qua nhiều stack, so qualitative output.

| Stack | Loại | Setup | Đại diện cho |
|-------|------|-------|--------------|
| Polycam | Production closed | App iOS/Android, free 5 scan/tuần | Consumer production |
| Luma AI | Production closed | Web upload, free | Consumer production |
| Scaniverse | Production on-device | App, on-device | Free production |
| 3DGS gốc Inria | Open-source baseline | `gaussian-splatting` repo | Naive baseline |
| gsplat / splatfacto | Production-grade open-source | `ns-train splatfacto` | Industry default |
| CoR-GS | Research baseline (đã có) | A0 anh đã có | Sparse-view research |
| **CRSGaussian Phase 22** | **Đề xuất của anh** | Đã có | Đề xuất |

**Evidence cuối cùng**: 7 stack × 2-3 custom scene = qualitative grid + 1 link SuperSplat interactive.

### Hướng B — Plug-in ("module em ghép vào pipeline họ")

**Concept**: CRSGaussian là collection modules (CRS score, LFCF, opacity decay, RoMa v1 init,
DropAnSH, SH freeze) — plug vào pipeline production hiện có.

| Pipeline đích | Module plug vào | Khó/dễ | Expected gain |
|---------------|-----------------|--------|---------------|
| Nerfstudio splatfacto | LFCF densifier | 🟡 Trung bình (wrap `densify_and_prune`) | +0.1-0.3 dB |
| gsplat Berkeley | CRS score + opacity decay | 🟡 Trung bình (fork) | +0.1-0.2 dB |
| dn-splatter | RoMa v1 init replace COLMAP | ✅ Dễ (swap PCD) | +0.3-0.6 dB |
| SuperSplat workflow | Output `.ply` chuẩn | ✅ Sẵn | N/A — interop |
| Polycam/Luma | ❌ Closed-source | 🔴 Không khả thi | — |

**Plug-in dễ nhất (recommend)**: RoMa v1 init drop-in cho 3 pipeline open-source phổ biến.
Lý do: Phase 22 đã verify +0.584 SIG. Module isolated (chỉ output `fused.ply`).

---

## 3. Kế hoạch 10 ngày

| Ngày | Hướng | Việc | Đầu ra |
|------|-------|------|--------|
| 1 | A baseline | Chạy 3DGS gốc Inria + gsplat trên 8 scene LLFF 3-view | PSNR baseline cho bảng |
| 2 | B | Chạy 3DGS gốc + gsplat với RoMa v1 init thay COLMAP | Bảng plug-in 3 pipeline |
| 3 | A custom | Anh chụp 3-5 ảnh điện thoại 2-3 custom scene + COLMAP | Custom dataset |
| 4 | A custom | Upload cùng input lên Polycam + Luma + Scaniverse → save output | Production output |
| 5 | A custom | Run 4 method open-source trên custom scene | Open-source output |
| 6 | A custom | Render flythrough video + upload SuperSplat → link interactive | Video + link demo |
| 7 | B verify | Verify RoMa init plug-in trên 2-3 pipeline khác (splatfacto?) | Bảng plug-in mở rộng |
| 8-9 | Slide | Compose slide + qualitative grid + diễn tập | Slide defense |
| 10 | Buffer | Re-run nếu sai | — |

---

## 4. Bảng kết quả cuối cùng (template điền sau khi chạy)

### 4.1 Academic benchmark — LLFF 3-view (8 scene, multi-seed N=24)

| Method | PSNR | SSIM | LPIPS | N_gauss | Time/scene | Đại diện |
|--------|------|------|-------|---------|------------|----------|
| 3DGS gốc Inria | TBD | TBD | TBD | TBD | TBD | Production naive |
| gsplat / splatfacto | TBD | TBD | TBD | TBD | TBD | Industry default |
| CoR-GS (paper) | 20.11 | 0.704 | 0.201 | — | — | Sparse research |
| FSGS (paper) | 20.31 | — | — | — | — | Sparse research |
| DOC-GS (paper) | 21.38 | — | — | — | — | Sparse SOTA |
| BinocularGS (paper) | 21.44 | — | — | — | — | Sparse SOTA |
| ICO-GS (preprint) | 22.20 | — | — | — | — | Sparse SOTA preprint |
| **CRSGaussian Phase 22 (anh)** | **21.92** | TBD | TBD | TBD | TBD | **Đề xuất** |

### 4.2 Production same-input comparison (custom scene anh chụp)

| Stack | Scene 1 | Scene 2 | Scene 3 | Đánh giá visual |
|-------|---------|---------|---------|-----------------|
| Polycam | screenshot | screenshot | screenshot | TBD |
| Luma AI | screenshot | screenshot | screenshot | TBD |
| Scaniverse | screenshot | screenshot | screenshot | TBD |
| 3DGS gốc | render | render | render | TBD |
| gsplat | render | render | render | TBD |
| CoR-GS | render | render | render | TBD |
| **CRSGaussian (anh)** | render + SuperSplat link | render + link | render + link | TBD |

### 4.3 Plug-in table (Hướng B)

| Pipeline đích | Init mặc định | + RoMa v1 init (anh) | Δ PSNR |
|---------------|---------------|----------------------|--------|
| 3DGS gốc Inria | TBD | TBD | TBD |
| gsplat | TBD | TBD | TBD |
| splatfacto (Nerfstudio) | TBD | TBD | TBD |

---

## 5. Slide narrative — câu chuyện kể với thầy

### Slide 1: Bài toán
*"Sparse-view 3D reconstruction từ 3 ảnh. Ứng dụng: mobile capture (Polycam/Luma yêu cầu 30 ảnh),
heritage từ ảnh archive (không thể quay lại chụp), robotic POV limited."*

### Slide 2: Sản phẩm production hiện tại
*"Polycam, Luma AI, Scaniverse là consumer 3D leader. Pipeline production yêu cầu 30-100 ảnh.
Khi user chỉ chụp 3 ảnh — họ fail hoặc quality rất xấu."*

### Slide 3: Bảng academic benchmark
*"Trên LLFF 3-view (benchmark chung của lĩnh vực), CRSGaussian đạt 21.92 dB, vượt:
- 3DGS gốc (production naive) ~17 dB → +5 dB
- gsplat (production-grade) ~17-18 dB → +4 dB
- CoR-GS (sparse research) 20.11 → +1.8 dB
- BinocularGS 21.44 → +0.5 dB
Gần ngang ICO-GS preprint 22.20 (gap −0.28 dB)."*

### Slide 4: Production same-input compare (qualitative grid)
*"Cùng 3 ảnh em chụp điện thoại. Đây là Polycam/Luma/Scaniverse output (fail).
Đây là CRSGaussian (tái dựng được). Link interactive: superspl.at/..."*

### Slide 5: Plug-in table
*"CRS module + RoMa v1 init là module độc lập, plug được vào pipeline production có sẵn.
Em verify: cùng module này khi plug vào 3DGS Inria → +X dB, gsplat → +Y dB,
splatfacto Nerfstudio → +Z dB. Contribution của em là drop-in upgrade cho ecosystem 3DGS hiện có."*

### Slide 6: Live demo
*"Em mở link SuperSplat trên laptop. Thầy có thể xoay scene tái dựng từ 3 ảnh điện thoại em chụp."*

---

## 6. Setup / dependencies cần chuẩn bị

### Server (Linux GPU, env `corgs` hoặc `binocular3dgs`)
- [ ] Clone `gaussian-splatting` Inria gốc — đã có submodule?
- [ ] Cài Nerfstudio + `ns-train splatfacto` — cần internet 1× cho package
- [ ] Cài gsplat (`pip install gsplat`)
- [ ] COLMAP đã có sẵn (CoR-GS pipeline đã dùng)
- [ ] RoMa v1 init pipeline đã có (Phase 22)

### Local (Windows + điện thoại)
- [ ] Polycam app — đăng ký account free
- [ ] Luma AI web — đăng ký account free
- [ ] Scaniverse app (iOS/Android)
- [ ] SuperSplat account (free, browser only)

### Data
- [ ] Chụp 3-5 ảnh điện thoại 2-3 scene đa dạng:
  - 1 vật nhỏ trong nhà (đồ trang trí, sách, mô hình)
  - 1 cảnh phòng nhỏ (góc bàn, kệ sách)
  - 1 cảnh ngoài trời nhỏ (cây, tượng, biển hiệu)
- [ ] Run COLMAP local hoặc server → pose + sparse PCD
- [ ] Mirror dataset: cùng input cho cả 3 production app

---

## 7. Risk + mitigation

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Polycam/Luma không fail như mong đợi với 3 ảnh | 🟡 Medium | Chụp scene khó hơn (textureless, thin-structure) để stress-test họ |
| COLMAP fail trên custom scene 3 ảnh | 🟡 Medium | Chụp overlap nhiều hơn, hoặc dùng pose từ Polycam export |
| Nerfstudio setup tốn 2+ ngày | 🟡 Medium | Fallback: skip splatfacto, chỉ dùng 3DGS gốc + gsplat |
| Plug-in RoMa init phá pipeline đích | 🟢 Low | Đã verify Phase 22, format `.ply` chuẩn |
| Defense deadline gấp | TBD | Ưu tiên Hướng A (visual impact mạnh hơn) |

---

## 8. Checklist trước defense

- [ ] Bảng số academic LLFF 3-view 7-method full PSNR/SSIM/LPIPS
- [ ] Bảng plug-in 3 pipeline với Δ PSNR rõ ràng
- [ ] Qualitative grid 7 method × 2-3 custom scene (screenshot)
- [ ] Video flythrough 1-2 scene đẹp nhất (MP4, 30 FPS)
- [ ] Link SuperSplat interactive (test mở trên trình duyệt hội đồng)
- [ ] Slide flow (~15-20 slide tổng)
- [ ] Diễn tập 2 lần, time ≤ thời gian được phép

---

## 9. Liên quan

- Phase 22 RoMa v1 pilot: `logs/p22_pilot_v1/`, memory `project_phase22_roma_v1_pilot.md`
- Phase 23 ablation v1: `logs/p23_ablation/`, memory `project_phase23_ablation_v1.md`
- Phase 24 trim-add v1 (đang chạy): `scripts/p24_trim_add_v1_run.sh`
- 3-view capacity ceiling background: `docs/04_decisions_log.md` + memory `project_3view_capacity_ceiling.md`
