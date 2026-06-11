# thesis_progress.md — Outline + Trạng thái + Quyết định

> File chính cập nhật mỗi lượt. Xem `CLAUDE.md` cho nguyên tắc tổng.

---

## 1. OUTLINE KHUNG TOÀN QUYỂN

> Khung dưới đây bám **template** (6 chương + phụ lục) và sẽ được tinh chỉnh/ánh xạ
> vào nội dung CRSGaussian thật **sau khi đọc docs + code**. Phần ánh xạ CRSGaussian
> hiện chỉ là DỰ KIẾN dựa trên memory dự án, CHƯA verify → không đưa số liệu vào đây.

### Phần đầu
- **Abstract** (200–350 từ): vấn đề sparse-view 3DGS → hạn chế hiện có → hướng chọn (CRS) →
  tổng quan giải pháp → đóng góp + kết quả cuối. [trạng thái: chưa viết]
- Acknowledgment. [chưa viết]

### CHƯƠNG 1 — INTRODUCTION (3–6 trang)
- 1.1 Problem Statement — bài toán novel view synthesis từ sparse views, vì sao khó.
- 1.2 Background and Problems of Research — 3DGS, sparse-view degradation (floater,
  overdensification, overfit), hạn chế hướng hiện có.
- 1.3 Research Objectives and Conceptual Framework — mục tiêu + định hướng giải pháp.
- 1.4 Contributions — danh sách đóng góp (DỰ KIẾN map tới 4–5 contribution dự án).
- 1.5 Organization of Thesis — mô tả bố cục bằng đoạn văn.

### CHƯƠNG 2 — LITERATURE REVIEW (≤ 10 trang)
- 2.1 Scope of Research.
- 2.2 Related Work — CoR-GS, FSGS, DropGaussian/Co-Adapt, Binocular3DGS, DOC-GS,
  ICO-GS... phân tích ưu/nhược → động lực.
- 2.3 Background 1 — 3D Gaussian Splatting (biểu diễn, rasterization, densification).
- 2.4 Background 2 — depth/correspondence priors (DepthAnythingV2, RoMa, COLMAP MVS).

### CHƯƠNG 3 — METHODOLOGY (đóng góp cốt lõi)
- 3.1 Overview — sơ đồ luồng giải pháp [CẦN HÌNH pipeline].
- 3.x Các module: CRS module (D_i + R_i), CRS-guided densification/pruning, depth loss,
  dense init (RoMa v1)... (sắp xếp lại sau khi đọc code).

### CHƯƠNG 4 — THEORETICAL ANALYSIS (không bắt buộc, DỰ KIẾN GIỮ)
- Phân tích: info-conservation / 3-view capacity ceiling, compute cost, atomicAdd variance.
- (Xác nhận giữ hay bỏ sau khi đọc docs.)

### CHƯƠNG 5 — NUMERICAL RESULTS
- 5.1 Evaluation Parameters — dataset (LLFF 3-view), metric (PSNR/SSIM/LPIPS), N_gauss/FPS/MB/time.
- 5.2 Simulation Method — baselines + lý do chọn, multi-seed N=24, protocol đánh giá.
- 5.x Kết quả chính + ablation từng thành phần + so sánh SOTA.

### CHƯƠNG 6 — CONCLUSIONS
- 6.1 Summary. 6.2 Future Works.

### Phụ lục
- Appendix A/B: tùy nội dung (mẫu có sẵn — cân nhắc thêm chi tiết thí nghiệm/cấu hình).

---

## 2. TRẠNG THÁI TỪNG PHẦN

| Phần | Trạng thái | Ghi chú |
|------|-----------|---------|
| Outline tổng | 🟡 KHUNG SƠ BỘ | Chờ đọc docs/code để chốt chi tiết |
| Abstract | ⬜ chưa | Viết sau cùng |
| Ch.1 Introduction | ⬜ chưa | |
| Ch.2 Literature Review | ⬜ chưa | |
| Ch.3 Methodology | ⬜ chưa | Trọng tâm — cần đọc code kỹ |
| Ch.4 Theoretical Analysis | ⬜ chưa | Chưa chốt giữ/bỏ |
| Ch.5 Numerical Results | ⬜ chưa | Cần verify mọi số từ docs/code |
| Ch.6 Conclusions | ⬜ chưa | |

Chú thích: ⬜ chưa bắt đầu · 🟡 đang làm/nháp · ✅ xong (chờ duyệt) · ✔️ user đã duyệt

---

## 3. QUYẾT ĐỊNH ĐÃ CHỐT

| Ngày | Quyết định | Lý do |
|------|-----------|-------|
| 2026-06-11 | Dùng template LaTeX + Phụ lục A làm chuẩn format–hành văn | Quy định Trường bắt buộc |
| 2026-06-11 | Logs viết tiếng Việt, thesis viết tiếng Anh | Theo yêu cầu user |

---

## 4. NHẬT KÝ LƯỢT LÀM VIỆC

- **2026-06-11**: Đọc xong template + quy định. Tạo bộ docs `docs/thesis/`. Chưa viết thesis.
