# thesis_glossary.md — Thuật ngữ / ký hiệu / viết tắt thống nhất

> Mục tiêu: nhất quán xuyên suốt quyển + đồng bộ với glossary LaTeX (`thesis/latex/glossary.tex`).
> Mọi mục dưới đây là DỰ KIẾN, phải verify tên biến/ký hiệu đúng như code trước khi chốt.

---

## 1. Viết tắt (Abbreviations) — định nghĩa ở lần xuất hiện đầu

| Viết tắt | Đầy đủ | Trạng thái |
|----------|--------|------------|
| 3DGS | 3D Gaussian Splatting | ⏳ |
| NVS | Novel View Synthesis | ⏳ |
| CRS | Confidence-Reliability Score | ⏳ verify tên chính thức trong code |
| PSNR | Peak Signal-to-Noise Ratio | ⏳ |
| SSIM | Structural Similarity Index Measure | ⏳ |
| LPIPS | Learned Perceptual Image Patch Similarity | ⏳ |
| MVS | Multi-View Stereo | ⏳ |
| SH | Spherical Harmonics | ⏳ |

## 2. Ký hiệu toán học

| Ký hiệu | Ý nghĩa | Nguồn/code | Trạng thái |
|---------|---------|-----------|------------|
| $D_i$ | Depth consistency của Gaussian i | crs_module | ⏳ |
| $R_i$ | Reprojection consistency của Gaussian i | crs_module | ⏳ |
| CRS$_i$ | sigmoid(scale·(w1·D_i + w2·R_i − 0.5)) | crs_module | ⏳ verify công thức cuối |

## 3. Tên module/biến (đúng như code — KHÔNG đổi)

| Tên trong code | Vai trò | Trạng thái |
|----------------|---------|------------|
| `_crs_score` | attribute lưu logit CRS | ⏳ |
| _(bổ sung sau khi đọc code)_ | | |

---

## Quy ước thống nhất khác
- Tên phương pháp đề xuất: **CRSGaussian** (verify cách viết hoa/cách dùng trong docs).
- Dataset chính: LLFF (verify số scene + setting 3-view).
