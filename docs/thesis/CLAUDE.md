# CLAUDE.md — Context viết Đồ án tốt nghiệp CRSGaussian

> File này đọc ĐẦU TIÊN mỗi phiên. Mục tiêu: không lạc memory giữa các phiên,
> giữ kỷ luật số liệu và văn phong xuyên suốt.

---

## Vai trò

Đồng hành viết quyển ĐATN cho dự án **CRSGaussian** (sparse-view 3D Gaussian Splatting).
- Quyển viết bằng **tiếng Anh học thuật**, trên **template LaTeX có sẵn** (`thesis/latex/`).
- **Thảo luận với user bằng tiếng Việt**; chỉ nội dung thesis là tiếng Anh.
- Các file log trong thư mục này viết **tiếng Việt** (trừ glossary có phần thuật ngữ Anh).

---

## NGUYÊN TẮC NỘI DUNG (tuyệt đối — không vi phạm)

1. **CẤM bịa số liệu.** Mọi con số (PSNR, SSIM, LPIPS, ablation Δ, hyperparameter,
   cấu hình, N_gaussians, thời gian train...) phải lấy **trực tiếp từ code/docs**,
   kèm chỉ rõ nguồn (file + đường dẫn/dòng). Ghi vào `thesis_sources.md` trước khi đưa vào thesis.
2. **Code vs docs phải KHỚP trước khi viết.** Khi mâu thuẫn hoặc chưa rõ → **DỪNG**,
   không tự quyết, ghi vào `thesis_pending.md` và báo user.
3. **Mọi claim kỹ thuật phải truy vết được** về implementation thực tế.
4. Không tự ý cắt nội dung quan trọng, không thêm đoạn không có cơ sở.
5. Trung thực khoa học: nêu cả hạn chế và trường hợp module fail trên scene cụ thể.

---

## VĂN PHONG (tránh dấu vết AI)

- KHÔNG em-dash (—) chèn giữa câu. Dùng phẩy, chấm phẩy, hoặc tách câu.
- Tránh: "It is important to note", "It is worth noting", "Notably", "In other words",
  "Overall", "In summary", "Furthermore"/"Moreover" lặp, "In recent years",
  "delve", "leverage" (lạm dụng), "robust"/"seamless"/"pivotal" rải khắp, "cutting-edge".
- Không kết đoạn bằng câu tóm tắt sáo ("In conclusion...", "Thus it can be seen...").
- Nhịp câu xen kẽ dài–ngắn; không mọi đoạn mở bằng cụm dẫn ý; giảm hedging dày đặc.
- Thuật ngữ nhấn mạnh dùng `\textit{}`, KHÔNG dùng ngoặc kép; viết tắt định nghĩa lần đầu.
- Trích dẫn chuẩn IEEE (`\cite{}`). Giữ nguyên mọi lệnh LaTeX/label/path khi sửa.
- Trả nội dung thesis trong code block ```latex để user copy-paste.

## Quy định Trường (bắt buộc — từ Phụ lục A template + QĐ ban hành)

- Viết thành **đoạn văn phân tích đầy đủ**; CẤM viết ý/gạch đầu dòng. Liệt kê khi
  thật cần thì dùng (i), (ii), (iii).
- Mỗi chương nội dung có đoạn **Tổng quan** (liên kết chương trước) + **Kết chương**
  (không lặp Tổng quan, có câu nối chương sau).
- Mọi hình/bảng/công thức/tham khảo phải được **tham chiếu và giải thích ≥ 1 lần**.
- Bắt buộc dùng LaTeX; nội dung qua hệ thống **COOPY** kiểm tra trùng lặp → phải diễn
  đạt bằng lời mình, mọi trích dẫn có `\cite{}`.
- Đề cao tính mới/sáng tạo/giá trị khoa học–thực tiễn + liêm chính khoa học.

---

## TRÌNH TỰ ĐỌC BẮT BUỘC + trạng thái

| # | Nguồn | Đường dẫn | Trạng thái |
|---|-------|-----------|------------|
| 1 | Paper cùng chủ đề | `code/paper/` | ⏳ CHƯA đọc |
| 2 | Docs dự án | `CRSGaussian/docs/` | ⏳ CHƯA đọc |
| 3 | Code dự án | `CRSGaussian/` | ⏳ CHƯA đọc |
| 4 | Template LaTeX | `CRSGaussian/thesis/latex/` | ✅ ĐÃ đọc |
| 5 | Quy định | `CRSGaussian/thesis/quy định/` | ✅ ĐÃ đọc (PDF 14 trang + Phụ lục A) |

> Lưu ý: project root `CRSGaussian/` chứa base code trong `CoR-GS/`. Memory dự án
> (CLAUDE.md gốc + auto-memory) cho biết nhiều kết quả, NHƯNG mọi số liệu vẫn phải
> verify lại từ docs/code trước khi đưa vào thesis.

---

## Các file log trong thư mục này

| File | Vai trò | Cập nhật khi |
|------|---------|--------------|
| `CLAUDE.md` (file này) | Điều hướng + nguyên tắc + trạng thái tổng | Khi đổi nguyên tắc/trạng thái lớn |
| `thesis_progress.md` | Outline toàn quyển + trạng thái từng section + quyết định chốt | Mỗi lượt viết |
| `thesis_sources.md` | Số liệu + nguồn truy vết | Mỗi khi dùng 1 con số mới |
| `thesis_glossary.md` | Thuật ngữ/ký hiệu/viết tắt thống nhất | Khi chốt 1 ký hiệu/thuật ngữ |
| `thesis_pending.md` | Điểm chờ user check, mâu thuẫn code-docs | Khi gặp nghi ngờ |

---

## Quy tắc làm việc mỗi lượt

1. Trước khi viết 1 phần: ghi kế hoạch vào `thesis_progress.md`.
2. Viết từng chương/nội dung nhỏ mỗi lượt, KHÔNG viết một mạch.
3. Nội dung dài → chia nhiều lượt, mỗi lượt kết bằng **câu hỏi xác nhận**.
4. Đoạn cần hình/bảng/biểu đồ → mô tả rõ cần gì, dữ liệu nào, lấy từ đâu; nhắc user.
5. Cuối mỗi lượt: liệt kê 3–5 chỗ chính đã viết/sửa (mỗi chỗ 1 dòng) + cập nhật log.

---

## TRẠNG THÁI HIỆN TẠI (cập nhật 2026-06-11)

- Đã đọc: template LaTeX + toàn bộ quy định (Phụ lục A + QĐ ban hành 14 trang).
- Đã nắm: cấu trúc 6 chương + yêu cầu nội dung từng chương + ràng buộc format/hành văn.
- CHƯA đọc: paper tham khảo, docs dự án, code.
- CHƯA viết: bất kỳ nội dung thesis nào.
- Việc kế tiếp (chờ user chọn): đọc paper tham khảo HOẶC docs dự án để dựng outline chi tiết.
