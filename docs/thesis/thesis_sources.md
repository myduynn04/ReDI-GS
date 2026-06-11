# thesis_sources.md — Số liệu & nguồn truy vết

> NGUYÊN TẮC: không con số nào vào thesis nếu chưa có dòng tương ứng ở đây với nguồn
> (file + đường dẫn/dòng) và trạng thái VERIFY. Khi code vs docs lệch → ghi `thesis_pending.md`.

Trạng thái: ⏳ chờ verify từ code/docs · ✅ đã verify (code & docs khớp) · ⚠️ lệch (xem pending)

---

## A. Kết quả chính (PSNR/SSIM/LPIPS...)

| Đại lượng | Giá trị (claim từ memory) | Nguồn cần verify | Trạng thái |
|-----------|---------------------------|------------------|------------|
| _(điền sau khi đọc docs/code)_ | | | |

> Các con số trong memory dự án (vd PSNR ~21.89, RoMa v1 +0.58...) hiện CHƯA verify.
> Tuyệt đối không trích vào thesis trước khi đối chiếu docs (`docs/04_decisions_log.md`,
> `docs/05_results...`) và/hoặc log thí nghiệm thực tế.

## B. Cấu hình & hyperparameter

| Tham số | Giá trị | Nguồn (file:dòng) | Trạng thái |
|---------|---------|-------------------|------------|
| _(điền sau khi đọc arguments/__init__.py + train.py)_ | | | |

## C. Dataset & protocol đánh giá

| Mục | Nội dung | Nguồn | Trạng thái |
|-----|----------|-------|------------|
| _(điền sau)_ | | | |

## D. Compute / chi phí (theo feedback: luôn đo time + N_gaussians)

| Mục | Giá trị | Nguồn | Trạng thái |
|-----|---------|-------|------------|
| _(điền sau)_ | | | |
