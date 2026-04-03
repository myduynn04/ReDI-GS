# Prompt cho Session mới — T2.3: D_i Function

## Dùng prompt này để bắt đầu session mới với Claude Code

---

Đọc file CLAUDE.md.
Đọc docs/03_task_queue.md.
Đọc docs/04_decisions_log.md.

Sau khi đọc xong, tóm tắt:
1. Dự án làm gì (2-3 câu)
2. Trạng thái hiện tại: task nào đã done, task nào tiếp theo
3. File nào sẽ đọc/sửa trong task tiếp theo
4. Tiêu chí pass/fail của task tiếp theo

Chờ tôi confirm rồi mới làm bất cứ thứ gì.

---

## Context cho session mới

### Trạng thái dự án
- Phase 0 (Setup + Baseline): DONE
- Phase 1 (Depth Alignment): DONE
- Phase 2 (CRS Module): IN PROGRESS — T2.1, T2.2 done, T2.3 tiếp theo

### Task tiếp theo: T2.3 — Implement D_i function

D_i (Depth consistency) = tín hiệu đo Gaussian có đứng đúng độ sâu so với depth prior không.

Công thức (từ CLAUDE.md):
```python
D_i = 1 - abs(d_render_i - d_prior_i) / depth_range
# d_render_i: project Gaussian 3D position qua camera → depth (dùng world_view_transform)
# d_prior_i: aligned depth tại pixel tương ứng (từ aligned_depth_dict)
# Chỉ tính cho visible Gaussians (radii > 0 sau rasterization)
# Gaussian không visible → D_i = 0.5 (neutral)
```

### File cần đọc trước khi code
1. `utils/depth/depth_alignment.py` — hàm `_project_points_to_camera()` đã viết projection logic. Tái dụng pattern nhưng D_i cần tính cho TẤT CẢ Gaussians (hàng chục nghìn), không chỉ COLMAP points.
2. `scene/gaussian_model.py` — đã thêm `_crs_score`, `get_crs` property ở T2.2
3. `gaussian_renderer/__init__.py` — `render()` trả về `visibility_filter` (radii > 0), `depth`, `color` per-Gaussian
4. `CLAUDE.md` mục "Công thức cốt lõi" — tất cả formulas

### Quyết định thiết kế đã confirm
- D_i dùng **3D projection** (project xyz qua camera matrix), KHÔNG dùng rendered depth per-pixel
- Chỉ update **visible Gaussians** (visibility_filter = radii > 0)
- Gaussian không visible → giữ D_i = 0.5 (neutral)
- depth_range đã tính sẵn từ Phase 1 (fern: 30.56)

### File tạo mới
- `utils/depth/crs_module.py` — chứa D_i, R_i, update_crs()
- Hoặc tạo folder `crs/` riêng — hỏi user preference

### Lưu ý kỹ thuật quan trọng
- Projection dùng `cam.world_view_transform.cpu().float().T` (row-major W2C 4x4)
  KHÔNG dùng R.T @ P + T trực tiếp — vì getWorld2View2 có translate+scale adjustment
- `aligned_depth_dict` là {cam.uid: Tensor (H,W)} trên CPU — cần .cuda() khi dùng
- `depth_range` là float, đã tính = median(far) - median(near)
- `gaussians.get_xyz` trả về (N, 3) positions trên GPU

### Workflow rules (áp dụng mọi session)
- Claude Code chạy trên máy local, server tách biệt
- KHÔNG chạy bash/python trực tiếp — viết lệnh, báo user chạy trên server
- Đọc file gốc trước khi sửa
- Tóm tắt hiểu biết, chờ confirm trước khi code
- Header block + inline comments bắt buộc
- KHÔNG sửa FSGS/, DNGaussian/, LoopSparseGS/, DepthRegularizedGS/
- KHÔNG implement quá 1 task mỗi session (trừ khi task nhỏ có thể gom)
- GFS metric (C3) — HOLD

---

## Checklist sau khi T2.3 done
- [ ] File tạo đúng vị trí
- [ ] Header block có đủ: Task, File, Mục đích, Được gọi từ
- [ ] Inline comments tại: visibility filter, 3D projection, neutral default, depth_range normalization
- [ ] Function signature rõ ràng: input types, output shape
- [ ] Chỉ tính cho visible Gaussians
- [ ] User chạy test trên server: không break baseline
- [ ] Tick [x] trong docs/03_task_queue.md
