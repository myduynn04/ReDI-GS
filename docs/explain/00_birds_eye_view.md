# Doc 00 — Bird's-eye view: CRSGaussian giải quyết bài toán gì?

> **Mục đích doc này**: Cho bạn 1 bức tranh tổng thể về dự án trước khi nhìn vào code.
> Đọc xong doc này, bạn phải trả lời được: "CRSGaussian giải bài toán gì? Đầu vào? Đầu ra?".

---

## 1. Bài toán: cho 3 ảnh, render được view mới

Hãy tưởng tượng bạn đứng trong 1 căn phòng và chụp **3 tấm ảnh** từ 3 vị trí khác nhau.
Bây giờ tôi muốn hỏi bạn: "Nếu tôi đứng ở **vị trí khác** (chưa chụp), tôi sẽ thấy gì?"

Đây gọi là bài toán **Novel View Synthesis** (NVS) — tạm dịch "tổng hợp ảnh từ góc nhìn mới":

```
INPUT:                          OUTPUT:
  📸 Ảnh 1 (từ vị trí A)          🖼️ Ảnh mới (từ vị trí D, E, F, ...)
  📸 Ảnh 2 (từ vị trí B)          (tạo ra bằng thuật toán, không chụp)
  📸 Ảnh 3 (từ vị trí C)
  + biết camera ở đâu (pose)
```

**Tại sao khó?** Vì:
- Bạn chỉ có **3 góc nhìn** → có rất nhiều thứ camera chưa "nhìn thấy"
- Cần **suy luận** ra hình dạng 3D của scene từ thông tin hạn chế đó
- Render ảnh mới phải **chính xác** ở từng pixel

---

## 2. Cách tiếp cận: dựng scene 3D rồi render lại

Idea cốt lõi:
```
3 ảnh input ──► [Thuật toán A] ──► Mô hình 3D của scene ──► [Render] ──► Ảnh mới
                                    (collection of "blobs")
```

Mô hình 3D ở đây = **tập các "blob" Gaussian 3D** — đó là 3D Gaussian Splatting (3DGS).

### Mỗi Gaussian "blob" trông như gì?

Tưởng tượng 1 **đốm sương màu** trôi trong không khí:
- Có **vị trí** (x, y, z) trong không gian 3D
- Có **kích thước + hình dạng** (rộng/hẹp theo từng trục)
- Có **màu sắc** (RGB)
- Có **độ trong suốt** (giống alpha trong Photoshop, 0 = trong suốt, 1 = đặc)

Một scene = **hàng trăm nghìn** đốm sương như vậy, "lơ lửng" trong không gian, mỗi đốm góp 1 chút màu vào ảnh cuối.

### Render = chiếu các blob lên màn hình

Khi muốn render từ góc nhìn mới:
```
1. Chiếu (project) tất cả blob 3D lên màn hình 2D (theo phép chiếu camera)
2. Mỗi blob trở thành 1 "vệt" mờ trên màn hình
3. Sort theo độ sâu (gần→xa)
4. Blend chồng lên nhau theo alpha (giống vẽ tranh nhiều lớp)
5. → ra 1 ảnh hoàn chỉnh
```

---

## 3. Vấn đề khi chỉ có 3 ảnh (sparse-view)

3DGS gốc (SIGGRAPH 2023) thiết kế cho **dense view** (100+ ảnh). Khi chỉ có **3 ảnh**:

### Vấn đề 1 — "Floater" (vật bay)
Thuật toán có thể đặt 1 đốm sương màu **lơ lửng giữa không khí** mà vẫn khớp đủ với 3 ảnh input. Nhưng khi nhìn từ góc khác → đốm này hiện ra ở chỗ vô lý → ảnh xấu.

```
Góc nhìn A: ✓ pass (đốm nằm sau cánh cửa, không thấy)
Góc nhìn B: ✓ pass (đốm nằm cùng màu với background)
Góc nhìn C: ✓ pass (cùng lý do)
─────────────
Góc nhìn D (mới): ❌ ĐỐM LƠ LỬNG giữa không trung — ảnh bị "bẩn"
```

### Vấn đề 2 — Overfitting
Thuật toán "học thuộc" 3 ảnh input → render lại 3 ảnh đó hoàn hảo, nhưng render góc nhìn mới thì tệ.

### Vấn đề 3 — Init kém
3DGS cần điểm khởi tạo Gaussian. Với 3 ảnh, COLMAP (tool tìm điểm 3D từ ảnh) chỉ ra **rất ít điểm** (vài nghìn) → Gaussian bắt đầu từ "ít vốn" → khó học scene.

---

## 4. CRSGaussian giải quyết bằng cách nào?

Project có **2 idea chính** + nhiều cải tiến phụ:

### Idea 1 — CRS (Confidence-Reliability Score)
Cho **mỗi Gaussian blob** 1 điểm số **độ tin cậy** trong khoảng [0, 1]:
- Điểm cao = "blob này đáng tin, nó đặt đúng chỗ"
- Điểm thấp = "blob này nghi ngờ, có thể là floater"

Cách tính điểm = tổng hợp **3 tín hiệu**:
1. **Depth signal**: blob đặt đúng "khoảng cách" với camera không?
2. **Reprojection signal**: nhìn từ camera khác, blob có "ăn khớp" không?
3. **Stability signal**: màu của blob đã ổn định chưa, hay còn đang "lắc lư"?

Dùng điểm này để **điều khiển training**:
- Blob điểm thấp → freeze màu lại (đừng cho update tự do)
- Blob điểm cao → cho update bình thường

### Idea 2 — Init dense bằng RoMa v1
Thay vì để COLMAP cho ra vài nghìn điểm khởi tạo (sparse), dùng **AI matcher** (RoMa v1) để tìm **hàng chục nghìn điểm** matching giữa các cặp ảnh → init Gaussian dày hơn → học tốt hơn.

### Các cải tiến phụ (kết hợp vào recipe)
| Module | Vai trò |
|---|---|
| **Depth prior** | Dùng AI (DepthAnything-V2) đoán depth cho mỗi ảnh → ép Gaussian khớp depth đó |
| **DropAnSH** | Random "drop" 2% blob mỗi iter (như dropout NN) → ngăn overfit |
| **Opacity decay** | Mỗi iter giảm độ đặc x0.999 → blob phải "earn" độ đặc |
| **LFCF + AbsGS** | Densification thông minh hơn (đẻ blob mới ở chỗ cần) |

---

## 5. Kết quả

Trên benchmark chuẩn LLFF (8 scenes, 3 ảnh input):
- **CRSGaussian: 21.92 PSNR** (chỉ số chất lượng càng cao càng tốt)
- CoR-GS (paper gốc mà ta xây dựng trên): 20.11 → ta hơn **+1.81 dB**
- FSGS (ICLR'24): 20.31 → ta hơn **+1.61 dB**
- Binocular3DGS (NeurIPS'24): 21.44 → ta hơn **+0.48 dB**

---

## 6. Roadmap học code

Sau khi hiểu bức tranh tổng thể, ta sẽ đào sâu code theo thứ tự:

| Doc | Topic |
|---|---|
| **00 (file này)** | Bird's-eye view |
| **01** | `python train.py` — chạy thế nào? |
| **02** | Imports + seed (dòng 1-76) |
| **03** | Setup phase: load scene, init Gaussian (dòng 78-185) |
| **04** | Main training loop — flow tổng thể (dòng 187-870) |
| **05+** | Đào sâu từng bước trong loop |

---

## 7. Câu hỏi để check hiểu

Trước khi sang doc 01, bạn trả lời mấy câu này (nếu chưa rõ, hỏi tôi):

| # | Câu hỏi |
|---|---|
| **Q0.1** | Đầu vào của project là gì? Đầu ra là gì? |
| **Q0.2** | "Gaussian blob" có những thuộc tính gì? |
| **Q0.3** | "Floater" là gì? Tại sao nó xuất hiện trong sparse-view? |
| **Q0.4** | CRS dùng để làm gì? (1 câu, đại ý) |
| **Q0.5** | Tại sao dùng RoMa v1 thay vì COLMAP? |

---

**Khi đã hiểu doc này → ta qua [Doc 01](01_train_py_entry_point.md): chạy `python train.py` thì điều gì xảy ra.**
