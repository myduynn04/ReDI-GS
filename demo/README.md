# ReDI-GS — Gradio Demo

Ứng dụng Gradio 2 tab dùng cho phần demo bảo vệ luận văn.

Design chi tiết: `../docs/gradio/00_design.md`
Log implement: `../docs/gradio/01_implementation_log.md`

---

## 1. Chạy trên server (chế độ mặc định)

```bash
ssh aidev@<server>
cd ~/workspace/representation-3d/duyen/CoR-GS

# Tạo env riêng cho demo (chạy 1 lần, clone từ corgs để giữ CUDA + 3DGS rasterizer)
conda create -n gradio_demo --clone corgs
conda activate gradio_demo
pip install -r demo/requirements.txt

# Các lần sau chỉ cần activate + launch
conda activate gradio_demo
bash demo/run_server.sh
# Hoặc trực tiếp:
python demo/app.py --port 7860
```

Không cài gradio vào env `corgs` gốc để tránh conflict pydantic/fastapi
có thể ảnh hưởng production training.

Access qua SSH port forwarding từ laptop
```bash
ssh -L 7860:localhost:7860 aidev@<server>
```
Mở browser → `http://localhost:7860`.

---

## 2. Chạy local backup (Windows, không GPU)

Chỉ dùng khi server sập vào ngày demo.

```powershell
# Trên laptop (đã sync cache từ server)
cd D:\Dowload\Paper\3D representation\code\CRSGaussian
pip install gradio pillow imageio

# Chạy chế độ local (đọc từ cache/)
python demo\app.py --local-mode --port 7860
```

Cache đầy đủ khoảng 300 MB, sync từ server bằng
```bash
scp -r aidev@<server>:~/workspace/representation-3d/duyen/CoR-GS/demo/cache ./demo/
```

---

## 3. Cấu trúc thư mục

```
demo/
├── app.py                 Gradio entry point
├── render_utils.py        Load .ply + render 1 view (Phase 1)
├── trajectory_utils.py    Camera orbit interpolation (Phase 1)
├── metrics_utils.py       PSNR/SSIM/LPIPS wrappers (Phase 1)
├── ui_precompute.py       Tab 1 (Phase 2 — TODO)
├── ui_live_training.py    Tab 2 (Phase 3 — TODO)
├── populate_cache.py      Helper sinh cache pre-computed (Phase 2 — TODO)
├── requirements.txt       Gradio + imageio
├── run_server.sh          Launch script
├── README.md              File này
└── cache/                 Nội dung pre-generated (git-ignored)
```

---

## 4. Smoke test cho Phase 1

Kiểm tra render pipeline hoạt động trước khi build UI đầy đủ.

### 4.1 Render 1 test view của fern

```bash
cd ~/workspace/representation-3d/duyen/CoR-GS
conda activate corgs

python -m demo.render_utils \
    --source data/nerf_llff_data/fern \
    --ply output/p28_crs_boost/tau65/A3_seed42_fern/point_cloud/iteration_10000/point_cloud.ply \
    --test-idx 0 \
    --out /tmp/fern_test0.png
```

Kết quả kỳ vọng
- `/tmp/fern_test0.png` — render test view idx 0
- `/tmp/fern_test0_gt.png` — ground truth
- `/tmp/fern_test0_diff.png` — difference map

### 4.2 Test orbit trajectory

```bash
python -m demo.trajectory_utils --source data/nerf_llff_data/fern --n-frames 60
```

Kỳ vọng
- Print `[demo] 20 cameras total (data/nerf_llff_data/fern)`
- Print `[demo] built trajectory with 60 MiniCam frames`

### 4.3 Test metrics

```bash
python -m demo.metrics_utils
```

Kỳ vọng
- Print PSNR ≈ inf cho 2 ảnh giống hệt
- Print PSNR khoảng 25-35 cho 2 ảnh khác nhau nhẹ

### 4.4 Launch app scaffold

```bash
bash demo/run_server.sh
```

Kỳ vọng
- Gradio launch trên port 7860
- 2 tab hiện ra (chưa có content thật, chỉ placeholder Phase 1)

---

## 5. Roadmap

- [x] Phase 1 — Foundation (render + trajectory + metrics + scaffold)
- [ ] Phase 2 — Pre-computed tab (populate_cache + full Tab 1 UI)
- [ ] Phase 3 — Live training tab (subprocess launcher + poll)
- [ ] Phase 4 — Local backup (video recording + local mode polish)
