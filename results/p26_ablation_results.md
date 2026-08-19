# Phase 26 — Kết quả ablation (A1/C1/A2 vs baseline `trim_full`)

Ghi lại kết quả các lần chạy thật trên Kaggle (2× Tesla T4), LLFF 3-view, 10.000
iteration/run, dataset 8 scene chuẩn (fern, flower, fortress, horns, leaves,
orchids, room, trex). Baseline `trim_full` = recipe production hiện tại của
ReDI-GS (khớp "Proposed ReDI-GS" trong DATN_NguyenMyDuyen_20225967.pdf).

Bug đã fix trước khi có bất kỳ kết quả nào ở dưới: `train.py:835` dùng nhầm
`opt.opacity_decay_freq_modulate` (không tồn tại ở OptimizationParams) thay vì
đúng `dataset.opacity_decay_freq_modulate` (ModelParams) — khiến mọi lần train
trước đó crash tại iteration ~500.

## 0. A1 / C1 / A2 là gì?

Cả 3 module đều là extension thêm vào CRS module (Confidence/Reliability Score)
và loss function của recipe production `trim_full`, bật/tắt độc lập qua flag
riêng trong `arguments/__init__.py`, default OFF (không đổi baseline khi tắt).

### A1 — V_stability (tín hiệu ổn định hình học)
File: `utils/crs/v_stability.py`, `utils/crs/geom_freeze.py`.

Theo dõi EMA mean + variance của 3 tham số hình học mỗi Gaussian (`_xyz`,
`_scaling`, `_rotation`) qua các iteration. Variance cao = Gaussian đang "trôi
dạt"/méo hình học bất thường (dấu hiệu nó đang dùng SH bậc cao để "diễn" 2 màu
cạnh nhau thay vì tách thành 2 Gaussian riêng). Từ đó tính điểm ổn định:

```
V = 1 − sigmoid(scale_xyz·log1p(std_xyz) + scale_shape·log1p(std_shape))
```

V thấp → hình học bất ổn → không đáng tin. V được cộng vào công thức CRS tổng
như signal thứ 4, bên cạnh D (depth consistency), R (reprojection), S (SH
stability). Có thêm tuỳ chọn "geom freeze" (`geom_freeze.py`): zero gradient
`_xyz/_scaling/_rotation` cho Gaussian có V dưới ngưỡng, ép nó ngừng di
chuyển/biến dạng tới khi ổn định lại — nhưng **config test trong file này
(`trim_full_a1`) chỉ bật phần tín hiệu V vào CRS, CHƯA bật geom freeze** (đó
là config `trim_full_a1freeze` riêng, chưa được chạy).

### C1 — Density-as-frequency modulation (điều biến theo mật độ)
File: `utils/regularizer/density_freq_modulate.py`.

DropAnSH (SH dropout) và opacity-decay hiện áp cường độ **đều tay** cho mọi
Gaussian. C1 dùng mật độ Gaussian cục bộ + kích thước (scale) làm proxy cho
"tần số không gian 3D": vùng dày đặc + Gaussian nhỏ = chi tiết mịn/biên sắc
(tần số cao) → cần giữ nguyên, giảm cường độ regularization; vùng thưa +
Gaussian to = mặt phẳng trơn (tần số thấp) → regularize mạnh như cũ:

```
freq_i = normalize(density_i) × normalize(1 / scale_i)      # ∈ [0, 1]
p_i    = base_prob × (1 − strength × freq_i)                # xác suất per-Gaussian
```

Config test (`trim_full_c1`) chỉ bật modulation cho **xác suất dropout SH của
DropAnSH**, chưa bật modulation cho opacity-decay factor (đó là
`trim_full_c1_full`, cần thêm flag `--opacity_decay_freq_modulate`, chưa
chạy).

### A2 — Temporal parameter regularization (phạt lệch tham số theo thời gian)
File: `utils/loss/temporal_reg.py`.

Toàn bộ loss khác trong recipe (L1, D-SSIM, depth Pearson, C1-normal) đều là
**loss 2D** — so sánh ảnh render với ảnh GT tại pixel. A2 là loss đầu tiên tác
động **trực tiếp lên không gian tham số 3D**: phạt độ lệch giữa
`_xyz/_scaling/_rotation` hiện tại và EMA của chính nó (kiểu "velocity
penalty"), giữ Gaussian ổn định qua iteration thay vì để nó trôi dạt tự do:

```
L_temporal = mean( w_i · [ λ_xyz·‖μ_i − ema_μ_i‖² + λ_shape·(‖Δlog_s_i‖² + ‖Δq_i‖²) ] )
```

`w_i` = điểm CRS hiện tại nếu `temporal_crs_weighted=True` (Gaussian đã đáng
tin bị phạt mạnh hơn khi di chuyển nhiều — ổn định hoá vùng đã học đúng;
Gaussian CRS thấp — đang cần tự sửa vị trí — được nới lỏng). Cộng thẳng vào
tổng loss trước `.backward()`, mỗi iteration (không phải mỗi
`crs_update_interval` như A1/C1).

## 1. So với baseline paper (DATN, trang 21/24 — 3 views)

| Nguồn | PSNR↑ | SSIM↑ | LPIPS↓ | N_gauss |
|---|---|---|---|---|
| DATN (paper, RTX A4000) | 21.920 | 0.769 | 0.158 | 95603 |
| `trim_full` (Kaggle T4, seed 42) | 21.739 | 0.767 | 0.160 | 93019 |
| Chênh lệch | −0.181 | −0.002 | +0.002 | −2584 (−2.7%) |

Reproduce sát paper (lệch nhỏ, hợp lý do khác hardware A4000/T4 + mới 1 seed).
Xác nhận pipeline không có bug ngầm làm lệch baseline.

## 2. Kết quả single-seed đầu tiên (seed 42, N=8/config)

Chạy full `kaggle_run_all.sh` mặc định (4 config × 8 scene × seed 42).

| Config | PSNR↑ | SSIM↑ | LPIPS↓ | N_gauss | Chênh lệch PSNR |
|---|---|---|---|---|---|
| `trim_full` (baseline) | 21.739 | 0.767 | 0.160 | 93019 | — |
| `trim_full_a1` (V_stability) | 21.779 | 0.768 | 0.160 | 94577 | +0.040 |
| `trim_full_c1` (density-freq mod.) | 21.835 | 0.768 | 0.159 | 93966 | +0.096 |
| `trim_full_a2` (temporal reg) | 21.843 | 0.768 | 0.159 | 90104 | +0.104 |

A1 yếu nhất, bị loại khỏi vòng multi-seed tiếp theo (không đủ tín hiệu để đáng
chạy thêm 2 seed × 8 scene × 10k iter).

## 3. Multi-seed cho `trim_full` / `trim_full_c1` / `trim_full_a2` (N=24/config)

Chạy thêm seed 137 (GPU0) + seed 9999 (GPU1) song song, session Kaggle riêng
(session chạy seed 42 ban đầu đã bị mất do ngắt kết nối trước khi tải kết quả
— bài học: luôn tải kết quả ngay sau Phase 5, hoặc dùng Save & Run All/Commit).

### 3a. PSNR theo từng seed

| Config | PSNR seed 42 | PSNR seed 137 | PSNR seed 9999 | PSNR trung bình (3 seed) | Độ lệch chuẩn (3 seed) |
|---|---|---|---|---|---|
| `trim_full` (baseline) | 21.739 | 21.865 | 21.854 | 21.819 | ±0.070 |
| `trim_full_c1` | 21.835 | 21.867 | 21.885 | 21.862 | ±0.025 |
| `trim_full_a2` | 21.843 | 21.882 | 21.902 | 21.876 | ±0.030 |

### 3b. Chênh lệch PSNR so với baseline, theo từng seed

| Config | Chênh lệch seed 42 | Chênh lệch seed 137 | Chênh lệch seed 9999 | Chênh lệch trung bình (3 seed) |
|---|---|---|---|---|
| `trim_full_c1` | +0.096 | +0.002 | +0.031 | +0.043 |
| `trim_full_a2` | +0.104 | +0.017 | +0.048 | +0.057 |

### 3c. SSIM theo từng seed

| Config | SSIM seed 42 | SSIM seed 137 | SSIM seed 9999 | SSIM trung bình (3 seed) |
|---|---|---|---|---|
| `trim_full` (baseline) | 0.767 | 0.769 | 0.768 | 0.768 |
| `trim_full_c1` | 0.768 | 0.768 | 0.769 | 0.768 |
| `trim_full_a2` | 0.768 | 0.769 | 0.769 | 0.769 |

### 3d. LPIPS theo từng seed

| Config | LPIPS seed 42 | LPIPS seed 137 | LPIPS seed 9999 | LPIPS trung bình (3 seed) |
|---|---|---|---|---|
| `trim_full` (baseline) | 0.160 | 0.159 | 0.159 | 0.159 |
| `trim_full_c1` | 0.159 | 0.159 | 0.158 | 0.159 |
| `trim_full_a2` | 0.159 | 0.158 | 0.158 | 0.158 |

### 3e. N_gauss theo từng seed

| Config | N_gauss seed 42 | N_gauss seed 137 | N_gauss seed 9999 | N_gauss trung bình (3 seed) |
|---|---|---|---|---|
| `trim_full` (baseline) | 93019 | 93392 | 96256 | 94222 |
| `trim_full_c1` | 93966 | 94767 | 93977 | 94237 |
| `trim_full_a2` | 90104 | 93440 | 93171 | 92238 |

## 4. Nhận định

- **Cả C1 và A2 đều dương ở cả 3/3 seed độc lập** — không seed nào đảo dấu
  (baseline không thắng ngược lại lần nào). Tín hiệu nhiều khả năng là thật,
  không phải thuần nhiễu ngẫu nhiên.
- Nhưng biên độ dao động mạnh theo seed, đặc biệt **C1 ở seed 137 chỉ +0.002dB**
  — về bản chất bằng 0, nằm trong nhiễu CUDA rasterizer (`atomicAdd`
  non-determinism, ghi chú tác giả ở `train.py:1095`: ±1.3dB/scene ngay cả
  giữ nguyên 1 seed). Không nên chỉ nêu số trung bình mà không kèm biên độ dao
  động khi viết vào báo cáo.
- **A2 (temporal parameter regularization) là module đáng tin cậy nhất**:
  dương ở cả 3 seed với biên độ hẹp hơn C1 (+0.017 đến +0.104, chưa bao giờ
  gần 0), PSNR trung bình cao nhất (21.876), đồng thời **giảm ~2% số Gaussian**
  trung bình so baseline (94222 → 92238) — đạt chất lượng tương đương/tốt hơn
  với model gọn hơn, không phải overfit bằng cách sinh thêm Gaussian.
- A1 (V_stability) yếu nhất trong 3 module ở vòng single-seed đầu (+0.040dB),
  chưa được test multi-seed — nếu cần kết luận về A1 cho báo cáo, nên chạy bổ
  sung seed 137/9999 tương tự trước khi đưa vào so sánh cuối.

## 5. Ghi chú tái lập

- Protocol cố định: `--eval -r 8 --n_views 3 --random_background --iterations
  10000 --densify_until_iter 5000 --densify_grad_threshold 0.0005
  --gaussiansN 1 --sample_pseudo_interval 1 --start_sample_pseudo 500
  --test_iterations 10000` (xem `scripts/p26_ablation_run.sh`).
- `--seed` set toàn bộ `random`/`numpy`/`torch` (CPU+CUDA) qua `seed_everything()`
  (`train.py:67`), nhưng KHÔNG đảm bảo full determinism vì rasterizer CUDA
  dùng `atomicAdd` (không kết hợp được theo thứ tự thread) — đây là lý do dự
  án chọn chuẩn 3-seed (42/137/9999) thay vì tin 1 lần chạy.
- Log raw từng scene/seed: `logs/p26_ablation/{CONFIG}/{CONFIG}_seed{SEED}_{SCENE}.log`.
- Tổng hợp bằng `scripts/analyze.py` (`LOG_DIR=... PREFIX=... SEEDS=... python
  scripts/analyze.py`), chạy được cả trên máy dev (không cần GPU/Kaggle) miễn
  có log raw.
