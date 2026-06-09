# Baseline 04 — NexusGS (CVPR 2025)

> Zheng et al., "NexusGS: Sparse View Synthesis with Epipolar Depth Priors in 3D Gaussian Splatting", CVPR 2025.
> [Paper](https://arxiv.org/abs/2503.18794) · base 3DGS + epipolar depth (từ optical flow FlowFormer++).
> Mục tiêu: reproduce LLFF 3-view, images_8 (≈-r8), **30k iter** → unified eval + unified FPS.

**Trạng thái (2026-06-08): 🔧 SETUP — chờ env `nexus` + HF download.**

---

## 1. Method tóm tắt
Base 3DGS, **30k iter**, init + depth từ **epipolar depth priors**:
- Optical flow (FlowFormer++) giữa các view → **flow-derived depth** (`flow_depth`) → depth loss (epipolar).
- `split_num`, `valid_dis_threshold`, `drop_rate`, `near_n` = tham số epipolar/densify đặc thù.
- Init dense từ flow-depth (không phải COLMAP-MVS thuần).

Paper LLFF 3-view: PSNR **_điền từ paper NexusGS_** (CVPR'25, newer — không có trong bảng CoR-GS).

---

## 2. Hai route setup

| Route | Cần gì | Đụng data Phase 22? | Khuyến nghị |
|---|---|---|---|
| **A — HuggingFace** (`run_llff_hf.sh --huggingface`) | tải model `Yukinoo/NexusGS-llff` (per-scene revision) = init pcd + cameras + **flow_depth bake sẵn** | ❌ KHÔNG | ✅ **dùng cái này** |
| B — Raw (`run_llff.sh`) | download `.flo` (FlowFormer++) + `3_views/dense/fused.ply` MVS + COLMAP sparse | ⚠️ dùng fused.ply (cần data tree riêng) | tránh — phức tạp |

→ **Route A**: self-contained, không cần flow/MVS/COLMAP, không động data gốc. Chỉ cần internet tải HF model (init + cam + flow_depth). Đây là init NATIVE của NexusGS → fair.

---

## 3. Setup (env + build)

Env `nexus` (official: python 3.10, torch 2.0.0 cu118). Server chỉ CUDA 12.4 → **hướng B (torch cu121 như fsgs)**:
```bash
conda create -n nexus python=3.10 -y
conda activate nexus
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install plyfile tqdm timm torchmetrics open3d opencv-python imageio matplotlib scipy huggingface_hub
export CUDA_HOME=/usr/local/cuda
cd /home/aidev/workspace/representation-3d/duyen/NexusGS
pip install --no-build-isolation ./submodules/diff-gaussian-rasterization-confidence
pip install --no-build-isolation ./submodules/simple-knn
python -c "import torch, diff_gaussian_rasterization, simple_knn; print('BUILD OK', torch.__version__)"
```
> `huggingface_hub` để tải model HF. Cần internet.

---

## 4. Diagnostic server (chạy trước)
```bash
ROOT=/home/aidev/workspace/representation-3d/duyen
ls -d $ROOT/NexusGS 2>/dev/null && echo "repo OK" || echo "NexusGS repo MISSING (cần clone)"
ls $ROOT/NexusGS/submodules/diff-gaussian-rasterization-confidence/setup.py $ROOT/NexusGS/submodules/simple-knn/setup.py 2>/dev/null
conda env list | grep -i nexus || echo "env nexus chưa có"
ls $ROOT/NexusGS/hf_models/NexusGS-llff/ 2>/dev/null || echo "HF models chưa tải"
```

---

## 5. Pipeline (route A — HF)
Per scene (init từ HF, KHÔNG đụng data gốc):
1. train: `python train.py --source_path <hf_or_repo> --model_path $OUT/<scene>/3_views --eval --n_views 3 --iterations 30000 --save_iterations 30000 --densify_until_iter 30000 --position_lr_max_steps 30000 --dataset_type llff --images images_8 --split_num 4 --valid_dis_threshold 1.0 --drop_rate 1.0 --near_n 2 --huggingface --revision <scene>`
2. render: `python render.py --source_path <hf> --model_path $OUT/<scene>/3_views --iteration 30000 --render_depth --huggingface`
3. **unified eval**: `corgs/metrics.py -m $OUT/<scene>/3_views`
4. **unified FPS**: bench (render signature NexusGS)
- Source: `Yukinoo/NexusGS-llff` (auto-download) hoặc `./hf_models/NexusGS-llff/<scene>` (tải sẵn).
- Output: `NexusGS/output/llff/<scene>/3_views/` · Log: `CoR-GS/logs/ablation/nexusgs/`

---

## 6. Kết quả (điền sau)
| Scene | PSNR | SSIM | SSIM_sk | LPIPS | AVGE | N_gauss | FPS | Train(s) |
|-------|------|------|---------|-------|------|---------|-----|----------|
| (8) | | | | | | | | |
| **Avg** | | | | | | | | |

vs ours 21.918 · Binocular 21.356 · FSGS 20.407 · CoR-GS 20.110.

---

## 7. Caveats
- 30k iter (như Binocular) → cross-budget vs ours 10k.
- Init = NexusGS native (epipolar/flow), tải từ HF → fair, không đụng data ta.
- images_8 = -r8 ta → so trực tiếp.
- Cần internet tải HF model + (cu121 build như fsgs).
