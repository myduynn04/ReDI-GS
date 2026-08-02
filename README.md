# ReDI-GS

**ReDI-GS: Reliable Dense Initialization for Sparse-view 3D Gaussian Splatting**

Code accompanying the thesis. ReDI-GS is a sparse-view Novel View Synthesis (NVS) method
built on top of 3D Gaussian Splatting that combines a dense initialization, a per-Gaussian
Confidence-Reliability Score (CRS), an adaptive densification strategy, and two regularizers,
into a single radiance field trained for 10,000 iterations.

On LLFF with 3 training views, ReDI-GS reaches **21.89 PSNR / 0.769 SSIM / 0.158 LPIPS**
(mean of N=24 paired runs, 3 seeds × 8 scenes), outperforming prior 3DGS-based sparse-view
methods including CoR-GS (+1.78 dB), FSGS (+1.58 dB), and Binocular3DGS (+0.45 dB).

![ReDI-GS pipeline](assets/redi_pipeline.png)

*Overall ReDI-GS pipeline. Phase 1 takes three training images, the COLMAP poses, and the
sparse SfM point cloud to produce a dense initialization with RoMa v1 and a scale-aligned
depth prior with DepthAnything V2. Phase 2 then optimizes a single radiance field for
10,000 iterations under the CRS, the adaptive densification, and the two regularizers.*

## Method

ReDI-GS addresses three limitations of sparse-view 3DGS through five contributions,
organized into two phases.

### Phase 1 — Dense initialization and depth alignment

![Phase 1](assets/phase1.png)

1. **Dense Initialization** with the pre-trained RoMa v1 matcher and a triangulation
   filter, replacing the impoverished SfM point cloud.
2. **Aligned Depth Prior** using DepthAnything V2 (ViT-L) with a weighted-least-squares
   scale alignment to COLMAP sparse depth.

### Phase 2 — Per-scene optimization

![Phase 2](assets/phase2.png)

3. **Confidence-Reliability Score (CRS)** — a per-Gaussian reliability signal that fuses
   depth-cycle consistency, cross-view color consistency, and SH coefficient stability,
   smoothed by an EMA in logit space. The CRS drives a *selective appearance freezing* that
   zeros the high-degree SH gradients of unreliable Gaussians.

   ![CRS module](assets/crs_module.png)

4. **Adaptive Densification (EFA-GS)** combining LFCF and AbsGS for low-frequency-first
   detail growth.
5. **Regularizers** — DropAnSH (anchor + SH coarse-to-fine dropout) at render time and
   opacity decay at the end of each iteration.

The full architecture, equations, and ablation results are presented in the thesis.

## Directory convention

The instructions below assume the following sibling layout. All command lines are run from
inside the repository root unless stated otherwise.

```
<workspace>/
├── CRSGaussian/              this repository
└── Depth-Anything-V2/        cloned from https://github.com/DepthAnything/Depth-Anything-V2
```

RoMa v1 does **not** need to be cloned — it is installed from PyPI by
``environment_roma.yml`` and imported as ``from romatch import roma_outdoor``.

## Installation

The training environment was tested on Ubuntu 22.04 with CUDA 12.x and a single
NVIDIA RTX A4000 (16 GB). The exact conda environments used in our experiments are
exported as ``environment.yml`` (training) and ``environment_roma.yml`` (dense matcher).

### 1. Training environment (`redigs`)

```bash
conda env create --file environment.yml
conda activate redigs

# Build the rasterizer (provides depth + confidence + alpha outputs)
pip install submodules/diff-gaussian-rasterization-confidence

# Build the k-NN extension used by DropAnSH anchor sampling
pip install submodules/simple-knn
```

Key dependencies resolved by the env file: Python 3.8, PyTorch 2.1.0 + cu121,
`torchmetrics`, `open3d`, `opencv-python`, `plyfile`, `kmeans1d`, `matplotlib`,
`scikit-image`.

### 2. Dense-matcher environment (`roma_v1`)

RoMa v1 needs newer Python and PyTorch than the training environment, so it lives in a
separate conda env to avoid library conflicts:

```bash
conda env create --file environment_roma.yml
conda activate roma_v1
```

The first run downloads the RoMa and DINOv2 weights, so the machine needs internet access
once.

### 3. Monocular depth prior — DepthAnything V2

Clone DepthAnything V2 as a sibling of this repository (the training code looks for it at
``../Depth-Anything-V2``):

```bash
cd ..
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
cd Depth-Anything-V2
mkdir -p checkpoints

# Download the ViT-L checkpoint from
#   https://huggingface.co/depth-anything/Depth-Anything-V2-Large
# and place it at:
#   ../Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth
```

The path and encoder are configurable through ``--dav2_path`` and ``--dav2_encoder``;
the code loads ``<dav2_path>/checkpoints/depth_anything_v2_<encoder>.pth``.

### 4. COLMAP

A COLMAP binary on the ``PATH`` is required to build the per-N-view subsets. Any recent
COLMAP (3.7+) providing ``feature_extractor``, ``exhaustive_matcher``,
``point_triangulator``, ``image_undistorter``, ``patch_match_stereo`` and
``stereo_fusion`` works.

## Dataset — LLFF

Download LLFF from the
[NeRF authors](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)
and place it under ``data/`` so that each scene keeps the COLMAP output shipped with the
release:

```
data/
└── nerf_llff_data/
    ├── fern/
    │   ├── images/                       Full-resolution training views
    │   ├── images_4/  images_8/          Downsampled copies (used at -r 4 / -r 8)
    │   ├── sparse/0/                     COLMAP outputs over all views
    │   │   ├── cameras.bin
    │   │   ├── images.bin
    │   │   └── points3D.bin
    │   └── poses_bounds.npy
    ├── flower/ fortress/ horns/ leaves/ orchids/ room/ trex/
```

The official release ships ``sparse/0/`` computed over *all* views, but the sparse-view
experiments additionally need a per-N-view COLMAP triangulation and MVS under
``<scene>/3_views/``. Build them with:

```bash
conda activate redigs
python tools/colmap_llff.py
```

The script is configured by environment variables, so no file editing is needed:

| Variable | Default | Meaning |
|----------|---------|---------|
| `DATA_ROOT` | `data/nerf_llff_data` | Directory holding the scenes |
| `N_VIEWS` | `3` | Number of training views (`3`, `6`, or `9`) |
| `SCENES` | all 8 LLFF scenes | Whitespace-separated scene list |

For example, ``N_VIEWS=6 python tools/colmap_llff.py`` builds the 6-view subsets. Each run
produces, per scene:

```
<scene>/3_views/
├── images/             The N selected training images (a deterministic subset)
├── database.db         COLMAP database for the subset
├── created/            Intermediate files used by point_triangulator
├── triangulated/       Sparse output: cameras.bin, images.bin, points3D.bin
└── dense/
    └── fused.ply       COLMAP MVS dense point cloud over the selected views
```

This step is the slowest part of the setup (patch-match stereo over 8 scenes).

## Reproducing the main result (LLFF 3-view)

Three stages, in this order. Stage 2 depends on the output of Stage 1, and training reads
whatever ``fused.ply`` currently contains — so the order matters.

### Stage 1 — Dense initialization with RoMa v1

For every scene this matches all pairs of the 3 training images with RoMa v1,
triangulates the correspondences using the COLMAP poses, filters by chirality and
reprojection error (``<= 2 px``), and writes
``data/nerf_llff_data/<scene>/3_views/dense/fused.ply.romav1``.

```bash
conda activate roma_v1

# Option A -- all 8 scenes split across 2 GPUs (~1 minute total)
bash scripts/preprocess_all.sh

# Option B -- one scene at a time on a single GPU
SCENE=fern python scripts/preprocess.py
```

### Stage 2 — Place the dense init and train

Training reads the initial point cloud from ``fused.ply`` itself, so the placement script
**overwrites** that file. The first run backs the original COLMAP MVS cloud up to
``fused.ply.colmap_mvs_backup``; later runs keep that backup untouched.

```bash
conda activate redigs

# Swap fused.ply <- fused.ply.romav1 for all 8 scenes
python scripts/place_init.py

# Train the production recipe (3 seeds x 8 scenes = 24 runs, ~5 h on 2 GPUs)
bash scripts/run.sh
```

``run.sh`` first verifies that ``fused.ply`` really is the RoMa v1 cloud, then splits the
8 scenes across two GPUs and calls ``scripts/trainer.sh`` with ``CONFIG=trim_full`` for
every (scene, seed) pair. That config enables the depth prior, D-cycle, SH stability,
SH-CRS freezing, DropAnSH, opacity decay, LFCF and AbsGS.

To restore the original COLMAP MVS initialization at any time:

```bash
RESTORE=1 python scripts/place_init.py
```

### Stage 3 — Extract metrics

```bash
python scripts/analyze.py
```

The script aggregates PSNR, SSIM, LPIPS and the Gaussian count over the 24 logs in
``logs/run/`` and reports the per-scene means together with the overall N=24 mean. Point it
at a different directory with ``LOG_DIR=... python scripts/analyze.py``.

## Quick smoke test

Before launching the full 5-hour training, this one-scene, 100-iteration run confirms that
every dependency resolves and the data is wired correctly:

```bash
conda activate redigs

python train.py \
    --source_path data/nerf_llff_data/fern \
    --model_path output/smoke_test \
    --eval -r 8 --n_views 3 --random_background \
    --iterations 100 --test_iterations 100 \
    --use_depth_prior --dav2_path ../Depth-Anything-V2
```

It should finish in about a minute on a modern GPU. If the log ends with a ``PSNR`` line,
the installation is healthy. Note this smoke test uses only the depth prior — it checks
the plumbing, not the full recipe.

## Reproducing the ablation study

``trainer.sh`` trains any single cell of the per-component leave-one-out ablation; select
the cell through the ``CONFIG`` environment variable:

```bash
GPU=0 CONFIG=trim_no_dcycle \
    SCENES_OVERRIDE="fern flower fortress horns" \
    SEEDS_OVERRIDE="42" \
    LOG_DIR=logs/ablation/trim_no_dcycle \
    OUT_DIR=output/ablation/trim_no_dcycle \
    bash scripts/trainer.sh
```

| `CONFIG` | Recipe |
|----------|--------|
| `trim_full` | Full production recipe (reference) |
| `trim_no_efa` | minus LFCF and AbsGS |
| `trim_no_drop` | minus DropAnSH |
| `trim_no_opacity` | minus opacity decay |
| `trim_no_dcycle` | minus the D-cycle CRS component |
| `trim_no_shcrs` | minus SH-modulated freezing and SH reliability |
| `trim_no_depthcrs` | minus depth supervision and the whole CRS |
| `base` | Vanilla 3DGS, no ReDI-GS additions |

Aggregate a cell by pointing ``analyze.py`` at its log directory:

```bash
LOG_DIR=logs/ablation/trim_no_dcycle python scripts/analyze.py
```

## Codebase layout

```
train.py                          Main training loop with all hooks
render.py                         Evaluation renderer
metrics.py                        PSNR / SSIM / LPIPS computation
arguments/__init__.py             Hyperparameter definitions and CLI flags
scene/
├── __init__.py                   Scene loading and camera setup
├── dataset_readers.py            COLMAP reader exposing reprojection errors
├── gaussian_model.py             Gaussian model with _crs_score, AbsGS,
│                                 opacity decay, LFCF integration
├── cameras.py                    Camera utilities
└── colmap_loader.py              COLMAP binary/text parsers
gaussian_renderer/__init__.py     Differentiable renderer with DropAnSH
utils/
├── crs/
│   ├── crs_module.py             update_crs, cross-view R, fusion, EMA
│   ├── d_cycle.py                D-cycle (round-trip 3D) signal
│   ├── sh_stability.py           S signal (EMA variance of SH high-degree)
│   ├── sh_freeze.py              Selective appearance freezing
│   └── crs_diagnostics.py        Optional CRS logging / heat maps
├── densify/lfcf.py               LFCF action decision + volume-preserving scale
├── regularizer/dropansh.py       DropAnSH anchor + SH coarse-to-fine dropout
└── depth/
    ├── depth_model.py            DepthAnything V2 ViT-L wrapper
    └── depth_alignment.py        Weighted-least-squares scale alignment
lpipsPyTorch/                     LPIPS metric used by train.py and metrics.py
scripts/
├── preprocess.py                 Dense init via RoMa v1 for a single scene
├── preprocess_all.sh             Runs preprocess.py over all 8 scenes on 2 GPUs
├── place_init.py                 Swaps fused.ply with the RoMa v1 dense init
├── run.sh                        Main training driver (24 runs across 2 GPUs)
├── trainer.sh                    Inner trainer, also used for the ablation
└── analyze.py                    Aggregates per-run logs into metric tables
tools/colmap_llff.py              Per-N-view COLMAP pipeline for LLFF
submodules/
├── diff-gaussian-rasterization-confidence/  Differentiable tile rasterizer
└── simple-knn/                              k-NN utility for DropAnSH anchors
```

The repository also carries the research history used while developing the method —
`scripts/p*.py` and `scripts/p*.sh` (per-phase experiments), `external/` (Nerfstudio
plug-in), `demo/` (Gradio demo), `tests/`, and the DTU / Blender / 360 code paths. None of
it is needed to reproduce the results above, and some of it targets flags that no longer
exist.

## Key hyperparameters

These are the values `scripts/trainer.sh` passes for `CONFIG=trim_full`.

```
Iterations                    10,000
Densification window          iter 500 -- 5,000, interval 100
Positional gradient threshold 5e-4
Resolution                    -r 8
CRS warm-up                   1,000 iter
CRS update interval           every 100 iter
CRS EMA decay                 0.3
D-cycle reprojection sigma    5.0 px
SH-CRS freeze threshold tau   0.65
CRS triplet weight w_s        0.33
SH stability EMA beta         0.95
DropAnSH (p_a, p_sh)          (0.02, 0.20)
Opacity decay factor          0.999
LFCF init scale (max / min)   1.5 / 1.0
LFCF interval                 every 2 densification steps
AbsGS                         enabled
```

> **Note on the freeze threshold.** The N=24 figure reported below was measured with
> `--crs_freeze_tau 0.5`. A later single-seed sweep favoured `0.65`, which is the value
> shipped in `trainer.sh`; the observed gain (+0.045 dB) is within the ±0.10 dB
> run-to-run noise floor of the rasterizer and has not yet been re-measured at N=24. Pass
> `--crs_freeze_tau 0.5` to reproduce the published number exactly.

## Results

Quantitative comparison on LLFF with 3 training views (mean over N=24 paired runs).

| Method            | PSNR  | SSIM  | LPIPS |
|-------------------|-------|-------|-------|
| 3DGS              | 15.52 | 0.405 | 0.408 |
| DNGaussian        | 19.12 | 0.591 | 0.294 |
| FSGS              | 20.31 | 0.652 | 0.288 |
| CoR-GS            | 20.45 | 0.712 | 0.196 |
| LoopSparseGS      | 20.85 | 0.717 | 0.205 |
| NexusGS           | 21.07 | 0.738 | 0.177 |
| Binocular3DGS     | 21.44 | 0.751 | 0.168 |
| **ReDI-GS (ours)**| **21.92** | **0.769** | **0.158** |

Single-scene PSNR on this benchmark carries roughly ±1.3 dB of run-to-run variance from
non-deterministic atomic adds in the rasterizer, so every comparison above is a paired
mean over 3 seeds × 8 scenes rather than a single run.

ReDI-GS also leads at 6 and 9 views; see the thesis Results chapter for the full tables
and the per-component contribution analysis.

## Acknowledgement

This project builds on the following open-source works:

- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) — base rasterizer and training framework
- [CoR-GS](https://github.com/jiaw-z/CoR-GS) — codebase scaffold
- [FSGS](https://github.com/VITA-Group/FSGS) — Pearson depth-loss formulation
- [DepthAnything V2](https://github.com/DepthAnything/Depth-Anything-V2) — monocular depth prior
- [RoMa](https://github.com/Parskatt/RoMa) — dense feature matcher
- [DropAnSH](https://arxiv.org/abs/2405.17890) — dropout regularizer (DropAnSH-GS)
- [Binocular3DGS](https://github.com/hanl2010/Binocular3DGS) — opacity-decay regularizer
- [AbsGS](https://github.com/TY424/AbsGS) — absolute-gradient densification
- [EFA-GS](https://github.com/lin-jiawei/EFA-GS) — LFCF densification with volume-preserving scaling
- [DepthRegularizedGS](https://github.com/robot0321/DepthRegularizedGS) — weighted scale-alignment idea
