# ============================================================
# [CRSGaussian Path A B3 split-fix] DataParser cho method `crsgaussian`
# File: crsgaussian_plugin/corgs_dataparser.py  (KEEP LOCAL — upload server)
#
# Mục đích: Subclass ColmapDataParser →
#   1. Load fused.ply.romav1 (Phase 22 RoMa init)
#   2. Apply Phase 22 split logic PORTED FROM:
#      CRSGaussian/scene/dataset_readers.py:356-366 (readColmapSceneInfo)
#         - Step 1: llffhold=8 → test=idx%8==0 (3 cams), train_pool=idx%8!=0 (21 cams)
#         - Step 2: n_views=3 → train = linspace(0, len(pool)-1, 3).round() = 3 cams
#      → Final: 3 train + 3 test (KHÔNG phải 21 test)
#
# Data folder convention (full fern/, NOT fern/3_views/):
#   fern/
#   ├── sparse/0/         (COLMAP cameras.bin, images.bin, points3D.bin — 24 cams)
#   ├── images/           (24 ảnh gốc)
#   ├── images_8/         (24 ảnh downscaled 1/8 — Phase 22 args.resolution=8)
#   ├── 3_views/
#   │   ├── dense/fused.ply.romav1   (RoMa v1 init — 24543 points)
#   │   └── aligned_depth_a23/       (DAV2 aligned depth)
#   └── ...
#
# Caveat camera ordering:
#   - ns ColmapDataParser sort cameras qua `cameras_unsorted` rồi sort theo... ? cần check
#   - Phase 22 sort by image_name (dataset_readers.py:353)
#   - Em assume ns cũng sort tương tự (verify trong smoke)
# ============================================================
"""[Path A B3 split-fix] DataParser load RoMa init + Phase 22 split (3 train + 3 test)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import numpy as np
import torch

from nerfstudio.data.dataparsers.colmap_dataparser import (
    ColmapDataParser,
    ColmapDataParserConfig,
)


@dataclass
class CrsGaussianDataParserConfig(ColmapDataParserConfig):
    """[Path A B3 split-fix] Config — Phase 22 protocol exact."""

    _target: Type = field(default_factory=lambda: CrsGaussianDataParser)

    # ── [Path A B3 split-fix v2 2026-06-04] Resolution: in-memory resize ──
    # Phase 22 standalone resize ảnh 1/8 in-memory (KHÔNG cần pre-gen images_8/).
    # ns ColmapDataParser dùng `downscale_factor` thì REQUIRE pre-gen folder
    # images_{factor}/ với SAME naming COLMAP. CRSGaussian fern/images_8/
    # naming DIFFERENT (image000.png vs IMG_4027.JPG) → mismatch.
    #
    # Solution: downscale_factor=1 (load fullres) + FullImageDatamanagerConfig
    # camera_res_scale_factor=0.125 (in-memory resize via PIL inside InputDataset,
    # base_dataset.py:88-91 + 59 rescale cameras).
    downscale_factor: int = 1

    # ── [Path A B3 split-fix] Phase 22 split protocol ──
    # Step 1: ColmapDataParser eval_mode="interval" + eval_interval=llffhold
    #   → test = idx%llffhold==0 (3 cams cho fern 24)
    #   → train_pool = idx%llffhold!=0 (21 cams)
    # Step 2: post-process trong _generate_dataparser_outputs
    #   → train = linspace subsample n_views_phase22 cams từ train_pool
    n_views_phase22: int = 3
    """Phase 22 n_views — subsample train_pool xuống N cam. 0 = no subsample."""

    llffhold: int = 8
    """LLFF every-Nth as test. ColmapDataParser eval_interval = llffhold."""

    image_scale_factor: float = 0.125
    """[B3 split-fix v2] In-memory image scale (Phase 22 res=1/8). Phải khớp
    FullImageDatamanagerConfig.camera_res_scale_factor. Em manually rescale
    metadata train_cameras để CoR-GS callback cams match InputDataset rescaled."""

    # Override ColmapDataParser defaults để khớp Phase 22
    eval_mode: str = "interval"
    eval_interval: int = 8
    colmap_path: Path = Path("sparse/0")
    """Default cho LLFF full data folder (vs `triangulated/` cho 3_views/)."""

    use_roma_init: bool = True
    """Master switch — True load RoMa PLY init, False fallback COLMAP."""

    roma_ply_relpath: str = "3_views/dense/fused.ply.romav1"
    """Relative path từ --data root (fern/) tới RoMa PLY."""

    aligned_depth_relpath: str = "3_views/aligned_depth_a23"
    """Relative path tới aligned depth folder (relative từ --data root)."""


class CrsGaussianDataParser(ColmapDataParser):
    """[Path A B3 split-fix] DataParser — Phase 22 split + RoMa init."""

    config: CrsGaussianDataParserConfig

    def _generate_dataparser_outputs(self, split="train", **kwargs):
        """[Path A B3 split-fix] Apply Phase 22 split + inject metadata.

        PORTED FROM: CRSGaussian/scene/dataset_readers.py:362-366
        Step 2 train subsample via linspace AFTER eval_mode="interval" split.

        ColmapDataParser handles Step 1 (eval_interval=8 → test=3, train_pool=21).
        Em handle Step 2 (linspace subsample train_pool → 3 train cams).
        """
        outputs = super()._generate_dataparser_outputs(split=split, **kwargs)

        # ── Step 2: subsample train pool → n_views_phase22 cams ──
        if split == "train" and self.config.n_views_phase22 > 0:
            n_pool = outputs.cameras.size
            if n_pool > self.config.n_views_phase22:
                idx_sub = np.linspace(0, n_pool - 1, self.config.n_views_phase22)
                idx_sub = sorted(set(int(round(i)) for i in idx_sub))

                # Slice cameras — TensorDataclass __getitem__ (tensor_dataclass.py:149-166):
                #   torch.Tensor → x[indices] direct (line 150-151) ← em dùng path này
                #   list[int] → assertion fail (line 154 expects tuple)
                #   tuple → indices + (slice(None),)
                # Use LongTensor cho clean indexing.
                idx_tensor = torch.tensor(idx_sub, dtype=torch.long)
                outputs.cameras = outputs.cameras[idx_tensor]
                # Slice image_filenames
                outputs.image_filenames = [outputs.image_filenames[i] for i in idx_sub]
                # Slice mask_filenames nếu có
                if outputs.mask_filenames is not None:
                    outputs.mask_filenames = [outputs.mask_filenames[i] for i in idx_sub]

                print(
                    f"[Path A B3 split-fix] Subsampled train: "
                    f"{n_pool} train_pool → {len(idx_sub)} train cams (indices {idx_sub})"
                )

        # ── Inject metadata cho Model.populate_modules ──
        if outputs.metadata is None:
            outputs.metadata = {}

        if split == "train":
            # [B3 split-fix v2 2026-06-04] Rescale metadata cameras để match
            # InputDataset rescaled (base_dataset.py:58 deepcopy + 59 rescale).
            # Em's CoR-GS callback cams phải cùng resolution với render cams.
            import copy
            metadata_cams = copy.deepcopy(outputs.cameras)
            metadata_cams.rescale_output_resolution(
                scaling_factor=self.config.image_scale_factor
            )
            outputs.metadata["train_cameras"] = metadata_cams
            outputs.metadata["data_root"] = str(self.config.data)
            outputs.metadata["aligned_depth_dir"] = str(
                self.config.data / self.config.aligned_depth_relpath
            )
            outputs.metadata["image_filenames"] = list(outputs.image_filenames)

        return outputs

    def _load_3D_points(
        self,
        colmap_path: Path,
        transform_matrix: torch.Tensor,
        scale_factor: float,
    ):
        """Override: load RoMa PLY (Phase 22 init) thay points3D.bin COLMAP."""

        if not self.config.use_roma_init:
            print("[Path A] use_roma_init=False → fallback COLMAP points3D.bin")
            return super()._load_3D_points(colmap_path, transform_matrix, scale_factor)

        ply_path = self.config.data / self.config.roma_ply_relpath
        if not ply_path.is_file():
            raise FileNotFoundError(
                f"[Path A] RoMa PLY not found: {ply_path}\n"
                f"  Required: run RoMa v1 preprocess for scene first.\n"
                f"  Or set use_roma_init=False to fallback COLMAP."
            )

        from plyfile import PlyData

        plydata = PlyData.read(str(ply_path))
        vertex = plydata["vertex"]

        positions = np.stack(
            [vertex["x"], vertex["y"], vertex["z"]], axis=-1
        ).astype(np.float32)

        if all(k in vertex.data.dtype.names for k in ("red", "green", "blue")):
            colors = np.stack(
                [vertex["red"], vertex["green"], vertex["blue"]], axis=-1
            ).astype(np.uint8)
        else:
            colors = np.full_like(positions, 128, dtype=np.uint8)

        pts3d = torch.from_numpy(positions).float()
        pts3d = (transform_matrix[:3, :3] @ pts3d.T).T + transform_matrix[:3, 3]
        pts3d *= scale_factor

        print(
            f"[Path A B3 split-fix] Loaded {pts3d.shape[0]} points from {self.config.roma_ply_relpath}"
        )

        return {
            "points3D_xyz": pts3d,
            "points3D_rgb": torch.from_numpy(colors),
        }
