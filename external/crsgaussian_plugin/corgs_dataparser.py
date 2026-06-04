# ============================================================
# [CRSGaussian Path A B1] DataParser cho method `crsgaussian`
# File: crsgaussian_plugin/corgs_dataparser.py  (KEEP LOCAL — upload server)
#
# Mục đích: Subclass ColmapDataParser → load fused.ply.romav1 (Phase 22 init)
#           + inject train_cameras + depth dict vào metadata cho CrsGaussianModel.
#
# Khác RomaDataParser (A1) ở chỗ:
#   - Inject thêm `aligned_depth_dir` path vào metadata (B2+ depth loss)
#   - Inject thêm `corgs_source_path` cho Model build CoR-GS Scene
#
# Verify B1:
#   - Init Gaussians = 23420 points (fern fused.ply.romav1)
#   - metadata['train_cameras'] không None
# ============================================================
"""[Path A B1] DataParser load RoMa init + inject metadata cho CrsGaussianModel."""

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
    """[Path A B1] Config DataParser cho method `crsgaussian`.

    Extends ColmapDataParserConfig với 3 field:
      - use_roma_init: load fused.ply.romav1 thay points3D.bin
      - roma_ply_relpath: relative path tới PLY
      - aligned_depth_relpath: depth .npy folder (B2 sẽ dùng)
    """

    _target: Type = field(default_factory=lambda: CrsGaussianDataParser)

    # ── [Path A B2] FIX 1 — Align downscale với Phase 22 ──
    # Phase 22 args.resolution=8 → ảnh load 1/8.
    # ns ColmapDataParser default downscale_factor=None (auto).
    # → Override = 8 để ns Cameras image_width/height match PIL load của plug-in.
    # → CRS update + LFCF image lookup KHÔNG còn off-by-N.
    downscale_factor: int = 8
    """[Path A B2] Match Phase 22 args.resolution=8. ns load images_8/ folder."""

    use_roma_init: bool = True
    """[Path A] Master switch — True load RoMa PLY init, False fallback COLMAP."""

    roma_ply_relpath: str = "dense/fused.ply.romav1"
    """Relative path tới PLY từ --data root."""

    aligned_depth_relpath: str = "aligned_depth_a23"
    """Relative path tới aligned depth folder (DAV2 + COLMAP align, .npy per image)."""


class CrsGaussianDataParser(ColmapDataParser):
    """[Path A B1] DataParser — override _load_3D_points + inject metadata.

    Mọi method khác (camera, image, eval split) inherit nguyên ColmapDataParser.
    """

    config: CrsGaussianDataParserConfig

    def _generate_dataparser_outputs(self, split="train", **kwargs):
        """Inject train_cameras + aligned_depth_dir + corgs_source_path vào metadata."""
        outputs = super()._generate_dataparser_outputs(split=split, **kwargs)

        if outputs.metadata is None:
            outputs.metadata = {}

        if split == "train":
            outputs.metadata["train_cameras"] = outputs.cameras
            outputs.metadata["data_root"] = str(self.config.data)
            outputs.metadata["aligned_depth_dir"] = str(
                self.config.data / self.config.aligned_depth_relpath
            )
            # [B2] Inject image_filenames để Model build idx→stem→depth mapping
            outputs.metadata["image_filenames"] = list(outputs.image_filenames)

        return outputs

    def _load_3D_points(
        self,
        colmap_path: Path,
        transform_matrix: torch.Tensor,
        scale_factor: float,
    ):
        """Override: load RoMa PLY (Phase 22 init) thay points3D.bin COLMAP."""

        # OFF flag → fallback parent (vanilla splatfacto byte-identical)
        if not self.config.use_roma_init:
            print(
                "[Path A] use_roma_init=False → fallback COLMAP points3D.bin"
            )
            return super()._load_3D_points(colmap_path, transform_matrix, scale_factor)

        # ON flag → load fused.ply.romav1
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

        # Color: Phase 22 PLY có red/green/blue (uint8), fallback grey nếu thiếu
        if all(k in vertex.data.dtype.names for k in ("red", "green", "blue")):
            colors = np.stack(
                [vertex["red"], vertex["green"], vertex["blue"]], axis=-1
            ).astype(np.uint8)
        else:
            colors = np.full_like(positions, 128, dtype=np.uint8)

        # Apply nerfstudio transform_matrix + scale_factor
        pts3d = torch.from_numpy(positions).float()
        pts3d = (transform_matrix[:3, :3] @ pts3d.T).T + transform_matrix[:3, 3]
        pts3d *= scale_factor

        print(
            f"[Path A B1] Loaded {pts3d.shape[0]} points from {self.config.roma_ply_relpath}"
        )

        return {
            "points3D_xyz": pts3d,
            "points3D_rgb": torch.from_numpy(colors),
        }
