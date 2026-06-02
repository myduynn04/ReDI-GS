# ============================================================
# [CRSGaussian Plug-in A1] RoMa v1 dense init DataParser for Nerfstudio
# File: crsgaussian_plugin/roma_dataparser.py  (KEEP LOCAL — upload server)
#
# Mục đích: Subclass ColmapDataParser → load dense/fused.ply.romav1
#           (output Phase 22 preprocess) thay points3D.bin của COLMAP.
#
# Method registered: splatfacto-roma
#
# Hooks:
#   - Override _load_3D_points() chỉ khi use_roma_init=True
#   - use_roma_init=False → fallback parent (vanilla splatfacto byte-identical)
#
# Tier 2 verify: anh sẽ test
#   2.1 use_roma_init=False → PSNR khớp splatfacto vanilla ±0.10
#   2.2 use_roma_init=True  → N_gauss ≈ 17K (RoMa) vs ~vài trăm (COLMAP)
#   2.3 First 5 positions khớp load_ply(fused.ply.romav1) ±1e-4
#   2.4 Missing PLY → FileNotFoundError, KHÔNG silent fallback COLMAP
# ============================================================
"""[CRSGaussian Plug-in A1] Custom DataParser dùng RoMa v1 dense PLY làm init."""

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
class RomaDataParserConfig(ColmapDataParserConfig):
    """[CRSGaussian Plug-in A1] Config — extends ColmapDataParserConfig.

    Thêm 2 field, KHÔNG đổi behavior parent khi use_roma_init=False.
    """

    _target: Type = field(default_factory=lambda: RomaDataParser)

    use_roma_init: bool = True
    """[CRSGaussian Plug-in A1] Master switch.
    True  → load fused.ply.romav1 (Phase 22 RoMa v1 output) làm init points.
    False → fallback ColmapDataParser parent (đọc points3D.bin) — byte-identical
            với splatfacto vanilla → Tier 2.1 test."""

    roma_ply_relpath: str = "dense/fused.ply.romav1"
    """Đường dẫn PLY tương đối từ --data root.
    Phase 22 preprocess output: data/nerf_llff_data/<scene>/3_views/dense/fused.ply.romav1"""


class RomaDataParser(ColmapDataParser):
    """[CRSGaussian Plug-in A1] DataParser — override _load_3D_points.

    Mọi method khác (camera, image, eval split) inherit nguyên ColmapDataParser.
    """

    config: RomaDataParserConfig

    def _load_3D_points(self, colmap_path: Path, transform_matrix: torch.Tensor, scale_factor: float):
        # ── [CRSGaussian Plug-in A1] Override: load RoMa PLY thay points3D.bin ──
        # OFF flag → fallback parent (Tier 2.1 byte-identical vanilla)
        if not self.config.use_roma_init:
            print("[CRSGaussian Plug-in A1] use_roma_init=False → fallback COLMAP points3D")
            return super()._load_3D_points(colmap_path, transform_matrix, scale_factor)

        # ON flag → load fused.ply.romav1
        ply_path = self.config.data / self.config.roma_ply_relpath
        if not ply_path.is_file():
            # KHÔNG silent fallback — Tier 2.4: missing PLY phải raise error
            raise FileNotFoundError(
                f"[CRSGaussian Plug-in A1] RoMa PLY not found: {ply_path}\n"
                f"  Required: run scripts/p22_romav1_preprocess.py for scene first.\n"
                f"  Or set use_roma_init=False to fallback COLMAP."
            )

        from plyfile import PlyData
        plydata = PlyData.read(str(ply_path))
        vertex = plydata["vertex"]

        # Phase 22 format: x,y,z (float32) + nx,ny,nz (zeros) + red,green,blue (uint8)
        positions = np.stack(
            [vertex["x"], vertex["y"], vertex["z"]], axis=-1
        ).astype(np.float32)
        colors = np.stack(
            [vertex["red"], vertex["green"], vertex["blue"]], axis=-1
        ).astype(np.uint8)

        n_points = len(positions)
        print(
            f"[CRSGaussian Plug-in A1] Loaded {n_points} points from {ply_path.name} "
            f"(vs splatfacto vanilla ~hundreds from COLMAP points3D.bin)"
        )

        # Apply same transform parent class dùng cho COLMAP points
        # (orientation + auto_scale_poses + scale_factor)
        points3D = torch.from_numpy(positions)
        points3D = (
            torch.cat(
                [points3D, torch.ones_like(points3D[..., :1])],
                dim=-1,
            )
            @ transform_matrix.T
        )
        points3D *= scale_factor

        points3D_rgb = torch.from_numpy(colors)

        # PLY của Phase 22 KHÔNG có per-point reproj error / num_2D_views info
        # Splatfacto không dùng 2 field này (chỉ dùng xyz + rgb làm seed_points)
        # Trả synthetic placeholder để pass schema check của parent
        return {
            "points3D_xyz": points3D,
            "points3D_rgb": points3D_rgb,
            "points3D_error": torch.zeros(n_points, dtype=torch.float32),
            "points3D_num_points2D": torch.full((n_points,), 3, dtype=torch.int64),
        }
