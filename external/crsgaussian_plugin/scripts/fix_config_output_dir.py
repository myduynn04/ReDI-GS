#!/usr/bin/env python
"""
[B6 viewer-fix] Sửa output_dir trong config.yml để ns reconstruct ckpt path
đúng folder mới (splatfacto_3view_crs_view), KHÔNG load ckpt từ folder gốc.

Bug: cp -r clone folder config.yml giữ nguyên PosixPath cũ → ns viewer
load ckpt từ original folder (vanilla baseline), KHÔNG load converted ckpt.

Cách dùng:
    python fix_config_output_dir.py <path/to/config.yml> <new_output_dir>

Ví dụ:
    python fix_config_output_dir.py \\
        outputs/splatfacto_3view_crs_view/leaves/splatfacto-sparse/.../config.yml \\
        outputs/splatfacto_3view_crs_view
"""

import sys
import yaml
from pathlib import Path


def fix_config(config_path: Path, new_output_dir: Path):
    print(f"═══ Patching {config_path} ═══")

    # Read raw config (use unsafe_load để parse PosixPath + dataclass objects)
    with open(config_path) as f:
        cfg = yaml.unsafe_load(f)

    # cfg is TrainerConfig dataclass → dùng attribute access (KHÔNG subscript)
    old_output_dir = getattr(cfg, "output_dir", None)
    print(f"  Old output_dir: {old_output_dir}")

    # Override output_dir attribute — use ABSOLUTE path để ns resolve đúng
    # dù CWD khác (gốc config dùng absolute, em match convention)
    cfg.output_dir = Path(new_output_dir).resolve()
    print(f"  New output_dir: {cfg.output_dir}")

    # Backup
    backup = config_path.with_suffix(".yml.original_outputdir")
    if not backup.exists():
        import shutil
        shutil.copy(config_path, backup)
        print(f"  Backup: {backup}")

    # Write back
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)
    print(f"  ✅ Written")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <config.yml> <new_output_dir>")
        sys.exit(1)
    fix_config(Path(sys.argv[1]), Path(sys.argv[2]))
