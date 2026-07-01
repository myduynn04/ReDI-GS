#!/usr/bin/env python
"""
[Thesis 4.6 Figure] Qualitative comparison 3x2 grid.

Layout: 2 rows (trex + room) × 3 cols (GT + Splatfacto + ReDI-GS).
Skip Pure-3DGS để compact.

Output: outputs/figures/fig_qualitative_3x2.png (300 DPI)

Cách dùng:
    cd /home/aidev/workspace/representation-3d/duyen/nerfstudio
    conda activate nerfstudio
    python crsgaussian_plugin/scripts/gen_qualitative_3x2.py

EDIT TEST_CAM_PER_SCENE nếu muốn đổi ảnh test khác.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


def main():
    BASE = Path("outputs/demo3way/demo3way")
    OUT_DIR = Path("outputs/figures")
    OUT_DIR.mkdir(exist_ok=True)

    # ── EDIT: chọn scene + test cam ──
    SCENES = ["trex", "room"]   # 2 rows

    # Test cam mỗi scene — em pick 1 representative cam.
    # Để None → auto-pick first cam.
    TEST_CAM_PER_SCENE = {
        "trex": None,    # auto-pick
        "room": None,    # auto-pick
    }

    # 3 columns (skip Pure-3DGS)
    COLUMNS = [
        ("Ground Truth",   "crsgaussian",      "gt-rgb"),
        ("Splatfacto",     "splatfacto-sparse", "rgb"),
        ("ReDI-GS (ours)", "crsgaussian",      "rgb"),
    ]

    LABEL_HEIGHT = 50      # space cho label cột top
    SCENE_LABEL_W = 80     # space cho label scene bên trái

    # ── Load images ──
    rows_images = []
    for scene in SCENES:
        row = []
        for label, method, subdir in COLUMNS:
            d = BASE / f"{scene}_{method}" / "test" / subdir
            cam_file = TEST_CAM_PER_SCENE.get(scene)
            if cam_file:
                img_path = d / cam_file
            else:
                img_path = next(iter(sorted(d.glob("*.jpg"))), None)

            if img_path and img_path.exists():
                img = np.array(Image.open(img_path).convert("RGB"))
                row.append(img)
                print(f"  Loaded {scene}/{method}: {img_path.name} (shape={img.shape})")
            else:
                print(f"  ❌ Missing: {d}/{cam_file if cam_file else '?'}")
                row.append(None)
        rows_images.append(row)

    if not rows_images or not rows_images[0][0] is not None:
        print("❌ No images loaded — check paths")
        return

    # ── Determine target size from first valid cell ──
    first_img = next(img for row in rows_images for img in row if img is not None)
    h, w = first_img.shape[:2]
    print(f"\n  Target cell size: {w} × {h}")

    n_rows = len(SCENES)
    n_cols = len(COLUMNS)

    canvas_h = LABEL_HEIGHT + n_rows * h
    canvas_w = SCENE_LABEL_W + n_cols * w
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    # ── Paste images ──
    for r, row in enumerate(rows_images):
        for c, img in enumerate(row):
            if img is None:
                continue
            # Resize to common size
            if img.shape[:2] != (h, w):
                img = np.array(Image.fromarray(img).resize((w, h), Image.BILINEAR))
            y = LABEL_HEIGHT + r * h
            x = SCENE_LABEL_W + c * w
            canvas[y:y+h, x:x+w] = img

    # ── Labels: column header (top) + scene name (left) ──
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    try:
        font_col = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
        font_row = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except Exception:
        font_col = font_row = ImageFont.load_default()

    # Column labels (top, centered in each col)
    for c, (label, _, _) in enumerate(COLUMNS):
        bbox = draw.textbbox((0, 0), label, font=font_col)
        text_w = bbox[2] - bbox[0]
        x = SCENE_LABEL_W + c * w + (w - text_w) // 2
        draw.text((x, 12), label, fill=(0, 0, 0), font=font_col)

    # Scene labels (left, vertical-ish, centered in each row)
    for r, scene in enumerate(SCENES):
        bbox = draw.textbbox((0, 0), scene, font=font_row)
        text_h = bbox[3] - bbox[1]
        y = LABEL_HEIGHT + r * h + (h - text_h) // 2
        draw.text((10, y), scene, fill=(0, 0, 0), font=font_row)

    # ── Save ──
    out_path = OUT_DIR / "fig_qualitative_3x2.png"
    pil.save(out_path, "PNG", dpi=(300, 300))
    print(f"\n✅ Saved: {out_path}")
    print(f"   Size: {canvas_w} × {canvas_h} pixels")
    print(f"   LaTeX: \\includegraphics[width=\\textwidth]{{fig_qualitative_3x2.png}}")


if __name__ == "__main__":
    main()
