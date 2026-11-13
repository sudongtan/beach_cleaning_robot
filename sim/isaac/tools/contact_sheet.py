#!/usr/bin/env python3
"""Contact sheet: one row per scene, one column per camera, from folders of PNGs.

    python sim/isaac/tools/contact_sheet.py --root results/d0_prep/blender --out results/d0_prep/blender_contact_sheet.png

Finds <root>/scene_<seed>/<camera>.png, or the newest rgb*.png under <root>/scene_<seed>/rgb/<camera>/
(render_check.py's layout), so the same script serves the Blender previews, the Isaac session
results and the Gazebo references. Cells are scaled to --cell-height; missing images are grey.
"""

import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw

CAMERAS = ('top', 'oblique', 'robot_eye')


def find(scene_dir, camera):
    direct = scene_dir / f'{camera}.png'
    if direct.is_file():
        return direct
    rgb = sorted((scene_dir / 'rgb' / camera).rglob('rgb*.png')) if (scene_dir / 'rgb' / camera).is_dir() else []
    return rgb[-1] if rgb else None


def seed_of(p):
    m = re.fullmatch(r'scene_(\d+)', p.name)
    return int(m.group(1)) if m else None


def sheet(root, cameras=CAMERAS, cell_h=240, label_w=90):
    scenes = sorted((p for p in Path(root).iterdir() if p.is_dir() and seed_of(p) is not None), key=seed_of)
    if not scenes:
        raise SystemExit(f'no scene_<seed> folders under {root}')
    widths = {c: int(cell_h * {'top': 4 / 3, 'oblique': 16 / 9, 'robot_eye': 4 / 3}.get(c, 4 / 3)) for c in cameras}
    header = 24
    img = Image.new('RGB', (label_w + sum(widths.values()), header + cell_h * len(scenes)), (255, 255, 255))
    d = ImageDraw.Draw(img)
    x = label_w
    for c in cameras:
        d.text((x + 6, 6), c, fill=(0, 0, 0))
        x += widths[c]
    for r, sd in enumerate(scenes):
        y = header + r * cell_h
        d.text((8, y + cell_h // 2 - 6), f'seed {seed_of(sd)}', fill=(0, 0, 0))
        x = label_w
        for c in cameras:
            f = find(sd, c)
            if f is None:
                d.rectangle([x, y, x + widths[c] - 2, y + cell_h - 2], fill=(160, 160, 160))
                d.text((x + 6, y + 6), 'missing', fill=(0, 0, 0))
            else:
                im = Image.open(f).convert('RGB')
                im.thumbnail((widths[c] - 2, cell_h - 2))
                img.paste(im, (x, y))
            x += widths[c]
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--cell-height', type=int, default=240)
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    sheet(a.root, cell_h=a.cell_height).save(a.out)
    print(a.out)


if __name__ == '__main__':
    main()
