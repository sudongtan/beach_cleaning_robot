"""The contact sheet finds both layouts (Blender <camera>.png and render_check rgb/<camera>/rgb_*.png)."""

import sys

import numpy as np
from PIL import Image

from conftest import ISAAC

sys.path.insert(0, str(ISAAC / 'tools'))
import contact_sheet  # noqa: E402


def test_contact_sheet_layouts(tmp_path):
    red = np.zeros((30, 40, 3), np.uint8)
    red[..., 0] = 255
    (tmp_path / 'scene_0').mkdir()
    Image.fromarray(red).save(tmp_path / 'scene_0' / 'top.png')
    d = tmp_path / 'scene_101' / 'rgb' / 'oblique'
    d.mkdir(parents=True)
    Image.fromarray(red).save(d / 'rgb_0000.png')
    (tmp_path / 'not_a_scene').mkdir()
    img = contact_sheet.sheet(tmp_path, cell_h=60, label_w=50)
    a = np.asarray(img)
    assert img.size == (50 + 80 + 106 + 80, 24 + 2 * 60)
    assert tuple(a[24 + 10, 50 + 10]) == (255, 0, 0)           # seed 0 top (rows sorted by seed)
    assert tuple(a[24 + 60 + 10, 50 + 80 + 10]) == (255, 0, 0)  # seed 101 oblique, render_check layout
    assert tuple(a[24 + 45, 50 + 80 + 20]) == (160, 160, 160)   # seed 0 oblique missing
