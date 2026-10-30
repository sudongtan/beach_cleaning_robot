#!/usr/bin/env python3
"""Test variant of a scene with its inner dunes scaled so the steepest slope on a
given line is TARGET degrees. Writes a one-scene scenes file and prints the
climbing line (start, steepest point, end) for goto.py.

    slope_variant.py <seed> <target_deg> <out_scenes.yaml>
    target_deg = 0: keep the scene as generated (only find the line)
"""
import math
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'sim' / 'common'))
import terrain as T  # noqa: E402

seed, target, out = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
doc = yaml.safe_load(open(REPO / 'bench' / 'scenes.yaml'))
s = [s for s in doc['scenes'] if s['seed'] == seed][0]
t = s['terrain']
inner = [d for d in t['dunes'] if d['height'] < 1.0]


def steepest(terr):
    g = T.traversability({'terrain': terr, 'area': s['area']}, 90, 0.25)['slope_deg']
    ys = (np.arange(g.shape[0]) + .5) * .25
    xs = (np.arange(g.shape[1]) + .5) * .25
    (x0, y0), _, (_, y1), _ = s['geofence']['polygon']
    m = (ys[:, None] > y0 + 2) & (ys[:, None] < y1 - 2) & (xs[None, :] > 3) & (xs[None, :] < 37)
    g = np.where(m, g, 0)
    j, i = np.unravel_index(g.argmax(), g.shape)
    return g.max(), xs[i], ys[j]


base = [d['height'] for d in inner]
lo, hi = 0.1, 3.0
k = 1.0
for _ in range(40 if target > 0 else 0):     # bisection on the dune height scale
    k = (lo + hi) / 2
    for d, h in zip(inner, base):
        d['height'] = h * k
    if steepest(t)[0] > target:
        hi = k
    else:
        lo = k
mx, x, y = steepest(t)
e = 0.05
gx = (T.height_at(t, x + e, y) - T.height_at(t, x - e, y)) / (2 * e)
gy = (T.height_at(t, x, y + e) - T.height_at(t, x, y - e)) / (2 * e)
n = math.hypot(gx, gy)
ux, uy = gx / n, gy / n
doc['scenes'] = [s]
Path(out).write_text(yaml.safe_dump(doc, sort_keys=False))
print(f'scale {k:.3f}: steepest {mx:.1f} deg at ({x:.2f},{y:.2f})')
print(f'LINE {x - 4 * ux:.2f},{y - 4 * uy:.2f} {x - 2.5 * ux:.2f},{y - 2.5 * uy:.2f} {x + 2.5 * ux:.2f},{y + 2.5 * uy:.2f}')
