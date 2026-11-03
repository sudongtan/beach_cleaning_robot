#!/usr/bin/env python3
"""Seeded, simulator-independent beach scenes -> bench/scenes.yaml.

    python3 sim/common/generate_scene.py            # benchmark seeds 0-9 + the dev scenes
    python3 sim/common/generate_scene.py --seeds 0-9

Each scene uses only random.Random(seed), so a scene depends on its seed alone,
never on the other seeds in the list. Pure Python + numpy, no ROS, no simulator.
The output format is documented in bench/SCENES_SCHEMA.md.
"""

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import terrain as terrain_mod  # noqa: E402

SCHEMA_VERSION = 1
MATERIALS = ('pet', 'metal', 'glass', 'residual')


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_inputs():
    return {
        'params': load_yaml(HERE / 'scene_params.yaml'),
        'objects': load_yaml(HERE / 'objects.yaml'),
        'geofence': load_yaml(REPO / 'config' / 'geofence.yaml')['geofence'],
        'traversability': load_yaml(REPO / 'config' / 'terrain.yaml')['traversability'],
    }


def r3(v):
    return round(float(v), 3)


def r4(v):
    return round(float(v), 4)


def uniform(rng, lo_hi):
    lo, hi = lo_hi
    return rng.uniform(lo, hi)


def merge(base, over):
    """base with the keys of `over` replaced (nested mappings merged)."""
    out = dict(base)
    for k, v in over.items():
        out[k] = merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def params_for(seed, params):
    """Scene parameters for a seed: the dev-scene overrides for dev seeds."""
    dev = params.get('dev_scenes', {})
    return merge(params, dev['overrides']) if seed in dev.get('seeds', []) else params


def make_terrain(rng, p, area_w, area_h):
    tp = p['terrain']
    wl = uniform(rng, tp['waterline_y'])
    wet_w = uniform(rng, tp['wet_zone_width'])
    rough = []
    for _ in range(tp['roughness']['components']):
        rough.append({
            'amplitude': r4(uniform(rng, tp['roughness']['amplitude'])),
            'wavelength': r3(uniform(rng, tp['roughness']['wavelength'])),
            'direction_deg': r3(rng.uniform(0.0, 180.0)),
            'phase': r4(rng.uniform(0.0, 2 * math.pi)),
        })
    dunes = []
    inner = tp['inner_dunes']
    for _ in range(rng.randint(*inner['count'])):
        s = uniform(rng, inner['sigma'])
        dunes.append({'x': r3(rng.uniform(6.0, area_w - 6.0)),
                      'y': r3(wl + wet_w + 2.5 * s + uniform(rng, inner['extra_y'])),
                      'height': r3(uniform(rng, inner['height'])),
                      'sigma_x': r3(s), 'sigma_y': r3(s)})
    back = tp['back_dunes']
    for _ in range(rng.randint(*back['count'])):
        dunes.append({'x': r3(rng.uniform(-8.0, area_w + 8.0)),
                      'y': r3(area_h + uniform(rng, back['y_behind_area'])),
                      'height': r3(uniform(rng, back['height'])),
                      'sigma_x': r3(uniform(rng, back['sigma']) * 2.0),
                      'sigma_y': r3(uniform(rng, back['sigma']))})
    hm = tp['heightmap']
    return {
        'z_min': tp['z_min'], 'z_max': tp['z_max'],
        'heightmap': {'pixels': hm['pixels'], 'size': hm['size'],
                      'center': [area_w / 2.0, area_h / 2.0]},
        'waterline_y': r3(wl),
        'wet_zone_width': r3(wet_w),
        'wet_zone_height': tp['wet_zone_height'],
        'beach_slope': r4(uniform(rng, tp['beach_slope'])),
        'roughness': rough,
        'dunes': dunes,
        'friction': dict(tp['friction']),
    }


class Placer:
    """Rejection sampler for non-overlapping circles on gentle terrain."""

    def __init__(self, rng, terrain, max_slope_deg, clearance):
        self.rng = rng
        self.terrain = terrain
        self.max_slope = max_slope_deg
        self.clearance = clearance
        self.placed = []          # (x, y, radius)

    def free(self, x, y, r, extra=0.0):
        for px, py, pr in self.placed:
            if math.hypot(x - px, y - py) < r + pr + self.clearance + extra:
                return False
        return True

    def gentle(self, x, y, r):
        pts = [(x, y)] + [(x + r * math.cos(a), y + r * math.sin(a))
                          for a in (0, math.pi / 2, math.pi, 3 * math.pi / 2)]
        return all(terrain_mod.slope_deg_at(self.terrain, px, py) <= self.max_slope for px, py in pts)

    def add(self, x, y, r):
        self.placed.append((x, y, r))

    def sample(self, box, r, tries=2000):
        x0, y0, x1, y1 = box
        for _ in range(tries):
            x = self.rng.uniform(x0 + r, x1 - r)
            y = self.rng.uniform(y0 + r, y1 - r)
            if self.free(x, y, r) and self.gentle(x, y, r):
                self.add(x, y, r)
                return x, y
        raise RuntimeError(f'could not place an object of radius {r} in {box}')


def seg_point_dist(a, b, p):
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(ax + t * dx - px, ay + t * dy - py)


def max_geofence_slope(terrain, fence, res=0.1):
    xs = np.arange(fence['x_min'], fence['x_max'] + 1e-9, res)
    ys = np.arange(fence['y_min'], fence['y_max'] + 1e-9, res)
    X, Y = np.meshgrid(xs, ys)
    return float(terrain_mod.slope_deg_at(terrain, X, Y).max())


def limit_slopes(terrain, fence, max_deg):
    """Scale every dune down until the geofence is walkable everywhere."""
    for _ in range(50):
        if max_geofence_slope(terrain, fence) <= max_deg:
            return
        for d in terrain['dunes']:
            d['height'] = r3(d['height'] * 0.9)
    raise RuntimeError(f'cannot bring the geofence slope under {max_deg} deg')


def make_scene(seed, inp):
    p = params_for(seed, inp['params'])
    objs = inp['objects']
    gf = inp['geofence']
    rng = random.Random(seed)
    area_w, area_h = p['area_size']

    t = make_terrain(rng, p, area_w, area_h)
    wl = t['waterline_y']
    wet_top = wl + t['wet_zone_width']

    fence_y = area_h - 0.5
    fence = {'x_min': gf['edge_margin_m'], 'x_max': area_w - gf['edge_margin_m'],
             'y_min': wl + gf['water_margin_m'], 'y_max': fence_y - gf['back_margin_m']}
    limit_slopes(t, fence, inp['traversability']['max_slope_deg'])
    box = (fence['x_min'], fence['y_min'], fence['x_max'], fence['y_max'])
    polygon = [[r3(fence['x_min']), r3(fence['y_min'])], [r3(fence['x_max']), r3(fence['y_min'])],
               [r3(fence['x_max']), r3(fence['y_max'])], [r3(fence['x_min']), r3(fence['y_max'])]]

    pl = p['placement']
    placer = Placer(rng, t, inp['traversability']['placement_max_slope_deg'], pl['clearance_m'])
    keepout = pl['keepout_m']

    # Bins on the west edge of the dry beach, the robot starts next to them.
    bins = []
    bin_x = fence['x_min'] + 1.0
    bin_y0 = wet_top + uniform(rng, pl['bin_offset_y'])
    for i, name in enumerate(('bin_recyclable', 'bin_residual')):
        x, y = bin_x, bin_y0 + 1.2 * i
        if not placer.gentle(x, y, objs['bins'][name]['footprint_radius']):
            raise RuntimeError(f'seed {seed}: bin site too steep')
        placer.add(x, y, objs['bins'][name]['footprint_radius'] + keepout)
        bins.append({'id': name, 'model': name, 'bin_id': objs['bins'][name]['bin_id'],
                     'pose': [r3(x), r3(y), 0.0]})
    start = (bin_x + 2.5, bin_y0 + 0.6, 0.0)
    placer.add(start[0], start[1], keepout)

    # Landmarks: lifeguard tower at the back, fence posts along the back edge.
    landmarks = []
    tr = objs['landmarks']['lifeguard_tower']['footprint_radius']
    tm = pl['tower_edge_margin_m']
    tx, ty = placer.sample((tm, fence['y_max'] - 5.0, area_w - tm, fence['y_max']), tr)
    placer.placed[-1] = (tx, ty, tr + keepout)
    landmarks.append({'id': 'lifeguard_tower', 'model': 'lifeguard_tower',
                      'pose': [r3(tx), r3(ty), r4(rng.uniform(-0.3, 0.3) - math.pi / 2)]})
    spacing = p['counts']['fence_post_spacing']
    n_posts = int(area_w // spacing) + 1
    for i in range(n_posts):
        landmarks.append({'id': f'fence_post_{i:02d}', 'model': 'fence_post',
                          'pose': [r3(i * spacing), r3(fence_y), 0.0]})

    # Trash: one item of each material first, so every scene has all four.
    trash_models = sorted(objs['trash'])
    n_trash = rng.randint(*p['counts']['trash'])
    chosen = []
    for m in MATERIALS:
        chosen.append(rng.choice([k for k in trash_models if objs['trash'][k]['material'] == m]))
    while len(chosen) < n_trash:
        chosen.append(rng.choice(trash_models))
    rng.shuffle(chosen)
    trash = []
    for i, model in enumerate(chosen):
        o = objs['trash'][model]
        x, y = placer.sample(box, o['footprint_radius'])
        buried = bool(o['buriable'] and rng.random() < pl['half_buried_probability'])
        item = {'id': f'trash_{i:02d}', 'model': model, 'material': o['material'],
                'pose': [r3(x), r3(y), r4(rng.uniform(-math.pi, math.pi))],
                'half_buried': buried}
        if buried:
            item['bury_fraction'] = r3(uniform(rng, pl['bury_fraction']))
        trash.append(item)

    # Distractors; seaweed mostly washes up in the wet zone.
    dist_models = sorted(objs['distractors'])
    distractors = []
    for i in range(rng.randint(*p['counts']['distractors'])):
        model = rng.choice(dist_models)
        o = objs['distractors'][model]
        b = box
        if model == 'seaweed' and rng.random() < pl['seaweed_wet_zone_probability']:
            b = (box[0], box[1], box[2], wet_top)
        x, y = placer.sample(b, o['footprint_radius'])
        distractors.append({'id': f'distractor_{i:02d}', 'model': model, 'type': o['class'],
                            'pose': [r3(x), r3(y), r4(rng.uniform(-math.pi, math.pi))]})

    # Person: odd seeds. Walks east across the whole area, turns, walks back.
    person = None
    if seed % 2 == 1:
        pp = p['person']
        out = pp['start_x_outside']
        # (x, y, radius) of things the person must walk around
        obstacles = [(b['pose'][0], b['pose'][1], 0.4) for b in bins] + [(tx, ty, tr)]
        for _ in range(1000):
            ys = [rng.uniform(wl + pp['y_from_waterline_m'], fence['y_max'] - 1.0) for _ in range(4)]
            wps = [(-out, ys[0]), (area_w + out, ys[1]), (area_w + out, ys[2]), (-out, ys[3])]
            segs = list(zip(wps, wps[1:] + wps[:1]))
            if all(seg_point_dist(a, b, (ox, oy)) >= pp['clearance_m'] + orad
                   for a, b in segs for ox, oy, orad in obstacles):
                break
        else:
            raise RuntimeError(f'seed {seed}: no clear person path')
        person = {'id': 'person_0', 'speed': r3(uniform(rng, pp['speed'])), 'loop': True,
                  'waypoints': [[r3(x), r3(y)] for x, y in wps]}

    return {
        'seed': seed,
        'area': {'size': [area_w, area_h]},
        'geo': {'latitude_deg': p['geo']['latitude_deg'], 'longitude_deg': p['geo']['longitude_deg'],
                'elevation_m': p['geo']['elevation_m']},
        'terrain': t,
        'geofence': {'polygon': polygon},
        'robot_start': {'pose': [r3(start[0]), r3(start[1]), start[2]]},
        'bins': bins,
        'landmarks': landmarks,
        'trash': trash,
        'distractors': distractors,
        'person': person,
    }


def parse_seeds(text, params=None):
    """'0-9' / '0,3,5' / 'all' (benchmark seeds 0-9 + the dev seeds)."""
    if text == 'all':
        return list(range(10)) + (params or load_inputs()['params'])['dev_scenes']['seeds']
    seeds = []
    for part in text.split(','):
        if '-' in part:
            a, b = part.split('-')
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    return seeds


def generate(seeds, inp=None):
    inp = inp or load_inputs()
    return {'schema_version': SCHEMA_VERSION,
            'scenes': [make_scene(s, inp) for s in seeds]}


def dump(doc):
    header = ('# GENERATED by sim/common/generate_scene.py. Do not edit by hand.\n'
              '# Schema: bench/SCENES_SCHEMA.md\n')
    return header + yaml.safe_dump(doc, sort_keys=False, default_flow_style=None, width=100)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--seeds', default='all', help='e.g. 0-9 or 0,3,5; all = 0-9 + dev scenes')
    ap.add_argument('--out', default=str(REPO / 'bench' / 'scenes.yaml'))
    args = ap.parse_args()
    doc = generate(parse_seeds(args.seeds))
    Path(args.out).write_text(dump(doc))
    print(f'wrote {len(doc["scenes"])} scenes to {args.out}')


if __name__ == '__main__':
    main()
