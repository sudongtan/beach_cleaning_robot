#!/usr/bin/env python3
"""Terrain exports for one scene: heightmap PNG (Gazebo), OBJ mesh (Isaac),
traversability grid (A2's baseline /map).

    python3 sim/common/export_terrain.py --seed 3

Output: sim/common/generated/scene_<seed>/ (git-ignored, regenerated on demand).
"""

import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import terrain as terrain_mod  # noqa: E402

OUT_ROOT = HERE / 'generated'


def load_scene(seed, scenes_path=REPO / 'bench' / 'scenes.yaml'):
    with open(scenes_path) as f:
        doc = yaml.safe_load(f)
    for s in doc['scenes']:
        if s['seed'] == seed:
            return s
    raise SystemExit(f'seed {seed} not in {scenes_path}; run `make generate`')


def export(scene, mesh=True):
    out = OUT_ROOT / f"scene_{scene['seed']}"
    out.mkdir(parents=True, exist_ok=True)
    with open(REPO / 'config' / 'terrain.yaml') as f:
        trav = yaml.safe_load(f)['traversability']
    terrain_mod.write_heightmap_png(scene['terrain'], out / 'heightmap.png')
    if mesh:
        terrain_mod.write_obj_mesh(scene['terrain'], out / 'terrain.obj')
    terrain_mod.write_traversability(scene, trav['max_slope_deg'], trav['grid_resolution_m'], out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--no-mesh', action='store_true', help='skip the OBJ terrain mesh')
    args = ap.parse_args()
    print(f'wrote {export(load_scene(args.seed), mesh=not args.no_mesh)}')


if __name__ == '__main__':
    main()
