#!/usr/bin/env python3
"""Probe world for the Gazebo heightmap: a scene's terrain only, plus 0.1 m
boxes dropped from z = 6.5 m on a grid of known points. Compare where they
rest with terrain.height_at (see run_heightmap_probe.sh).

    heightmap_probe.py --detector ode --seed 3 --out probe.sdf --expected expected.txt
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'sim' / 'common'))
sys.path.insert(0, str(REPO / 'sim' / 'gazebo'))
import export_terrain  # noqa: E402
import load_scene  # noqa: E402
import terrain as T  # noqa: E402

GRID = [(x, y) for x in (2.0, 20.0, 38.0) for y in (-5.0, 2.0, 8.0, 14.0, 20.0, 26.0, 32.0, 38.0)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--detector', default='ode', help='ode | bullet | fcl | default (no <dart> element)')
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--out', required=True)
    ap.add_argument('--expected', required=True, help='writes "name x y terrain_height" per probe')
    a = ap.parse_args()

    scene = export_terrain.load_scene(a.seed)
    with open(load_scene.COMMON / 'objects.yaml') as f:
        xml = load_scene.build_world(scene, yaml.safe_load(f))
    root = ET.fromstring(xml.split('\n', 1)[1])
    w = root.find('world')
    for m in list(w.findall('model')) + list(w.findall('actor')):
        if m.get('name') != 'terrain':
            w.remove(m)
    if a.detector == 'default':
        w.find('physics').remove(w.find('physics/dart'))
    else:
        w.find('physics/dart/collision_detector').text = a.detector

    lines = []
    for i, (x, y) in enumerate(GRID):
        name = f'probe_{i:02d}'
        lines.append(f'{name} {x} {y} {float(T.height_at(scene["terrain"], x, y)):.3f}')
        m = ET.SubElement(w, 'model', name=name)
        ET.SubElement(m, 'pose').text = f'{x} {y} 6.5 0 0 0'
        link = ET.SubElement(m, 'link', name='link')
        ine = ET.SubElement(link, 'inertial')
        ET.SubElement(ine, 'mass').text = '0.2'
        it = ET.SubElement(ine, 'inertia')
        for k in ('ixx', 'iyy', 'izz'):
            ET.SubElement(it, k).text = '3e-4'
        c = ET.SubElement(link, 'collision', name='c')
        ET.SubElement(ET.SubElement(ET.SubElement(c, 'geometry'), 'box'), 'size').text = '0.1 0.1 0.1'
    Path(a.out).write_text('<?xml version="1.0" ?>\n' + ET.tostring(root, encoding='unicode'))
    Path(a.expected).write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
