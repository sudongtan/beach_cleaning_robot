#!/usr/bin/env python3
"""Markdown table of physics-grasp results, from physics_grasp_test.py JSON files.

    physics_grasp_table.py results/physics_grasp/baseline.json ... > results/physics_grasp/table.md

Later files replace earlier ones per item. Only items still in sim/common/objects.yaml are
listed. Trials from runs made before the 'invalid' class existed are re-classified with the
same rule as physics_grasp_test.py (jaw tips above the item's top after the descent).
"""

import json
import sys
from pathlib import Path

import yaml

from physics_grasp_test import DESCENT_MARGIN, REPO, item_height


def main():
    objects = yaml.safe_load(open(REPO / 'sim' / 'common' / 'objects.yaml'))['trash']
    runs = {}
    for f in sys.argv[1:]:
        for name, v in json.loads(Path(f).read_text())['items'].items():
            runs[name] = (Path(f).name, v['trials'])
    print('| item | height | held | slipped | dropped | carried / valid | invalid | jaw tips after descent | source |')
    print('|---|---|---|---|---|---|---|---|---|')
    for name in sorted(objects):
        if name not in runs:
            print(f'| {name} | | not run | | | | | | |')
            continue
        src, trials = runs[name]
        h = item_height(objects[name])
        res = ['invalid' if t['result'] == 'invalid' or t.get('tips_above_ground_cm', 0) > (h + DESCENT_MARGIN) * 100
               else t['result'] for t in trials]
        n = {c: res.count(c) for c in ('held', 'slipped', 'dropped', 'invalid')}
        valid = n['held'] + n['slipped'] + n['dropped']
        tips = [t['tips_above_ground_cm'] for t, r in zip(trials, res) if r != 'invalid']
        print(f'| {name} | {h * 100:.1f} cm | {n["held"]} | {n["slipped"]} | {n["dropped"]} | '
              f'{n["held"] + n["slipped"]}/{valid} | {n["invalid"]} | {min(tips):.1f}–{max(tips):.1f} cm | {src} |')


if __name__ == '__main__':
    main()
