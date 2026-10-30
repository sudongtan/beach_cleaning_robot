#!/usr/bin/env python3
"""Compare settled poses (a pose/info dump) with the spawn poses of the
dynamic models in a world SDF.

    compare_poses.py <world.sdf> <poses.txt>
"""

import math
import sys
import xml.etree.ElementTree as ET

import gz_poses


def main():
    sdf, dump = sys.argv[1], sys.argv[2]
    spawn = {}
    for m in ET.parse(sdf).getroot().iter('model'):
        p, st = m.find('pose'), m.find('static')
        if p is not None and st is not None and st.text == 'false':
            spawn[m.get('name')] = list(map(float, p.text.split()))
    t, got = gz_poses.parse(open(dump).read())
    print(f'sim time {t:.1f} s')
    worst = 0.0
    for k in sorted(spawn):
        if k not in got:
            print('MISSING', k)
            continue
        dx, dy, dz = (g - s for g, s in zip(got[k], spawn[k][:3]))
        dxy = math.hypot(dx, dy)
        worst = max(worst, dxy)
        flag = '  <-- large' if abs(dz) > 0.05 or dxy > 0.05 else ''
        print(f'{k:16s} dz {dz:+.3f}  dxy {dxy:.3f}{flag}')
    print('worst horizontal drift', round(worst, 3))


if __name__ == '__main__':
    main()
