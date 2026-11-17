#!/usr/bin/env python3
"""B5.5 offline check: the stowed arm (`rest`, `up`) against the goal bin's collision boxes while the body
stands where escapes and bin steps reach (base_link `along` out on the bin's axis, `lateral` beside it,
`heading` off facing it; both body heights). Not a planned move: the body moves while the arm rests, so
this only reports the minimum distance (grasping.collision, every pair) and where it is below clearance_m.

Usage (sourced ~/wt/install, PYTHONPATH with packages/grasping):
  python3 packages/grasping/scripts/stowed_bin_check.py [--out ~/wt/logs/b5_stowed.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from collision_sweep import CFG, LIMITS, OBJECTS, bin_pose  # noqa: E402

from grasping.arm import Arm  # noqa: E402
from grasping.collision import Collision, State, bin_boxes  # noqa: E402

ALONG = np.round(np.arange(0.70, 0.951, 0.025), 3)
LATERAL = np.round(np.arange(-0.30, 0.301, 0.05), 3)
HEADING = np.round(np.arange(-0.30, 0.301, 0.05), 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out')
    args = ap.parse_args()
    arm = Arm(LIMITS['variant'], CFG['ik_restarts'], CFG['ik_tolerance_m'])
    clearance = CFG['collision']['clearance_m']
    col = Collision.from_config(arm, CFG['collision'])
    rows = []
    for pose in ('rest', 'up'):
        q = np.asarray(arm.poses[pose]['arm'], dtype=float)
        for body_height in LIMITS['bin_drop']['body_heights_m']:
            for along in ALONG:
                for lateral in LATERAL:
                    for heading in HEADING:
                        opening, byaw = bin_pose(along, lateral, heading, body_height)
                        state = State(bins=bin_boxes(OBJECTS['bins']['bin_recyclable'], opening, byaw))
                        d, pair = col.distance(q, state)
                        rows.append({'pose': pose, 'body_height': body_height, 'along': float(along),
                                     'lateral': float(lateral), 'heading': float(heading), 'd': float(d),
                                     'pair': list(pair)})
    out = {'clearance_m': clearance, 'n': len(rows)}
    for pose in ('rest', 'up'):
        r = [x for x in rows if x['pose'] == pose]
        below = [x for x in r if x['d'] < clearance]
        out[pose] = {'n': len(r), 'min': min(r, key=lambda x: x['d']), 'below_clearance': len(below),
                     'max_along_below': max((x['along'] for x in below), default=None),
                     'min_by_along': {str(a): min(x['d'] for x in r if x['along'] == a) for a in ALONG}}
        print(pose, json.dumps(out[pose]), flush=True)
    if args.out:
        json.dump({**out, 'rows': rows}, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
