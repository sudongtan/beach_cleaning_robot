#!/usr/bin/env python3
"""Does /people match the rendered actor? (A1b DoD 5)

Gazebo doesn't publish the actor's pose, so /people is computed from the
person's timeline in the manifest. This runs the scene's debug world (built
with load_scene.py --debug-camera, no robot), takes frames from the top-down
segmentation camera, finds the person's pixels (label 40), projects their
centroid onto the ground and compares it with the timeline position at the
image's sim time. Uses the same interpolation as ground_truth.py.

    check_people.py <seed> [--frames 15]
"""

import argparse
import bisect
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from gz.msgs10.image_pb2 import Image
from gz.transport13 import Node as GzNode

REPO = Path(__file__).resolve().parents[3]
GEN = REPO / 'sim' / 'gazebo' / 'worlds' / 'generated'
PERSON_LABEL = 40
# debug top camera (load_scene.debug_cameras): above the area centre, looking down
CAM_Z, HFOV, W, H = 42.0, 1.2, 1280, 960


def timeline_pose(person, t):
    tl = person['timeline']
    times = [s[0] for s in tl]
    if person['loop']:
        t = t % times[-1]
    k = max(0, bisect.bisect_right(times, t) - 1)
    k = min(k, len(tl) - 2)
    t0, t1 = times[k], times[k + 1]
    f = 0.0 if t1 == t0 else min(1.0, (t - t0) / (t1 - t0))
    a, b = tl[k], tl[k + 1]
    return a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2]), a[3] + f * (b[3] - a[3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('seed', type=int)
    ap.add_argument('--frames', type=int, default=15)
    a = ap.parse_args()
    subprocess.run([sys.executable, str(REPO / 'sim' / 'gazebo' / 'load_scene.py'), '--seed', str(a.seed),
                    '--debug-camera'], check=True, capture_output=True)
    sdf = GEN / f'scene_{a.seed}_debug.sdf'
    man = yaml.safe_load(open(GEN / f'scene_{a.seed}_debug.manifest.yaml'))
    person = man['person']
    if not person:
        sys.exit(f'seed {a.seed} has no person (odd seeds do)')
    with open(REPO / 'bench' / 'scenes.yaml') as f:
        area = [s for s in yaml.safe_load(f)['scenes'] if s['seed'] == a.seed][0]['area']['size']
    cx, cy = area[0] / 2.0, area[1] / 2.0
    m_per_px = 2 * CAM_Z * math.tan(HFOV / 2) / W        # at z = 0

    srv = subprocess.Popen(['gz', 'sim', '-s', '-r', '--headless-rendering', str(sdf)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    frames = []
    node = GzNode()

    def cb(msg):
        if len(frames) < a.frames:
            frames.append(msg)
    try:
        node.subscribe(Image, '/debug/top/seg/labels_map', cb)
        t0 = time.monotonic()
        while len(frames) < a.frames and time.monotonic() - t0 < 300:
            time.sleep(0.5)
    finally:
        srv.send_signal(2)
        try:
            srv.wait(15)
        except subprocess.TimeoutExpired:
            srv.kill()

    errs = []
    for msg in frames:
        t = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
        v, u = np.nonzero(img[..., 0] == PERSON_LABEL)
        px, py, gz = timeline_pose(person, t)
        if len(u) == 0:
            in_view = abs(px - cx) < W / 2 * m_per_px and abs(py - cy) < H / 2 * m_per_px
            print(f't={t:6.2f}  person not visible (timeline at ({px:.2f}, {py:.2f}), '
                  f'{"inside" if in_view else "outside"} the camera view)')
            continue
        zc = gz + 0.9                                        # centroid ~ mid-body height
        s = m_per_px * (CAM_Z - zc) / CAM_Z
        x = cx + (u.mean() - W / 2) * s
        y = cy - (v.mean() - H / 2) * s                      # image up = north
        e = math.hypot(x - px, y - py)
        errs.append(e)
        print(f't={t:6.2f}  rendered ({x:6.2f}, {y:6.2f})  timeline ({px:6.2f}, {py:6.2f})  error {e:.2f} m  ({len(u)} px)')
    if errs:
        print(f'RESULT {len(errs)} frames with the person in view: mean error {np.mean(errs):.2f} m, max {max(errs):.2f} m')


if __name__ == '__main__':
    main()
