#!/usr/bin/env python3
"""Save /sim/ground_truth/objects and /people at a given sim time to YAML, to compare
two launches of the same scene (A1b DoD 2).

    dump_objects.py <out.yaml> [--at 20.0]
    dump_objects.py --compare a.yaml b.yaml
"""
import argparse
import math
import sys

import yaml


def dump(out, at):
    import rclpy
    from rclpy.node import Node
    from vision_msgs.msg import Detection3DArray
    rclpy.init()
    n = Node('dump_objects')
    got = {}

    def cb(key):
        def f(m):
            t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            if key not in got and t >= at:
                got[key] = (t, {d.id: [d.results[0].pose.pose.position.x, d.results[0].pose.pose.position.y,
                                       d.results[0].pose.pose.position.z] for d in m.detections})
        return f
    n.create_subscription(Detection3DArray, '/sim/ground_truth/objects', cb('objects'), 10)
    n.create_subscription(Detection3DArray, '/people', cb('people'), 10)
    while len(got) < 2:
        rclpy.spin_once(n, timeout_sec=0.1)
    yaml.safe_dump({k: {'t': v[0], 'poses': v[1]} for k, v in got.items()}, open(out, 'w'))
    print(f"saved objects at t={got['objects'][0]:.2f} s, people at t={got['people'][0]:.2f} s -> {out}")
    rclpy.try_shutdown()


def compare(a, b):
    A, B = yaml.safe_load(open(a)), yaml.safe_load(open(b))
    for k in ('objects', 'people'):
        pa, pb = A[k]['poses'], B[k]['poses']
        d = [math.dist(pa[i], pb[i]) for i in pa if i in pb]
        print(f"{k}: {len(pa)} vs {len(pb)} ids, t {A[k]['t']:.3f} vs {B[k]['t']:.3f} s, "
              f"max difference {max(d) * 1000 if d else float('nan'):.2f} mm")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    ap.add_argument('--at', type=float, default=20.0)
    ap.add_argument('--compare', action='store_true')
    a = ap.parse_args()
    compare(*a.files) if a.compare else dump(a.files[0], a.at)
    sys.exit(0)
