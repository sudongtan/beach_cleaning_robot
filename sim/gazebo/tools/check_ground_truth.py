#!/usr/bin/env python3
"""Compare the ground-truth topics with Gazebo's own pose stream (A1b DoD 5),
and the objects with their spawn poses in the world SDF (did anything get
dragged at startup?).

Run while a scene is up (source scene_env.sh first):
    check_ground_truth.py <world.sdf> [--world beach] [--samples 20]
"""

import argparse
import math
import threading
import time
import xml.etree.ElementTree as ET

import rclpy
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node as GzNode
from nav_msgs.msg import Odometry
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray


def stamp_s(st):
    return st.sec + st.nanosec * 1e-9


class Checker(Node):
    def __init__(self, world):
        super().__init__('check_ground_truth')
        self.lock = threading.Lock()
        self.gz_hist = []        # (t, {name: (x, y, z)})
        self.odom, self.objs = [], []
        self.gz = GzNode()
        self.gz.subscribe(Pose_V, f'/world/{world}/pose/info', self.on_gz)
        self.create_subscription(Odometry, '/sim/ground_truth/pose', lambda m: self.odom.append(m), 10)
        self.create_subscription(Detection3DArray, '/sim/ground_truth/objects', lambda m: self.objs.append(m), 10)

    def on_gz(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
        d = {}
        for p in msg.pose:
            d.setdefault(p.name, (p.position.x, p.position.y, p.position.z))
        with self.lock:
            self.gz_hist.append((t, d))
            self.gz_hist = self.gz_hist[-2000:]

    def gz_at(self, t):
        with self.lock:
            return min(self.gz_hist, key=lambda e: abs(e[0] - t)) if self.gz_hist else (None, {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sdf')
    ap.add_argument('--world', default='beach')
    ap.add_argument('--samples', type=int, default=20)
    a = ap.parse_args()

    spawn = {}
    for m in ET.parse(a.sdf).getroot().iter('model'):
        st = m.find('static')
        if st is not None and st.text == 'false':
            spawn[m.get('name')] = list(map(float, m.find('pose').text.split()))[:3]

    rclpy.init()
    n = Checker(a.world)
    t0 = time.monotonic()
    while (len(n.odom) < a.samples or len(n.objs) < a.samples) and time.monotonic() - t0 < 60:
        rclpy.spin_once(n, timeout_sec=0.1)

    worst_robot = 0.0
    for o in n.odom[-a.samples:]:
        t = stamp_s(o.header.stamp)
        tg, d = n.gz_at(t)
        if 'beach_robot' in d:
            p = o.pose.pose.position
            worst_robot = max(worst_robot, math.dist((p.x, p.y, p.z), d['beach_robot']))
    print(f'/sim/ground_truth/pose vs Gazebo model pose: worst {worst_robot * 1000:.2f} mm '
          f'over {min(a.samples, len(n.odom))} msgs (frame {n.odom[-1].header.frame_id} -> '
          f'{n.odom[-1].child_frame_id})')

    worst_obj, count = 0.0, set()
    for arr in n.objs[-a.samples:]:
        tg, d = n.gz_at(stamp_s(arr.header.stamp))
        for det in arr.detections:
            count.add(det.id)
            p = det.results[0].pose.pose.position
            if det.id in d:
                worst_obj = max(worst_obj, math.dist((p.x, p.y, p.z), d[det.id]))
    missing = sorted(set(spawn) - count)
    print(f'/sim/ground_truth/objects vs Gazebo: worst {worst_obj * 1000:.2f} mm, '
          f'{len(count)} ids published, missing: {missing or "none"}')

    last = n.objs[-1]
    drift = []
    for det in last.detections:
        if det.id in spawn:
            p = det.results[0].pose.pose.position
            sx, sy, sz = spawn[det.id]
            drift.append((math.hypot(p.x - sx, p.y - sy), p.z - sz, det.id))
    drift.sort(reverse=True)
    print(f'objects vs spawn pose (sim t = {stamp_s(last.header.stamp):.1f} s): '
          f'worst horizontal {drift[0][0] * 1000:.1f} mm ({drift[0][2]}), '
          f'worst dz {min(d[1] for d in drift) * 1000:+.1f} mm')
    n.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
