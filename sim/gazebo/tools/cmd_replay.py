#!/usr/bin/env python3
"""Replay recorded velocity commands (A2b problem E): reads a command recording (JSON lines with
'topic', 'sim', 'msg' as written by the A2b recorder), takes the /cmd_vel_nav messages ('nav')
between --start and --end sim seconds, and publishes them on /cmd_vel_nav with their original
sim-time spacing, --repeat times with 2 s of zero in between. Stops at the first fall (tilt > 30).

    cmd_replay.py rec.jsonl --start 2920 --end 2936 [--repeat 10]
"""
import argparse
import json
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('recording')
    ap.add_argument('--start', type=float, required=True)
    ap.add_argument('--end', type=float, required=True)
    ap.add_argument('--repeat', type=int, default=10)
    a = ap.parse_args()
    cmds = []
    for line in open(a.recording):
        r = json.loads(line)
        if r['topic'] == 'nav' and a.start <= r['sim'] <= a.end:
            m = r['msg']
            cmds.append((r['sim'] - a.start, m['linear']['x'], m['linear']['y'], m['angular']['z']))
    rclpy.init()
    n = Node('cmd_replay')
    pub = n.create_publisher(Twist, '/cmd_vel_nav', 10)
    st = {'p': None}
    n.create_subscription(Odometry, '/sim/ground_truth/pose', lambda m: st.update(p=m), 10)
    while st['p'] is None:
        rclpy.spin_once(n, timeout_sec=0.1)

    def sim_t():
        return st['p'].header.stamp.sec + st['p'].header.stamp.nanosec * 1e-9

    def tilt():
        q = st['p'].pose.pose.orientation
        return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (q.x * q.x + q.y * q.y)))))

    def publish(vx, vy, wz):
        m = Twist()
        m.linear.x, m.linear.y, m.angular.z = vx, vy, wz
        pub.publish(m)

    max_tilt, done, fell = 0.0, 0, False
    for k in range(a.repeat):
        t0, i = sim_t(), 0
        while i < len(cmds) and not fell:
            while i < len(cmds) and cmds[i][0] <= sim_t() - t0:
                publish(*cmds[i][1:])
                i += 1
            rclpy.spin_once(n, timeout_sec=0.002)
            max_tilt = max(max_tilt, tilt())
            fell = tilt() > 30.0
        t1 = sim_t()
        while sim_t() - t1 < 2.0 and not fell:
            publish(0.0, 0.0, 0.0)
            end = time.monotonic() + 0.05
            while time.monotonic() < end:
                rclpy.spin_once(n, timeout_sec=0.01)
        if fell:
            print(f'FALL in repeat {k + 1} at replay t {sim_t() - t0:.1f} s', flush=True)
            break
        done += 1
        print(f'repeat {k + 1}: max tilt so far {max_tilt:.1f} deg', flush=True)
    publish(0.0, 0.0, 0.0)
    print(f'RESULT commands={len(cmds)} repeats_ok={done}/{a.repeat} fell={fell} max_tilt={max_tilt:.1f}', flush=True)


if __name__ == '__main__':
    main()
