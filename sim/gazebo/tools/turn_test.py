#!/usr/bin/env python3
"""Abrupt walk -> turn-on-the-spot cycles (A1b issue 23, A1c item 3): walk forward at
0.3 m/s, then switch straight to turning at 0.5 rad/s, repeat. Reports the max body
tilt per phase from the ground-truth pose. Publishes /cmd_vel (test tool, A1 only).

    turn_test.py [--cycles 3] [--walk 8] [--turn 6]      (wall seconds per phase)
"""
import argparse
import math
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cycles', type=int, default=3)
    ap.add_argument('--walk', type=float, default=8.0)
    ap.add_argument('--turn', type=float, default=6.0)
    a = ap.parse_args()
    rclpy.init()
    n = Node('turn_test')
    pub = n.create_publisher(Twist, '/cmd_vel', 10)
    st = {}
    n.create_subscription(Odometry, '/sim/ground_truth/pose', lambda m: st.update(p=m), 10)

    def tilt():
        q = st['p'].pose.pose.orientation
        r = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
        p = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
        return math.degrees(max(abs(r), abs(p)))

    def run(vx, wz, secs):
        m = Twist()
        m.linear.x, m.angular.z = float(vx), float(wz)
        t0, mt = time.monotonic(), 0.0
        while time.monotonic() - t0 < secs:
            pub.publish(m)
            rclpy.spin_once(n, timeout_sec=0.02)
            if 'p' in st:
                mt = max(mt, tilt())
        return mt

    while 'p' not in st:
        rclpy.spin_once(n, timeout_sec=0.1)
    worst, fell = 0.0, False
    for k in range(a.cycles):
        w = run(0.3, 0.0, a.walk)
        t = run(0.0, 0.5, a.turn)
        worst = max(worst, w, t)
        print(f'cycle {k}: walk max tilt {w:.1f} deg, abrupt turn max tilt {t:.1f} deg', flush=True)
        if t > 30 or w > 30:
            fell = True
            break
    run(0.0, 0.0, 1.0)
    print(f'RESULT {"FELL" if fell else "ok"}: max tilt {worst:.1f} deg over {k + 1} cycles')
    rclpy.try_shutdown()
    sys.exit(1 if fell else 0)


if __name__ == '__main__':
    main()
