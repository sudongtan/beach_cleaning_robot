#!/usr/bin/env python3
"""Turn on the spot, then start walking (A2b problem E: falls when Nav2 started walking while
still turning). Cycles of: turn at --wz for --turn s, then ramp vx at --accel up to --vx for
--walk s, then stop for 2 s. --mode combined keeps turning while walking off (as RPP did);
--mode separate stops turning first. Publishes /cmd_vel_nav (through the safety supervisor).
Stops at the first fall (tilt > 30 deg). Prints one RESULT line.

    turn_walk_test.py [--mode combined|separate] [--cycles 10] [--wz 0.33] [--vx 0.18]
"""
import argparse
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['combined', 'separate'], default='combined')
    ap.add_argument('--cycles', type=int, default=10)
    ap.add_argument('--wz', type=float, default=0.33)
    ap.add_argument('--vx', type=float, default=0.18)
    ap.add_argument('--accel', type=float, default=0.3)
    ap.add_argument('--turn', type=float, default=3.0)
    ap.add_argument('--walk', type=float, default=3.0)
    a = ap.parse_args()
    rclpy.init()
    n = Node('turn_walk_test')
    pub = n.create_publisher(Twist, '/cmd_vel_nav', 10)
    st = {'p': None}
    n.create_subscription(Odometry, '/sim/ground_truth/pose', lambda m: st.update(p=m), 10)

    def sim_t():
        return st['p'].header.stamp.sec + st['p'].header.stamp.nanosec * 1e-9

    def tilt():
        q = st['p'].pose.pose.orientation
        return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (q.x * q.x + q.y * q.y)))))

    while st['p'] is None:
        rclpy.spin_once(n, timeout_sec=0.1)
    max_tilt, fell = 0.0, None

    def phase(secs, cmd):
        """Publish cmd(t since phase start) at 20 Hz (wall) for secs of sim time."""
        nonlocal max_tilt, fell
        t0 = sim_t()
        while sim_t() - t0 < secs:
            vx, wz = cmd(sim_t() - t0)
            m = Twist()
            m.linear.x, m.angular.z = vx, wz
            pub.publish(m)
            end = time.monotonic() + 0.05
            while time.monotonic() < end:
                rclpy.spin_once(n, timeout_sec=0.01)
            max_tilt = max(max_tilt, tilt())
            if tilt() > 30.0:
                fell = True
                return False
        return True

    done = 0
    for k in range(a.cycles):
        sign = 1 if k % 2 == 0 else -1
        ok = phase(a.turn, lambda t: (0.0, sign * a.wz))
        walk_wz = sign * a.wz if a.mode == 'combined' else 0.0
        ok = ok and phase(a.walk, lambda t: (min(a.vx, a.accel * t), walk_wz))
        ok = ok and phase(2.0, lambda t: (0.0, 0.0))
        if not ok:
            print(f'FALL in cycle {k + 1} (tilt {tilt():.0f} deg)', flush=True)
            break
        done += 1
        print(f'cycle {k + 1}: max tilt so far {max_tilt:.1f} deg', flush=True)
    pub.publish(Twist())
    print(f'RESULT mode={a.mode} wz={a.wz} vx={a.vx} cycles_ok={done}/{a.cycles} '
          f'fell={bool(fell)} max_tilt={max_tilt:.1f}', flush=True)


if __name__ == '__main__':
    main()
