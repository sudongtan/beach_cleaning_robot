#!/usr/bin/env python3
"""Does the robot stay up standing still? (A2b: `a2b_s3full4` fell with all commands zero.)

Sends no command for --minutes sim minutes and logs position, height, tilt, roll and pitch
from the ground-truth pose every 30 sim s; stops at a fall (tilt > 30 deg). Walk to the spot
first, e.g. at the `a2b_s3full4` fall spot, heading north:

    LOCO=legubiao_rl GOTO="2.74,14.37 --yaw 1.5708" TOOL=hold_test.py turn_walk_test.sh 3 --minutes 5

Prints one RESULT line: seconds held, fall, max tilt, drift from the start position.
"""
import argparse
import math

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--minutes', type=float, default=5.0)
    a, _ = ap.parse_known_args()      # turn_walk_test.sh passes its own args too
    rclpy.init()
    n = Node('hold_test', parameter_overrides=[Parameter('use_sim_time', value=True)])
    st = {}
    n.create_subscription(Odometry, '/sim/ground_truth/pose', lambda m: st.update(p=m), 10)

    def now():
        return n.get_clock().now().nanoseconds * 1e-9

    while 'p' not in st or now() == 0:
        rclpy.spin_once(n, timeout_sec=0.1)
    t0, last, max_tilt, start, fell = now(), -1e9, 0.0, None, False
    while now() - t0 < a.minutes * 60:
        rclpy.spin_once(n, timeout_sec=0.05)
        p = st['p'].pose.pose.position
        q = st['p'].pose.pose.orientation
        roll = math.degrees(math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y)))
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x)))))
        yaw = math.degrees(math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (q.x * q.x + q.y * q.y)))))
        start = start or (p.x, p.y)
        max_tilt = max(max_tilt, tilt)
        if now() - last >= 30 or tilt > 30:
            last = now()
            print(f't {now() - t0:6.1f} pos {p.x:6.2f} {p.y:6.2f} z {p.z:.3f} yaw {yaw:5.0f} tilt {tilt:4.1f} '
                  f'roll {roll:+5.1f} pitch {pitch:+5.1f}', flush=True)
        if tilt > 30:
            fell = True
            break
    drift = math.hypot(p.x - start[0], p.y - start[1])
    print(f'RESULT held_s={now() - t0:.0f} fell={fell} max_tilt={max_tilt:.1f} drift_m={drift:.2f}', flush=True)


if __name__ == '__main__':
    main()
