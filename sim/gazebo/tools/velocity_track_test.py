#!/usr/bin/env python3
"""How well does the walking controller follow velocity commands? (A2b HIMLoco test.)

Holds each (vx, vy, wz) step on /cmd_vel_nav (through the safety supervisor) for --step-s sim
seconds and measures the body-frame velocity from the ground-truth pose over the second half
of the step. Prints one line per step with the command, what the supervisor passed on
(/cmd_vel at the end of the step) and the measured velocity. Stops at a fall (tilt > 30 deg).
Run it on a stack without the task, e.g. through turn_walk_test.sh:

    LOCO=legubiao_rl TOOL=velocity_track_test.py turn_walk_test.sh 101 [--sweep] [--step-s 6]

Default steps: a short mix (walk, turn both ways, walk + turn, sideways). --sweep: slow commands
(vx 0.05-0.25, wz +-0.1 to +-0.5, vy 0.05-0.2), each followed by a stop. The steps walk the robot
forward a few metres: rows after it has run into something are not about the controller (check
z and the scene).
"""
import argparse
import math
import sys

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter

MIX = [(0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.3, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.3), (0.0, 0.0, 0.0),
       (0.0, 0.0, -0.3), (0.0, 0.0, 0.0), (0.2, 0.0, 0.2), (0.0, 0.1, 0.0), (0.0, 0.0, 0.0)]


def sweep_steps():
    steps = [(0.0, 0.0, 0.0)]
    for v in (0.05, 0.08, 0.1, 0.15, 0.2, 0.25):
        steps += [(v, 0.0, 0.0), (0.0, 0.0, 0.0)]
    for w in (0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.5, -0.5):
        steps += [(0.0, 0.0, w), (0.0, 0.0, 0.0)]
    for v in (0.05, 0.1, 0.2):
        steps += [(0.0, v, 0.0), (0.0, 0.0, 0.0)]
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', action='store_true', help='slow-command sweep instead of the short mix')
    ap.add_argument('--step-s', type=float, default=6.0, help='sim seconds per step')
    a, _ = ap.parse_known_args()      # turn_walk_test.sh passes its own args too
    steps = sweep_steps() if a.sweep else MIX
    rclpy.init()
    n = Node('velocity_track_test', parameter_overrides=[Parameter('use_sim_time', value=True)])
    st = {'p': None, 'out': Twist()}
    n.create_subscription(Odometry, '/sim/ground_truth/pose', lambda m: st.update(p=m), 10)
    n.create_subscription(Twist, '/cmd_vel', lambda m: st.update(out=m), 10)
    pub = n.create_publisher(Twist, '/cmd_vel_nav', 10)

    def now():
        return n.get_clock().now().nanoseconds * 1e-9

    def pose():
        p = st['p'].pose.pose
        q = p.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (q.x * q.x + q.y * q.y)))))
        return p.position.x, p.position.y, yaw, tilt, p.position.z

    while st['p'] is None:
        rclpy.spin_once(n, timeout_sec=0.1)
    max_tilt = 0.0
    for vx, vy, wz in steps:
        t0, mid = now(), None
        while now() - t0 < a.step_s:
            c = Twist()
            c.linear.x, c.linear.y, c.angular.z = float(vx), float(vy), float(wz)
            pub.publish(c)
            rclpy.spin_once(n, timeout_sec=0.02)
            x, y, yaw, tilt, _ = pose()
            max_tilt = max(max_tilt, tilt)
            if mid is None and now() - t0 >= a.step_s / 2:
                mid = (now(), x, y, yaw)
            if tilt > 30.0:
                print(f'FELL at sim t {now():.1f}, tilt {tilt:.0f} deg', flush=True)
                print(f'RESULT fell=True max_tilt={max_tilt:.1f}', flush=True)
                sys.exit(1)
        x, y, yaw, tilt, z = pose()
        dt = now() - mid[0]
        dx, dy = x - mid[1], y - mid[2]
        bx = (dx * math.cos(mid[3]) + dy * math.sin(mid[3])) / dt
        by = (-dx * math.sin(mid[3]) + dy * math.cos(mid[3])) / dt
        bw = math.remainder(yaw - mid[3], 2 * math.pi) / dt
        o = st['out']
        print(f'cmd ({vx:+.2f},{vy:+.2f},{wz:+.2f})  passed ({o.linear.x:+.2f},{o.linear.y:+.2f},{o.angular.z:+.2f})  '
              f'meas ({bx:+.2f},{by:+.2f},{bw:+.2f})  z {z:.3f} tilt {tilt:.1f}', flush=True)
    print(f'RESULT fell=False max_tilt={max_tilt:.1f}', flush=True)


if __name__ == '__main__':
    main()
