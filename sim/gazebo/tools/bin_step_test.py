#!/usr/bin/env python3
"""Sideways step between the two bins' place poses (B2), on its own, before the task uses it.

Same control law and limits as the task's bin step (packages/task/task/fsm.py bin_step, config/task.yaml
final_approach + bin_step): command towards the target place pose in the robot frame, heading held
facing the bins, rate-limited like the task's own moves. Publishes /cmd_vel_nav (through the safety
supervisor). Aligns at the first bin's place pose, then steps to the other one and back, --reps times.
One line per step (wall / sim duration, final error, max tilt); stops at a fall (tilt > 30 deg).
Run it on a stack without the task, after walking to the first place pose:

    GOTO="3.5,<y_rec> 2.37,<y_rec> --yaw 3.1416" TOOL=bin_step_test.py LABEL=binstep \
        turn_walk_test.sh <seed> --seed <seed> --reps 3
"""
import argparse
import math
import sys
from pathlib import Path

import rclpy
import yaml
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.parameter import Parameter

REPO = Path(__file__).resolve().parents[3]


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--reps', type=int, default=3, help='there-and-back pairs')
    ap.add_argument('--timeout', type=float, default=60.0, help='sim seconds per step')
    a, _ = ap.parse_known_args()
    cfg = yaml.safe_load(open(REPO / 'config' / 'task.yaml'))['task']['fsm']
    fa, bs = cfg['final_approach'], cfg['bin_step']
    scene = next(s for s in yaml.safe_load(open(REPO / 'bench' / 'scenes.yaml'))['scenes'] if s['seed'] == a.seed)
    d = cfg['bin_place_distance_m']
    poses = {b['bin_id']: (b['pose'][0] + d * math.cos(b['pose'][2]), b['pose'][1] + d * math.sin(b['pose'][2]),
                           wrap(b['pose'][2] + math.pi)) for b in scene['bins']}

    rclpy.init()
    n = Node('bin_step_test', parameter_overrides=[Parameter('use_sim_time', value=True)])
    st = {'p': None}
    n.create_subscription(Odometry, '/sim/ground_truth/pose', lambda m: st.update(p=m), 10)
    pub = n.create_publisher(Twist, '/cmd_vel_nav', 10)

    def now():
        return n.get_clock().now().nanoseconds * 1e-9

    def pose():
        p = st['p'].pose.pose
        q = p.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (q.x * q.x + q.y * q.y)))))
        return p.position.x, p.position.y, yaw, tilt

    def send(vx, vy, wz):
        c = Twist()
        c.linear.x, c.linear.y, c.angular.z = float(vx), float(vy), float(wz)
        pub.publish(c)

    while st['p'] is None:
        rclpy.spin_once(n, timeout_sec=0.1)
    clip = lambda v, m: max(-m, min(m, v))         # noqa: E731
    max_tilt_all = 0.0

    def step_to(name, max_vy):
        nonlocal max_tilt_all
        tx, ty, tyaw = poses[name]
        cmd, t_cmd, t0, max_tilt = [0.0, 0.0, 0.0], None, now(), 0.0
        while now() - t0 < a.timeout:
            rclpy.spin_once(n, timeout_sec=0.02)
            x, y, yaw, tilt = pose()
            max_tilt = max(max_tilt, tilt)
            if tilt > 30.0:
                send(0, 0, 0)
                print(f'FELL at sim t {now():.1f} stepping to {name} after {now() - t0:.1f} s, tilt {tilt:.0f} deg, '
                      f'at ({x:.3f}, {y:.3f}) yaw {math.degrees(yaw):+.1f} deg, target ({tx:.3f}, {ty:.3f}), '
                      f'last command ({cmd[0]:+.3f}, {cmd[1]:+.3f}, {cmd[2]:+.3f})', flush=True)
                print(f'RESULT fell=True max_tilt={max(max_tilt_all, max_tilt):.1f}', flush=True)
                sys.exit(1)
            dx, dy = tx - x, ty - y
            ex, ey = math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy
            if math.hypot(ex, ey) <= fa['tolerance_m']:
                send(0, 0, 0)
                break
            want = (clip(fa['gain'] * ex, fa['max_vx']), clip(fa['gain'] * ey, max_vy),
                    clip(fa['gain'] * wrap(tyaw - yaw), fa['max_wz']))
            t = now()
            dt = 0.1 if t_cmd is None else min(max(t - t_cmd, 0.0), 0.5)
            cmd = [c + max(-lim * dt, min(lim * dt, w - c)) for c, w, lim in zip(cmd, want, fa['max_accel'])]
            t_cmd = t
            send(*cmd)
            # ~10 Hz like the task
            end = now() + 0.1
            while now() < end:
                rclpy.spin_once(n, timeout_sec=0.02)
        x, y, yaw, _ = pose()
        err = math.hypot(tx - x, ty - y)
        max_tilt_all = max(max_tilt_all, max_tilt)
        print(f'to {name:10s} sim {now() - t0:5.1f} s  err {err:.3f} m  yaw err {math.degrees(wrap(tyaw - yaw)):+.1f} deg  '
              f'max tilt {max_tilt:.1f} deg  {"ok" if err <= fa["tolerance_m"] + 0.01 else "TIMEOUT"}', flush=True)
        # settle like the task before the arm moves
        t1 = now()
        while now() - t1 < fa['settle_s']:
            send(0, 0, 0)
            rclpy.spin_once(n, timeout_sec=0.05)

    first, other = 'recyclable', 'residual'
    step_to(first, fa['max_vy'])                    # fine alignment, as the task's bin final approach
    for _ in range(a.reps):
        step_to(other, bs['max_vy'])
        step_to(first, bs['max_vy'])
    print(f'RESULT fell=False max_tilt={max_tilt_all:.1f}', flush=True)


if __name__ == '__main__':
    main()
