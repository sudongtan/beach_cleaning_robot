#!/usr/bin/env python3
"""How well does the walking controller follow velocity commands? (A2b HIMLoco test.)

Holds each (vx, vy, wz) step on /cmd_vel_nav (through the safety supervisor) for --step-s sim
seconds and measures the body-frame velocity from the ground-truth pose over the second half
of the step. Prints one line per step with the command, what the supervisor passed on
(/cmd_vel at the end of the step) and the measured velocity. Stops at a fall (tilt > 30 deg).
Run it on a stack without the task, e.g. through turn_walk_test.sh:

    LOCO=legubiao_rl TOOL=velocity_track_test.py turn_walk_test.sh 101 [--sweep] [--step-s 6]

--tiny: final-approach sized steps (vx 0.02-0.05, vy 0.02-0.1, both signs).
--tiny-yaw: escape-waypoint sized steps (wz 0.03-0.1 both signs; vx / vy 0.08 backward, sideways, mixed). Default steps: a short mix (walk, turn both ways, walk + turn, sideways). --sweep: slow commands
(vx 0.05-0.25, wz +-0.1 to +-0.5, vy 0.05-0.2), each followed by a stop. The steps walk the robot
forward a few metres: rows after it has run into something are not about the controller (check
z and the scene).

--brake (B4 gate G3): --brake-n times walk at 0.3 m/s for --step-s, then command zero for --step-s.
Speed = ground-truth distance over 0.5 s windows centred on each sample (as in A3); deceleration =
(speed at the zero command - 0.05) / time until the speed is <= 0.05 m/s.
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


def tiny_steps():
    """Final-approach sized commands (task final_approach: gain 0.8 on a few cm of error)."""
    steps = [(0.0, 0.0, 0.0)]
    for v in (0.02, 0.03, 0.05, -0.03):
        steps += [(v, 0.0, 0.0), (0.0, 0.0, 0.0)]
    for v in (0.02, 0.03, 0.05, 0.1, -0.03, -0.05):
        steps += [(0.0, v, 0.0), (0.0, 0.0, 0.0)]
    return steps


def tiny_yaw_steps():
    """Escape-waypoint sized commands: yaw alone (task gain 0.8 on a heading error of ~0.04-0.12 rad),
    then planar at the task's minimum speed 0.08 backward / sideways, and mixed with a small yaw."""
    steps = [(0.0, 0.0, 0.0)]
    for c in ((0, 0, 0.03), (0, 0, 0.05), (0, 0, 0.08), (0, 0, 0.1), (0, 0, -0.03), (0, 0, -0.05), (0, 0, -0.08),
              (0, 0, -0.1), (-0.08, 0, 0), (0, 0.08, 0), (0, -0.08, 0), (-0.057, 0.057, 0), (-0.08, 0, 0.05),
              (0, 0.08, -0.05)):
        steps += [tuple(float(v) for v in c), (0.0, 0.0, 0.0)]
    return steps


def window_speed(samples, t, half=0.25):
    """Speed from (t, x, y) samples over [t - half, t + half] (nearest samples); None at the ends."""
    if not samples or t - half < samples[0][0] or t + half > samples[-1][0]:
        return None
    a = min(samples, key=lambda s: abs(s[0] - (t - half)))
    b = min(samples, key=lambda s: abs(s[0] - (t + half)))
    return math.hypot(b[1] - a[1], b[2] - a[2]) / (b[0] - a[0]) if b[0] > a[0] else None


def braking(samples, t_zero, stopped=0.05):
    """(speed at t_zero, seconds until the windowed speed is <= stopped, deceleration) or None."""
    v0 = window_speed(samples, t_zero)
    if v0 is None:
        return None
    for t, _, _ in samples:
        if t >= t_zero:
            v = window_speed(samples, t)
            if v is not None and v <= stopped:
                dt = t - t_zero
                return v0, dt, (v0 - stopped) / dt if dt > 0 else float('inf')
    return v0, None, 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', action='store_true', help='slow-command sweep instead of the short mix')
    ap.add_argument('--step-s', type=float, default=6.0, help='sim seconds per step')
    ap.add_argument('--tiny', action='store_true', help='final-approach sized vx / vy steps')
    ap.add_argument('--tiny-yaw', action='store_true', help='escape-waypoint sized yaw steps')
    ap.add_argument('--brake', action='store_true', help='braking test (B4 G3) instead of steps')
    ap.add_argument('--brake-n', type=int, default=3)
    a, _ = ap.parse_known_args()      # turn_walk_test.sh passes its own args too
    steps = sweep_steps() if a.sweep else tiny_steps() if a.tiny else tiny_yaw_steps() if a.tiny_yaw else MIX
    if a.brake:
        steps = [(0.0, 0.0, 0.0)] + [(0.3, 0.0, 0.0), (0.0, 0.0, 0.0)] * a.brake_n
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
    samples, zeros, prev = [], [], None
    for vx, vy, wz in steps:
        t0, mid = now(), None
        if a.brake and prev is not None and prev[0] > 0.0 and vx == 0.0:
            zeros.append(t0)
        prev = (vx, vy, wz)
        while now() - t0 < a.step_s:
            c = Twist()
            c.linear.x, c.linear.y, c.angular.z = float(vx), float(vy), float(wz)
            pub.publish(c)
            rclpy.spin_once(n, timeout_sec=0.02)
            x, y, yaw, tilt, _ = pose()
            samples.append((now(), x, y))
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
    if a.brake:
        decels = []
        for tz in zeros:
            b = braking(samples, tz)
            if b is None:
                continue
            decels.append(b[2])
            print(f'brake at t {tz:.1f}: v0 {b[0]:.3f} m/s, '
                  f'{"never stopped" if b[1] is None else f"<= 0.05 after {b[1]:.2f} s"}, decel {b[2]:.3f} m/s^2',
                  flush=True)
        print(f'RESULT fell=False max_tilt={max_tilt:.1f} min_decel={min(decels, default=0.0):.3f}', flush=True)
        return
    print(f'RESULT fell=False max_tilt={max_tilt:.1f}', flush=True)


if __name__ == '__main__':
    main()
