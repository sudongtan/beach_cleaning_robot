"""Closed-loop final approach (the task's APPROACH step 2) to virtual items, without the task:
the same law as fsm.Task (gain, max_vx / vy / wz, minimum speed, max_accel, tolerance, settle,
mean-error settle, safety pause, progress rule) from config/task.yaml, on /cmd_vel_nav through the safety supervisor. Each case
puts a virtual item at (target_x + dx, dy) in the robot frame at its start; prints the error
every 0.5 s and one RESULT line per case (in reach / no_progress / cap).
  approach_test.py [--locomotion legubiao_robot_lab] [--cases "dx,dy;dx,dy;..."] [--cap-s 40]
"""
import argparse
import math
import sys
from pathlib import Path

import rclpy
import yaml
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String

from task.fsm import min_planar, to_robot

REPO = Path(__file__).resolve().parents[3]
LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
CASES = '-0.04,0;0.04,0;0,-0.04;0,0.04;-0.06,0.03;-0.10,-0.10;-0.89,-0.53'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--locomotion', default='legubiao_robot_lab')
    ap.add_argument('--cases', default=CASES)
    ap.add_argument('--cap-s', type=float, default=40.0)
    a, _ = ap.parse_known_args()      # turn_walk_test.sh passes its own args too
    fa = yaml.safe_load(open(REPO / 'config' / 'task.yaml'))['task']['fsm']['final_approach']
    vmin = fa.get('min_speed_by_locomotion_mps', {}).get(a.locomotion, 0.0)
    rclpy.init()
    n = Node('approach_test', parameter_overrides=[Parameter('use_sim_time', value=True)])
    st = {'p': None, 'safety': 'ok'}
    n.create_subscription(Odometry, '/sim/ground_truth/pose', lambda m: st.update(p=m), 10)
    n.create_subscription(String, '/safety/state', lambda m: st.update(safety=m.data), LATCHED)
    pub = n.create_publisher(Twist, '/cmd_vel_nav', 10)

    def now():
        return n.get_clock().now().nanoseconds * 1e-9

    def pose():
        p = st['p'].pose.pose
        q = p.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (q.x * q.x + q.y * q.y)))))
        return (p.position.x, p.position.y, yaw), tilt

    def send(c):
        m = Twist()
        m.linear.x, m.linear.y, m.angular.z = c
        pub.publish(m)

    def wait(s):
        t0 = now()
        while now() - t0 < s:
            send((0.0, 0.0, 0.0))
            rclpy.spin_once(n, timeout_sec=0.05)

    while st['p'] is None:
        rclpy.spin_once(n, timeout_sec=0.1)
    print(f'min speed {vmin} m/s ({a.locomotion}), tolerance {fa["tolerance_m"]} m', flush=True)
    clip = lambda v, m: max(-m, min(m, v))         # noqa: E731
    for case in a.cases.split(';'):
        dx, dy = (float(v) for v in case.split(','))
        wait(2.0)
        r, _ = pose()
        c, s = math.cos(r[2]), math.sin(r[2])
        ix, iy = fa['target_x_m'] + dx, dy
        item = (r[0] + c * ix - s * iy, r[1] + s * ix + c * iy)
        cmd, t_cmd, best, best_t, settle, t0, last_print, out = (0.0, 0.0, 0.0), now(), None, now(), None, now(), -1.0, 'cap'
        while now() - t0 < a.cap_s:
            rclpy.spin_once(n, timeout_sec=0.02)
            t = now()
            if t - t_cmd < 0.1:
                continue
            r, tilt = pose()
            if tilt > 30.0:
                print(f'FELL at sim t {t:.1f}', flush=True)
                print('RESULT fell=True', flush=True)
                sys.exit(1)
            x, y = to_robot(r, *item)
            ex = x - fa['target_x_m']
            if st['safety'] != 'ok':                   # the task pauses: zero velocity, not "no progress"
                best_t, settle, cmd, t_cmd = t, None, (0.0, 0.0, 0.0), t
                send(cmd)
                continue
            dist = math.hypot(ex, y)
            if best is None or dist < best - fa['progress_min_m']:
                best, best_t = dist, t
            elif t - best_t > fa['progress_timeout_s']:
                out = 'no_progress'
                break
            tol, band = fa['tolerance_m'], fa['settle_band_m']
            if settle is None and abs(ex) <= tol and abs(y) <= tol:
                settle, acc = t, [0.0, 0.0, 0]
            if settle is not None:
                acc = [acc[0] + ex, acc[1] + y, acc[2] + 1]
                want = (0.0, 0.0, 0.0)
                if t - settle >= fa['settle_s']:
                    settle = None
                    mx, my = acc[0] / acc[2], acc[1] / acc[2]
                    print(f'  settle mean err ({mx:+.3f},{my:+.3f})', flush=True)
                    if abs(mx) <= tol + band and abs(my) <= tol + band:
                        out = 'in_reach'
                        break
            if settle is None:
                vx, vy = min_planar(clip(fa['gain'] * ex, fa['max_vx']), clip(fa['gain'] * y, fa['max_vy']), vmin)
                want = (vx, vy, clip(fa['gain'] * math.atan2(y, x), fa['max_wz']) if x > 0.2 else 0.0)
            dt = min(t - t_cmd, 0.5)
            cmd = tuple(c0 + max(-l * dt, min(l * dt, w - c0)) for c0, w, l in zip(cmd, want, fa['max_accel']))
            t_cmd = t
            send(cmd)
            if t - last_print >= 0.5:
                last_print = t
                print(f'  t {t - t0:5.1f}  err ({ex:+.3f},{y:+.3f})  cmd ({cmd[0]:+.3f},{cmd[1]:+.3f},{cmd[2]:+.3f})  {st["safety"]}',
                      flush=True)
        send((0.0, 0.0, 0.0))
        r, _ = pose()
        x, y = to_robot(r, *item)
        print(f'RESULT case ({dx:+.2f},{dy:+.2f}): {out} after {now() - t0:.1f} s, '
              f'final err ({x - fa["target_x_m"]:+.3f},{y:+.3f})', flush=True)
    wait(1.0)


if __name__ == '__main__':
    main()
