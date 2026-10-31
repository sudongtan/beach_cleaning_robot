#!/usr/bin/env python3
"""Walk the robot through map-frame waypoints using the ground-truth pose, and
report what happened on the way (A1b DoD 6: dry sand, wet sand, dune slope).

A test tool, not navigation: turn towards the waypoint, walk straight, repeat.
Publishes --topic, default /cmd_vel_nav (from A2 on test drivers go through the safety
supervisor, CLAUDE.md; A1 used /cmd_vel directly).

    goto.py x1,y1 [x2,y2 ...] [--yaw Y] [--speed 0.3] [--scene bench/scenes.yaml --seed N]
            [--accel 0.3 --ang-accel 0.5]

--accel / --ang-accel: limit how fast the command changes (m/s^2, rad/s^2), like
Nav2's velocity smoother; without them the driver switches straight from walking to
turning on the spot.

With --scene/--seed, every second it also logs the terrain zone, height and
slope under the robot (from sim/common/terrain.py). Exit code 1 on a fall, 2 if the
ground-truth pose stops updating for STALE_S wall seconds (result INVALID: e.g. the
simulator died or the network went away; such a run says nothing about the robot).
"""

import argparse
import math
import sys
import time
from pathlib import Path

import rclpy
import yaml
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'sim' / 'common'))
import terrain as T  # noqa: E402

FALL_TILT = 0.6        # rad, roll or pitch (same criterion as A0's drive test)
STALE_S = 10.0         # wall seconds without a new ground-truth pose -> INVALID


def rpy(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class Goto(Node):
    def __init__(self, topic):
        super().__init__('goto_test')
        self.pose = None
        self.last_update = time.monotonic()
        self.create_subscription(Odometry, '/sim/ground_truth/pose', self.cb, 10)
        self.pub = self.create_publisher(Twist, topic, 10)

    def cb(self, m):
        self.pose = m
        self.last_update = time.monotonic()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('waypoints', nargs='+')
    ap.add_argument('--yaw', type=float, help='final heading (rad)')
    ap.add_argument('--speed', type=float, default=0.3)
    ap.add_argument('--tol', type=float, default=0.15)
    ap.add_argument('--timeout', type=float, default=240.0, help='wall seconds')
    ap.add_argument('--scene', default=str(REPO / 'bench' / 'scenes.yaml'))
    ap.add_argument('--seed', type=int)
    ap.add_argument('--accel', type=float, help='max linear acceleration, m/s^2 (default: none)')
    ap.add_argument('--ang-accel', type=float, help='max angular acceleration, rad/s^2 (default: none)')
    ap.add_argument('--topic', default='/cmd_vel_nav', help='velocity topic (default /cmd_vel_nav: through the '
                                                          'safety supervisor)')
    a = ap.parse_args()
    wps = [tuple(map(float, w.split(','))) for w in a.waypoints]
    terrain = None
    if a.seed is not None:
        terrain = [s for s in yaml.safe_load(open(a.scene))['scenes'] if s['seed'] == a.seed][0]['terrain']

    rclpy.init()
    n = Goto(a.topic)
    while n.pose is None:
        rclpy.spin_once(n, timeout_sec=0.1)
    max_tilt, fell, last_log, zones = 0.0, False, 0.0, set()
    t0 = time.monotonic()
    i = 0
    phase = 'goto'
    prev = Twist()
    prev_t = time.monotonic()

    def smooth(cmd):
        """Rate-limit the command (per wall second; RTF < 1 makes it gentler in sim time)."""
        nonlocal prev, prev_t
        now = time.monotonic()
        dt = max(1e-3, now - prev_t)
        prev_t = now
        if a.accel:
            d = a.accel * dt
            cmd.linear.x = max(prev.linear.x - d, min(prev.linear.x + d, cmd.linear.x))
        if a.ang_accel:
            d = a.ang_accel * dt
            cmd.angular.z = max(prev.angular.z - d, min(prev.angular.z + d, cmd.angular.z))
        prev = cmd
        return cmd
    stale = False
    while rclpy.ok() and time.monotonic() - t0 < a.timeout:
        rclpy.spin_once(n, timeout_sec=0.02)
        if time.monotonic() - n.last_update > STALE_S:
            stale = True
            break
        p = n.pose.pose.pose
        r, pi, yaw = rpy(p.orientation)
        max_tilt = max(max_tilt, abs(r), abs(pi))
        if abs(r) > FALL_TILT or abs(pi) > FALL_TILT:
            fell = True
            break
        cmd = Twist()
        if phase == 'goto':
            tx, ty = wps[i]
            dx, dy = tx - p.position.x, ty - p.position.y
            dist = math.hypot(dx, dy)
            if dist < a.tol:
                i += 1
                if i == len(wps):
                    phase = 'align' if a.yaw is not None else 'done'
                continue
            err = wrap(math.atan2(dy, dx) - yaw)
            cmd.angular.z = max(-0.5, min(0.5, 1.5 * err))
            if abs(err) < 0.35:
                cmd.linear.x = min(a.speed, 0.8 * dist)
        elif phase == 'align':
            err = wrap(a.yaw - yaw)
            if abs(err) < 0.05:
                phase = 'done'
            cmd.angular.z = max(-0.5, min(0.5, 1.5 * err))
        if phase == 'done':
            break
        n.pub.publish(smooth(cmd))
        st = n.pose.header.stamp.sec + n.pose.header.stamp.nanosec * 1e-9
        if st - last_log >= 1.0:
            last_log = st
            info = ''
            if terrain:
                x, y = p.position.x, p.position.y
                wl, wt = terrain['waterline_y'], terrain['waterline_y'] + terrain['wet_zone_width']
                zone = 'water' if y < wl else 'wet' if y < wt else 'dry'
                zones.add(zone)
                info = (f' zone={zone} ground={float(T.height_at(terrain, x, y)):+.2f} m '
                        f'slope={float(T.slope_deg_at(terrain, x, y)):4.1f} deg')
            print(f't={st:6.1f} x={p.position.x:6.2f} y={p.position.y:6.2f} yaw={yaw:+.2f} '
                  f'roll={math.degrees(r):+5.1f} pitch={math.degrees(pi):+5.1f}{info}', flush=True)
    for _ in range(10):
        n.pub.publish(Twist())
        time.sleep(0.05)
    p = n.pose.pose.pose
    ok = phase == 'done' and not fell and not stale
    verdict = 'INVALID (pose stream stopped)' if stale else 'ok' if ok else 'FELL' if fell else 'TIMEOUT'
    print(f'RESULT {verdict}: reached {i}/{len(wps)} waypoints, '
          f'final ({p.position.x:.2f}, {p.position.y:.2f}), max tilt {math.degrees(max_tilt):.1f} deg'
          + (f', zones {sorted(zones)}' if zones else ''))
    n.destroy_node()
    rclpy.try_shutdown()
    sys.exit(2 if stale else 0 if ok else 1)


if __name__ == '__main__':
    main()
