#!/usr/bin/env python3
"""Long Nav2 walks for the fall rate (A2b problem E): sends the task's search waypoints for the
scene to /nav/goal one after another, in a loop, until --minutes of sim time or a fall
(tilt > 30 deg). Nav2's commands go through the safety supervisor as in a full run. Needs a
running stack without the task (nav_walk_test.sh). Prints progress and one RESULT line with
the sim minutes walked, distance and falls.

    nav_walk_test.py --seed 3 [--minutes 60] [--repo $BEACH_REPO]
"""
import argparse
import math
import os
import time
from pathlib import Path

import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node

from mapping.grid import lethal_at
from mapping.ground_truth import load as load_map
from task.search import lane_spacing, plan


def waypoints(repo, seed):
    """The task's search waypoints for the scene (as in task.node)."""
    s = yaml.safe_load(open(repo / 'config' / 'task.yaml'))['task']['search']
    camera = yaml.safe_load(open(repo / 'config' / 'sensors.yaml'))['camera']
    scene, occ, res = load_map(str(repo), seed)
    spacing = lane_spacing(camera['horizontal_fov_rad'], s['view_distance_m'], s['overlap'])
    return plan(scene['geofence']['polygon'], scene['robot_start']['pose'][:2], spacing,
                s['edge_margin_m'], s['waypoint_spacing_m'], lambda x, y: lethal_at(occ, res, x, y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--minutes', type=float, default=60.0)
    ap.add_argument('--repo', default=os.environ.get('BEACH_REPO', ''))
    a = ap.parse_args()
    wps = waypoints(Path(a.repo), a.seed)
    rclpy.init()
    n = Node('nav_walk_test')
    st = {'p': None}
    n.create_subscription(Odometry, '/sim/ground_truth/pose', lambda m: st.update(p=m), 10)
    client = ActionClient(n, NavigateToPose, '/nav/goal')
    while st['p'] is None or not client.wait_for_server(timeout_sec=1.0):
        rclpy.spin_once(n, timeout_sec=0.1)

    def sim_t():
        return st['p'].header.stamp.sec + st['p'].header.stamp.nanosec * 1e-9

    def pose():
        p = st['p'].pose.pose
        q = p.orientation
        return p.position.x, p.position.y, math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (q.x * q.x + q.y * q.y)))))

    t0, dist, fell, k, goals = sim_t(), 0.0, False, 0, 0
    last = pose()[:2]
    while sim_t() - t0 < a.minutes * 60 and not fell:
        x, y = wps[k % len(wps)]
        rx, ry, _ = pose()
        g = NavigateToPose.Goal()
        g.pose = PoseStamped()
        g.pose.header.frame_id = 'map'
        g.pose.pose.position.x, g.pose.pose.position.y = float(x), float(y)
        yaw = math.atan2(y - ry, x - rx)
        g.pose.pose.orientation.z, g.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        fut = client.send_goal_async(g)
        while not fut.done():
            rclpy.spin_once(n, timeout_sec=0.05)
        handle = fut.result()
        res = handle.get_result_async() if handle.accepted else None
        start = sim_t()
        while res is not None and not res.done():
            rclpy.spin_once(n, timeout_sec=0.05)
            px, py, tilt = pose()
            dist += math.hypot(px - last[0], py - last[1])
            last = (px, py)
            if tilt > 30.0:
                fell = True
                handle.cancel_goal_async()
                print(f'FALL at sim t {sim_t():.1f} ({px:.2f}, {py:.2f}), tilt {tilt:.0f} deg', flush=True)
                break
            if sim_t() - start > 120.0:
                handle.cancel_goal_async()
                break
        goals += 1
        k += 1
        if goals % 10 == 0:
            print(f'{goals} goals, {(sim_t() - t0) / 60:.1f} sim min, {dist:.0f} m', flush=True)
    print(f'RESULT seed={a.seed} sim_minutes={(sim_t() - t0) / 60:.1f} distance_m={dist:.0f} '
          f'goals={goals} fell={fell}', flush=True)


if __name__ == '__main__':
    main()
