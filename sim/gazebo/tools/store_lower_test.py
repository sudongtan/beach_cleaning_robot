#!/usr/bin/env python3
"""Arm contact while storing (c1_axis s3 arm-rule violation): the real grasping node picks scene items
and stores them in one basket compartment; per grasping phase, the largest gap between the arm
controller's reference and the measured joints (a held item pressed on a wall shows as a gap; static
p99 while still is <= 0.047 rad). Run on the full skeleton with the task stopped:
`KEEP=1 TOOL=store_lower_test.py nav_walk_test.sh 3 lower --items trash_05,trash_06,trash_08`.
Per item: walk 0.49 m behind it facing +x (goto.py, through the supervisor), pick, store.

    store_lower_test.py --seed 3 --items trash_05,trash_06 [--bin recyclable] [--gap-limit 0.05]
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import rclpy
from beach_interfaces.action import PickObject, PlaceObject
from beach_interfaces.msg import WorldModel
from control_msgs.msg import JointTrajectoryControllerState
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState

REPO = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
ARM = ['arm_waist', 'arm_shoulder', 'arm_elbow', 'arm_forearm_roll', 'arm_wrist_angle', 'arm_wrist_rotate']
SETTLE_S = 0.3          # skip the start of each phase (a new goal starts from the measured state)


def scene_pose(seed, scene_id):
    sdf = (REPO / 'sim' / 'gazebo' / 'worlds' / 'generated' / f'scene_{seed}.sdf').read_text()
    m = re.search(r'<model name="%s">.*?<pose>([^<]*)</pose>' % scene_id, sdf, re.S)
    return [float(v) for v in m.group(1).split()[:2]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--items', required=True)
    ap.add_argument('--bin', default='recyclable')
    ap.add_argument('--gap-limit', type=float, default=0.05)
    a, _ = ap.parse_known_args()
    rclpy.init()
    n = Node('store_lower_test')
    st = {'q': None, 'ref': None, 'phase': None, 'gaps': {}, 'wm': {}}

    def on_js(m):
        d = dict(zip(m.name, m.position))
        if all(j in d for j in ARM):
            st['q'] = d
            ph = st['phase']
            if ph and st['ref'] and time.monotonic() - ph[1] > SETTLE_S:
                g = max((abs(st['ref'][j] - d[j]), j) for j in ARM)
                st['gaps'][ph[0]] = max(st['gaps'].get(ph[0], (0.0, '')), g)

    def on_wm(m):
        st['wm'] = {it.scene_id: it for it in m.items if it.scene_id}
    n.create_subscription(JointState, '/joint_states', on_js, 50)
    n.create_subscription(JointTrajectoryControllerState, '/arm_controller/controller_state',
                          lambda m: st.update(ref=dict(zip(m.joint_names, m.reference.positions))), 10)
    n.create_subscription(WorldModel, '/world_model', on_wm, 10)
    pick = ActionClient(n, PickObject, '/grasp/pick')
    place = ActionClient(n, PlaceObject, '/grasp/place')
    pick.wait_for_server(timeout_sec=60.0)
    place.wait_for_server(timeout_sec=60.0)

    def feedback(prefix):
        def cb(fb):
            st['phase'] = (f'{prefix}:{fb.feedback.phase}', time.monotonic())
        return cb

    def run(client, goal, prefix):
        f = client.send_goal_async(goal, feedback_callback=feedback(prefix))
        rclpy.spin_until_future_complete(n, f)
        r = f.result().get_result_async()
        rclpy.spin_until_future_complete(n, r, timeout_sec=900.0)
        st['phase'] = None
        return r.result().result if r.done() else None

    worst = (0.0, '')
    for sid in a.items.split(','):
        x, y = scene_pose(a.seed, sid)
        subprocess.run([sys.executable, str(TOOLS / 'goto.py'), f'{x - 0.49},{y}', '--yaw', '0', '--tol', '0.12',
                        '--seed', str(a.seed), '--accel', '0.3', '--ang-accel', '0.5'], check=False)
        t0 = time.monotonic()
        while sid not in st['wm'] and time.monotonic() - t0 < 30.0:
            rclpy.spin_once(n, timeout_sec=0.1)
        it = st['wm'].get(sid)
        if it is None:
            print(f'{sid}: not in the world model', flush=True)
            continue
        st['gaps'] = {}
        g = PickObject.Goal(id=it.id, pose=it.pose, size=it.size)
        r = run(pick, g, 'pick')
        print(f'{sid} ({it.id}) pick: {r.success if r else None} {r.message if r else "no result"}', flush=True)
        if not r or not r.success:
            continue
        r = run(place, PlaceObject.Goal(target=PlaceObject.Goal.COMPARTMENT, item_id=it.id, bin_id=a.bin), 'store')
        print(f'{sid} store: {r.success if r else None} {r.message if r else "no result"}', flush=True)
        for ph, (gap, j) in st['gaps'].items():
            print(f'  {ph:22s} max gap {gap:.4f} rad ({j})', flush=True)
            if ph.startswith('store') and gap > worst[0]:
                worst = (gap, f'{sid} {ph} {j}')
    print(f'RESULT worst store gap {worst[0]:.4f} rad at {worst[1]} (limit {a.gap_limit}): '
          f'{"PASS" if worst[0] <= a.gap_limit else "FAIL"}', flush=True)
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
