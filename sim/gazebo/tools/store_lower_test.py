#!/usr/bin/env python3
"""Arm contact while storing (c1_axis s3 arm-rule violation): the real grasping node picks scene items
and stores them in one basket compartment; per grasping phase, the largest gap between the arm
controller's reference and the measured joints (a held item pressed on a wall shows as a gap; static
p99 while still is <= 0.047 rad). Run on the full skeleton with the task stopped:
`KEEP=1 TOOL=store_lower_test.py nav_walk_test.sh 3 lower --items trash_05,trash_06,trash_08`.
Per item: walk 0.49 m behind it facing +x (goto.py, through the supervisor), pick, store.

    store_lower_test.py --seed 3 --items trash_05,trash_06 [--bin recyclable] [--gap-limit 0.05]

--plan (B5b S4, one item per compartment): steps in order, `<scene id>:<compartment>` picks and stores an
item, `unload` takes every stored item to its bin: Nav2 to bin_standoff_m in front of it, goto.py onto the
bin's axis at bin_place_distance_m facing it, /grasp/place BIN (the task's bin pose: centre, rim, yaw).
    store_lower_test.py --seed 3 --plan trash_05:recyclable,trash_00:residual,unload,trash_06:recyclable,unload
"""

import argparse
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pinocchio as pin
import rclpy
import yaml
from beach_interfaces.action import PickObject, PlaceObject
from beach_interfaces.msg import WorldModel
from control_msgs.msg import JointTrajectoryControllerState
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node as GzNode
from nav_msgs.msg import Odometry
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState

from physics_grasp_test import quat_to_R, rc

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
    ap.add_argument('--items', default='')
    ap.add_argument('--plan', help='steps <scene id>:<compartment> | unload (see the docstring)')
    ap.add_argument('--bin', default='recyclable')
    ap.add_argument('--gap-limit', type=float, default=0.05)
    ap.add_argument('--world', default='beach')
    a, _ = ap.parse_known_args()
    rclpy.init()
    n = Node('store_lower_test')
    st = {'q': None, 'ref': None, 'phase': None, 'gaps': {}, 'wm': {}, 'end': {}, 'poses': {}, 'base': None}

    def on_js(m):
        d = dict(zip(m.name, m.position))
        if all(j in d for j in ARM):
            st['q'] = d
            ph = st['phase']
            if ph and st['ref'] and time.monotonic() - ph[1] > SETTLE_S:
                g = max((abs(st['ref'][j] - d[j]), j) for j in ARM)
                st['gaps'][ph[0]] = max(st['gaps'].get(ph[0], (0.0, '')), g)
                st['end'][ph[0]] = [st['ref'][j] - d[j] for j in ARM]     # signed, last sample of the phase

    def on_wm(m):
        st['wm'] = {it.scene_id: it for it in m.items if it.scene_id}
    n.create_subscription(JointState, '/joint_states', on_js, 50)
    n.create_subscription(JointTrajectoryControllerState, '/arm_controller/controller_state',
                          lambda m: st.update(ref=dict(zip(m.joint_names, m.reference.positions))), 10)
    n.create_subscription(WorldModel, '/world_model', on_wm, 10)
    n.create_subscription(Odometry, '/sim/ground_truth/pose', lambda m: st.update(base=m), 10)
    gz = GzNode()
    gz.subscribe(Pose_V, f'/world/{a.world}/pose/info', lambda m: st.update(poses={
        p.name: np.array([p.position.x, p.position.y, p.position.z]) for p in reversed(m.pose)}))
    model = rc.build_model('custom')
    data = model.createData()
    pin.framesForwardKinematics(model, data, pin.neutral(model))
    margin = yaml.safe_load(open(REPO / 'config' / 'grasping.yaml'))['grasping']['store']['landing_margin_m']
    margin = margin if isinstance(margin, list) else [margin, margin]
    pick = ActionClient(n, PickObject, '/grasp/pick')
    place = ActionClient(n, PlaceObject, '/grasp/place')
    pick.wait_for_server(timeout_sec=60.0)
    place.wait_for_server(timeout_sec=60.0)

    def in_tips(sid):
        """B5b S4 run 4 (held 6.7-9 cm off the jaw centre): sid's centre (map, ground truth) and in the jaw-tips
        frame (ground-truth base pose + measured arm joints); None if not known yet."""
        if sid not in st['poses'] or st['base'] is None or st['q'] is None:
            return None
        qv = pin.neutral(model)
        for j in ARM:
            qv[model.joints[model.getJointId(j)].idx_q] = st['q'][j]
        pin.framesForwardKinematics(model, data, qv)
        b = st['base'].pose.pose
        bR = quat_to_R(b.orientation.x, b.orientation.y, b.orientation.z, b.orientation.w)
        T = data.oMf[model.getFrameId('arm_jaw_tips_link')]
        tips, tR = np.array([b.position.x, b.position.y, b.position.z]) + bR @ T.translation, bR @ T.rotation
        base = np.array([b.position.x, b.position.y, math.atan2(bR[1, 0], bR[0, 0])])
        return st['poses'][sid].copy(), tR.T @ (st['poses'][sid] - tips), base

    def feedback(prefix):
        def cb(fb):
            st['phase'] = (f'{prefix}:{fb.feedback.phase}', time.monotonic())
            if prefix == 'pick' and st.get('track'):
                st['snaps'].append((fb.feedback.phase, in_tips(st['track'])))
        return cb

    def pick_report(sid, goal):
        """Per pick phase start: how far the item moved (map, cm) and where it sat in the jaw-tips frame (cm;
        the planned hold is ~[4.5, 0, 0]: centred between the jaws, below the tips)."""
        gp = goal.pose.pose.position
        first = next((s[0] for _, s in st['snaps'] if s is not None), None)
        b0 = next((s[2] for _, s in st['snaps'] if s is not None), None)
        if first is not None:
            print(f'  {sid} world model vs ground truth at the pick goal: '
                  f'{((np.array([gp.x, gp.y, gp.z]) - first) * 100).round(1).tolist()} cm', flush=True)
        for ph, s in st['snaps']:
            if s is not None:
                d, c, sn = s[2] - b0, math.cos(b0[2]), math.sin(b0[2])      # body shift since the goal, goal frame
                print(f'  {sid} pick:{ph:10s} item moved {((s[0] - first) * 100).round(1).tolist()} cm, '
                      f'in tips {(s[1] * 100).round(1).tolist()} cm, body moved fwd {(c * d[0] + sn * d[1]) * 100:+.1f} '
                      f'left {(-sn * d[0] + c * d[1]) * 100:+.1f} cm yaw {math.remainder(d[2], 2 * math.pi):+.3f} rad',
                      flush=True)
        st['snaps'], st['track'] = [], None

    def run(client, goal, prefix):
        f = client.send_goal_async(goal, feedback_callback=None if prefix == 'nav' else feedback(prefix))
        rclpy.spin_until_future_complete(n, f)
        r = f.result().get_result_async()
        rclpy.spin_until_future_complete(n, r, timeout_sec=900.0)
        st['phase'] = None
        return r.result().result if r.done() else None

    def walk(x, y, tol):
        # B5b S4 run 1: arriving from the side, the final turn to yaw 0 on wet sand moved the robot ~0.1 m and
        # the item ended outside the reach band twice; a waypoint 0.3 m behind makes the robot arrive facing +x.
        subprocess.run([sys.executable, str(TOOLS / 'goto.py'), f'{x - 0.79},{y}', f'{x - 0.49},{y}', '--yaw', '0',
                        '--tol', str(tol),
                        '--seed', str(a.seed), '--accel', '0.3', '--ang-accel', '0.5'], check=False)

    worst, failed, stored = (0.0, ''), [], []

    def landed(sid, comp, when):
        """B5b S4: the item's centre in its compartment frame (rim centre, base_link axes), as store_drop_test.py.
        plan_store's centre placement plans it within 0.4 cm of (0, 0) in x and y (S1 planned_rel_cm), so the
        landing error is taken against (0, 0) and checked per axis against landing_margin_m."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < 3.0:
            rclpy.spin_once(n, timeout_sec=0.1)
        if sid not in st['poses'] or st['base'] is None:
            print(f'{sid} landed in {comp} ({when}): no pose', flush=True)
            failed.append(f'{sid} landing {when}')
            return
        b = st['base'].pose.pose
        bR = quat_to_R(b.orientation.x, b.orientation.y, b.orientation.z, b.orientation.w)
        M = data.oMf[model.getFrameId(f'basket_{comp}_link')]
        c = np.array([b.position.x, b.position.y, b.position.z]) + bR @ M.translation
        rel = (bR @ M.rotation).T @ (st['poses'][sid] - c)
        ok = abs(rel[0]) <= margin[0] and abs(rel[1]) <= margin[1]
        print(f'{sid} landed in {comp} ({when}): rel_cm {(rel * 100).round(1).tolist()}, margin_cm '
              f'{[round(m * 100, 1) for m in margin]}: {"ok" if ok else "OUTSIDE"}', flush=True)
        if not ok:
            failed.append(f'{sid} landing {when}')

    def report(sid):
        nonlocal worst
        for ph, (gap, j) in st['gaps'].items():
            end = ' '.join(f'{v:+.3f}' for v in st['end'].get(ph, []))
            print(f'  {ph:22s} max gap {gap:.4f} rad ({j})  at end [{end}]', flush=True)
            if ph.startswith(('store', 'unload', 'place')) and gap > worst[0]:
                worst = (gap, f'{sid} {ph} {j}')

    def unload_all():
        from mapping.ground_truth import load as load_map
        from task.node import bin_rims, pose_stamped
        fsm = yaml.safe_load(open(REPO / 'config' / 'task.yaml'))['task']['fsm']
        objects = yaml.safe_load(open(REPO / 'sim' / 'common' / 'objects.yaml'))
        bins = bin_rims(REPO, load_map(str(REPO), a.seed)[0], objects, fsm)
        nav = ActionClient(n, NavigateToPose, '/nav/goal')
        nav.wait_for_server(timeout_sec=60.0)
        for sid, iid, comp in list(stored):
            bx, by, rim, byaw = bins[comp]
            out = lambda d: (bx + d * math.cos(byaw), by + d * math.sin(byaw))  # noqa: E731
            face = byaw + math.pi
            sx, sy = out(fsm['bin_standoff_m'])
            r = run(nav, NavigateToPose.Goal(pose=pose_stamped('map', sx, sy, yaw=face)), 'nav')
            print(f'{sid} nav to the {comp} bin standoff: {"done" if r is not None else "no result"}', flush=True)
            landed(sid, comp, 'before unload')
            px, py = out(fsm['bin_place_distance_m'])
            subprocess.run([sys.executable, str(TOOLS / 'goto.py'), f'{sx},{sy}', f'{px},{py}', '--yaw', str(face),
                            '--tol', '0.03', '--speed', '0.15', '--seed', str(a.seed), '--accel', '0.3',
                            '--ang-accel', '0.5'], check=False)
            st['gaps'], st['end'] = {}, {}
            t0 = time.monotonic()
            while time.monotonic() - t0 < 1.0:               # drain the poses queued while goto.py ran
                rclpy.spin_once(n, timeout_sec=0.1)
            if st['base'] is not None:                       # B5b S4 run 4: goto.py timed out 0.95 m out
                b = st['base'].pose.pose
                yaw = math.atan2(2 * (b.orientation.w * b.orientation.z + b.orientation.x * b.orientation.y),
                                 1 - 2 * (b.orientation.y ** 2 + b.orientation.z ** 2))
                d = np.array([b.position.x - bx, b.position.y - by])
                along, lateral = d @ [math.cos(byaw), math.sin(byaw)], d @ [-math.sin(byaw), math.cos(byaw)]
                print(f'{sid} at the {comp} bin: along {along:.3f} m (planned {fsm["bin_place_distance_m"]}), '
                      f'lateral {lateral:+.3f} m, heading {math.remainder(yaw - face, 2 * math.pi):+.3f} rad', flush=True)
            r = run(place, PlaceObject.Goal(target=PlaceObject.Goal.BIN, item_id=iid, bin_id=comp,
                                            pose=pose_stamped('map', bx, by, rim, byaw)), 'unload')
            print(f'{sid} unload into {comp}: {r.success if r else None} {r.message if r else "no result"}', flush=True)
            if not r or not r.success:
                failed.append(f'{sid} unload')
            report(sid)
            stored.remove((sid, iid, comp))
            subprocess.run([sys.executable, str(TOOLS / 'goto.py'), f'{sx},{sy}', '--seed', str(a.seed),
                            '--accel', '0.3', '--ang-accel', '0.5'], check=False)

    steps = a.plan.split(',') if a.plan else [f'{sid}:{a.bin}' for sid in a.items.split(',')]
    for step in steps:
        if step == 'unload':
            unload_all()
            continue
        sid, comp = step.split(':')
        x, y = scene_pose(a.seed, sid)
        walk(x, y, 0.04)                                     # B5b S4 run 2: 0.12 m left both items 0.57-0.60 m ahead
        t0 = time.monotonic()
        while sid not in st['wm'] and time.monotonic() - t0 < 30.0:
            rclpy.spin_once(n, timeout_sec=0.1)
        it = st['wm'].get(sid)
        if it is None:
            print(f'{sid}: not in the world model', flush=True)
            continue
        st['gaps'], st['end'] = {}, {}
        g = PickObject.Goal(id=it.id, pose=it.pose, size=it.size)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 1.0:                   # fresh base pose (the walk ran in a subprocess)
            rclpy.spin_once(n, timeout_sec=0.1)
        st['snaps'], st['track'] = [('goal', in_tips(sid))], sid
        r = run(pick, g, 'pick')
        st['snaps'].append(('end', in_tips(sid)))
        pick_report(sid, g)
        print(f'{sid} ({it.id}) pick: {r.success if r else None} {r.message if r else "no result"}', flush=True)
        if r and not r.success and r.message.startswith('outside_envelope'):
            # B5.4 runs 5, 6: the 0.12 m walk tolerance is wider than the reach band (reach_x_m 0.42-0.56):
            # the jar ended 0.562 m ahead. Walk again, tighter, and pick once more.
            walk(x, y, 0.04)
            rclpy.spin_once(n, timeout_sec=1.0)
            it = st['wm'].get(sid, it)
            r = run(pick, PickObject.Goal(id=it.id, pose=it.pose, size=it.size), 'pick')
            print(f'{sid} ({it.id}) pick after a tighter walk: {r.success if r else None} '
                  f'{r.message if r else "no result"}', flush=True)
        if not r or not r.success:
            failed.append(f'{sid} pick')
            continue
        r = run(place, PlaceObject.Goal(target=PlaceObject.Goal.COMPARTMENT, item_id=it.id, bin_id=comp), 'store')
        print(f'{sid} store: {r.success if r else None} {r.message if r else "no result"}', flush=True)
        if not r or not r.success:
            failed.append(f'{sid} store')
            # B5b S4 runs 4, 5: still held at `show`, the arm then walked to the bin and its unload check (right)
            # refused the extended arm at the bin wall. Put it back like the task (fsm state_put_back).
            r = run(place, PlaceObject.Goal(target=PlaceObject.Goal.GROUND, item_id=it.id, bin_id=''), 'put_back')
            print(f'{sid} put back: {r.success if r else None} {r.message if r else "no result"}', flush=True)
        else:
            stored.append((sid, it.id, comp))
            landed(sid, comp, 'after store')
        report(sid)
    print(f'RESULT worst store/unload gap {worst[0]:.4f} rad at {worst[1]} (limit {a.gap_limit}), failed: '
          f'{failed or "none"}: {"PASS" if worst[0] <= a.gap_limit and not failed else "FAIL"}', flush=True)
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
