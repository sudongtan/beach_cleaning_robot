#!/usr/bin/env python3
"""Physics-grasp feasibility (A1c item 7): can the robot hold the A1 trash items by
contact forces alone (grasp_mode:=physics), or does A2 start with attach mode?

Run with the robot standing on flat ground (world:=empty), e.g. via
physics_grasp_run.sh. Per trial:
  1. IK (Pinocchio, robot model + current leg joints) for the jaw tips 5 mm above the
     ground in front of the robot, gripper pointing down
  2. spawn the item there with a random offset (+-1 cm, +-10 deg), its narrow side
     across the jaws
  3. open, pre-grasp 10 cm above, descend, close fully (the 5 N joint limit sets the
     squeeze), wait 0.5 s
  4. lift 20 cm, hold 5 s, turn the wrist +90 / -90 deg and back, walk 1 m
  5. classify from the item's true pose relative to arm_ee_gripper_link:
     held (drift < 2 cm), slipped (still off the ground), dropped (on the ground / away);
     invalid if the jaw tips stopped above the item's top after the descent (an arm-controller
     fault of the rig, not a grasp result; jaws resting ON the item still count as a failure)

    physics_grasp_test.py --items all --trials 10 --out results/physics_grasp/run.json
"""

import argparse
import json
import math
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pinocchio as pin
import rclpy
import yaml
from geometry_msgs.msg import Twist
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.entity_factory_pb2 import EntityFactory
from gz.msgs10.entity_pb2 import Entity
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node as GzNode
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parents[2]
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(REPO / 'packages' / 'beach_description' / 'scripts'))
sys.path.insert(0, str(REPO / 'sim' / 'gazebo'))
import arm_pose  # noqa: E402
import load_scene  # noqa: E402
import reach_check as rc  # noqa: E402

TIPS, TCP = 'arm_jaw_tips_link', 'arm_ee_gripper_link'
REACH_X = 0.48           # grasp point ahead of base_link (as ground_front)
TIP_CLEARANCE = 0.005    # jaw tips above the ground at the grasp
HELD_DRIFT = 0.02
DESCENT_MARGIN = 0.02    # jaw tips higher than item top + this after the descent: the arm didn't get down


def quat_to_R(x, y, z, w):
    return pin.Quaternion(w, x, y, z).toRotationMatrix()


class Rig(Node):
    def __init__(self, world, variant):
        super().__init__('physics_grasp_test')
        self.world = world
        self.base = None
        self.js = {}
        self.poses = {}
        self.poses_t = time.monotonic()
        self.create_subscription(Odometry, '/sim/ground_truth/pose', lambda m: setattr(self, 'base', m), 10)
        self.create_subscription(JointState, '/joint_states', self.on_js, 10)
        self.cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.gz = GzNode()
        self.gz.subscribe(Pose_V, f'/world/{world}/pose/info', self.on_poses)
        self.model = rc.build_model(variant)
        self.data = self.model.createData()

    def on_js(self, m):
        self.js.update(dict(zip(m.name, m.position)))

    def on_poses(self, msg):
        d = {}
        for p in msg.pose:
            if p.name not in d:
                d[p.name] = (np.array([p.position.x, p.position.y, p.position.z]),
                             quat_to_R(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w))
        self.poses = d
        self.poses_t = time.monotonic()

    def fresh(self, max_age=3.0):
        """Is the Gazebo pose stream alive? (A dead stream gives stale poses: nonsense drifts.)"""
        return time.monotonic() - self.poses_t < max_age

    def sim_now(self):
        st = self.base.header.stamp
        return st.sec + st.nanosec * 1e-9

    def spin(self, secs):
        """Wait `secs` of SIM time (ground-truth stamps), not wall time."""
        rclpy.spin_once(self, timeout_sec=0.02)
        t0 = self.sim_now()
        w0 = time.monotonic()
        while self.sim_now() - t0 < secs and time.monotonic() - w0 < 20 * secs + 5:
            rclpy.spin_once(self, timeout_sec=0.02)

    def base_pose(self):
        p = self.base.pose.pose
        return (np.array([p.position.x, p.position.y, p.position.z]),
                quat_to_R(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w))

    def q_now(self):
        q = pin.neutral(self.model)
        for name, val in self.js.items():
            if self.model.existJointName(name):
                j = self.model.joints[self.model.getJointId(name)]
                if j.nq == 1:
                    q[j.idx_q] = val
        return q

    def frame_world(self, frame, q=None):
        """World pose of a robot frame: ground-truth base pose x model FK."""
        q = self.q_now() if q is None else q
        pin.framesForwardKinematics(self.model, self.data, q)
        M = self.data.oMf[self.model.getFrameId(frame)]
        bp, bR = self.base_pose()
        return bp + bR @ M.translation, bR @ M.rotation

    def ik(self, target_world, q0):
        """Arm joints putting the jaw tips at target_world, tool pointing down."""
        bp, bR = self.base_pose()
        t = bR.T @ (target_world - bp)
        ok, q, err, tilt = rc.solve(self.model, self.data, q0, t, 5.0, np.random.default_rng(0),
                                    restarts=20, frame=TIPS)
        arm = [float(q[self.model.joints[self.model.getJointId(n)].idx_q]) for n in rc.ARM]
        return ok, arm, q, err, tilt

    def spawn(self, name, sdf_model, tries=5):
        """Create the item and wait until it shows up in the pose stream (retries: the create
        service can time out when the CPU is busy)."""
        req = EntityFactory()
        req.sdf = f'<sdf version="1.9">{sdf_model}</sdf>'
        req.allow_renaming = False
        for _ in range(tries):
            self.gz.request(f'/world/{self.world}/create', req, EntityFactory, Boolean, 15000)
            w0 = time.monotonic()
            while time.monotonic() - w0 < 10.0:
                rclpy.spin_once(self, timeout_sec=0.05)
                if name in self.poses:
                    return True
        return False

    def remove(self, name, tries=5):
        req = Entity()
        req.name = name
        req.type = Entity.MODEL
        for _ in range(tries):
            self.gz.request(f'/world/{self.world}/remove', req, Entity, Boolean, 15000)
            w0 = time.monotonic()
            while time.monotonic() - w0 < 10.0:
                rclpy.spin_once(self, timeout_sec=0.05)
                if name not in self.poses:
                    return True
        return False

    def walk(self, dist, speed=0.3):
        bp0, _ = self.base_pose()
        m = Twist()
        m.linear.x = speed
        t0 = time.monotonic()
        while np.linalg.norm(self.base_pose()[0][:2] - bp0[:2]) < dist and time.monotonic() - t0 < 30:
            self.cmd.publish(m)
            self.spin(0.05)
        for _ in range(10):
            self.cmd.publish(Twist())
            self.spin(0.05)


def item_height(spec):
    return max(c['pose'][2] + (c['size'][2] if c['type'] == 'box' else c['length']) / 2 for c in spec['collision'])


def item_model_sdf(name, spec, x, y, yaw, class_id):
    root = ET.Element('world')
    load_scene.object_model(root, name, spec, (x, y, yaw), 0.005, class_id, static=False)
    return ET.tostring(root.find('model'), encoding='unicode')


def rel_pose(rig, name):
    """Item position in the arm_ee_gripper_link frame (m), and its height above the ground."""
    if name not in rig.poses:
        return None, None
    ip, _ = rig.poses[name]
    gp, gR = rig.frame_world(TCP)
    return gR.T @ (ip - gp), ip[2]


def trial(rig, name, spec, class_id, rng, closed, opened, log):
    up = arm_pose.load_poses('custom')['up']['arm']
    arm_pose.move_arm(rig, up, 2.5)
    arm_pose.set_fingers(rig, opened)
    rig.spin(0.5)
    bp, bR = rig.base_pose()
    fwd = bR[:, 0].copy()
    fwd[2] = 0
    fwd /= np.linalg.norm(fwd)
    nominal = bp + REACH_X * fwd
    nominal[2] = TIP_CLEARANCE                       # flat ground at z = 0
    ok, arm_grasp, qg, err, tilt = rig.ik(nominal, rig.q_now())
    if not ok:
        return {'result': 'ik_failed', 'err_cm': err * 100}
    _, gR = rig.frame_world(TCP, qg)
    jaw_axis = gR[:, 1]                              # fingers slide along tool y
    item_yaw = math.atan2(jaw_axis[1], jaw_axis[0]) - math.pi / 2 + math.radians(rng.uniform(-10, 10))
    off = rng.uniform(-0.01, 0.01, 2)
    item_xy = nominal[:2] + off
    pre = nominal + np.array([0, 0, 0.10])
    _, arm_pre, _, _, _ = rig.ik(pre, qg)
    trial.count = getattr(trial, 'count', 0) + 1
    item = f'grasp_{name}_{trial.count:03d}'
    if not rig.spawn(item, item_model_sdf(item, spec, item_xy[0], item_xy[1], item_yaw, class_id)):
        return {'result': 'spawn_failed'}
    rig.spin(1.5)
    arm_pose.move_arm(rig, arm_pre, 2.5)
    arm_pose.move_arm(rig, arm_grasp, 2.0)
    rig.spin(0.3)
    tips, _ = rig.frame_world(TIPS)
    if tips[2] > item_height(spec) + DESCENT_MARGIN:
        arm_pose.move_arm(rig, arm_pre, 2.0)
        rig.remove(item)
        return {'result': 'invalid', 'reason': 'arm did not descend',
                'tips_above_ground_cm': round(float(tips[2]) * 100, 2)}
    arm_pose.set_fingers(rig, closed, 1.0)
    rig.spin(0.5)
    finger = rig.js.get('arm_left_finger', float('nan'))
    rel0, _ = rel_pose(rig, item)
    lift = nominal + np.array([0, 0, 0.20])
    _, arm_lift, ql, _, _ = rig.ik(lift, qg)
    arm_pose.move_arm(rig, arm_lift, 2.0)
    stages = {}

    def check(stage):
        if not rig.fresh():
            raise RuntimeError(f'Gazebo pose stream stale at stage {stage}: aborting (results would be invalid)')
        rel, z = rel_pose(rig, item)
        drift = float(np.linalg.norm(rel - rel0)) if rel is not None else None
        stages[stage] = {'drift_cm': None if drift is None else round(drift * 100, 1),
                         'item_z_cm': None if z is None else round(z * 100, 1)}
        return drift, z

    check('lift')
    rig.spin(5.0)
    check('hold_5s')
    for ang in (math.pi / 2, -math.pi / 2, 0.0):
        arm_lift[5] = ang
        arm_pose.move_arm(rig, arm_lift, 1.5)
    check('wrist_pm90')
    rig.walk(1.0)
    drift, z = check('walk_1m')
    if drift is None or z is None or z < 0.06 or drift > 0.15:
        result = 'dropped'
    elif drift < HELD_DRIFT:
        result = 'held'
    else:
        result = 'slipped'
    first_loss = next((s for s, v in stages.items() if v['drift_cm'] is None or v['drift_cm'] >= HELD_DRIFT * 100),
                      None)
    arm_pose.set_fingers(rig, opened)
    rig.spin(0.5)
    rig.remove(item)
    return {'result': result, 'first_loss_stage': first_loss, 'stages': stages,
            'offset_cm': (off * 100).round(2).tolist(), 'tips_above_ground_cm': round(float(tips[2]) * 100, 2),
            'finger_after_close_m': round(finger, 4), 'ik_tilt_deg': round(float(tilt), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--items', default='all', help='comma-separated trash models, or all')
    ap.add_argument('--trials', type=int, default=10)
    ap.add_argument('--world', default='empty')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--label', default='baseline', help='name of the configuration (e.g. jaw_mu_1.5)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    objects = yaml.safe_load(open(REPO / 'sim' / 'common' / 'objects.yaml'))
    names = sorted(objects['trash']) if a.items == 'all' else a.items.split(',')
    closed, opened = arm_pose.finger_limits('custom')
    rclpy.init()
    rig = Rig(a.world, 'custom')
    while rig.base is None or not rig.js or not rig.poses:
        rclpy.spin_once(rig, timeout_sec=0.2)
    rng = np.random.default_rng(a.seed)
    out = {'label': a.label, 'variant': 'custom', 'world': a.world, 'trials_per_item': a.trials,
           'finger_effort_limit_N': 5.0, 'items': {}}
    for name in names:
        spec = objects['trash'][name]
        res = []
        for k in range(a.trials):
            r = trial(rig, name, spec, objects['classes'][spec['material']], rng, closed, opened, None)
            res.append(r)
            drifts = ' '.join(f'{s}:{v["drift_cm"]}' for s, v in r.get('stages', {}).items())
            print(f'{name:18s} trial {k}: {r["result"]:8s} drift cm {drifts}', flush=True)
        counts = {c: sum(r['result'] == c for r in res)
                  for c in ('held', 'slipped', 'dropped', 'invalid', 'ik_failed', 'spawn_failed')}
        out['items'][name] = {'counts': counts, 'trials': res}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
    print('\n| item | held | slipped | dropped | invalid |\n|---|---|---|---|---|')
    for name, v in out['items'].items():
        c = v['counts']
        print(f'| {name} | {c["held"]} | {c["slipped"]} | {c["dropped"]} | {c["invalid"]} |')
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
