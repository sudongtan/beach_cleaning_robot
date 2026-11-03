#!/usr/bin/env python3
"""Do stored items stay in the basket? (A2b basket issues from `a2b_full1`.)

Run on the full skeleton with the task, perception, world model and grasping stopped, e.g.
`TOOL=store_drop_test.py nav_walk_test.sh 101 store --rounds 3`. Per round, both compartments
are filled to the limit with the store motion of the grasping node, except that the release
spawns the item where a held item hangs (top hover_above_item_m below the jaw tips):
  --mode fixed:   rest -> up -> over_<comp> -> open -> release -> up -> rest (the store until
                  2026-11-03); item long side across the jaw axis, +-1 cm / +-10 deg
  --mode planned: grasping.arm.plan_store with the true poses of the items already in the basket
                  (as the grasping node), the item at a random yaw to the jaws (as after a pick)
After each drop the item settles for 3 s; then the robot walks out and back
with on-the-spot turns (through /cmd_vel_nav and the safety supervisor), and every item is
checked again. The arm only moves while /safety/state is ok (as the grasping node).

    store_drop_test.py [--mode planned] [--rounds 3] [--walk 2.0] [--out results/store_drop/run.json]
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Twist
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node as GzNode
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import String

import physics_grasp_test as pg
from physics_grasp_test import REPO, TIPS, arm_pose, item_height, item_model_sdf, rc
from grasping.arm import Arm, plan_store

FILL = {'recyclable': ['pet_bottle_500', 'food_tin', 'can_330', 'glass_bottle_330', 'glass_jar'],
        'residual': ['food_container', 'poly_cup', 'cigarette_pack', 'food_container', 'poly_cup']}
SETTLE_S = 3.0
ACCEL, ANG_ACCEL = 0.3, 0.5        # m/s^2, rad/s^2 (goto.py --accel / --ang-accel)
FALL_TILT_DEG = 30.0


class Rig(pg.Rig):
    """physics_grasp_test's rig without its /cmd_vel publisher: drives via /cmd_vel_nav."""

    def __init__(self, world):
        Node.__init__(self, 'store_drop_test')
        self.world = world
        self.base, self.js, self.poses, self.poses_t, self.safety = None, {}, {}, time.monotonic(), None
        self.v = (0.0, 0.0)
        self.create_subscription(Odometry, '/sim/ground_truth/pose', lambda m: setattr(self, 'base', m), 10)
        self.create_subscription(JointState, '/joint_states', self.on_js, 10)
        self.create_subscription(String, '/safety/state', lambda m: setattr(self, 'safety', m.data),
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.cmd = self.create_publisher(Twist, '/cmd_vel_nav', 10)
        self.gz = GzNode()
        self.gz.subscribe(Pose_V, f'/world/{world}/pose/info', self.on_poses)
        self.model = rc.build_model('custom')
        self.data = self.model.createData()

    def safe(self):
        return self.safety == 'ok'

    def wait_safe(self):
        while not self.safe():
            self.spin(0.2)

    def arm(self, q, secs=2.5):
        self.wait_safe()
        arm_pose.move_arm(self, q, secs)

    def fingers(self, v):
        self.wait_safe()
        arm_pose.set_fingers(self, v)

    def drive(self, vx, wz, until, limit_s):
        """Ramp to (vx, wz) (ACCEL, ANG_ACCEL, as Nav2's velocity smoother: abrupt walk/turn
        changes tip the robot, A1c), hold until until() or limit_s of sim time, ramp to zero."""
        t0 = self.sim_now()
        while not until() and self.sim_now() - t0 < limit_s:
            self.ramp(vx, wz)
        while self.v != (0.0, 0.0):
            self.ramp(0.0, 0.0)
        for _ in range(10):
            self.ramp(0.0, 0.0)

    def ramp(self, vx, wz, dt=0.05):
        if self.tilt() > FALL_TILT_DEG:
            raise RuntimeError(f'FALL: tilt {self.tilt():.0f} deg')
        step = lambda cur, want, a: cur + max(-a * dt, min(a * dt, want - cur))   # noqa: E731
        self.v = (step(self.v[0], vx, ACCEL), step(self.v[1], wz, ANG_ACCEL))
        m = Twist()
        m.linear.x, m.angular.z = self.v
        self.cmd.publish(m)
        self.spin(dt)

    def tilt(self):
        _, R = self.base_pose()
        return math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2]))))

    def yaw(self):
        _, R = self.base_pose()
        return math.atan2(R[1, 0], R[0, 0])


def in_compartment(rig, comp, name):
    """Item centre in the compartment frame (rim centre, base_link axes) and whether it is inside."""
    if name not in rig.poses:
        return None
    p, _ = rig.poses[name]
    c, cR = rig.frame_world(f'basket_{comp}_link')
    rel = cR.T @ (p - c)
    length, width, height, wall = 0.34, 0.30, 0.15, 0.003
    inside = abs(rel[0]) <= length / 2 - wall and abs(rel[1]) <= width / 4 and -height <= rel[2] <= 0.0
    return {'rel_cm': (rel * 100).round(1).tolist(), 'inside': bool(inside), 'world_z_cm': round(float(p[2]) * 100, 1)}


def held_spec(spec, hover, yaw):
    """An item held after an attach-mode pick (plan_store's `held`): centre below the jaw tips
    (tool x points down), turned by yaw about the vertical relative to the jaws (tool y)."""
    size = spec['collision'][0]['size']
    down, jaw = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    side = np.cross(down, jaw)
    ix = math.cos(yaw) * jaw + math.sin(yaw) * side
    iz = -down
    return {'p': down * (hover + size[2] / 2), 'R': np.column_stack([ix, np.cross(iz, ix), iz]), 'size': size}


def basket_items(rig, items, objects):
    """plan_store's `others`: the dropped items in base_link (true poses)."""
    bp, bR = rig.base_pose()
    out = []
    for it in items:
        if 'name' not in it or it['name'] not in rig.poses:
            continue
        p, R = rig.poses[it['name']]
        size = np.asarray(objects['trash'][it['model']]['collision'][0]['size'], dtype=float)
        Rb = bR.T @ R
        c = bR.T @ (p - bp) + Rb @ np.array([0.0, 0.0, size[2] / 2])     # model origin is the bottom
        ext = np.abs(Rb) @ size
        out.append({'centre': c, 'half': ext[:2] / 2, 'top': c[2] + ext[2] / 2})
    return out


def walk_loop(rig, dist):
    """Out dist m, turn on the spot by pi, back, turn by pi (the search pattern's turns)."""
    for _ in range(2):
        p0, _ = rig.base_pose()
        rig.drive(0.3, 0.0, lambda: np.linalg.norm(rig.base_pose()[0][:2] - p0[:2]) >= dist, 40.0)
        turned = [0.0, rig.yaw()]

        def done():
            y = rig.yaw()
            turned[0] += abs(math.remainder(y - turned[1], 2 * math.pi))
            turned[1] = y
            return turned[0] >= math.pi
        rig.drive(0.0, 0.5, done, 40.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['fixed', 'planned'], default='planned')
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--walk', type=float, default=2.0, help='out-and-back distance, m')
    ap.add_argument('--world', default='beach')
    ap.add_argument('--seed', type=int, default=0, help='scene seed (unused; for nav_walk_test.sh)')
    ap.add_argument('--out', default=str(REPO / 'results' / 'store_drop' / 'run.json'))
    a = ap.parse_args()
    objects = yaml.safe_load(open(REPO / 'sim' / 'common' / 'objects.yaml'))
    gcfg = yaml.safe_load(open(REPO / 'config' / 'grasping.yaml'))['grasping']
    hover = gcfg['hover_above_item_m']
    model_arm = Arm('custom', gcfg['ik_restarts'], gcfg['ik_tolerance_m']) if a.mode == 'planned' else None
    poses = arm_pose.load_poses('custom')
    closed, opened = arm_pose.finger_limits('custom')
    rclpy.init()
    rig = Rig(a.world)
    while rig.base is None or not rig.js or not rig.poses:
        rclpy.spin_once(rig, timeout_sec=0.2)
    rng = np.random.default_rng(0)
    out = {'rounds': []}
    for r in range(a.rounds):
        items = []
        for comp, models in FILL.items():
            for model in models:
                spec = objects['trash'][model]
                rig.arm(poses['up']['arm'])
                rig.arm(poses[f'over_{comp}']['arm'])
                plan = None
                if a.mode == 'planned':
                    held = held_spec(spec, hover, rng.uniform(-math.pi, math.pi))
                    others = basket_items(rig, items, objects)
                    ok, why, plan, info = plan_store(model_arm, comp, held, others, gcfg)
                    if not ok:
                        print(f'round {r} {comp:10s} {model:18s} plan: {why}', flush=True)
                        items.append({'model': model, 'comp': comp, 'refused': why})
                        rig.arm(poses['up']['arm'])
                        rig.arm(poses['rest']['arm'])
                        continue
                    rig.arm(plan['above'])
                    rig.arm(plan['release'])
                else:
                    rig.fingers(opened)
                rig.spin(0.5)
                tips, tR = rig.frame_world(TIPS)
                if plan:
                    c = tips + tR @ held['p']
                    Ri = tR @ held['R']
                    yaw = math.atan2(Ri[1, 0], Ri[0, 0])             # the model's x axis
                    xy, z = c[:2], c[2] - item_height(spec) / 2
                else:
                    jaw = tR[:, 1]
                    yaw = math.atan2(jaw[1], jaw[0]) - math.pi / 2 + math.radians(rng.uniform(-10, 10))
                    xy = tips[:2] + rng.uniform(-0.01, 0.01, 2)
                    z = tips[2] - hover - item_height(spec)          # model origin = bottom of the item
                name = f'drop_{r}_{comp}_{len(items):02d}'
                sdf = item_model_sdf(name, spec, xy[0], xy[1], yaw, objects['classes'][spec['material']])
                sdf = _set_z(sdf, z)
                if not rig.spawn(name, sdf):
                    print(f'{name}: spawn failed', flush=True)
                    continue
                rig.spin(SETTLE_S)
                st = in_compartment(rig, comp, name)
                items.append({'name': name, 'model': model, 'comp': comp, 'drop': st})
                print(f'round {r} {comp:10s} {model:18s} after drop: {st}', flush=True)
                if plan:
                    rig.arm(plan['above'])
                else:
                    rig.fingers(closed)
                rig.arm(poses['up']['arm'])
                rig.arm(poses['rest']['arm'])
        walk_loop(rig, a.walk)
        rig.spin(2.0)
        for it in [i for i in items if 'name' in i]:
            it['walk'] = in_compartment(rig, it['comp'], it['name'])
            print(f'round {r} {it["comp"]:10s} {it["model"]:18s} after walk: {it["walk"]}', flush=True)
        out['rounds'].append(items)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
        for it in [i for i in items if 'name' in i]:
            rig.remove(it['name'])
    refused = [it['model'] for rd in out['rounds'] for it in rd if 'refused' in it]
    drops = [it for rd in out['rounds'] for it in rd if 'name' in it]
    lost_drop = [it['model'] for it in drops if not (it['drop'] or {}).get('inside')]
    lost_walk = [it['model'] for it in drops
                 if (it['drop'] or {}).get('inside') and not (it['walk'] or {}).get('inside')]
    print(f'RESULT items={len(drops)} out_after_drop={len(lost_drop)} {lost_drop} '
          f'out_after_walk={len(lost_walk)} {lost_walk} refused={refused}', flush=True)
    rclpy.try_shutdown()


def _set_z(sdf, z):
    """Replace the z of the model pose (item_model_sdf puts the model 5 mm above z = 0)."""
    head, rest = sdf.split('<pose>', 1)
    vals, tail = rest.split('</pose>', 1)
    v = vals.split()
    v[2] = f'{z:.4f}'
    return f'{head}<pose>{" ".join(v)}</pose>{tail}'


if __name__ == '__main__':
    main()
