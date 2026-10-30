#!/usr/bin/env python3
"""Check the named arm poses in config/arm_poses.yaml (A0).

For each pose: TCP position (in base_link and above ground), gripper tilt from vertical, and
collisions of the arm with the robot body, basket and itself (mesh collision, Pinocchio/coal).
With --search-rest, also samples compact collision-free tucked poses.

  ros2 run beach_description arm_pose_check.py [--variant custom|stock] [--body-height 0.285]
      [--search-rest] [--legs]

--legs also checks the basket (and plate) against the legs over a sweep of leg joint angles
(hip -0.4..0.4, thigh 0..1.6, calf -2.6..-0.9 rad), i.e. whether the legs can hit the basket.
"""
import argparse
import math
import subprocess

import numpy as np
import pinocchio as pin
import yaml
from ament_index_python.packages import get_package_share_directory

SHARE = get_package_share_directory('beach_description')
ARM = ['arm_waist', 'arm_shoulder', 'arm_elbow', 'arm_forearm_roll', 'arm_wrist_angle', 'arm_wrist_rotate']
FINGERS = ['arm_left_finger', 'arm_right_finger']
STAND = {'hip': 0.0, 'thigh': 0.67, 'calf': -1.3}
TCP = 'arm_ee_gripper_link'
CLEARANCE = 0.01  # m, minimum distance counted as "no collision"


def poses_file(variant):
    return SHARE + ('/config/arm_poses_custom.yaml' if variant == 'custom' else '/config/arm_poses.yaml')


def load(variant='custom'):
    urdf = subprocess.run(['xacro', SHARE + '/urdf/beach_robot.urdf.xacro', f'variant:={variant}'],
                          check=True, capture_output=True, text=True).stdout.replace('file://', '')
    model = pin.buildModelFromXML(urdf)
    geom = pin.buildGeomFromUrdfString(model, urdf, pin.GeometryType.COLLISION)
    arm_links = {f.name for f in model.frames if f.name.startswith('arm_') and f.name != 'arm_plate_link'}
    names = [g.name for g in geom.geometryObjects]

    def link_of(gname):
        return gname.rsplit('_', 1)[0]  # pinocchio names geometries <link>_<index>

    # arm (except its base, which sits on the plate) vs everything not on the arm; plus arm self-collision
    # between links at least 2 joints apart
    chain = ['arm_base_link', 'arm_shoulder_link', 'arm_upper_arm_link', 'arm_upper_forearm_link',
             'arm_lower_forearm_link', 'arm_wrist_link', 'arm_gripper_link']
    hand = {'arm_gripper_link', 'arm_gripper_prop_link', 'arm_gripper_bar_link',
            'arm_left_finger_link', 'arm_right_finger_link'}

    def rank(link):
        return len(chain) - 1 if link in hand else chain.index(link)

    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            la, lb = link_of(a), link_of(names[j])
            in_a, in_b = la in arm_links, lb in arm_links
            if in_a and in_b:
                if abs(rank(la) - rank(lb)) >= 2:
                    geom.addCollisionPair(pin.CollisionPair(i, j))
            elif in_a != in_b:
                arm_l = la if in_a else lb
                if arm_l != 'arm_base_link':
                    geom.addCollisionPair(pin.CollisionPair(i, j))
    return model, geom


def config(model, arm_q, fingers=0.037):
    q = pin.neutral(model)
    for leg in ['FR', 'FL', 'RR', 'RL']:
        for part, val in STAND.items():
            q[model.joints[model.getJointId(f'{leg}_{part}_joint')].idx_q] = val
    for n, v in zip(ARM, arm_q):
        q[model.joints[model.getJointId(n)].idx_q] = v
    q[model.joints[model.getJointId('arm_left_finger')].idx_q] = fingers
    q[model.joints[model.getJointId('arm_right_finger')].idx_q] = -fingers
    return q


def collisions(model, data, geom, gdata, q):
    pin.computeDistances(model, data, geom, gdata, q)
    hits = []
    for k, cp in enumerate(geom.collisionPairs):
        d = gdata.distanceResults[k].min_distance
        if d < CLEARANCE:
            a = geom.geometryObjects[cp.first].name.rsplit('_', 1)[0]
            b = geom.geometryObjects[cp.second].name.rsplit('_', 1)[0]
            hits.append((a, b, d))
    return hits


def tcp(model, data, q):
    pin.framesForwardKinematics(model, data, q)
    M = data.oMf[model.getFrameId(TCP)]
    tilt = math.degrees(math.acos(np.clip(M.rotation[:, 0] @ np.array([0, 0, -1.0]), -1, 1)))
    return M.translation.copy(), tilt


def leg_clearance(model, variant):
    """Minimum distance between the legs and the basket/plate over a sweep of leg angles."""
    urdf = subprocess.run(['xacro', SHARE + '/urdf/beach_robot.urdf.xacro', f'variant:={variant}',
                           'basket_load_kg:=0.0'], check=True, capture_output=True, text=True).stdout
    geom = pin.buildGeomFromUrdfString(model, urdf.replace('file://', ''), pin.GeometryType.COLLISION)
    names = [g.name for g in geom.geometryObjects]
    body = [i for i, n in enumerate(names) if n.startswith(('basket_link', 'arm_plate_link'))]
    legs = [i for i, n in enumerate(names) if n[:3] in ('FR_', 'FL_', 'RR_', 'RL_')]
    for i in body:
        for j in legs:
            geom.addCollisionPair(pin.CollisionPair(min(i, j), max(i, j)))
    data, gdata = model.createData(), pin.GeometryData(geom)
    worst = (1.0, None)
    for hip in np.linspace(-0.4, 0.4, 5):
        for thigh in np.linspace(0.0, 1.6, 9):
            for calf in np.linspace(-2.6, -0.9, 6):
                q = pin.neutral(model)
                for leg in ['FR', 'FL', 'RR', 'RL']:
                    for part, val in (('hip', hip), ('thigh', thigh), ('calf', calf)):
                        q[model.joints[model.getJointId(f'{leg}_{part}_joint')].idx_q] = val
                pin.computeDistances(model, data, geom, gdata, q)
                for k, cp in enumerate(geom.collisionPairs):
                    d = gdata.distanceResults[k].min_distance
                    if d < worst[0]:
                        worst = (d, (names[cp.first].rsplit('_', 1)[0], names[cp.second].rsplit('_', 1)[0],
                                     round(hip, 2), round(thigh, 2), round(calf, 2)))
    print(f'\nlegs vs basket/plate over the leg sweep: min distance {worst[0]*1000:.0f} mm '
          f'({worst[1][0]} / {worst[1][1]} at hip {worst[1][2]}, thigh {worst[1][3]}, calf {worst[1][4]})'
          + ('  OK' if worst[0] >= CLEARANCE else '  COLLISION'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--body-height', type=float, default=0.285, help='base_link above ground, m')
    ap.add_argument('--search-rest', action='store_true')
    ap.add_argument('--legs', action='store_true')
    ap.add_argument('--variant', choices=['stock', 'custom'], default='custom')
    args = ap.parse_args()

    model, geom = load(args.variant)
    data, gdata = model.createData(), pin.GeometryData(geom)
    cfg = yaml.safe_load(open(poses_file(args.variant)))
    poses = cfg['poses']
    print(f'variant {args.variant}, poses from {poses_file(args.variant)}')
    print(f'{len(geom.collisionPairs)} collision pairs, body height {args.body_height} m\n')
    for name, p in poses.items():
        q = config(model, p['arm'], p.get('fingers', 0.037))
        pos, tilt = tcp(model, data, q)
        hits = collisions(model, data, geom, gdata, q)
        print(f'{name:16s} TCP base_link=({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})  '
              f'above ground {pos[2] + args.body_height:.3f} m  tilt {tilt:5.1f} deg  '
              f'{"OK" if not hits else "COLLISION " + ", ".join(f"{a}/{b} {d*1000:.0f}mm" for a, b, d in hits[:4])}')

    seq = cfg.get('test_sequence', [])
    if seq:
        print('\ntest sequence moves (joint-space interpolation, 30 samples each):')
        for a, b in zip(seq, seq[1:]):
            qa, qb = np.array(poses[a]['arm']), np.array(poses[b]['arm'])
            fa, fb = poses[a].get('fingers', 0.037), poses[b].get('fingers', 0.037)
            worst = None
            for s_ in np.linspace(0, 1, 30):
                hits = collisions(model, data, geom, gdata, config(model, qa + s_ * (qb - qa), fa + s_ * (fb - fa)))
                if hits and worst is None:
                    worst = (s_, hits)
            print(f'  {a:16s} -> {b:16s} ' + ('OK' if worst is None else
                  f'COLLISION at {worst[0]:.2f}: ' + ', '.join(f'{x}/{y} {d*1000:.0f}mm' for x, y, d in worst[1][:3])))

    if args.legs:
        leg_clearance(model, args.variant)

    if args.search_rest:
        rng = np.random.default_rng(1)
        best = []
        for _ in range(20000):
            q_arm = [0.0, rng.uniform(-1.9, 1.0), rng.uniform(-2.1, 1.6), 0.0, rng.uniform(-1.7, 2.1), 0.0]
            q = config(model, q_arm)
            pos, _ = tcp(model, data, q)
            # compact: TCP low above the plate, not in front of the head (camera), not over the basket
            if not (0.0 < pos[0] < 0.30 and pos[2] < 0.30):
                continue
            if collisions(model, data, geom, gdata, q):
                continue
            pin.framesForwardKinematics(model, data, q)
            top = max(data.oMf[model.getFrameId(f)].translation[2] for f in
                      ['arm_upper_arm_link', 'arm_upper_forearm_link', 'arm_wrist_link', TCP])
            best.append((top, q_arm, pos))
        best.sort(key=lambda b: b[0])
        print('\nlowest collision-free tucked poses (max link-origin height in base_link):')
        for top, q_arm, pos in best[:5]:
            joints = ', '.join(f'{v:+.2f}' for v in q_arm)
            print(f'  top {top:.3f} m  TCP ({pos[0]:+.3f}, {pos[2]:+.3f})  arm=[{joints}]')


if __name__ == '__main__':
    main()
