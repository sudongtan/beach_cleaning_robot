#!/usr/bin/env python3
"""Geometric reach check for the arm (A0): can the gripper reach the ground in front of the
front feet and both basket compartments?

Builds the robot from the xacro with Pinocchio, puts the legs in the controller's stand pose,
places the ground under the feet (or at a given body height), and solves position IK with the
gripper's approach axis (tool x) pointing straight down, or tilted up to --max-tilt deg.
Self-collision is not checked here (check the poses in RViz).

  ros2 run beach_description reach_check.py [--variant custom|stock] [--body-height 0.285] [--max-tilt 45]
"""
import argparse
import math
import subprocess

import numpy as np
import pinocchio as pin
from ament_index_python.packages import get_package_share_directory

STAND = {'hip': 0.0, 'thigh': 0.67, 'calf': -1.3}   # basic_quadruped_controller default stand_pos
ARM = ['arm_waist', 'arm_shoulder', 'arm_elbow', 'arm_forearm_roll', 'arm_wrist_angle', 'arm_wrist_rotate']
TCP = 'arm_ee_gripper_link'


def build_model(variant='custom'):
    xacro = get_package_share_directory('beach_description') + '/urdf/beach_robot.urdf.xacro'
    urdf = subprocess.run(['xacro', xacro, f'variant:={variant}'], check=True, capture_output=True, text=True).stdout
    return pin.buildModelFromXML(urdf)


def stand_config(model):
    q = pin.neutral(model)
    for leg in ['FR', 'FL', 'RR', 'RL']:
        for part, val in STAND.items():
            j = model.getJointId(f'{leg}_{part}_joint')
            q[model.joints[j].idx_q] = val
    return q


def frame_pos(model, data, q, name):
    pin.framesForwardKinematics(model, data, q)
    return data.oMf[model.getFrameId(name)].translation.copy()


def solve(model, data, q0, target, down_tilt_max, rng, restarts=40, iters=300, frame=TCP):
    """Position IK of `frame` (default the TCP) + keep tool x within down_tilt_max of straight
    down. Returns (ok, q, pos_err, tilt_deg)."""
    fid = model.getFrameId(frame)
    idx = [model.joints[model.getJointId(n)].idx_q for n in ARM]
    vidx = [model.joints[model.getJointId(n)].idx_v for n in ARM]
    lo = model.lowerPositionLimit[idx]
    hi = model.upperPositionLimit[idx]
    down = np.array([0.0, 0.0, -1.0])
    best = (False, None, 1e9, 180.0)
    for r in range(restarts):
        q = q0.copy()
        q[idx] = rng.uniform(lo, hi) if r else 0.0
        for _ in range(iters):
            pin.framesForwardKinematics(model, data, q)
            M = data.oMf[fid]
            a = M.rotation[:, 0]
            e_pos = target - M.translation
            tilt = math.degrees(math.acos(np.clip(a @ down, -1, 1)))
            # orientation task only beyond the allowed tilt (pull towards straight down)
            w_rot = 1.0 if tilt > down_tilt_max else 0.0
            e_rot = np.cross(a, down) * w_rot
            if np.linalg.norm(e_pos) < 0.003 and tilt <= down_tilt_max + 0.5:
                break
            J = pin.computeFrameJacobian(model, data, q, fid, pin.LOCAL_WORLD_ALIGNED)[:, vidx]
            Jw = np.vstack([J[:3], 0.3 * w_rot * J[3:]])
            e = np.concatenate([e_pos, 0.3 * e_rot])
            dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + 1e-4 * np.eye(6), e)
            q[idx] = np.clip(q[idx] + np.clip(dq, -0.2, 0.2), lo, hi)
        pin.framesForwardKinematics(model, data, q)
        M = data.oMf[fid]
        err = np.linalg.norm(target - M.translation)
        tilt = math.degrees(math.acos(np.clip(M.rotation[:, 0] @ down, -1, 1)))
        ok = err < 0.005 and tilt <= down_tilt_max + 0.5
        if ok and (not best[0] or tilt < best[3]):
            best = (True, q.copy(), err, tilt)
            if tilt < 1.0:
                break
        elif not best[0] and err < best[2]:
            best = (False, q.copy(), err, tilt)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--body-height', type=float, default=None,
                    help='height of base_link above ground (default: from the stand pose, feet on the ground)')
    ap.add_argument('--max-tilt', type=float, default=45.0, help='allowed gripper tilt from vertical, deg')
    ap.add_argument('--tcp-clearance', type=float, default=None,
                    help='TCP height above the ground target, m (default: stock 0.02, custom 0.045 = '
                         'jaw tips 1.5 cm above ground)')
    ap.add_argument('--variant', choices=['stock', 'custom'], default='custom')
    args = ap.parse_args()
    if args.tcp_clearance is None:
        args.tcp_clearance = 0.045 if args.variant == 'custom' else 0.02

    model = build_model(args.variant)
    data = model.createData()
    q0 = stand_config(model)
    feet = {f: frame_pos(model, data, q0, f'{f}_foot') for f in ['FR', 'FL']}
    foot_x = max(p[0] for p in feet.values())
    foot_z = min(p[2] for p in feet.values()) - 0.022   # Go2 foot sphere radius ~0.022
    ground_z = -args.body_height if args.body_height is not None else foot_z
    arm_base = frame_pos(model, data, q0, 'arm_base_link')
    print(f'front feet x = {foot_x:.3f} m, base_link height = {-ground_z:.3f} m '
          f'({"given" if args.body_height else "from stand pose"}), arm_base_link at x={arm_base[0]:.3f} '
          f'z={arm_base[2] - ground_z:.3f} m above ground')

    rng = np.random.default_rng(0)
    print(f'\nGround targets (TCP {args.tcp_clearance*100:.0f} cm above ground), tool pointing down:')
    print('  dist in front of feet  y=0            y=+-0.10')
    for d in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        cells = []
        for y in [0.0, 0.10]:
            t = np.array([foot_x + d, y, ground_z + args.tcp_clearance])
            ok, _, err, tilt = solve(model, data, q0, t, 2.0, rng)
            if ok:
                cells.append('vertical   ')
            else:
                ok2, _, err2, tilt2 = solve(model, data, q0, t, args.max_tilt, rng)
                cells.append(f'tilt {tilt2:4.1f} deg' if ok2 else f'NO ({err2*100:.0f} cm)')
        print(f'  {d:.2f} m                 {cells[0]:13s}  {cells[1]}')

    print('\nBasket compartments (TCP 5 cm above the rim centre), tool pointing down:')
    for comp in ['basket_recyclable_link', 'basket_residual_link']:
        t = frame_pos(model, data, q0, comp) + np.array([0, 0, 0.05])
        ok, q, err, tilt = solve(model, data, q0, t, 2.0, rng)
        if not ok:
            ok, q, err, tilt = solve(model, data, q0, t, args.max_tilt, rng)
        joints = ' '.join(f'{q[model.joints[model.getJointId(n)].idx_q]:+.2f}' for n in ARM) if q is not None else ''
        print(f'  {comp:24s} {"OK" if ok else "NO"}  tilt {tilt:4.1f} deg  err {err*1000:.1f} mm  q=[{joints}]')


if __name__ == '__main__':
    main()
