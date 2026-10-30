"""Arm kinematics and grasp planning, plain Python + Pinocchio (no ROS nodes).

The robot model is the beach_description xacro of the configured variant, with the legs
in the leg controller's stand pose; targets are in base_link. IK is position IK of a
frame (default the jaw tips) with the tool's approach axis (tool x) kept within a tilt of
straight down; adapted from A0's beach_description/scripts/reach_check.py. Self-collision
is not checked (the named poses in arm_poses_custom.yaml were checked in A0).
"""

import math
import subprocess

import numpy as np
import pinocchio as pin
import yaml
from ament_index_python.packages import get_package_share_directory

STAND = {'hip': 0.0, 'thigh': 0.67, 'calf': -1.3}   # basic_quadruped_controller stand_pos
ARM = ['arm_waist', 'arm_shoulder', 'arm_elbow', 'arm_forearm_roll', 'arm_wrist_angle', 'arm_wrist_rotate']
TCP, TIPS = 'arm_ee_gripper_link', 'arm_jaw_tips_link'
DOWN = np.array([0.0, 0.0, -1.0])


class Arm:

    def __init__(self, variant='custom', restarts=30, tolerance=0.005, seed=0):
        share = get_package_share_directory('beach_description')
        urdf = subprocess.run(['xacro', share + '/urdf/beach_robot.urdf.xacro', f'variant:={variant}'],
                              check=True, capture_output=True, text=True).stdout
        self.model = pin.buildModelFromXML(urdf)
        self.data = self.model.createData()
        m = self.model
        self.q0 = pin.neutral(m)
        for leg in ('FR', 'FL', 'RR', 'RL'):
            for part, val in STAND.items():
                self.q0[m.joints[m.getJointId(f'{leg}_{part}_joint')].idx_q] = val
        self.idx = [m.joints[m.getJointId(n)].idx_q for n in ARM]
        self.vidx = [m.joints[m.getJointId(n)].idx_v for n in ARM]
        self.lo, self.hi = m.lowerPositionLimit[self.idx], m.upperPositionLimit[self.idx]
        f = m.joints[m.getJointId('arm_left_finger')].idx_q
        self.finger_closed, self.finger_open = float(m.lowerPositionLimit[f]), float(m.upperPositionLimit[f])
        self.restarts, self.tol = restarts, tolerance
        self.rng = np.random.default_rng(seed)
        self.poses = yaml.safe_load(open(f'{share}/config/arm_poses{"_custom" if variant == "custom" else ""}.yaml'))
        self.poses = self.poses['poses']

    def full(self, arm_q):
        q = self.q0.copy()
        q[self.idx] = arm_q
        return q

    def fk(self, arm_q, frame=TCP):
        """(position, tool x axis) of a frame in base_link."""
        pin.framesForwardKinematics(self.model, self.data, self.full(arm_q))
        M = self.data.oMf[self.model.getFrameId(frame)]
        return M.translation.copy(), M.rotation[:, 0].copy()

    def frame(self, name):
        """Position of a robot frame (e.g. basket_recyclable_link) in base_link."""
        pin.framesForwardKinematics(self.model, self.data, self.q0)
        return self.data.oMf[self.model.getFrameId(name)].translation.copy()

    def ik(self, target, max_tilt_deg, frame=TIPS, seed_q=None, iters=300):
        """Returns (ok, arm_q, position error m, tilt deg). First try starts at seed_q (or zeros)."""
        m, d = self.model, self.data
        fid = m.getFrameId(frame)
        target = np.asarray(target, dtype=float)
        best = (False, None, 1e9, 180.0)
        for r in range(self.restarts):
            q = self.q0.copy()
            q[self.idx] = self.rng.uniform(self.lo, self.hi) if r else (0.0 if seed_q is None else seed_q)
            for _ in range(iters):
                pin.framesForwardKinematics(m, d, q)
                M = d.oMf[fid]
                a = M.rotation[:, 0]
                e_pos = target - M.translation
                tilt = math.degrees(math.acos(np.clip(a @ DOWN, -1, 1)))
                w_rot = 1.0 if tilt > max_tilt_deg else 0.0      # pull towards down only beyond the limit
                if np.linalg.norm(e_pos) < self.tol * 0.6 and tilt <= max_tilt_deg + 0.5:
                    break
                J = pin.computeFrameJacobian(m, d, q, fid, pin.LOCAL_WORLD_ALIGNED)[:, self.vidx]
                Jw = np.vstack([J[:3], 0.3 * w_rot * J[3:]])
                e = np.concatenate([e_pos, 0.3 * w_rot * np.cross(a, DOWN)])
                dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + 1e-4 * np.eye(6), e)
                q[self.idx] = np.clip(q[self.idx] + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
            pin.framesForwardKinematics(m, d, q)
            M = d.oMf[fid]
            err = float(np.linalg.norm(target - M.translation))
            tilt = math.degrees(math.acos(np.clip(M.rotation[:, 0] @ DOWN, -1, 1)))
            ok = err < self.tol and tilt <= max_tilt_deg + 0.5
            if ok and (not best[0] or tilt < best[3]):
                best = (True, q[self.idx].copy(), err, tilt)
                if tilt < 1.0:
                    break
            elif not best[0] and err < best[2]:
                best = (False, q[self.idx].copy(), err, tilt)
        return best


def graspable(size, finger_closed, finger_open, jaw_margin, max_height):
    """Can the jaws take an item of bbox size (x, y, z)? Opening = 2 x finger joint value.
    Returns (ok, reason)."""
    narrow, height = min(size[0], size[1]), size[2]
    if narrow > 2 * finger_open - jaw_margin:
        return False, 'too_wide'
    if narrow < 2 * finger_closed:
        return False, 'too_narrow'
    if height > max_height:
        return False, 'too_tall'
    return True, ''


def in_envelope(x, y, cfg):
    return cfg['reach_x_m'][0] <= x <= cfg['reach_x_m'][1] and abs(y) <= cfg['reach_y_max_m']


def plan_pick(arm, centre, height, cfg):
    """Arm joint targets for a top-down attach-mode pick of an item whose bbox centre (base_link)
    and height are given. Returns (ok, reason, {phase: arm_q}) with phases above, hover, lift."""
    if not in_envelope(centre[0], centre[1], cfg):
        return False, 'outside_envelope', {}
    hover = np.array([centre[0], centre[1], centre[2] + height / 2 + cfg['hover_above_item_m']])
    targets = {'hover': hover,
               'above': hover + [0, 0, cfg['pregrasp_above_m']],
               'lift': hover + [0, 0, cfg['lift_above_m']]}
    plan, seed = {}, None
    for phase in ('hover', 'above', 'lift'):           # solve the lowest first, seed the others with it
        ok, q, err, tilt = arm.ik(targets[phase], cfg['approach_max_tilt_deg'], seed_q=seed)
        if not ok:
            return False, 'unreachable', {}
        plan[phase], seed = q, q
    return True, '', plan


def bin_release_target(bin_rim_centre, near_wall_x, bin_drop):
    """Jaw-tip point over a bin (base_link): clearance above the rim, inside_wall past the near
    wall (the bin is ahead of the robot along x), as checked by the cross-phase tests."""
    return np.array([near_wall_x + bin_drop['inside_wall_m'], bin_rim_centre[1],
                     bin_rim_centre[2] + bin_drop['clearance_above_rim_m']])
