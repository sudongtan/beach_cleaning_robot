"""Arm kinematics and grasp planning, plain Python + Pinocchio (no ROS nodes).

The robot model is the beach_description xacro of the configured variant, with the legs
in the leg controller's stand pose; targets are in base_link. IK is position IK of a
frame (default the jaw tips) with the tool's approach axis (tool x) kept within a tilt of
straight down; adapted from A0's beach_description/scripts/reach_check.py. Self-collision
is not checked (the named poses in arm_poses_custom.yaml were checked in A0).
"""

import math
import subprocess
import xml.etree.ElementTree as ET

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
        self.basket = basket_size(urdf)
        self.bar = box_sizes(urdf, 'arm_gripper_bar_link')[0]      # gripper bar (tool x, y = jaw axis, z)

    def full(self, arm_q):
        q = self.q0.copy()
        q[self.idx] = arm_q
        return q

    def fk(self, arm_q, frame=TCP):
        """(position, tool x axis) of a frame in base_link."""
        pin.framesForwardKinematics(self.model, self.data, self.full(arm_q))
        M = self.data.oMf[self.model.getFrameId(frame)]
        return M.translation.copy(), M.rotation[:, 0].copy()

    def jaw_axis(self, arm_q):
        """Direction the fingers open along (tool y) in base_link."""
        pin.framesForwardKinematics(self.model, self.data, self.full(arm_q))
        return self.data.oMf[self.model.getFrameId(TIPS)].rotation[:, 1].copy()

    def frame(self, name):
        """Position of a robot frame (e.g. basket_recyclable_link) in base_link."""
        pin.framesForwardKinematics(self.model, self.data, self.q0)
        return self.data.oMf[self.model.getFrameId(name)].translation.copy()

    def ik(self, target, max_tilt_deg, frame=TIPS, seed_q=None, iters=300, bounds=None):
        """Returns (ok, arm_q, position error m, tilt deg). First try starts at seed_q (or zeros).
        bounds: (lo, hi) arrays narrowing the joint limits for this solve (e.g. no flipped arm)."""
        m, d = self.model, self.data
        fid = m.getFrameId(frame)
        target = np.asarray(target, dtype=float)
        lo, hi = (self.lo, self.hi) if bounds is None else (np.maximum(self.lo, bounds[0]),
                                                             np.minimum(self.hi, bounds[1]))
        best = (False, None, 1e9, 180.0)
        for r in range(self.restarts):
            q = self.q0.copy()
            q[self.idx] = self.rng.uniform(lo, hi) if r else np.clip(0.0 if seed_q is None else seed_q, lo, hi)
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
                q[self.idx] = np.clip(q[self.idx] + np.clip(dq, -0.2, 0.2), lo, hi)
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


def box_sizes(urdf, link_name):
    """Sizes (n x 3) of a link's collision boxes."""
    link = next(lk for lk in ET.fromstring(urdf).iter('link') if lk.get('name') == link_name)
    return np.array([[float(v) for v in box.get('size').split()]
                     for col in link.iter('collision') for box in col.iter('box')])


def basket_size(urdf):
    """(length, width, height, wall) of the basket from basket_link's collision boxes."""
    sizes = box_sizes(urdf, 'basket_link')
    return float(sizes[:, 0].max()), float(sizes[:, 1].max()), float(sizes[:, 2].max()), float(sizes.min())


def near(q, spread):
    """Joint bounds within spread (rad per joint, inf = free) of the pose q: keeps a solve on
    the same side as the pose the arm comes from, so a move never swings the arm half a turn
    (A2b a2b_bin4: a flipped release pose swung the attached item around and tipped the robot)."""
    q, spread = np.asarray(q, dtype=float), np.asarray(spread, dtype=float)
    return q - spread, q + spread


def align_jaws(arm, q, axis):
    """Turn the wrist (last joint) so the jaws open along the horizontal direction `axis`
    (either sign); the jaw tips stay on the wrist axis. None if the joint limit forbids it."""
    best = None
    want = math.atan2(axis[1], axis[0])
    for sign in (1, -1):
        q2 = np.array(q, dtype=float)
        for _ in range(3):                                 # the wrist axis is near vertical: converge
            a = arm.jaw_axis(q2)
            err = (want - math.atan2(a[1], a[0]) + math.pi / 2) % math.pi - math.pi / 2
            q2[-1] += sign * err
        for k in (0, 1, -1):                               # a half turn opens the jaws the same way
            q3 = q2.copy()
            q3[-1] += k * math.pi
            if not arm.lo[-1] <= q3[-1] <= arm.hi[-1]:
                continue
            a = arm.jaw_axis(q3)
            off = abs((want - math.atan2(a[1], a[0]) + math.pi / 2) % math.pi - math.pi / 2)
            if best is None or off < best[0]:
                best = (off, q3)
    return None if best is None or best[0] > math.radians(5) else best[1]


def compartment(arm, bin_id):
    """Inner box of a compartment in base_link: dict x (min, max), y (min, max), floor, rim."""
    length, width, height, wall = arm.basket
    rim = arm.frame(f'basket_{bin_id}_link')                # centre of the opening, at the rim
    half_y = width / 4 - wall
    return {'x': (rim[0] - length / 2 + wall, rim[0] + length / 2 - wall),
            'y': (rim[1] - half_y, rim[1] + half_y), 'floor': rim[2] - height + wall, 'rim': rim[2]}


def plan_unload(arm, bin_id, centre, height, cfg):
    """Arm joint targets to take an item (bbox centre in base_link, vertical extent `height`)
    out of compartment bin_id. The jaws (and the gripper bar, wider than a compartment) point
    along the compartment's length and stay closed: in attach mode the tips only hover above
    the item's top. The gripper is kept clear of the walls (the item's offset to it stays small),
    then lifts straight up until the item clears the rim. Returns (ok, reason, {phase: arm_q})
    with phases above, hover."""
    u = cfg['unload']
    c = compartment(arm, bin_id)
    half_x = arm.bar[1] / 2 + u['wall_clearance_m']           # bar along the compartment (x)
    half_y = arm.bar[2] / 2 + u['wall_clearance_m']           # its thickness across it
    if c['x'][1] - c['x'][0] < 2 * half_x or c['y'][1] - c['y'][0] < 2 * half_y:
        return False, 'gripper_wider_than_compartment', {}
    x = min(max(centre[0], c['x'][0] + half_x), c['x'][1] - half_x)
    y = min(max(centre[1], c['y'][0] + half_y), c['y'][1] - half_y)
    hover = np.array([x, y, centre[2] + height / 2 + cfg['hover_above_item_m']])
    above = np.array([x, y, c['rim'] + height + cfg['hover_above_item_m'] + u['rim_clearance_m']])
    over = arm.poses[f'over_{bin_id}']['arm']
    bounds = near(over, u['joint_spread_rad'])
    plan, seed = {}, over
    for phase, target, tilt_max in (('hover', hover, cfg['approach_max_tilt_deg']),
                                    ('above', above, u['above_max_tilt_deg'])):
        ok, q, err, tilt = arm.ik(target, tilt_max, seed_q=seed, bounds=bounds)
        if not ok:
            return False, f'unreachable ({phase})', {}
        q = align_jaws(arm, q, (1.0, 0.0))
        if q is None:
            return False, 'wrist_limit', {}
        plan[phase], seed = q, q
    return True, '', plan


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


def bin_release_target(bin_rim_centre, near_wall_x, bin_drop, hang=0.0):
    """Jaw-tip point over a bin (base_link): clearance above the rim, inside_wall past the near
    wall (the bin is ahead of the robot along x), as checked by the cross-phase tests. `hang`:
    how far the held item reaches below the jaw tips (raised by that, so the item clears the rim)."""
    return np.array([near_wall_x + bin_drop['inside_wall_m'], bin_rim_centre[1],
                     bin_rim_centre[2] + bin_drop['clearance_above_rim_m'] + hang])
