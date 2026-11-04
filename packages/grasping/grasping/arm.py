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

    def pose(self, arm_q, frame=TIPS):
        """(position, rotation) of a frame in base_link."""
        pin.framesForwardKinematics(self.model, self.data, self.full(arm_q))
        M = self.data.oMf[self.model.getFrameId(frame)]
        return M.translation.copy(), M.rotation.copy()

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


def in_basket(arm, p, margin=0.03):
    """Is point p (base_link) in the basket: inside its walls (+ margin), from the floor up to a
    basket height above the rim (items lying on others)?"""
    rec, res = compartment(arm, 'recyclable'), compartment(arm, 'residual')
    y_lo, y_hi = min(rec['y'][0], res['y'][0]), max(rec['y'][1], res['y'][1])
    height = arm.basket[2]
    return (rec['x'][0] - margin <= p[0] <= rec['x'][1] + margin and y_lo - margin <= p[1] <= y_hi + margin
            and rec['floor'] - margin <= p[2] <= rec['rim'] + height)


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


def held_item(arm, q, held):
    """Pose of a held item at arm pose q. held: item centre 'p' and rotation 'R' in the jaw-tips
    frame (fixed while attached), bbox 'size'. Returns (centre, rotation, extents along base x/y/z)."""
    t, R = arm.pose(q, TIPS)
    Rb = R @ np.asarray(held['R'])
    return t + R @ np.asarray(held['p']), Rb, np.abs(Rb) @ np.asarray(held['size'], dtype=float)


def item_yaw(Rb, size):
    """Yaw of an item's longest horizontal bbox axis (a line: defined modulo pi)."""
    i = max(range(3), key=lambda k: size[k] * np.hypot(Rb[0, k], Rb[1, k]))
    return math.atan2(Rb[1, i], Rb[0, i])


def align_item(arm, q, held, yaw=0.0):
    """Turn the wrist so the held item's long side points along `yaw` (base_link)."""
    _, Rb, _ = held_item(arm, q, held)
    a = arm.jaw_axis(q)
    jaw = math.atan2(a[1], a[0]) + (yaw - item_yaw(Rb, held['size']))
    return align_jaws(arm, q, (math.cos(jaw), math.sin(jaw)))


def pile_top(slot, half, others, floor, tolerance=0.0):
    """Highest top among items whose footprint overlaps a footprint of half extents `half` at slot
    by more than `tolerance` (others: dicts with 'centre', 'half' (x, y) and 'top'); the floor if none."""
    return max([o['top'] for o in others if abs(o['centre'][0] - slot[0]) < half[0] + o['half'][0] - tolerance
                and abs(o['centre'][1] - slot[1]) < half[1] + o['half'][1] - tolerance] + [floor])


def plan_store(arm, bin_id, held, others, cfg):
    """Arm joint targets to put a held item (see held_item) into compartment bin_id: long side
    along the compartment, at the lowest free spot (others: items already in the basket, see
    pile_top), released `drop_gap_m` above what lies there, but never so low that the gripper bar
    (across the compartment when the item is turned) reaches the rim. Spots: slots_x x slots_y
    along / across the compartment. An item that would stick out above the rim (less than
    top_below_rim_m below it) is refused as compartment_full. Returns (ok, reason, {'above',
    'release'}, info)."""
    st = cfg['store']
    c = compartment(arm, bin_id)
    over = arm.poses[f'over_{bin_id}']['arm']
    bounds = near(over, cfg['unload']['joint_spread_rad'])
    q0 = align_item(arm, over, held)
    if q0 is None:
        return False, 'wrist_limit', {}, {}
    t0, _ = arm.fk(q0, TIPS)
    c0, _, ext = held_item(arm, q0, held)
    d = c0 - t0                                              # item centre relative to the tips
    half = ext / 2
    bar_low = arm.fk(q0, 'arm_gripper_bar_link')[0][2] - arm.bar[0] / 2 - t0[2]   # bar underside above the tips
    tips_min = c['rim'] + st['bar_clearance_m'] - bar_low

    def spots(lo, hi, h, n):
        lo, hi = lo + h + st['wall_clearance_m'], hi - h - st['wall_clearance_m']
        return [(lo + hi) / 2] if hi <= lo else list(np.linspace(lo, hi, n))
    mid = np.array([(c['x'][0] + c['x'][1]) / 2, (c['y'][0] + c['y'][1]) / 2])
    cands = [(pile_top((x, y), half, others, c['floor'], st['overlap_tolerance_m']),
              float(np.hypot(x - mid[0], y - mid[1])), x, y)
             for x in spots(*c['x'], half[0], st['slots_x']) for y in spots(*c['y'], half[1], st['slots_y'])]
    for pile, _, x, y in sorted(cands):
        if pile + ext[2] > c['rim'] - st['top_below_rim_m']:           # its top once it has landed
            return False, 'compartment_full', {}, {'pile': pile}     # sorted: every other spot is higher
        tips_z = max(pile + st['drop_gap_m'] + half[2] - d[2], tips_min)
        target = np.array([x - d[0], y - d[1], tips_z])
        q, seed = None, q0
        for _ in range(2):                                   # the item's offset turns with the wrist: re-aim
            ok, q, err, tilt = arm.ik(target, cfg['approach_max_tilt_deg'], seed_q=seed, bounds=bounds)
            q = align_item(arm, q, held) if ok else None
            if q is None:
                break
            ci, _, _ = held_item(arm, q, held)
            target[:2] += np.array([x, y]) - ci[:2]
            seed = q
        if q is None:
            continue
        ci, _, e = held_item(arm, q, held)
        if (abs(ci[0] - x) > 0.015 or abs(ci[1] - y) > 0.015
                or arm.fk(q, 'arm_gripper_bar_link')[0][2] - arm.bar[0] / 2 < c['rim'] + st['bar_clearance_m'] - 0.005):
            continue
        above = target + [0, 0, max(0.0, c['rim'] + cfg['unload']['rim_clearance_m'] - (ci[2] - e[2] / 2))]
        ok, qa, err, tilt = arm.ik(above, cfg['unload']['above_max_tilt_deg'], seed_q=q, bounds=bounds)
        qa = align_item(arm, qa, held) if ok else None
        if qa is None:
            continue
        return True, '', {'above': qa, 'release': q}, {'slot': (x, y), 'pile': pile,
                                                       'drop': ci[2] - e[2] / 2 - pile}
    return False, 'no_spot', {}, {}


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


def bin_release_target(bin_rim_centre, near_wall_x, bin_drop, hang=0.0, clearance=None):
    """Jaw-tip point over a bin (base_link): `clearance` (default clearance_above_rim_m) above the rim,
    inside_wall past the near wall (the bin is ahead of the robot along x), as checked by the
    cross-phase tests. `hang`: how far the held item reaches below the jaw tips (raised by that, so
    the item clears the rim)."""
    if clearance is None:
        clearance = bin_drop['clearance_above_rim_m']
    return np.array([near_wall_x + bin_drop['inside_wall_m'], bin_rim_centre[1],
                     bin_rim_centre[2] + clearance + hang])


def release_clearances(bin_drop):
    """Clearances above the rim to try for the release, highest first, 1 cm apart."""
    hi, lo = bin_drop['clearance_above_rim_m'], bin_drop['min_clearance_above_rim_m']
    n = int(round((hi - lo) / 0.01))
    return [hi - k * (hi - lo) / max(n, 1) for k in range(n + 1)]
