"""Arm kinematics and grasp planning, plain Python + Pinocchio (no ROS nodes).

The robot model is the beach_description xacro of the configured variant, with the legs
in the leg controller's stand pose; targets are in base_link. IK is position IK of a
frame (default the jaw tips) with the tool's approach axis (tool x) kept within a tilt of
straight down; adapted from A0's beach_description/scripts/reach_check.py. Self-collision
is checked by grasping.collision (B5) when config `collision` is set; plans take a `check` callback.
"""

import itertools
import heapq
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
        self.urdf = urdf
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
        self.bar_origin = box_origins(urdf, 'arm_gripper_bar_link')[0]   # its box centre in the bar link

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

    def bar_underside(self, arm_q):
        """Height (base_link z) of the gripper bar's lowest collision-box corner."""
        t, R = self.pose(arm_q, 'arm_gripper_bar_link')
        corners = self.bar_origin + np.array(list(itertools.product(*[(-h, h) for h in self.bar / 2])))
        return float(np.min(corners @ R.T[:, 2] + t[2]))

    def jaw_axis(self, arm_q):
        """Direction the fingers open along (tool y) in base_link."""
        pin.framesForwardKinematics(self.model, self.data, self.full(arm_q))
        return self.data.oMf[self.model.getFrameId(TIPS)].rotation[:, 1].copy()

    def frame(self, name):
        """Position of a robot frame (e.g. basket_recyclable_link) in base_link."""
        pin.framesForwardKinematics(self.model, self.data, self.q0)
        return self.data.oMf[self.model.getFrameId(name)].translation.copy()

    def ik(self, target, max_tilt_deg, frame=TIPS, seed_q=None, iters=300, bounds=None, restarts=None):
        """Returns (ok, arm_q, position error m, tilt deg). First try starts at seed_q (or zeros).
        bounds: (lo, hi) arrays narrowing the joint limits for this solve (e.g. no flipped arm).
        restarts: tries for this solve (default the arm's)."""
        m, d = self.model, self.data
        fid = m.getFrameId(frame)
        target = np.asarray(target, dtype=float)
        lo, hi = (self.lo, self.hi) if bounds is None else (np.maximum(self.lo, bounds[0]),
                                                             np.minimum(self.hi, bounds[1]))
        best = (False, None, 1e9, 180.0)
        # the loop spells out np.cross / np.clip / np.vstack (same arithmetic, same results bit for bit):
        # their per-call overhead was most of the IK time (B5.3 profile)
        D0, D1, D2 = (float(v) for v in DOWN)
        damp = 1e-4 * np.eye(6)
        Jw, e = np.empty((6, len(self.vidx))), np.empty(6)
        for r in range(self.restarts if restarts is None else restarts):
            q = self.q0.copy()
            q[self.idx] = self.rng.uniform(lo, hi) if r else np.clip(0.0 if seed_q is None else seed_q, lo, hi)
            for _ in range(iters):
                pin.framesForwardKinematics(m, d, q)
                M = d.oMf[fid]
                a = M.rotation[:, 0]
                e_pos = target - M.translation
                tilt = math.degrees(math.acos(min(1.0, max(-1.0, float(a @ DOWN)))))
                w_rot = 1.0 if tilt > max_tilt_deg else 0.0      # pull towards down only beyond the limit
                if np.linalg.norm(e_pos) < self.tol * 0.6 and tilt <= max_tilt_deg + 0.5:
                    break
                J = pin.computeFrameJacobian(m, d, q, fid, pin.LOCAL_WORLD_ALIGNED)[:, self.vidx]
                w = 0.3 * w_rot
                Jw[:3], Jw[3:] = J[:3], w * J[3:]
                a0, a1, a2 = float(a[0]), float(a[1]), float(a[2])
                e[:3] = e_pos
                e[3], e[4], e[5] = w * (a1 * D2 - a2 * D1), w * (a2 * D0 - a0 * D2), w * (a0 * D1 - a1 * D0)
                dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + damp, e)
                q[self.idx] = np.minimum(np.maximum(q[self.idx] + np.minimum(np.maximum(dq, -0.2), 0.2), lo), hi)
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


def box_origins(urdf, link_name):
    """Origins (n x 3, in the link frame) of a link's collision boxes."""
    link = next(lk for lk in ET.fromstring(urdf).iter('link') if lk.get('name') == link_name)
    out = []
    for col in link.iter('collision'):
        o = col.find('origin')
        out.append([float(v) for v in (o.get('xyz', '0 0 0') if o is not None else '0 0 0').split()])
    return np.array(out)


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


def align_jaws(arm, q, axis, near_wrist=None):
    """Turn the wrist (last joint) so the jaws open along the horizontal direction `axis`
    (either sign); the jaw tips stay on the wrist axis. None if the joint limit forbids it.
    near_wrist: of the turns within 5 deg (k x pi apart, equal up to float noise), take the one
    nearest this wrist value; None picks by that noise (c1_axis s3: the stored bottle turned half
    a turn between `above` and `release`, through the compartment walls)."""
    cands = []
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
            if off <= math.radians(5):
                cands.append((off, q3))
    if not cands:
        return None
    if near_wrist is None:
        return min(cands, key=lambda c: c[0])[1]
    return min(cands, key=lambda c: abs(c[1][-1] - near_wrist))[1]


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


def plan_unload(arm, bin_id, centre, height, cfg, check=None):
    """Arm joint targets to take an item (bbox centre in base_link, vertical extent `height`)
    out of compartment bin_id. The jaws (and the gripper bar, wider than a compartment) point
    along the compartment's length and stay closed: in attach mode the tips only hover above
    the item's top. The gripper is kept clear of the walls (the item's offset to it stays small),
    then lifts straight up until the item clears the rim. Returns (ok, reason, {phase: arm_q})
    with phases above, hover. check(plan) -> None | failure text (collision check, B5): a failing plan
    is solved again from a random seed within the bounds, at most collision.restarts times."""
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
    nearest = cfg.get('wrist_branch', 'any') == 'nearest'
    n_via = u.get('lift_vias') or 0                         # IK points between hover and above (0 = off): with the
                                                            # store's edge spots (7d542b2) the joint line swung the
                                                            # gripper and the held item into the walls (sweep 7)

    def lift_vias(plan, seed):                              # top first; above and each re-aimed so the item centre
        t, R = arm.pose(plan['hover'], TIPS)                # stays over `centre` (it hangs below the tips)
        p = R.T @ (np.asarray(centre, dtype=float) - t)
        out, seed = [], plan['hover']
        for k in list(range(1, n_via + 1)) + [None]:
            target = above.copy() if k is None else hover + (above - hover) * k / (n_via + 1)
            for aim in range(2):
                ok, q, err, tilt = arm.ik(target, cfg['approach_max_tilt_deg'], seed_q=seed, bounds=bounds,
                                          restarts=1)       # one try from the point below at the hover's tilt
                if not ok:                                  # first (plan time; a one-try 25 deg solve tilted
                    ok, q, err, tilt = arm.ik(target, u['above_max_tilt_deg'], seed_q=seed, bounds=bounds,
                                              restarts=1 if aim else None)   # the item into the walls)
                q = align_jaws(arm, q, (1.0, 0.0), plan['hover'][-1] if nearest else None) if ok else None
                if q is None:
                    return None
                t, R = arm.pose(q, TIPS)
                target[:2] += np.asarray(centre[:2], dtype=float) - (t + R @ p)[:2]
                seed = q
            out.append(q)
        plan['above'] = out.pop()
        return out[::-1]

    why = ''
    for attempt in range(1 if check is None else 1 + restarts(cfg)):
        plan, seed = {}, (over if attempt == 0 else random_seed(arm, bounds))
        for phase, target, tilt_max in (('hover', hover, cfg['approach_max_tilt_deg']),
                                        ('above', above, u['above_max_tilt_deg'])):
            ok, q, err, tilt = arm.ik(target, tilt_max, seed_q=seed, bounds=bounds)
            if not ok:
                if attempt == 0:
                    return False, f'unreachable ({phase})', {}
                break
            q = align_jaws(arm, q, (1.0, 0.0), (over if attempt and phase == 'hover' else seed)[-1]
                           if nearest else None)
            if q is None:
                if attempt == 0:
                    return False, 'wrist_limit', {}
                break
            plan[phase], seed = q, q
        if len(plan) < 2:
            continue
        if n_via:
            vias = lift_vias(plan, seed)
            if vias is None:
                continue
            plan['via'] = vias
        why = check(plan) if check is not None else None
        if not why:
            return True, '', plan
    return False, why, {}


def first_plan_only(cfg):
    """cfg without collision restarts: plan_unload's first plan only (seeded at over, no random seeds),
    the one store.check_unload checks: the same plan comes back at the bin, and a failing spot costs
    one plan, not 1 + restarts (offline: store p95 1.43 -> 4.94 s with the restarts)."""
    return dict(cfg, collision=dict(cfg.get('collision') or {}, restarts=0))


def restarts(cfg):
    """Plan restarts after a collision (collision.restarts, default ik_restarts)."""
    return (cfg.get('collision') or {}).get('restarts', cfg['ik_restarts'])


def random_seed(arm, bounds):
    """A random arm pose within the joint limits and bounds (an IK restart after a collision)."""
    lo, hi = np.maximum(arm.lo, bounds[0]), np.minimum(arm.hi, bounds[1])
    return arm.rng.uniform(lo, hi)


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


def align_item(arm, q, held, yaw=0.0, near_wrist=None):
    """Turn the wrist so the held item's long side points along `yaw` (base_link). near_wrist: see
    align_jaws; then a footprint within 1 cm of square may also turn a quarter turn (which side is
    'long' flips with tilt: the jar turned pi/2 while lowered)."""
    _, Rb, _ = held_item(arm, q, held)
    a = arm.jaw_axis(q)
    jaw = math.atan2(a[1], a[0]) + (yaw - item_yaw(Rb, held['size']))
    q1 = align_jaws(arm, q, (math.cos(jaw), math.sin(jaw)), near_wrist)
    if near_wrist is None or q1 is None:
        return q1
    q2 = align_jaws(arm, q, (-math.sin(jaw), math.cos(jaw)), near_wrist)
    if q2 is None or abs(q2[-1] - near_wrist) >= abs(q1[-1] - near_wrist):
        return q1
    e1, e2 = held_item(arm, q1, held)[2], held_item(arm, q2, held)[2]
    along = (math.cos(yaw), math.sin(yaw))
    long1 = abs(along[0]) * e1[0] + abs(along[1]) * e1[1]
    long2 = abs(along[0]) * e2[0] + abs(along[1]) * e2[1]
    return q2 if long2 >= long1 - 0.01 else q1


def pile_top(slot, half, others, floor, tolerance=0.0):
    """Highest top among items whose footprint overlaps a footprint of half extents `half` at slot
    by more than `tolerance` (others: dicts with 'centre', 'half' (x, y) and 'top'); the floor if none."""
    return max([o['top'] for o in others if abs(o['centre'][0] - slot[0]) < half[0] + o['half'][0] - tolerance
                and abs(o['centre'][1] - slot[1]) < half[1] + o['half'][1] - tolerance] + [floor])


def supported(slot, others, pile, floor, margin):
    """Whether an item centred at slot rests flat on the pile at height `pile` (see pile_top): its
    centre, and margin either side of it along the compartment (x, the item's long side), lies over an item whose top is that pile. A pile
    under one end only tips it (devb5: a 0.23 m bottle stacked over the end of a 0.12 m one tipped
    to a height the unload could not lift clear of the rim, and the item under it was blocked)."""
    if pile <= floor:
        return True
    tops = [o for o in others if abs(o['top'] - pile) < 0.01]
    return all(any(abs(o['centre'][0] - px) <= o['half'][0] and abs(o['centre'][1] - py) <= o['half'][1]
                   for o in tops)
               for px, py in [(slot[0] + dx, slot[1]) for dx in (-margin, 0.0, margin)])


def plan_store(arm, bin_id, held, others, cfg, check=None, unload_check=None):
    """Arm joint targets to put a held item (see held_item) into compartment bin_id: long side
    along the compartment, at the lowest free spot (others: items already in the basket, see
    pile_top), released `drop_gap_m` above what lies there, but never so low that the gripper bar
    (across the compartment when the item is turned) reaches the rim. Spots: slots_x x slots_y
    along / across the compartment. An item that would stick out above the rim (less than
    top_below_rim_m below it) is refused as compartment_full. check(plan) -> None | failure text
    (collision check, B5): a failing spot is skipped (info 'rejected': [(slot, text)]); if no spot
    is clean the item is refused as compartment_full. unload_check(box, height) -> None | failure text
    (store.check_unload): a spot is used only if the item, landed there ({'centre', 'R', 'size'}), can be
    unloaded again (devb5b: a corner cube and a stacked wide item stored fine but no unload planned).
    Returns (ok, reason, {'above', 'release'}, info)."""
    st = cfg['store']
    c = compartment(arm, bin_id)
    over = arm.poses[f'over_{bin_id}']['arm']
    bounds = near(over, cfg['unload']['joint_spread_rad'])
    nearest = cfg.get('wrist_branch', 'any') == 'nearest'     # any: the wrist turn as align_jaws' noise picks
    q0 = align_item(arm, over, held, near_wrist=over[-1] if nearest else None)
    if q0 is None:
        return False, 'wrist_limit', {}, {}
    t0, _ = arm.fk(q0, TIPS)
    c0, _, ext = held_item(arm, q0, held)
    d = c0 - t0                                              # item centre relative to the tips
    half = ext / 2
    def bar_under(q):                                        # bar_box_exact false: the box centre (9ac4816)
        return arm.bar_underside(q) if st.get('bar_box_exact') else \
            arm.fk(q, 'arm_gripper_bar_link')[0][2] - arm.bar[0] / 2
    bar_low = bar_under(q0) - t0[2]                          # bar underside above the tips
    tips_min = c['rim'] + st['bar_clearance_m'] - bar_low

    def spots(lo, hi, h, n):                                 # [(spot, inward step for a retry)]
        lo, hi = lo + h + st['wall_clearance_m'], hi - h - st['wall_clearance_m']
        if hi <= lo:
            return [((lo + hi) / 2, 0.0)]
        v = np.linspace(lo, hi, n)
        return [(float(x), 1.0 if i == 0 else -1.0 if i == n - 1 else 0.0) for i, x in enumerate(v)]
    retry = st.get('wall_retry_m') or 0.0                    # an edge spot the wall rejected: again, further in,
    retries = st.get('wall_retries') or 1                    # at most this many times
    quick = st.get('seeded_ik')                              # one seeded try: pre-aim, and above for a first check
    margin = st.get('support_margin_m')                      # stack only flat on what lies there (null = off)
    n_via = st.get('lower_vias') or 0                        # IK points between above and release: the joint line
                                                             # drifted the item 2-3 cm sideways (B5.4 run 3)
    mid = np.array([(c['x'][0] + c['x'][1]) / 2, (c['y'][0] + c['y'][1]) / 2])
    corner = st.get('fill_corner')                           # [sx, sy]: from the (sx x, sy y) corner; null: centre first
    rank = (lambda x, y: (-corner[0] * x, -corner[1] * y)) if corner else \
        (lambda x, y: (float(np.hypot(x - mid[0], y - mid[1])), 0.0))
    ys = spots(*c['y'], half[1], st['slots_y'])
    middle = ys[len(ys) // 2][0] if len(ys) % 2 and len(ys) > 1 else None   # odd slots_y: a fallback row

    def solve(target, tilt, seed, once):
        if once:
            res = arm.ik(target, tilt, seed_q=seed, bounds=bounds, restarts=1)
            if res[0]:
                return res
        return arm.ik(target, tilt, seed_q=seed, bounds=bounds)

    def over(xy, target, max_tilt, seed, near_q, restarts=None):   # tips at target, re-aimed once so the item
        target = np.array(target, dtype=float)                     # centre is over xy (it hangs below the tips)
        for aim in range(2):
            ok, qt, err, tilt = arm.ik(target, max_tilt, seed_q=seed, bounds=bounds, restarts=restarts)
            qt = align_item(arm, qt, held, near_wrist=near_q[-1] if nearest else None) if ok else None
            if qt is None:
                return None
            target[:2] += np.asarray(xy) - held_item(arm, qt, held)[0][:2]
            seed = qt
        return qt

    def vias(xy, target, dz, q):                             # top first; seeded upward from the release
        out, seed = [], q
        for k in range(1, n_via + 1):
            qv = over(xy, target + [0, 0, dz * k / (n_via + 1)], cfg['approach_max_tilt_deg'], seed, q, 1)
            if qv is None:
                return None
            out.append(qv)
            seed = qv
        return out[::-1]

    def cand(x, y, inx, iny, again):                         # again: retries so far
        return (pile_top((x, y), half, others, c['floor'], st['overlap_tolerance_m']), y == middle,
                rank(x, y), again, x, y, inx, iny)          # a retry 5 mm in before the next spot along
    cands = [cand(x, y, inx, iny, 0) for x, inx in spots(*c['x'], half[0], st['slots_x']) for y, iny in ys]
    heapq.heapify(cands)
    rejected = []
    while cands:
        pile, _, _, again, x, y, inx, iny = heapq.heappop(cands)
        if pile + ext[2] > c['rim'] - st['top_below_rim_m']:           # its top once it has landed
            return False, 'compartment_full', {}, {'pile': pile, 'rejected': rejected, 'ext': ext}   # sorted: the rest is higher
        if margin is not None and not supported((x, y), others, pile, c['floor'], margin):
            rejected.append(((x, y), f'unsupported (pile {pile:.3f})'))
            continue
        tips_z = max(pile + st['drop_gap_m'] + half[2] - d[2], tips_min)
        target = np.array([x - d[0], y - d[1], tips_z])
        q, seed = None, q0
        for aim in range(2):                                 # the item's offset turns with the wrist: re-aim
            ok, q, err, tilt = solve(target, cfg['approach_max_tilt_deg'], seed, quick and aim == 0)
            q = align_item(arm, q, held, near_wrist=q0[-1] if nearest else None) if ok else None
            if q is None:
                break
            ci, _, _ = held_item(arm, q, held)
            target[:2] += np.array([x, y]) - ci[:2]
            seed = q
        if q is None:
            continue
        ci, _, e = held_item(arm, q, held)
        if (abs(ci[0] - x) > 0.015 or abs(ci[1] - y) > 0.015
                or bar_under(q) < c['rim'] + st['bar_clearance_m'] - 0.005):
            continue
        if st.get('inside_walls') and not (c['x'][0] <= ci[0] - e[0] / 2 and ci[0] + e[0] / 2 <= c['x'][1] and
                                           c['y'][0] <= ci[1] - e[1] / 2 and ci[1] + e[1] / 2 <= c['y'][1]):
            continue                                         # a tilted item at an edge spot reaches past the wall
        dz = max(0.0, c['rim'] + cfg['unload']['rim_clearance_m'] - (ci[2] - e[2] / 2))
        above = target + [0, 0, dz]
        qv = vias((x, y), target, dz, q)
        if qv is None:
            continue
        def solve_above(restarts=None):                      # with vias: re-aimed like them
            if n_via:
                return over((x, y), above, cfg['unload']['above_max_tilt_deg'], qv[0], q, restarts)
            ok, qa, err, tilt = arm.ik(above, cfg['unload']['above_max_tilt_deg'], seed_q=q, bounds=bounds,
                                       restarts=restarts)
            return align_item(arm, qa, held, near_wrist=q[-1] if nearest else None) if ok else None
        why = None
        if quick and check is not None:                      # a cheap seeded above first: most spots fail the
            qs = solve_above(1)                              # check anyway (the full solve only for the rest)
            why = check({'above': qs, 'via': qv, 'release': q}) if qs is not None else None
        if not why:
            qa = solve_above()
            if qa is None:
                continue
            why = check({'above': qa, 'via': qv, 'release': q}) if check is not None else None
        if not why and unload_check is not None:             # with unload_order last_first the items stored
            lc, Rb, le = held_item(arm, q, held)             # later are out by then: this layout comes back
            lc = lc - [0, 0, lc[2] - le[2] / 2 - pile]
            u = unload_check({'centre': lc, 'R': Rb, 'size': tuple(held['size'])}, float(le[2]))
            why = f'no_unload: {u}' if u else None
        info = {'slot': (x, y), 'pile': pile, 'drop': ci[2] - e[2] / 2 - pile, 'rejected': rejected, 'ext': ext}
        if why:
            rejected.append(((x, y), why))
            if retry and again < retries and (inx or iny) and 'basket_link' in why:
                heapq.heappush(cands, cand(x + inx * retry, y + iny * retry, inx, iny, again + 1))
            continue
        return True, '', dict({'above': qa, 'release': q}, **({'via': qv} if qv else {})), info
    if rejected:
        return False, 'compartment_full', {}, {'pile': float('nan'), 'rejected': rejected, 'ext': ext}
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


def plan_pick(arm, centre, height, cfg, check=None):
    """Arm joint targets for a top-down attach-mode pick of an item whose bbox centre (base_link)
    and height are given. Returns (ok, reason, {phase: arm_q}) with phases above, hover, lift.
    check(plan) -> None | failure text (collision check, B5): a failing plan is solved again with the
    hover pose from a random seed within the bounds, at most collision.restarts times."""
    if not in_envelope(centre[0], centre[1], cfg):
        return False, 'outside_envelope', {}
    hover = np.array([centre[0], centre[1], centre[2] + height / 2 + cfg['hover_above_item_m']])
    targets = {'hover': hover,
               'above': hover + [0, 0, cfg['pregrasp_above_m']],
               'lift': hover + [0, 0, cfg['lift_above_m']]}
    # pick_joint_spread_rad (None = unbounded): hover within it of `up`, above and lift within it of
    # hover. Unbounded, the lowest-tilt restart won: lift turned forearm roll 2 pi (or the waist pi)
    # from hover, ~21 s at the speed cap with the item hanging (0.14 rad lag, store_lower_test.py).
    spread = cfg.get('pick_joint_spread_rad')
    up = arm.poses['up']['arm']
    why = ''
    for attempt in range(1 if check is None else 1 + restarts(cfg)):
        plan, seed = {}, (up if spread is not None else None)
        if attempt:
            seed = random_seed(arm, near(up, spread) if spread is not None else (arm.lo, arm.hi))
        for phase in ('hover', 'above', 'lift'):           # solve the lowest first, seed the others with it
            bounds = near(up if phase == 'hover' else seed, spread) if spread is not None else None
            ok, q, err, tilt = arm.ik(targets[phase], cfg['approach_max_tilt_deg'], seed_q=seed, bounds=bounds)
            if not ok:
                if attempt == 0:
                    return False, 'unreachable', {}
                break
            plan[phase] = q
            if spread is None or phase == 'hover':         # bounded: above and lift both near hover
                seed = q
        if len(plan) < 3:
            continue
        why = check(plan) if check is not None else None
        if not why:
            return True, '', plan
    return False, why, {}


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
