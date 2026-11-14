"""D0b / D0c test logic, simulator-free: the stand + walk sequence and the attach-mode picks.

`io` is the simulator: run_d0.py gives an Isaac implementation (IsaacIO), the Mac tests a fake.
It must provide:
    dof_names            list of joint names in the simulator's order
    q(), dq()            joint positions / velocities (rad, m; rad/s, m/s), simulator order
    set_q(q)             teleport the joints (start pose)
    set_targets(q)       joint position targets, simulator order
    body_pose()          (position xyz, quaternion wxyz) of the body link (trunk) in the world
    body_ang_vel_world() angular velocity of the body (world frame)
    link_pose(name)      (xyz, wxyz) of a link (the gripper bar)
    item_pose(name)      (xyz, wxyz) of a pick item
    place_item(name, xyz, wxyz)       teleport an item (dynamic, at rest)
    hold_item(name, xyz, wxyz)        move a held item (kinematic follow)
    set_item_kinematic(name, on)
    set_item_collision(name, on)      collisions of a pick item's colliders
    root_pose(), reset_robot(pose, q) the articulation root's pose; teleport root + joints, zero velocities
    step(n)              n physics steps
    time()               simulation time (s)
    capture(tag)         optional frame (may do nothing)
"""

import math

import numpy as np

import pick as pick_mod
from walk import quat_rotate_inverse, speeds, tilt_deg, yaw_of


# ---------------------------------------------------------------- math

def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def matrix_quat(r):
    tr = np.trace(r)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = [(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s]
    else:
        s = math.sqrt(1 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def pose_matrix(xyz, wxyz):
    t = np.eye(4)
    t[:3, :3] = quat_matrix(wxyz)
    t[:3, 3] = xyz
    return t


def matrix_pose(t):
    return t[:3, 3].copy(), matrix_quat(t[:3, :3])


def gyro_from_quats(q_prev, q_now, dt):
    """Body angular velocity from two orientations dt apart (finite difference)."""
    dq = quat_mul(quat_conj(q_prev), q_now)
    if dq[0] < 0:
        dq = -dq
    angle = 2.0 * math.atan2(np.linalg.norm(dq[1:]), dq[0])
    if angle < 1e-12:
        return np.zeros(3)
    axis = dq[1:] / np.linalg.norm(dq[1:])
    return axis * angle / dt


# ---------------------------------------------------------------- the robot loop

class Robot:
    """Joint bookkeeping and the 200 Hz / 50 Hz loop shared by walk and pick."""

    def __init__(self, io, cfg, policy, shaper, gyro_source='api'):
        self.io = io
        self.cfg = cfg
        self.policy = policy
        self.shaper = shaper
        self.gyro_source = gyro_source
        r = cfg['robot']
        names = list(io.dof_names)
        missing = [j for j in r['leg_joints'] + r['arm_joints'] + r['finger_joints'] if j not in names]
        if missing:
            raise ValueError(f'joints missing in the simulator: {missing}')
        self.leg_idx = [names.index(j) for j in r['leg_joints']]
        self.arm_idx = [names.index(j) for j in r['arm_joints']]
        self.finger_idx = [names.index(j) for j in r['finger_joints']]
        self.n = len(names)
        self.dt = 1.0 / float(cfg['policy']['physics_hz'])
        self.decimation = policy.decimation
        self.arm_target = np.asarray(r['arm_rest'], float)
        self.finger_target = float(r['finger_rest'])
        self.leg_target = np.asarray(policy.default, float)
        self.q_prev = None

    def full_targets(self):
        t = np.zeros(self.n)
        t[self.leg_idx] = self.leg_target
        t[self.arm_idx] = self.arm_target
        t[self.finger_idx] = [self.finger_target, -self.finger_target]
        return t

    def start_pose(self):
        """Teleport the joints to the stand pose and hold them there (Gazebo: FixedStand)."""
        q = self.full_targets()
        self.io.set_q(q)
        self.io.set_targets(q)

    def gyro(self, quat):
        if self.gyro_source == 'fd':
            g = np.zeros(3) if self.q_prev is None else gyro_from_quats(self.q_prev, quat, self.dt * self.decimation)
        else:
            g = quat_rotate_inverse(quat, self.io.body_ang_vel_world())
        self.q_prev = np.asarray(quat, float)
        return g

    def tick(self, wanted, policy_on=True):
        """One policy period: read state, run the policy (or hold), step `decimation` physics steps."""
        pos, quat = self.io.body_pose()
        gyro = self.gyro(quat)
        if policy_on:
            cmd = self.shaper(*wanted, gyro[2], self.dt * self.decimation)
            q, dq = np.asarray(self.io.q()), np.asarray(self.io.dq())
            self.leg_target = self.policy.step(quat, gyro, cmd, q[self.leg_idx], dq[self.leg_idx])
        self.io.set_targets(self.full_targets())
        self.io.step(self.decimation)
        return pos, quat, gyro


def body_row(io, terrain_height):
    pos, quat = io.body_pose()
    return [io.time(), float(pos[0]), float(pos[1]), yaw_of(quat), float(pos[2] - terrain_height(pos[0], pos[1])),
            tilt_deg(quat)]


def run_walk(io, cfg, policy, shaper, terrain_height, gyro_source='api', capture_every=0.5):
    """Stand (PD hold, then policy at zero), then the walk sequence. Returns a result dict."""
    robot = Robot(io, cfg, policy, shaper, gyro_source)
    w = cfg['walk']
    fall_tilt = float(w['pass']['fall_tilt_deg'])
    robot.start_pose()
    trace, phases = [], []
    t_cap = -1e9

    def record(tag):
        nonlocal t_cap
        trace.append(body_row(io, terrain_height))
        if io.time() - t_cap >= capture_every:
            io.capture(f'{tag}_{io.time():07.2f}')
            t_cap = io.time()

    hold_ticks = int(round(float(cfg['stand']['hold_s']) / (robot.dt * robot.decimation)))
    for _ in range(hold_ticks):
        robot.tick((0.0, 0.0, 0.0), policy_on=False)
        record('hold')
    policy.reset()
    fell_at, over = None, 0
    for ph in w['sequence']:
        t0 = io.time()
        n = int(round(float(ph['seconds']) / (robot.dt * robot.decimation)))
        for _ in range(n):
            robot.tick(tuple(ph['cmd']))
            record(ph['name'])
            over = over + 1 if trace[-1][5] > fall_tilt else 0
            if over * robot.dt * robot.decimation >= 0.3 and fell_at is None:   # adapter fall_hold_s
                fell_at = io.time()
        phases.append({'name': ph['name'], 't0': t0, 't1': io.time(), 'cmd': ph['cmd']})
        if fell_at is not None:
            break
    out = {'phases': [], 'fell_at_s': fell_at, 'gyro_source': gyro_source,
           'max_tilt_deg': round(max(r[5] for r in trace), 2)}
    for ph in phases:
        mid = (ph['t0'] + ph['t1']) / 2.0          # second half, as c7_summary
        v, wz = speeds([(r[0], r[1], r[2], r[3]) for r in trace], mid, ph['t1'])
        hs = [r[4] for r in trace if mid <= r[0] <= ph['t1']]
        out['phases'].append({**ph, 'speed_mps': round(v, 4), 'yaw_rate_radps': round(wz, 4),
                              'trunk_above_ground_m': round(float(np.mean(hs)), 4) if hs else None})
    out['trace'] = trace
    return out


def judge_walk(result, reference, pass_cfg):
    """Compare with the Gazebo reference (c7 summary.yaml). Returns (ok, reasons, numbers)."""
    reasons, nums = [], {}
    ph = {p['name']: p for p in result['phases']}
    if result['fell_at_s'] is not None:
        reasons.append(f"fell at {result['fell_at_s']:.2f} s")
    st = ph.get('stand')
    ref_h = reference['standing']['trunk_above_ground_m']['mean']
    if st and st['trunk_above_ground_m'] is not None:
        nums['standing_height_m'] = (st['trunk_above_ground_m'], ref_h)
        if abs(st['trunk_above_ground_m'] - ref_h) > pass_cfg['standing_height_tol_m']:
            reasons.append(f"standing height {st['trunk_above_ground_m']:.3f} m vs Gazebo {ref_h:.3f}")
    else:
        reasons.append('no stand phase')
    for name, key, ref_key in (('walk_0.2', 'speed_mps', 'speed_mps'), ('turn_0.3', 'yaw_rate_radps', 'yaw_rate_radps')):
        if name not in ph:
            reasons.append(f'{name} not reached')
            continue
        got, ref = ph[name][key], reference[name][ref_key]
        nums[name] = (got, ref)
        if not (abs(got - ref) <= pass_cfg['speed_rel_tol'] * abs(ref)):
            reasons.append(f'{name}: {got:.3f} vs Gazebo {ref:.3f} (> {pass_cfg["speed_rel_tol"]:.0%})')
    return not reasons, reasons, nums


# ---------------------------------------------------------------- picks

class Attach:
    """Kinematic follow: at attach the item's pose relative to the TCP is kept until detach."""

    def __init__(self, io, gripper_link, bar_to_tcp, bar_to_tips=None, held_collides=False):
        self.io = io
        self.held_collides = held_collides
        self.gripper_link = gripper_link
        self.bar_to_tcp = bar_to_tcp
        self.bar_to_tips = bar_to_tcp if bar_to_tips is None else bar_to_tips
        self.held = None
        self.rel = None

    def tcp(self):
        return pose_matrix(*self.io.link_pose(self.gripper_link)) @ self.bar_to_tcp

    def attach(self, item, height):
        """Returns the distance from the jaw tips to the item's top centre (the plan hovers the
        tips hover_above_item_m above the item's top)."""
        t_tcp = self.tcp()
        t_item = pose_matrix(*self.io.item_pose(item))
        self.rel = np.linalg.inv(t_tcp) @ t_item
        self.held = item
        self.io.set_item_kinematic(item, True)
        if not self.held_collides:
            # a kinematic item has infinite mass: touching the basket or the robot it shoves the
            # robot over (GPU D0-3); Gazebo's attached item weighs its real ~25-200 g
            self.io.set_item_collision(item, False)
        tips = (pose_matrix(*self.io.link_pose(self.gripper_link)) @ self.bar_to_tips)[:3, 3]
        top = (t_item @ np.array([0.0, 0.0, height, 1.0]))[:3]
        return float(np.linalg.norm(tips - top))

    def follow(self):
        if self.held:
            self.io.hold_item(self.held, *matrix_pose(self.tcp() @ self.rel))

    def detach(self):
        if self.held:
            self.io.set_item_kinematic(self.held, False)
            if not self.held_collides:
                self.io.set_item_collision(self.held, True)
        self.held = None


def base_link_matrix(io, body_to_base):
    return pose_matrix(*io.body_pose()) @ body_to_base


def run_picks(io, cfg, policy, shaper, cases, comp, bar_to_tcp, body_to_base, ground_z_base,
              park, gyro_source='api', capture_every=0.5, bar_to_tips=None, legs='policy', held_collides=None):
    """Every case: from the same settled start (robot reset), place the item at its planned spot
    (base_link), play pick + store with the legs on the policy at zero command (as in Gazebo) or held
    by PD at the pick stance (`legs='hold'`, a diagnostic), let it settle, check, park the item."""
    robot = Robot(io, cfg, policy, shaper, gyro_source)
    pc = cfg['pick']
    if held_collides is None:
        held_collides = bool(pc.get('held_item_collides', False))
    att = Attach(io, cfg['robot']['gripper_link'], bar_to_tcp, bar_to_tips, held_collides)
    stance = pc.get('stance_legs')                              # pick plans' stand_legs, for legs='hold'
    if legs == 'hold' and stance:
        robot.leg_target = np.array([stance[j] for j in cfg['robot']['leg_joints']], float)
    robot.start_pose()
    tick_dt = robot.dt * robot.decimation
    for _ in range(int(round(float(cfg['stand']['hold_s']) / tick_dt))):
        robot.tick((0.0, 0.0, 0.0), policy_on=False)
    policy.reset()
    for _ in range(int(round(2.0 / tick_dt))):                 # settle on the policy
        robot.tick((0.0, 0.0, 0.0), policy_on=(legs == 'policy'))
    start = (io.root_pose(), np.asarray(io.q(), float).copy(), robot.leg_target.copy())
    results = []
    t_cap = -1e9
    for k, case in enumerate(cases):
        if k > 0:                                               # every case from the same start
            io.reset_robot(start[0], start[1])
            robot.leg_target = start[2].copy()
            policy.reset()
            for _ in range(int(round(1.0 / tick_dt))):
                robot.tick((0.0, 0.0, 0.0), policy_on=(legs == 'policy'))
        item = case['item']
        p_b, yaw_b = pick_mod.item_pose_base_link(case)
        t_base = base_link_matrix(io, body_to_base)
        t_item = t_base @ pose_matrix(p_b, (math.cos(yaw_b / 2), 0, 0, math.sin(yaw_b / 2)))
        io.place_item(item, *matrix_pose(t_item))
        sch = pick_mod.case_schedule(case, pc, cfg['robot']['arm_rest'], cfg['robot']['finger_rest'])
        t0 = io.time()
        trace = {'t': [], 'item_base_link': [], 'tilt_deg': [], 'fall_tilt_deg': cfg['walk']['pass']['fall_tilt_deg']}
        fall_tilt = float(cfg['walk']['pass']['fall_tilt_deg'])
        phase_tilt, fall_phase = {}, None
        events = sorted(sch.events())
        ev = 0
        lift_end = sch.phase_end('lift')
        trace['lift_end_t'] = lift_end
        while io.time() - t0 < sch.duration + 1.5:         # + 1.5 s for the item to settle in the basket
            rel_t = io.time() - t0
            while ev < len(events) and events[ev][0] <= rel_t:
                _, kind, name, _ = events[ev]
                if kind == 'attach':
                    trace['tips_item_top_at_attach'] = round(att.attach(item, case['size'][2]), 4)
                else:
                    att.detach()
                ev += 1
            robot.arm_target, robot.finger_target = sch.targets(rel_t)
            for _ in range(robot.decimation):             # the held item follows every physics step
                att.follow()
                robot_tick_targets(robot)
            pos, quat = io.body_pose()
            t_base = base_link_matrix(io, body_to_base)
            centre = pose_matrix(*io.item_pose(item)) @ np.array([0, 0, case['size'][2] / 2.0, 1.0])
            trace['t'].append(rel_t)
            trace['item_base_link'].append((np.linalg.inv(t_base) @ centre)[:3])
            trace['tilt_deg'].append(tilt_deg(quat))
            ph = sch.phase_at(rel_t)
            phase_tilt[ph] = max(phase_tilt.get(ph, 0.0), trace['tilt_deg'][-1])
            if fall_phase is None and trace['tilt_deg'][-1] > fall_tilt:
                fall_phase = (ph, round(rel_t, 2))
            if io.time() - t_cap >= capture_every:
                io.capture(f'pick{k}_{item}_{rel_t:06.2f}')
                t_cap = io.time()
            if legs == 'policy':
                policy_step(robot)
        att.detach()
        r = pick_mod.evaluate_case(case, trace, lambda b, p, m: comp(b, p, m), pc['pass'], ground_z_base)
        r['case'] = k
        r['fall_phase'] = fall_phase
        r['max_tilt_by_phase'] = {p: round(v, 1) for p, v in phase_tilt.items()}
        r['legs'] = legs
        results.append(r)
        io.place_item(item, *park[item])                   # out of the basket for the next case
        robot.arm_target, robot.finger_target = np.asarray(cfg['robot']['arm_rest'], float), float(cfg['robot']['finger_rest'])
        for _ in range(int(round(1.0 / tick_dt))):
            robot.tick((0.0, 0.0, 0.0), policy_on=(legs == 'policy'))
    return results


def robot_tick_targets(robot):
    robot.io.set_targets(robot.full_targets())
    robot.io.step(1)


def policy_step(robot):
    """The legs' policy at zero command, once per policy period (after its physics steps)."""
    pos, quat = robot.io.body_pose()
    gyro = robot.gyro(quat)
    cmd = robot.shaper(0.0, 0.0, 0.0, gyro[2], robot.dt * robot.decimation)
    q, dq = np.asarray(robot.io.q()), np.asarray(robot.io.dq())
    robot.leg_target = robot.policy.step(quat, gyro, cmd, q[robot.leg_idx], dq[robot.leg_idx])
