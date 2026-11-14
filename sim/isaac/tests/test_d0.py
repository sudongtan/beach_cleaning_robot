"""Mac tests for D0b / D0c (notes/briefs/D0.md): the policy port, the robot facts, the pick schedule,
the USD checks, the composed stage, and the whole walk / pick loop on a fake simulator.

Needs the laptop files (d0_from_laptop, notes/briefs/D0_laptop.md); skips without them.
"""

import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from conftest import ISAAC, REPO

import robot_model as rm

CFG = rm.load_robot_cfg()
LAPTOP = rm.laptop_dir(CFG)
if not (LAPTOP / 'MANIFEST.txt').is_file():
    pytest.skip(f'laptop files not found at {LAPTOP} (set BEACH_D0_LAPTOP_DIR)', allow_module_level=True)

import check_robot_usd as cru  # noqa: E402
import compose  # noqa: E402
import d0_runner  # noqa: E402
import pick as pick_mod  # noqa: E402
import walk  # noqa: E402

URDF_PATH = LAPTOP / CFG['urdf_import']['urdf']
URDF = rm.Urdf(URDF_PATH)
with open(LAPTOP / CFG['policy']['config']) as _f:
    PCFG = yaml.safe_load(_f)


# ------------------------------------------------------------------ policy (D0b)

def test_numpy_network_matches_torch_pairs():
    m = walk.Mlp(LAPTOP / CFG['policy']['weights'])
    io = np.load(LAPTOP / CFG['policy']['io_pairs'])
    out = m(io['obs'])
    assert out.shape == io['actions'].shape == (20, 12)
    assert np.allclose(out, io['actions'], rtol=1e-5, atol=1e-5)   # laptop's tolerance (MANIFEST)
    assert m.activation == 'elu'


def test_md5_of_policy_matches_the_logged_one():
    assert (LAPTOP / 'policy' / 'policy.pt.md5').read_text().split()[0].startswith('20a12e1f')


def _rot(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.array([math.cos(ang / 2), *(axis * math.sin(ang / 2))])


def test_quat_rotate_inverse_is_world_to_body():
    for axis, ang in (((0, 0, 1), 0.7), ((0, 1, 0), 0.3), ((1, 1, 0), -1.1)):
        q = _rot(axis, ang)
        r = d0_runner.quat_matrix(q)                      # body -> world
        v = np.array([0.2, -0.5, 0.9])
        assert np.allclose(walk.quat_rotate_inverse(q, v), r.T @ v, atol=1e-12)
    assert np.allclose(walk.quat_rotate_inverse((1, 0, 0, 0), (0, 0, -1)), (0, 0, -1))
    # nose down (pitch +0.3 about y): gravity seen in the body has a +x component
    g = walk.quat_rotate_inverse(_rot((0, 1, 0), 0.3), (0, 0, -1))
    assert g[0] > 0.25 and g[2] < -0.9


def _policy():
    pol, shaper, _ = walk.build_policy(LAPTOP, CFG)
    return pol, shaper


def test_observation_layout_and_scales():
    pol, _ = _policy()
    obs = pol.observation((1, 0, 0, 0), [0.4, 0, 0], [0.2, 0.0, 0.3], pol.default + 0.1, np.full(12, 2.0))
    assert obs.shape == (45,)
    assert np.allclose(obs[0:3], [0.4 * 0.25, 0, 0])                 # ang_vel * ang_vel_scale
    assert np.allclose(obs[3:6], [0, 0, -1])                          # projected gravity, level
    assert np.allclose(obs[6:9], [0.2 * 2.0, 0.0, 0.3 * 0.25])        # StateRL's (2, 2, 0.25), not the yaml
    assert np.allclose(obs[9:21], 0.1)                                 # (q - default) * 1.0
    assert np.allclose(obs[21:33], 2.0 * 0.05)                         # dq * 0.05
    assert np.allclose(obs[33:45], 0.0)                                # last actions after reset


def test_default_pose_and_joint_order_are_gazebos():
    ctrl = walk.load_controller()
    pol, _ = _policy()
    assert ctrl['joints'] == CFG['robot']['leg_joints']
    assert np.allclose(pol.default, [0.0, 0.8, -1.5] * 4)
    assert ctrl['update_rate'] / PCFG['decimation'] == 50.0


def test_hip_quirk_and_targets():
    pol, _ = _policy()
    pol.reset()
    obs = pol.observation((1, 0, 0, 0), [0, 0, 0], [0, 0, 0], pol.default, np.zeros(12))
    raw = np.clip(pol.mlp(obs[None])[0], pol.lo, pol.hi)
    t = pol.act(obs)
    want = raw.copy()
    want[[0, 3, 6, 9]] *= 0.5
    assert np.allclose(pol.actions, want, atol=1e-6)                   # the reduced action is fed back
    assert np.allclose(t, want * 0.25 + pol.default, atol=1e-6)


def test_command_shaping_matches_the_gazebo_adapter():
    _, sh = _policy()
    # C.7: 0.2 m/s wanted -> control_input ly 0.1041667 -> the policy sees 0.2083 (laptop summary)
    assert sh(0.2, 0.0, 0.0, 0.0, 0.02)[0] == pytest.approx(2.0 * 0.1041666641831398, abs=1e-6)
    sh2 = walk.CommandShaper(walk.load_adapter_params(REPO / CFG['policy']['adapter_config']))
    c1 = sh2(0.0, 0.0, 0.3, 0.0, 0.02)
    assert c1[2] == pytest.approx(0.3)                                 # first tick: no trim yet
    c2 = sh2(0.0, 0.0, 0.3, 0.0, 0.02)                                 # measured 0: trim grows
    assert c2[2] == pytest.approx(0.3 + 1.0 * 0.3 * 0.02)
    assert 'rclpy' not in sys.modules or hasattr(sys.modules['rclpy'], '__file__')


def test_scheduler_decimation():
    s = walk.Scheduler(4)
    assert [s.due() for _ in range(9)] == [True, False, False, False, True, False, False, False, True]


def test_speeds_from_trace():
    tr = [(t, 0.2 * t, 0.0, 0.3 * t) for t in np.arange(0, 5.01, 0.02)]
    v, wz = walk.speeds(tr, 2.5, 5.0)
    assert v == pytest.approx(0.2, rel=1e-3) and wz == pytest.approx(0.3, rel=1e-3)


# ------------------------------------------------------------------ robot facts

def test_urdf_matches_laptop_table():
    tab = yaml.safe_load((LAPTOP / 'urdf' / 'links_joints_custom.yaml').read_text())
    assert len(URDF.links) == tab['n_links'] and len(URDF.joints) == tab['n_joints']
    total = sum(v['mass'] or 0 for v in URDF.links.values())
    assert total == pytest.approx(tab['total_mass_kg'], abs=1e-5)
    assert URDF.root == CFG['robot']['root_link']
    assert sorted(URDF.movable()) == sorted(CFG['robot']['leg_joints'] + CFG['robot']['arm_joints']
                                            + CFG['robot']['finger_joints'])


def test_fixed_transforms():
    assert np.allclose(URDF.fixed_transform('arm_gripper_bar_link', 'arm_ee_gripper_link')[:3, 3], [0.08, 0, 0])
    assert np.allclose(URDF.fixed_transform('base_link', 'trunk'), np.eye(4))
    with pytest.raises(ValueError):
        URDF.fixed_transform('base_link', 'arm_gripper_bar_link')        # through movable joints


def _grasping_arm(monkeypatch):
    """grasping.arm.Arm('custom') on the laptop's URDF, without ROS (no xacro, no ament)."""
    pytest.importorskip('pinocchio')
    ament = types.ModuleType('ament_index_python')
    pk = types.ModuleType('ament_index_python.packages')
    pk.get_package_share_directory = lambda name: str(REPO / 'packages' / name)
    monkeypatch.setitem(sys.modules, 'ament_index_python', ament)
    monkeypatch.setitem(sys.modules, 'ament_index_python.packages', pk)
    monkeypatch.syspath_prepend(str(REPO / 'packages' / 'grasping'))
    import subprocess
    import grasping.arm as ga
    text = URDF_PATH.read_text()
    monkeypatch.setattr(ga.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=text))
    return ga


def test_compartments_match_the_planner(monkeypatch):
    ga = _grasping_arm(monkeypatch)
    arm = ga.Arm('custom', 30, 0.005, seed=0)
    mine = rm.compartments(URDF)
    for b in ('recyclable', 'residual'):
        theirs = ga.compartment(arm, b)
        for k in ('x', 'y'):
            assert np.allclose(mine[b][k], theirs[k], atol=1e-9)
        assert mine[b]['floor'] == pytest.approx(theirs['floor']) and mine[b]['rim'] == pytest.approx(theirs['rim'])
    for p in ([-0.1, 0.07, 0.15], [-0.1, -0.07, 0.15], [0.5, 0, 0], [-0.1, 0.0, 0.5]):
        assert rm.in_basket(mine, p) == ga.in_basket(arm, np.array(p))


def test_mac_planner_reproduces_the_laptop_plans(monkeypatch):
    """D0c: our planner on the Mac gives the laptop's joint targets (basic_stand: the planner's own legs)."""
    ga = _grasping_arm(monkeypatch)
    plans = json.loads((LAPTOP / CFG['pick']['plans']).read_text())
    cfg = plans['grasping_yaml']
    arm = ga.Arm('custom', cfg['ik_restarts'], cfg['ik_tolerance_m'], seed=0)
    # IK has random restarts, so the joint solution may differ; the jaw tips must land in the same
    # place with the same tool direction (the IK tolerance is 5 mm).
    for case in plans['stances']['basic_stand']['cases'][:3]:
        ok, why, plan = ga.plan_pick(arm, np.array(case['centre_base_link']), case['size'][2], cfg)
        assert ok == case['pick_ok'], why
        for k, q in case['pick_plan'].items():
            p_mac, x_mac = arm.fk(np.array(plan[k]), frame=ga.TIPS)
            p_lap, x_lap = arm.fk(np.array(q), frame=ga.TIPS)
            assert np.linalg.norm(p_mac - p_lap) < 2 * cfg['ik_tolerance_m'], (case['item'], k)
            assert np.dot(x_mac, x_lap) > math.cos(math.radians(cfg['approach_max_tilt_deg'] * 2)), (case['item'], k)


def test_drive_gains_and_units():
    g = rm.drive_gains(CFG, PCFG, 'custom')
    assert g['FR_hip_joint'] == (20.0, 0.5)
    assert g['arm_wrist_angle'] == (30.0, 0.9)                       # custom override
    assert g['arm_left_finger'] == (250.0, 8.0)
    assert rm.usd_angular_gain(180.0 / math.pi) == pytest.approx(1.0)


# ------------------------------------------------------------------ pick schedule (D0c)

def _cases():
    return pick_mod.load_cases(LAPTOP / CFG['pick']['plans'], CFG['pick']['stance'])


def test_schedule_follows_gazebo_timing():
    _, stance, cases = _cases()
    assert len(cases) == 9 and {c['item'] for c in cases} == {'pet_bottle_500', 'can_330', 'poly_cup'}
    case = cases[0]
    sch = pick_mod.case_schedule(case, CFG['pick'], CFG['robot']['arm_rest'], CFG['robot']['finger_rest'])
    steps = case['pick_steps'] + case['store_steps']
    n_arm = sum(s['kind'] == 'arm' for s in steps)
    n_fing = sum(s['kind'] == 'fingers' for s in steps)
    settle = CFG['pick']['settle_s']
    assert sch.duration == pytest.approx(n_arm * (2.5 + settle) + n_fing * (1.0 + settle))
    kinds = [e[1] for e in sch.events()]
    assert kinds == ['attach', 'detach']
    att = sch.events()[0][0]
    arm, fing = sch.targets(att)
    descend = [s for s in steps if s['phase'] == 'descend'][0]['value']
    assert np.allclose(arm, descend)                                  # attach after the descend
    arm, fing = sch.targets(sch.duration + 1)
    assert np.allclose(arm, steps[-1]['value'])


def test_schedule_is_smooth():
    sch = pick_mod.build_schedule([{'kind': 'arm', 'phase': 'a', 'value': [1, 0, 0, 0, 0, 0]}],
                                  np.zeros(6), 0.01, 2.0, 1.0, 0.0)
    assert sch.targets(0.0)[0][0] == 0.0 and sch.targets(1.0)[0][0] == pytest.approx(0.5)
    assert sch.targets(0.01)[0][0] < 0.001                             # starts at zero speed


# ------------------------------------------------------------------ USD checks

def fake_import(path, urdf=URDF, drop_joint=None, bad_mass=None, mesh_collider=False):
    """A minimal 'imported' robot USD from the URDF, for testing the checker and compose."""
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, '/beach_robot')
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    for name, link in urdf.links.items():
        p = UsdGeom.Xform.Define(stage, f'/beach_robot/{name}').GetPrim()
        if link['mass'] is not None and link['mass'] >= 0.01:
            UsdPhysics.RigidBodyAPI.Apply(p)
            UsdPhysics.MassAPI.Apply(p).CreateMassAttr(bad_mass if name == 'trunk' and bad_mass else link['mass'])
    if mesh_collider:
        m = UsdGeom.Mesh.Define(stage, '/beach_robot/FR_calf/collision')
        UsdPhysics.CollisionAPI.Apply(m.GetPrim())
    UsdGeom.Scope.Define(stage, '/beach_robot/joints')
    for name, j in urdf.joints.items():
        if name == drop_joint:
            continue
        path_ = f'/beach_robot/joints/{name}'
        if j['type'] in ('revolute', 'continuous'):
            jj = UsdPhysics.RevoluteJoint.Define(stage, path_)
            if j['lower'] is not None:
                jj.CreateLowerLimitAttr(math.degrees(j['lower']))
                jj.CreateUpperLimitAttr(math.degrees(j['upper']))
        elif j['type'] == 'prismatic':
            jj = UsdPhysics.PrismaticJoint.Define(stage, path_)
            jj.CreateLowerLimitAttr(j['lower'])
            jj.CreateUpperLimitAttr(j['upper'])
        else:
            jj = UsdPhysics.FixedJoint.Define(stage, path_)
    UsdPhysics.Scene.Define(stage, '/beach_robot/physicsScene')
    stage.GetRootLayer().Save()
    return stage


def test_check_accepts_a_faithful_import(tmp_path):
    rep = cru.check(fake_import(tmp_path / 'r.usda'), URDF)
    assert rep['ok'], rep['errors']
    assert any('physics scenes' in n for n in rep['notes'])


@pytest.mark.parametrize('kw,needle', [({'drop_joint': 'RL_calf_joint'}, 'joint RL_calf_joint missing'),
                                       ({'bad_mass': 5.0}, 'link trunk mass'),
                                       ({'mesh_collider': True}, 'triangle-mesh collider')])
def test_check_catches_bad_imports(tmp_path, kw, needle):
    rep = cru.check(fake_import(tmp_path / 'r.usda', **kw), URDF)
    assert not rep['ok'] and any(needle in e for e in rep['errors']), rep['errors']


def test_apply_drives(tmp_path):
    st = fake_import(tmp_path / 'r.usda')
    gains = rm.drive_gains(CFG, PCFG, 'custom')
    tg = cru.stand_targets(CFG, walk.load_controller()['stand_pos'])
    done = cru.apply_drives(st, URDF, gains, tg, CFG['leg_effort_limit'], CFG['robot']['leg_joints'])
    assert sorted(done) == sorted(URDF.movable())
    hip = st.GetPrimAtPath('/beach_robot/joints/FR_thigh_joint')
    d = UsdPhysics.DriveAPI(hip, 'angular')
    assert d.GetStiffnessAttr().Get() == pytest.approx(20.0 * math.pi / 180)
    assert d.GetMaxForceAttr().Get() == pytest.approx(23.5)
    assert d.GetTargetPositionAttr().Get() == pytest.approx(math.degrees(0.8))
    f = UsdPhysics.DriveAPI(st.GetPrimAtPath('/beach_robot/joints/arm_right_finger'), 'linear')
    assert f.GetStiffnessAttr().Get() == pytest.approx(250.0) and f.GetTargetPositionAttr().Get() == pytest.approx(-0.01)


# ------------------------------------------------------------------ composed stage

@pytest.mark.parametrize('scene,pick', [('flat', False), ('3', False), ('flat', True)])
def test_compose(tmp_path, scene, pick):
    robot = tmp_path / 'robot.usda'
    fake_import(robot)
    out, meta = compose.build(scene, robot, tmp_path / f'stage_{scene}.usda', pick)
    st = Usd.Stage.Open(str(out))
    assert st.GetPrimAtPath('/World/Terrain') or st.GetPrimAtPath('/World/Environment/terrain')
    rob = st.GetPrimAtPath('/World/Robot')
    assert st.GetPrimAtPath('/World/Robot/trunk'), 'the robot reference did not resolve'
    m = UsdGeom.Xformable(rob).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    x, y, z, yaw = meta['spawn_base_link']
    assert np.allclose(m.ExtractTranslation(), (x, y, z), atol=1e-6)
    ref = meta['reference']['standing']['trunk_above_ground_m']['mean']
    from terrain import height_at
    sc = compose.load_scene.export_terrain.load_scene(meta['seed'], Path(meta['scenes_yaml']))
    assert z - float(height_at(sc['terrain'], x, y)) == pytest.approx(ref + CFG['stand']['spawn_clearance_m'])
    ps = st.GetPrimAtPath('/World/PhysicsScene')
    assert ps.GetAttribute('physxScene:timeStepsPerSecond').Get() == 200
    # usd-core doesn't know PhysxSchema, so GetAppliedSchemas() hides it: read the authored list op
    assert 'PhysxSceneAPI' in ps.GetMetadata('apiSchemas').GetAddedOrExplicitItems()
    scenes = [p for p in st.Traverse() if p.IsA(UsdPhysics.Scene)]
    assert [str(p.GetPath()) for p in scenes] == ['/World/PhysicsScene']    # the robot's is inactive
    assert st.GetEndTimeCode() > st.GetStartTimeCode()
    if scene == 'flat':
        assert sc['terrain']['beach_slope'] == 0.0
    if pick:
        for n in meta['pick_items']:
            p = st.GetPrimAtPath(f'/World/PickItems/{n}')
            assert p.HasAPI(UsdPhysics.RigidBodyAPI)
        assert st.GetPrimAtPath('/World/Cameras/view_pick')
    else:
        assert st.GetPrimAtPath('/World/Cameras/view_walk')
    assert st.GetPrimAtPath('/World/Cameras/top')                      # the scene's cameras still there


# ------------------------------------------------------------------ the loop on a fake simulator

class FakeIO:
    """Kinematics only (Pinocchio, fixed base_link): joints follow their targets with a 50 ms lag,
    the body moves as a script says, a released item drops onto the basket floor or the ground."""

    def __init__(self, spawn, body_motion=None, shuffle=True):
        import pinocchio as pin
        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(URDF_PATH))
        self.data = self.model.createData()
        names = [n for n in self.model.names[1:]]
        self.dof_names = sorted(names) if shuffle else names           # simulator order != ours
        self.qmap = [self.model.joints[self.model.getJointId(n)].idx_q for n in self.dof_names]
        self.qv = np.zeros(len(self.dof_names))
        self.target = self.qv.copy()
        self.n = 0
        self.dt = 1.0 / 200
        self.spawn = spawn
        self.motion = body_motion or (lambda t: (0.0, 0.0, 0.0, 0.0))
        self.items = {}
        self.kin = {}
        self.comp = rm.compartments(URDF)
        self.frames = []

    def q(self):
        return self.qv.copy()

    def dq(self):
        return np.zeros_like(self.qv)

    def set_q(self, q):
        self.qv = np.asarray(q, float).copy()

    def set_targets(self, q):
        self.target = np.asarray(q, float).copy()

    def base(self):
        x, y, z, yaw = self.spawn
        dx, dy, dz, dyaw = self.motion(self.time())
        c, s = math.cos(yaw + dyaw), math.sin(yaw + dyaw)
        t = np.eye(4)
        t[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        t[:3, 3] = (x + dx, y + dy, z + dz)
        return t

    def body_pose(self):
        return d0_runner.matrix_pose(self.base())

    def body_ang_vel_world(self):
        return np.zeros(3)

    def link_pose(self, name):
        q = self.pin.neutral(self.model)
        for k, idx in enumerate(self.qmap):
            q[idx] = self.qv[k]
        self.pin.framesForwardKinematics(self.model, self.data, q)
        f = self.data.oMf[self.model.getFrameId(name)]
        t = np.eye(4)
        t[:3, :3], t[:3, 3] = f.rotation, f.translation
        return d0_runner.matrix_pose(self.base() @ t)

    def item_pose(self, name):
        return self.items[name]

    def place_item(self, name, xyz, wxyz):
        self.items[name] = (np.asarray(xyz, float), np.asarray(wxyz, float))

    def hold_item(self, name, xyz, wxyz):
        self.place_item(name, xyz, wxyz)

    def set_item_kinematic(self, name, on):
        self.kin[name] = on
        if not on:                                     # drop: basket floor if above it, else the ground
            p, q = self.items[name]
            pb = np.linalg.inv(self.base()) @ np.append(p, 1.0)
            inside = rm.in_basket(self.comp, [pb[0], pb[1], self.comp['recyclable']['floor']], 0.0)
            zb = self.comp['recyclable']['floor'] if inside else -self.spawn[2]
            p2 = (self.base() @ np.array([pb[0], pb[1], zb, 1.0]))[:3]
            self.items[name] = (p2, q)

    def step(self, n):
        a = min(1.0, n * self.dt / 0.05)
        self.qv = self.qv + a * (self.target - self.qv)
        self.n += n

    def time(self):
        return self.n * self.dt

    def capture(self, tag):
        self.frames.append(tag)


def test_walk_loop_and_judge_on_a_fake():
    pytest.importorskip('pinocchio')
    pol, sh = _policy()
    ref = yaml.safe_load((LAPTOP / 'c7' / 'flat' / 'summary.yaml').read_text())
    seq = CFG['walk']['sequence']
    t_walk = CFG['stand']['hold_s'] + seq[0]['seconds']

    def motion(t):                                     # what Gazebo did: 0.216 m/s, then 0.29 rad/s
        tw = min(max(t - t_walk, 0.0), 5.0)
        tt = min(max(t - t_walk - 5.0, 0.0), 5.0)
        return (0.216 * tw, 0.0, 0.0, 0.29 * tt)
    io = FakeIO((1.0, 2.0, 0.30, 0.0), motion)
    res = d0_runner.run_walk(io, CFG, pol, sh, lambda x, y: 0.0)
    assert [p['name'] for p in res['phases']] == ['stand', 'walk_0.2', 'turn_0.3', 'stop']
    ok, reasons, nums = d0_runner.judge_walk(res, ref, CFG['walk']['pass'])
    assert ok, reasons
    assert io.frames                                   # capture called on schedule
    # a robot that doesn't move fails
    io2 = FakeIO((1.0, 2.0, 0.30, 0.0))
    ok2, reasons2, _ = d0_runner.judge_walk(d0_runner.run_walk(io2, CFG, pol, sh, lambda x, y: 0.0), ref,
                                            CFG['walk']['pass'])
    assert not ok2 and any('walk_0.2' in r for r in reasons2)
    # the joint mapping: leg targets reach the simulator's leg slots
    assert np.allclose(io.target[[io.dof_names.index(j) for j in CFG['robot']['arm_joints']]], CFG['robot']['arm_rest'])


def test_falling_is_detected():
    pol, sh = _policy()
    ref = yaml.safe_load((LAPTOP / 'c7' / 'flat' / 'summary.yaml').read_text())

    class Tipping(FakeIO):
        def base(self):
            t = super().base()
            if self.time() > 4.0:
                t[:3, :3] = d0_runner.quat_matrix(_rot((1, 0, 0), 1.2))
            return t
    pytest.importorskip('pinocchio')
    res = d0_runner.run_walk(Tipping((0, 0, 0.3, 0)), CFG, pol, sh, lambda x, y: 0.0)
    assert res['fell_at_s'] is not None
    assert not d0_runner.judge_walk(res, ref, CFG['walk']['pass'])[0]


def test_all_pick_plans_succeed_on_a_fake():
    """The whole D0c chain on the Mac: plan steps -> schedule -> joint mapping -> TCP (URDF offset from
    the gripper bar) -> attach / follow / detach -> item in base_link -> the right compartment."""
    pytest.importorskip('pinocchio')
    pol, sh = _policy()
    plans, stance, cases = _cases()
    ground = float(stance['ground_z_in_base_link'])
    spawn = (3.0, 4.0, -ground, 0.4)                   # base_link at the stance height over flat ground z = 0
    io = FakeIO(spawn)
    items = sorted({c['item'] for c in cases})
    park = {}
    for k, n in enumerate(items):
        io.place_item(n, (0.0, 10.0 + k, 0.0), (1.0, 0, 0, 0))
        park[n] = io.item_pose(n)
    comp = rm.compartments(URDF)
    res = d0_runner.run_picks(io, CFG, pol, sh, cases, lambda b, p, m: rm.in_compartment(comp, b, p, m),
                              URDF.fixed_transform('arm_gripper_bar_link', 'arm_ee_gripper_link'),
                              URDF.fixed_transform('trunk', 'base_link'), ground, park,
                              bar_to_tips=URDF.fixed_transform('arm_gripper_bar_link', 'arm_jaw_tips_link'))
    assert len(res) == 9
    bad = [r for r in res if not r['ok']]
    assert not bad, bad
    for r in res:
        assert r['tips_item_top_at_attach_m'] < CFG['pick']['pass']['max_tips_item_top_m'], r
        assert r['height_after_lift_m'] > 0.10
