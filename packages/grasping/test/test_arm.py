"""Graspability, IK reach envelope, pick plans, compartments and the bin (robot model from the
xacro; needs a sourced ROS environment with beach_description built and Pinocchio)."""

from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip('pinocchio')
pytest.importorskip('ament_index_python')

from grasping.arm import (TCP, TIPS, Arm, bin_release_target, compartment, graspable, held_item,  # noqa: E402
                          in_basket, in_envelope, item_yaw, near, plan_pick, plan_store, plan_unload)

REPO = Path(__file__).resolve().parents[3]
CFG = yaml.safe_load(open(REPO / 'config' / 'grasping.yaml'))['grasping']
LIMITS = yaml.safe_load(open(REPO / 'config' / 'robot_limits.yaml'))['robot_limits']
OBJECTS = yaml.safe_load(open(REPO / 'sim' / 'common' / 'objects.yaml'))
MAX_H = 0.075                                    # = cross-phase limit: jaw length + tip height - bar clearance


def box_size(spec):
    return spec['collision'][0]['size']


@pytest.fixture(scope='module')
def arm():
    return Arm(LIMITS['variant'], CFG['ik_restarts'], CFG['ik_tolerance_m'])


def test_every_trash_item_is_graspable(arm):
    for name, spec in OBJECTS['trash'].items():
        ok, why = graspable(box_size(spec), arm.finger_closed, arm.finger_open, LIMITS['jaw_margin_m'], MAX_H)
        assert ok, f'{name}: {why}'


def test_graspability_limits(arm):
    args = (arm.finger_closed, arm.finger_open, LIMITS['jaw_margin_m'], MAX_H)
    assert graspable([0.32, 0.14, 0.06], *args) == (False, 'too_wide')
    assert graspable([0.10, 0.015, 0.015], *args) == (False, 'too_narrow')
    assert graspable([0.08, 0.08, 0.10], *args) == (False, 'too_tall')


@pytest.mark.parametrize('body_height', LIMITS['bin_drop']['body_heights_m'])
@pytest.mark.parametrize('item_height', [0.022, 0.075])
def test_whole_envelope_is_reachable(arm, body_height, item_height):
    """Every item centre on a grid over the envelope, on flat ground, gives a full pick plan."""
    fails = []
    for x in np.linspace(*CFG['reach_x_m'], 4):
        for y in np.linspace(-CFG['reach_y_max_m'], CFG['reach_y_max_m'], 3):
            centre = (x, y, -body_height + item_height / 2)
            ok, why, _ = plan_pick(arm, centre, item_height, CFG)
            if not ok:
                fails.append((round(x, 3), round(y, 3), why))
    assert not fails


def test_pick_plan_phases(arm):
    centre, h = (0.48, 0.0, -0.285 + 0.033), 0.066
    ok, why, plan = plan_pick(arm, centre, h, CFG)
    assert ok, why
    tips = {p: arm.fk(q, TIPS)[0] for p, q in plan.items()}
    top = centre[2] + h / 2
    assert tips['hover'][2] == pytest.approx(top + CFG['hover_above_item_m'], abs=CFG['ik_tolerance_m'])
    assert tips['hover'][:2] == pytest.approx(centre[:2], abs=CFG['ik_tolerance_m'])
    assert tips['above'][2] == pytest.approx(tips['hover'][2] + CFG['pregrasp_above_m'], abs=0.01)
    assert tips['lift'][2] == pytest.approx(tips['hover'][2] + CFG['lift_above_m'], abs=0.01)
    for q in plan.values():
        axis = arm.fk(q, TIPS)[1]
        assert np.degrees(np.arccos(-axis[2])) <= CFG['approach_max_tilt_deg'] + 0.5


def test_outside_envelope_is_refused(arm):
    assert plan_pick(arm, (0.9, 0.0, -0.25), 0.05, CFG)[1] == 'outside_envelope'
    assert plan_pick(arm, (-0.4, 0.0, -0.25), 0.05, CFG)[1] == 'outside_envelope'
    assert not in_envelope(0.48, CFG['reach_y_max_m'] + 0.01, CFG)


@pytest.mark.parametrize('comp', ['recyclable', 'residual'])
def test_named_pose_is_over_its_compartment(arm, comp):
    tcp, axis = arm.fk(arm.poses[f'over_{comp}']['arm'], TCP)
    centre = arm.frame(f'basket_{comp}_link')
    assert np.hypot(*(tcp[:2] - centre[:2])) < 0.03
    assert tcp[2] > centre[2] and axis[2] < -0.95                         # above the rim, pointing down


@pytest.mark.parametrize('body_height', LIMITS['bin_drop']['body_heights_m'])
def test_reaches_over_the_bin(arm, body_height):
    bd = LIMITS['bin_drop']
    rim = max(p['pose'][2] + p['size'][2] / 2 for p in OBJECTS['bins']['bin_recyclable']['collision'])
    place = yaml.safe_load(open(REPO / 'config' / 'task.yaml'))['task']['fsm']['bin_place_distance_m']
    near_wall = place - 0.3
    up = arm.poses['up']['arm']
    for hang in (0.0, MAX_H + CFG['hover_above_item_m']):            # empty jaws / the tallest item held
        target = bin_release_target((place, 0.0, -body_height + rim), near_wall, bd, hang)
        ok, q, err, tilt = arm.ik(target, bd['max_tilt_deg'], seed_q=up,
                                  bounds=near(up, CFG['release_joint_spread_rad']))
        assert ok, f'hang {hang}: err {err * 100:.1f} cm, tilt {tilt:.0f} deg'
        assert abs(q[0]) < 1.0 + 1e-6                                 # on the front side, no half-turn swing


def item_sizes():
    """(name, vertical extent, lying length) of every trash item as it lies (flat, largest face down)."""
    return [(n, box_size(s)[2], box_size(s)[0]) for n, s in OBJECTS['trash'].items()]


@pytest.mark.parametrize('comp', ['recyclable', 'residual'])
def test_items_can_be_taken_out_of_the_basket(arm, comp):
    """Anywhere it can lie in a compartment, every item gets an unload plan: tips hover above it,
    the gripper bar stays inside the compartment, and the lift takes the item over the rim."""
    c = compartment(arm, comp)
    u = CFG['unload']
    fails = []
    for name, h, length in item_sizes():
        half = min(length, c['x'][1] - c['x'][0]) / 2
        for x in np.linspace(c['x'][0] + half, c['x'][1] - half, 3):
            for y in np.linspace(c['y'][0] + 0.035, c['y'][1] - 0.035, 2):
                centre = (x, y, c['floor'] + h / 2)
                ok, why, plan = plan_unload(arm, comp, centre, h, CFG)
                if not ok:
                    fails.append((name, round(x, 3), round(y, 3), why))
                    continue
                tips = arm.fk(plan['hover'], TIPS)[0]
                assert tips[2] == pytest.approx(c['floor'] + h + CFG['hover_above_item_m'], abs=0.006)
                assert abs(arm.jaw_axis(plan['hover'])[0]) > 0.99            # bar along the compartment
                bar = arm.bar[1] / 2
                assert c['x'][0] + bar <= tips[0] + 1e-6 and tips[0] - 1e-6 <= c['x'][1] - bar
                lifted = arm.fk(plan['above'], TIPS)[0][2] - CFG['hover_above_item_m'] - h
                assert lifted >= c['rim'] + u['rim_clearance_m'] - 0.006       # item bottom over the rim
    assert not fails


@pytest.mark.parametrize('comp', ['recyclable', 'residual'])
def test_unload_moves_stay_on_one_side(arm, comp):
    """over -> above -> hover never swings a joint by more than the configured spread."""
    c = compartment(arm, comp)
    ok, why, plan = plan_unload(arm, comp, ((c['x'][0] + c['x'][1]) / 2, (c['y'][0] + c['y'][1]) / 2,
                                            c['floor'] + 0.033), 0.066, CFG)
    assert ok, why
    over = np.array(arm.poses[f'over_{comp}']['arm'])
    for q in plan.values():
        assert abs(q[0] - over[0]) <= 1.0 + 1e-6 and abs(q[3] - over[3]) <= 1.0 + 1e-6


@pytest.mark.parametrize('body_height', LIMITS['bin_drop']['body_heights_m'])
def test_release_moves_with_the_item_clear_the_bin_wall(arm, body_height):
    """up -> release -> up (joint space, as the controller runs it) with the tallest item hanging
    below the tips: neither the arm frames nor the item's lowest point pass the near wall below the rim."""
    import pinocchio as pin
    bd = LIMITS['bin_drop']
    rim = max(p['pose'][2] + p['size'][2] / 2 for p in OBJECTS['bins']['bin_recyclable']['collision']) - body_height
    place = yaml.safe_load(open(REPO / 'config' / 'task.yaml'))['task']['fsm']['bin_place_distance_m']
    wall = place - 0.3
    hang = MAX_H + CFG['hover_above_item_m']
    up = np.array(arm.poses['up']['arm'])
    ok, rel, *_ = arm.ik(bin_release_target((place, 0.0, rim), wall, bd, hang), bd['max_tilt_deg'], seed_q=up,
                         bounds=near(up, CFG['release_joint_spread_rad']))
    assert ok
    clear = bd['arm_wall_clearance_m']
    frames = [f.name for f in arm.model.frames if f.name.startswith('arm_') and f.type == pin.FrameType.BODY]
    for s in np.linspace(0, 1, 41):
        q = up + s * (rel - up)
        tips, axis = arm.fk(q, TIPS)
        points = {'item': tips + hang * axis}
        pin.framesForwardKinematics(arm.model, arm.data, arm.full(q))
        points.update({f: arm.data.oMf[arm.model.getFrameId(f)].translation for f in frames})
        for name, p in points.items():
            assert not (abs(p[0] - wall) < clear and p[2] < rim + clear), f'{name} at {p.round(3)}, s={s:.2f}'


def test_in_basket():
    from grasping.arm import compartment, in_basket
    arm = Arm('custom')
    for b in ('recyclable', 'residual'):
        c = compartment(arm, b)
        x, y = (c['x'][0] + c['x'][1]) / 2, (c['y'][0] + c['y'][1]) / 2
        assert in_basket(arm, (x, y, c['floor'] + 0.03))
        assert in_basket(arm, (x, y, c['rim'] + 0.05))                  # lying on other items
        assert not in_basket(arm, (x, y, c['floor'] - 0.30))             # on the ground below
        assert not in_basket(arm, (c['x'][1] + 0.10, y, c['floor'] + 0.03))
    assert not in_basket(arm, (-20.392, -2.813, -0.528))


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def picked(arm, spec, yaw):
    """An item as held after an attach-mode pick from flat ground at yaw (the jaws don't turn to it)."""
    size = box_size(spec)
    centre = np.array([0.48, 0.0, -0.25 + size[2] / 2])
    ok, why, plan = plan_pick(arm, centre, size[2], CFG)
    assert ok, why
    t, R = arm.pose(plan['hover'], TIPS)
    return {'p': R.T @ (centre - t), 'R': R.T @ rz(yaw), 'size': size}


def landed(arm, q, held, drop):
    c, Rb, e = held_item(arm, q, held)
    return {'centre': c - [0, 0, drop], 'half': e[:2] / 2, 'top': c[2] + e[2] / 2 - drop}, Rb, e


@pytest.mark.parametrize('comp', ['recyclable', 'residual'])
def test_store_lays_every_item_along_the_compartment_on_its_floor(arm, comp):
    """A2b a2b_full1 / store_drop_test: one fixed release point stacked the items to the rim; and a
    held bottle can hang across the 14.4 cm compartment (the pick doesn't turn the jaws to the item)."""
    c = compartment(arm, comp)
    st = CFG['store']
    for name, spec in OBJECTS['trash'].items():
        for yaw in (0.0, 0.7, 1.571, 2.5):
            held = picked(arm, spec, yaw)
            ok, why, plan, info = plan_store(arm, comp, held, [], CFG)
            assert ok, (name, yaw, why)
            item, Rb, e = landed(arm, plan['release'], held, info['drop'])
            # drop_gap_m, more only for items so flat that the gripper bar would reach the rim (cigarette pack)
            assert st['drop_gap_m'] - 0.006 <= info['drop'] <= 0.03, (name, yaw, info['drop'])
            assert item['centre'][2] - e[2] / 2 == pytest.approx(c['floor'], abs=0.006), (name, yaw)
            if abs(held['size'][0] - held['size'][1]) > 0.01:                     # square: any yaw
                assert abs(np.sin(item_yaw(Rb, held['size']))) < np.sin(np.radians(5)), (name, yaw)   # along x
            x, y = item['centre'][:2]
            assert c['x'][0] - 1e-3 <= x - e[0] / 2 and x + e[0] / 2 <= c['x'][1] + 1e-3, (name, yaw)
            assert c['y'][0] - 1e-3 <= y - e[1] / 2 and y + e[1] / 2 <= c['y'][1] + 1e-3, (name, yaw)
            for q in (plan['release'], plan['above']):
                bar = arm.fk(q, 'arm_gripper_bar_link')[0][2] - arm.bar[0] / 2
                assert bar >= c['rim'] + st['bar_clearance_m'] - 0.006, (name, yaw)


FILL = {'recyclable': ['pet_bottle_500', 'food_tin', 'can_330', 'glass_bottle_330', 'glass_jar'],
        'residual': ['food_container', 'poly_cup', 'cigarette_pack', 'food_container', 'poly_cup']}


@pytest.mark.parametrize('comp', ['recyclable', 'residual'])
def test_a_full_compartment_stays_below_the_rim(arm, comp):
    """Five items (the world model's limit) fit without anything sticking out above the rim; one
    more that wouldn't fit is refused as compartment_full."""
    c = compartment(arm, comp)
    others, rng = [], np.random.default_rng(3)
    for name in FILL[comp]:
        held = picked(arm, OBJECTS['trash'][name], rng.uniform(-np.pi, np.pi))
        ok, why, plan, info = plan_store(arm, comp, held, others, CFG)
        assert ok, (name, why)
        item, _, _ = landed(arm, plan['release'], held, info['drop'])
        assert item['top'] <= c['rim'] - CFG['store']['top_below_rim_m'] + 1e-3, name
        others.append(item)
    tall = [{'centre': (np.mean(c['x']), np.mean(c['y']), c['rim'] - 0.03), 'half': (0.2, 0.1), 'top': c['rim'] - 0.02}]
    ok, why, _, _ = plan_store(arm, comp, picked(arm, OBJECTS['trash']['food_tin'], 0.0), tall, CFG)
    assert not ok and why == 'compartment_full'


def lowest_over_basket(arm, qa, qb, held, rim, n=40):
    """Lowest held-item bottom above the rim along a joint-space move, while over the basket."""
    worst = np.inf
    for f in np.linspace(0.0, 1.0, n):
        c, _, e = held_item(arm, np.asarray(qa) + f * (np.asarray(qb) - np.asarray(qa)), held)
        if in_basket(arm, c, 0.10):
            worst = min(worst, c[2] - e[2] / 2 - rim)
    return worst


@pytest.mark.parametrize('comp', ['recyclable', 'residual'])
def test_store_move_keeps_the_held_item_above_the_rim(arm, comp):
    """A2b a2b_s3full: up -> over_<comp> -> above let the item hang 3 cm below the rim while the wrist
    turned; it hit the basket and the rigid attach pushed the robot over. up -> above stays clear."""
    c = compartment(arm, comp)
    up = arm.poses['up']['arm']
    for name, spec in OBJECTS['trash'].items():
        for yaw in (0.0, 0.7, 1.571, 2.5):
            held = picked(arm, spec, yaw)
            ok, why, plan, _ = plan_store(arm, comp, held, [], CFG)
            assert ok, why
            assert lowest_over_basket(arm, up, plan['above'], held, c['rim']) > 0.005, (name, yaw)
