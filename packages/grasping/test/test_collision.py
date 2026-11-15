"""Collision check of planned arm moves (B5, grasping.collision): known contacts are flagged, the check
picks clean plans, and with config `collision: null` the planner gives the 9ac4816 plans bit for bit.
Needs a sourced ROS environment with beach_description built, Pinocchio and coal."""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('pinocchio')
pytest.importorskip('coal')
pytest.importorskip('ament_index_python')

from test_arm import CFG, LIMITS, OBJECTS, ROW, box_size, landed, picked, rz  # noqa: E402

from grasping.arm import (TIPS, Arm, compartment, first_plan_only, held_item, plan_pick,  # noqa: E402
                          plan_store, plan_unload)
from grasping.collision import Collision, State, reason  # noqa: E402
from grasping.sequence import bin_steps, store_steps  # noqa: E402

DATA = Path(__file__).resolve().parent / 'data'
COL = CFG['collision']
SHOW = np.asarray(CFG['show_pose'], dtype=float)          # where a pick ends: a store starts here
OLD = {**CFG, 'store': {**CFG['store'], 'slots_x': 3, 'slots_y': 2,         # the 9ac4816 store planner
                        'bar_box_exact': False,                            # (B5.3: 4 x 3, exact bar;
                        'wall_retry_m': None, 'fill_corner': None,         # B5.4: 8 x 3 from a corner,
                        'seeded_ik': False, 'inside_walls': False,         # retry, IK tries, footprint;
                        'lower_vias': 0},                                  # run 3: lower waypoints)
       'unload': {**CFG['unload'], 'lift_vias': 0}}                         # sweep 7: lift waypoints


@pytest.fixture(scope='module')
def arm():
    return Arm(LIMITS['variant'], CFG['ik_restarts'], CFG['ik_tolerance_m'])


@pytest.fixture(scope='module')
def col(arm):
    return Collision.from_config(arm, COL)


def store_check(arm, col, comp, state, object_id):
    def check(plan):
        failure, _, _ = col.check_steps(store_steps(arm.poses, comp, arm.finger_open, object_id, plan),
                                        SHOW, state)
        return None if failure is None else reason(failure)
    return check


def test_geometry_cache_does_not_reuse_a_dead_state(arm, col):
    """B5.4: the cache was keyed by id(state.held); a new held dict made after the old one died got its
    id and the old item's box (a different size: wrong verdicts, order-dependent test failures)."""
    for i in range(50):
        size = (0.05 + 0.001 * i, 0.06, 0.07)
        geom, _ = col._geometry(State(held={'id': 'i', 'R': np.eye(3), 'p': np.zeros(3), 'size': size}))
        box = [g for g in geom.geometryObjects if g.name == 'held:i'][0].geometry
        assert np.allclose(2 * np.asarray(box.halfSide), size), (i, box.halfSide)


def test_named_moves_are_clear(arm, col):
    """Checked once at node start: every move between the named poses, empty basket, nothing held."""
    q = {**{k: v['arm'] for k, v in arm.poses.items()}, 'show': CFG['show_pose']}
    for a, b in [('rest', 'up'), ('up', 'rest'), ('up', 'show'), ('show', 'up'), ('up', 'over_recyclable'),
                 ('over_recyclable', 'up'), ('up', 'over_residual'), ('over_residual', 'up')]:
        assert col.check_move(q[a], q[b], State()) is None, (a, b)


@pytest.mark.parametrize('comp', ['recyclable', 'residual'])
def test_box_in_the_basket_wall_at_release_is_flagged(arm, col, comp):
    """B5.2 test 1: the held box moved onto the compartment's outer wall at the release pose."""
    held = picked(arm, OBJECTS['trash']['glass_jar'], 0.0)
    check = store_check(arm, col, comp, State(held=dict(held, id='jar')), 'jar')
    ok, why, plan, _ = plan_store(arm, comp, held, [], CFG, check)
    assert ok, why
    assert check(plan) is None                                             # where the planner puts it: clear
    c = compartment(arm, comp)
    ci, _, _ = held_item(arm, plan['release'], held)
    wall = c['y'][1] if comp == 'recyclable' else c['y'][0]
    _, R = arm.pose(plan['release'], TIPS)
    moved = dict(held, id='jar', p=np.asarray(held['p']) + R.T @ np.array([0.0, wall - ci[1], 0.0]))
    why = store_check(arm, col, comp, State(held=moved), 'jar')(plan)
    assert why is not None and why.startswith('collision:lower:') and 'basket_link' in why and 'held:jar' in why, why


@pytest.mark.parametrize('comp', ['recyclable', 'residual'])
def test_c1_axis_s3_half_turn_is_flagged(arm, col, comp):
    """B5.2 test 2: with wrist_branch any, `above` and `release` can take wrist turns ~pi apart and the
    bottle turns through the walls while lowered (c1_axis s3 arm-rule violation): every such plan fails on
    the 9ac4816 slot grid; on today's grid a turn the check passes is measured clear on every sample."""
    for cfg in ({**OLD, 'wrist_branch': 'any'}, {**CFG, 'wrist_branch': 'any'}):
        turns = half_turns(arm, col, comp, cfg)
        assert turns, 'no half turn left to reproduce'
        if cfg['store'] == OLD['store']:
            assert all(why is not None and why.startswith('collision:lower:') for *_, why, _ in turns), turns
        for name, yaw, why, plan_state in turns:
            if why is None:
                assert not col.scan_move(*plan_state, step=COL['step_rad'] / 4), (name, yaw)


@pytest.mark.parametrize('comp', ['recyclable', 'residual'])
def test_checked_store_lower_stays_inside_the_walls(arm, col, comp):
    """B5.4: stores fill from a corner, 5 mm from two walls; the joint-space arc of `lower` bulges past
    them unchecked (test_arm), so with the check every plan of every item and yaw is clear on a 4x finer
    scan of above -> release (through the store.lower_vias points the arm goes down by)."""
    for name, spec in OBJECTS['trash'].items():
        for yaw in np.linspace(-np.pi, np.pi, 7)[:-1]:
            held = picked(arm, spec, yaw)
            state = State(held=dict(held, id='i'))
            ok, why, plan, _ = plan_store(arm, comp, held, [], CFG, store_check(arm, col, comp, state, 'i'))
            assert ok, (name, yaw, why)
            path = [plan['above']] + list(plan.get('via') or []) + [plan['release']]
            for qa, qb in zip(path, path[1:]):
                assert not col.scan_move(qa, qb, state, step=COL['step_rad'] / 4), (name, yaw)


def half_turns(arm, col, comp, cfg):
    turns = []
    for name, spec in OBJECTS['trash'].items():
        if abs(box_size(spec)[0] - box_size(spec)[1]) < 0.01:            # square: any turn is clear
            continue
        for yaw in np.linspace(-np.pi, np.pi, 7)[:-1]:
            held = picked(arm, spec, yaw)
            ok, why, plan, _ = plan_store(arm, comp, held, [], cfg)
            assert ok, (name, yaw, why)
            if abs(plan['release'][-1] - plan['above'][-1]) > 2.5:
                state = State(held=dict(held, id='i'))
                turns.append((name, yaw, store_check(arm, col, comp, state, 'i')(plan),
                              (plan['above'], plan['release'], state)))
    return turns


def jar_on_bottle(arm):
    """pickfix4: a bottle lying flat along the recyclable compartment (centre (-0.125, 0.106), top 0.170),
    the jar stored next."""
    bottle = {'centre': np.array([-0.125, 0.106, 0.1375]), 'half': (0.105, 0.0325), 'top': 0.170}
    items = {'bottle': {'centre': bottle['centre'], 'R': np.eye(3), 'size': (0.21, 0.065, 0.065)}}
    return bottle, items


@pytest.mark.parametrize('yaw', [0.0, 1.2, -2.0])
def test_pickfix4_jar_on_bottle(arm, col, yaw):
    """B5.2 test 3: without the check the jar's slot (y 0.044 in pickfix4) lowers it into the bottle; with
    the check the planner refuses that slot and finds a clean one."""
    bottle, items = jar_on_bottle(arm)
    held = picked(arm, OBJECTS['trash']['glass_jar'], yaw)
    state = State(items, held=dict(held, id='jar'))
    check = store_check(arm, col, 'recyclable', state, 'jar')
    ok, why, plan, info = plan_store(arm, 'recyclable', held, [bottle], OLD)
    assert ok, why
    assert info['slot'][1] == pytest.approx(0.044, abs=0.005)
    assert check(plan) == 'collision:lower:item:bottle/held:jar'
    ok, why, plan, info = plan_store(arm, 'recyclable', held, [bottle], CFG, check)
    assert ok, (why, info['rejected'])
    assert ('collision:lower:item:bottle/held:jar', ) in {(w, ) for _, w in info['rejected']}
    assert check(plan) is None


def test_off_gives_the_9ac4816_plans():
    """B5.2 test 5: check=None (config collision: null) = the planner of 9ac4816, bit for bit, on a fixed
    seed (reference: test/data/plans_9ac4816.json, made with the 9ac4816 arm.py in the same order)."""
    ref = json.load(open(DATA / 'plans_9ac4816.json'))
    helper = Arm(LIMITS['variant'], CFG['ik_restarts'], CFG['ik_tolerance_m'], seed=1)
    helds = []
    for name, yaw in (('pet_bottle_500', 0.3), ('glass_jar', 1.2), ('food_container', -0.8)):
        size = box_size(OBJECTS['trash'][name])
        centre = np.array([0.48, 0.0, -0.25 + size[2] / 2])
        ok, why, plan = plan_pick(helper, centre, size[2], CFG)
        t, R = helper.pose(plan['hover'], TIPS)
        helds.append({'p': R.T @ (centre - t), 'R': R.T @ rz(yaw), 'size': size})
    for h, r in zip(helds, ref['held']):
        assert np.array_equal(h['p'], r['p']) and np.array_equal(h['R'], r['R'])
    arm = Arm(LIMITS['variant'], CFG['ik_restarts'], CFG['ik_tolerance_m'], seed=0)
    out = {'pick': [], 'unload': [], 'store': []}
    for c in ((0.45, 0.05, -0.25), (0.50, -0.08, -0.26), (0.55, 0.0, -0.24)):
        out['pick'].append(plan_pick(arm, c, 0.066, CFG)[2])
    for comp in ('recyclable', 'residual'):
        cc = compartment(arm, comp)
        centre = (np.mean(cc['x']), np.mean(cc['y']), cc['floor'] + 0.033)
        out['unload'].append(plan_unload(arm, comp, centre, 0.066, OLD)[2])
    for h, comp in zip(helds, ('recyclable', 'residual', 'recyclable')):
        out['store'].append(plan_store(arm, comp, h, [], OLD)[2])
    for kind, plans in out.items():
        assert len(plans) == len(ref[kind])
        for plan, r in zip(plans, ref[kind]):
            assert set(plan) == set(r), kind
            for k in plan:
                assert np.asarray(plan[k]).tolist() == r[k], (kind, k)


def unload_checker(arm, col, comp, obj, base=ROW):
    """node.unload_check offline: the unload of obj landed as box (alone in the basket), first plan only,
    checked from rest up to over_bin."""
    rest, cfg = np.asarray(arm.poses['rest']['arm'], dtype=float), first_plan_only(base)

    def unload_check(box, height):
        state = State({obj: box}, excluded={obj})

        def check(plan):
            steps = bin_steps(arm.poses, comp, plan, plan['above'], arm.finger_open, arm.finger_closed, obj)
            failure, _, _ = col.check_steps(steps[:next(i for i, st in enumerate(steps) if st.phase == 'over_bin')],
                                            rest, state)
            return None if failure is None else reason(failure)
        ok, why, _ = plan_unload(arm, comp, box['centre'], height, cfg, check)
        return None if ok else why
    return unload_check


def test_margin_refuses_the_devb5b_corner_cube(arm, col):
    """devb5b / devb5c: a 0.063 m cup stored at the residual corner spot (-0.223, -0.04) unloads from there,
    but landed 5-9 mm toward the divider (the sim) its lift touches the wall (sens.py). With
    check_unload_margin the corner spot the exact check accepts is refused, naming the offset."""
    c = compartment(arm, 'residual')
    size = box_size(OBJECTS['trash']['poly_cup'])
    uc = unload_checker(arm, col, 'residual', 'cup', CFG)                      # the c076fa5 lift (straight up)
    at = lambda dy: {'centre': np.array([-0.223, -0.04 + dy, c['floor'] + size[2] / 2]),  # noqa: E731
                     'R': np.eye(3), 'size': tuple(size)}
    assert uc(at(0.0), size[2]) is None
    assert uc(at(0.009), size[2]) == 'collision:lift:basket_link/held:cup'
    held = picked(arm, OBJECTS['trash']['poly_cup'], 0.0)
    corner = {**CFG, 'store': {**CFG['store'], 'fill_corner': [-1, 1], 'check_unload': True,
                               'landing_margin_m': 0.009}}
    ok, _, _, exact = plan_store(arm, 'residual', held, [], corner,
                                 unload_check=unload_checker(arm, col, 'residual', 'h', CFG))
    assert ok and exact['slot'][1] > c['y'][1] - 0.06                          # beside the divider
    margin = {**corner, 'store': {**corner['store'], 'check_unload_margin': True}}
    ok, _, _, info = plan_store(arm, 'residual', held, [], margin,
                                unload_check=unload_checker(arm, col, 'residual', 'h', CFG))
    assert any(xy == exact['slot'] and w.startswith('no_unload at (') for xy, w in info['rejected'])
    assert not ok or info['slot'] != exact['slot']


@pytest.mark.parametrize('comp', ['recyclable', 'residual'])
def test_a_narrow_bottle_at_the_centre_passes_the_margin(arm, col, comp):
    """B5b as configured: a 6.5 cm bottle at the centre unloads from every offset of landing_margin_m."""
    assert ROW['store']['placement'] == 'centre' and ROW['store']['check_unload_margin']
    seen = []
    uc = unload_checker(arm, col, comp, 'h')

    def counted(box, height):
        seen.append(box)
        return uc(box, height)
    held = picked(arm, OBJECTS['trash']['pet_bottle_500'], 0.0)
    ok, why, _, info = plan_store(arm, comp, held, [], ROW, unload_check=counted)
    assert ok, (why, info.get('rejected'))
    assert len(seen) == 9


@pytest.mark.parametrize('comp', ['recyclable', 'residual'])
def test_an_item_pushed_against_a_wall_unloads_with_the_lift_toward_the_middle(arm, col, comp):
    """B5b S1: the walk pushes items up to 4.3 cm sideways, against a wall. Starting below clearance_m, a lift
    straight up (c076fa5) drifts toward the wall and fails the separating-contacts rule; with
    unload.lift_toward_middle_m the item moves away from it (wall_lift2.py: poly cup 0.1 mm from the divider)."""
    assert ROW['unload']['lift_toward_middle_m'] > 0
    c = compartment(arm, comp)
    held = picked(arm, OBJECTS['trash']['poly_cup'], 0.0)
    ok, why, plan, info = plan_store(arm, comp, held, [], {**ROW, 'store': {**ROW['store'], 'check_unload_margin': False}})
    assert ok, why
    o, Rb, e = landed(arm, plan['release'], held, info['drop'])
    side = -1.0 if comp == 'recyclable' else 1.0                                 # toward the divider
    gap = (o['centre'][1] - e[1] / 2 - c['y'][0]) if side < 0 else (c['y'][1] - o['centre'][1] - e[1] / 2)
    box = {'centre': np.asarray(o['centre']) + [0.0, side * (gap - 0.0001), 0.0], 'R': Rb, 'size': tuple(held['size'])}
    assert unload_checker(arm, col, comp, 'h', CFG)(box, float(e[2])) == 'collision:lift:basket_link/held:h'
    assert unload_checker(arm, col, comp, 'h')(box, float(e[2])) is None
