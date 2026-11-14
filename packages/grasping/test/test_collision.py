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

from test_arm import CFG, LIMITS, OBJECTS, box_size, picked, rz  # noqa: E402

from grasping.arm import (TIPS, Arm, compartment, held_item, plan_pick, plan_store,  # noqa: E402
                          plan_unload)
from grasping.collision import Collision, State, reason  # noqa: E402
from grasping.sequence import store_steps  # noqa: E402

DATA = Path(__file__).resolve().parent / 'data'
COL = CFG['collision']
SHOW = np.asarray(CFG['show_pose'], dtype=float)          # where a pick ends: a store starts here
OLD = {**CFG, 'store': {**CFG['store'], 'slots_x': 3, 'slots_y': 2,         # the 9ac4816 store planner
                        'bar_box_exact': False}}                           # (B5.3: 4 x 3, exact bar)


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
        out['unload'].append(plan_unload(arm, comp, centre, 0.066, CFG)[2])
    for h, comp in zip(helds, ('recyclable', 'residual', 'recyclable')):
        out['store'].append(plan_store(arm, comp, h, [], OLD)[2])
    for kind, plans in out.items():
        assert len(plans) == len(ref[kind])
        for plan, r in zip(plans, ref[kind]):
            assert set(plan) == set(r), kind
            for k in plan:
                assert np.asarray(plan[k]).tolist() == r[k], (kind, k)
