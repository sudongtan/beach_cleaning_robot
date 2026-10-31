"""Graspability, IK reach envelope, pick plans, compartments and the bin (robot model from the
xacro; needs a sourced ROS environment with beach_description built and Pinocchio)."""

from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip('pinocchio')
pytest.importorskip('ament_index_python')

from grasping.arm import TCP, TIPS, Arm, bin_release_target, graspable, in_envelope, plan_pick  # noqa: E402

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
    target = bin_release_target((place, 0.0, -body_height + rim), near_wall, bd)
    ok, q, err, tilt = arm.ik(target, bd['max_tilt_deg'])
    assert ok, f'err {err * 100:.1f} cm, tilt {tilt:.0f} deg'
