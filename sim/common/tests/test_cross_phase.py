"""Cross-phase checks: do the robot (A0) and the scene objects (A1) still fit together?

The limits come from the robot model (xacro of beach_description, variant from
config/robot_limits.yaml) and from sim/common/objects.yaml, never from copied
numbers, so a change on either side fails here. Needs the ROS environment
(xacro, Pinocchio): `make test` sources it.
"""

import math
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import yaml

COMMON = Path(__file__).resolve().parents[1]
REPO = COMMON.parents[1]
DESC = REPO / 'packages' / 'beach_description'
OBJECTS = yaml.safe_load(open(COMMON / 'objects.yaml'))
LIMITS = yaml.safe_load(open(REPO / 'config' / 'robot_limits.yaml'))['robot_limits']
VARIANT = LIMITS['variant']

if shutil.which('xacro') is None:
    pytest.fail('xacro not found: source /opt/ros/jazzy/setup.bash and ~/ros2_ws/install/setup.bash '
                '(or run `make test`)', pytrace=False)


@lru_cache(maxsize=None)
def urdf(variant=VARIANT, load=0.0):
    out = subprocess.run(['xacro', str(DESC / 'urdf' / 'beach_robot.urdf.xacro'), f'variant:={variant}',
                          f'basket_load_kg:={load}'], check=True, capture_output=True, text=True).stdout
    return ET.fromstring(out)


def joint(root, name):
    return next(j for j in root.iter('joint') if j.get('name') == name)


def link(root, name):
    return next(lk for lk in root.iter('link') if lk.get('name') == name)


def size(spec):
    """Axis-aligned size [x, y, z] of an object's collision primitives."""
    lo, hi = [1e9] * 3, [-1e9] * 3
    for p in spec['collision']:
        ext = p['size'] if p['type'] == 'box' else [2 * p['radius'], 2 * p['radius'], p['length']]
        for k in range(3):
            lo[k] = min(lo[k], p['pose'][k] - ext[k] / 2)
            hi[k] = max(hi[k], p['pose'][k] + ext[k] / 2)
    return [h - lo_ for lo_, h in zip(lo, hi)]


def jaw_gap():
    """(closed, open) gap between the jaw faces. Custom jaws: the inner face sits at the
    finger joint position (wx250s_arm.xacro), so the gap is 2 x the joint value."""
    lim = joint(urdf(), 'arm_left_finger').find('limit')
    return 2 * float(lim.get('lower')), 2 * float(lim.get('upper'))


def compartment():
    """Inner (length, width, height) of one basket compartment, from basket_link's walls."""
    boxes = [tuple(map(float, c.find('geometry/box').get('size').split()))
             for c in link(urdf(), 'basket_link').findall('collision')]
    floor = max(boxes, key=lambda b: b[0] * b[1])          # the floor is the widest box
    length, width, wall = floor
    height = max(b[2] for b in boxes)
    return length - 2 * wall, (width - 3 * wall) / 2, height - wall


def fits(a, b, A, B, steps=900):
    """Does an a x b rectangle fit in A x B, at any rotation?"""
    for t in np.linspace(0, math.pi / 2, steps):
        if a * math.cos(t) + b * math.sin(t) <= A and a * math.sin(t) + b * math.cos(t) <= B:
            return True
    return False


TRASH = sorted(OBJECTS['trash'])


@pytest.mark.parametrize('name', TRASH)
def test_item_fits_the_jaws(name):
    closed, opened = jaw_gap()
    narrow = min(size(OBJECTS['trash'][name])[:2])
    assert narrow <= opened - LIMITS['jaw_margin_m'], \
        f'{name}: narrow side {narrow * 100:.1f} cm > open jaw gap {opened * 100:.1f} cm - margin'
    assert narrow >= closed, f'{name}: narrow side {narrow * 100:.1f} cm < closed gap {closed * 100:.1f} cm'


def jaw_length():
    """Distance from the gripper bar's face (arm_fingers_link) to the jaw tips."""
    return float(joint(urdf(), 'arm_jaw_tips').find('origin').get('xyz').split()[0])


@pytest.mark.parametrize('name', TRASH)
def test_item_clears_the_gripper_bar(name):
    height = size(OBJECTS['trash'][name])[2]
    limit = jaw_length() + LIMITS['grasp_tip_height_m'] - LIMITS['bar_clearance_m']
    assert height <= limit + 1e-9, \
        f'{name}: {height * 100:.1f} cm tall > {limit * 100:.1f} cm (the gripper bar would hit it)'


@pytest.mark.parametrize('name', TRASH)
def test_item_fits_a_compartment(name):
    L, W, H = compartment()
    sx, sy, sz = size(OBJECTS['trash'][name])
    assert fits(sx, sy, L, W), f'{name} ({sx:.3f} x {sy:.3f}) does not fit lying flat in {L:.3f} x {W:.3f}'
    assert sz <= H, f'{name} is {sz:.3f} m tall, compartment {H:.3f} m'


def test_heaviest_item_within_payload():
    heaviest = max(OBJECTS['trash'], key=lambda n: OBJECTS['trash'][n]['mass'])
    assert OBJECTS['trash'][heaviest]['mass'] <= LIMITS['payload_kg'], heaviest


def test_added_mass_within_budget():
    root = urdf(load=0.0)
    added = 0.0
    for lk in root.iter('link'):
        n = lk.get('name')
        m = lk.find('inertial/mass')
        if m is not None and (n.startswith('arm_') or n.startswith('basket_')):
            added += float(m.get('value'))
    total = added + LIMITS['basket_load_max_kg']
    assert added > 2.0, 'no arm/basket links found: naming changed?'
    assert total <= LIMITS['added_mass_budget_kg'], f'added mass {total:.2f} kg'


TASK = yaml.safe_load(open(REPO / 'config' / 'task.yaml'))['task']['fsm']


def bin_geometry(body_height):
    """(rim height, near-wall x) in base_link with the robot at the task's place distance, facing the bin."""
    boxes = [p for p in OBJECTS['bins']['bin_recyclable']['collision'] if p['type'] == 'box']
    rim = max(p['pose'][2] + p['size'][2] / 2 for p in boxes)
    half_depth = max(abs(p['pose'][0]) + p['size'][0] / 2 for p in boxes)
    return rim - body_height, TASK['bin_place_distance_m'] - half_depth


@lru_cache(maxsize=None)
def arm_model():
    pytest.importorskip('pinocchio')
    sys.path.insert(0, str(DESC / 'scripts'))
    import reach_check as rc
    model = rc.build_model(VARIANT)
    return rc, model, model.createData(), rc.stand_config(model)


def release_q(body_height):
    rc, model, data, q0 = arm_model()
    bd = LIMITS['bin_drop']
    rim, wall = bin_geometry(body_height)
    target = np.array([wall + bd['inside_wall_m'], 0.0, rim + bd['clearance_above_rim_m']])
    ok, q, err, tilt = rc.solve(model, data, q0, target, bd['max_tilt_deg'], np.random.default_rng(0),
                                frame='arm_jaw_tips_link')
    return ok, q, target, err, tilt


@pytest.mark.parametrize('body_height', LIMITS['bin_drop']['body_heights_m'])
def test_jaw_tips_reach_over_the_bin(body_height):
    """From the task's place distance the jaw tips get over the rim and past the near wall."""
    ok, _, target, err, tilt = release_q(body_height)
    assert ok, (f'jaw tips cannot reach {target.round(3)} from {TASK["bin_place_distance_m"]} m '
                f'(body height {body_height} m): error {err * 100:.1f} cm, tilt {tilt:.0f} deg')


@pytest.mark.parametrize('body_height', LIMITS['bin_drop']['body_heights_m'])
def test_arm_moves_at_the_bin_clear_its_wall(body_height):
    """rest -> up -> over the bin -> up -> rest (joint-space moves, as the controller runs them) never
    passes the bin's near wall below the rim. At 0.7 m the arm hit the bin and tipped the robot (A2b)."""
    import pinocchio as pin
    rc, model, data, q0 = arm_model()
    poses = yaml.safe_load(open(DESC / 'config' / 'arm_poses_custom.yaml'))['poses']
    ok, rel, *_ = release_q(body_height)
    assert ok
    idx = [model.joints[model.getJointId(n)].idx_q for n in rc.ARM]
    rest, up, rel = np.array(poses['rest']['arm']), np.array(poses['up']['arm']), rel[idx]
    rim, wall = bin_geometry(body_height)
    clear = LIMITS['bin_drop']['arm_wall_clearance_m']
    frames = [f.name for f in model.frames if f.name.startswith('arm_') and f.type == pin.FrameType.BODY]
    for name, a, b in (('rest->up', rest, up), ('up->bin', up, rel), ('bin->up', rel, up), ('up->rest', up, rest)):
        for s in np.linspace(0, 1, 81):
            q = q0.copy()
            q[idx] = a + s * (b - a)
            pin.framesForwardKinematics(model, data, q)
            for f in frames:
                p = data.oMf[model.getFrameId(f)].translation
                assert not (abs(p[0] - wall) < clear and p[2] < rim + clear and abs(p[1]) < 0.3), \
                    f'{name}: {f} at {p.round(3)} hits the bin wall (x {wall:.3f}, rim {rim:.3f}) at s={s:.2f}'
