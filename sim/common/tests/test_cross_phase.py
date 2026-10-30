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


@pytest.mark.parametrize('body_height', LIMITS['bin_drop']['body_heights_m'])
def test_jaw_tips_reach_over_the_bin(body_height):
    pin = pytest.importorskip('pinocchio')
    sys.path.insert(0, str(DESC / 'scripts'))
    import reach_check as rc
    bd = LIMITS['bin_drop']
    model = rc.build_model(VARIANT)
    data = model.createData()
    q0 = rc.stand_config(model)
    head_x = rc.frame_pos(model, data, q0, 'camera_link')[0]
    rim = max(p['pose'][2] + p['size'][2] / 2 for p in OBJECTS['bins']['bin_recyclable']['collision']
              if p['type'] == 'box')
    ground_z = -body_height
    target = np.array([head_x + bd['body_to_bin_m'] + bd['inside_wall_m'], 0.0,
                       ground_z + rim + bd['clearance_above_rim_m']])
    ok, _, err, tilt = rc.solve(model, data, q0, target, bd['max_tilt_deg'], np.random.default_rng(0),
                                frame='arm_jaw_tips_link')
    assert ok, (f'jaw tips cannot reach {target.round(3)} (bin rim {rim} m, body height {body_height} m): '
                f'error {err * 100:.1f} cm, tilt {tilt:.0f} deg')
