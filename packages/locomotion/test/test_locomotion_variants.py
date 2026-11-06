"""Every locomotion:= value has a leg controller block and an adapter yaml; the tuned basic
controller differs from the basic one only in its joint gains (B4)."""
import ast
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
CONTROLLERS = yaml.safe_load(open(REPO / 'packages/sim_bridge_gazebo/config/gazebo_controllers.yaml'))


def leg_controllers():
    tree = ast.parse(open(REPO / 'packages/sim_bridge_gazebo/launch/gazebo.launch.py').read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.targets[0].id == 'LEG_CONTROLLERS':
            return ast.literal_eval(node.value)
    raise AssertionError('LEG_CONTROLLERS not found')


def test_every_variant_is_wired():
    types = CONTROLLERS['controller_manager']['ros__parameters']
    for loco, ctrl in leg_controllers().items():
        assert ctrl in types and ctrl in CONTROLLERS, loco
        assert (REPO / f'packages/locomotion/config/{loco}_adapter.yaml').is_file(), loco


def test_tuned_differs_only_in_gains():
    basic = dict(CONTROLLERS['basic_quadruped_controller']['ros__parameters'])
    tuned = dict(CONTROLLERS['basic_tuned_controller']['ros__parameters'])
    gains = {k: tuned.pop(k) for k in ('swing_gains', 'stable_gains') if k in tuned}
    assert gains and tuned == basic
    assert all(len(v) == 2 and min(v) >= 0 for v in gains.values())
