import numpy as np
import pytest

from locomotion.leg_odometry import LEGS, Leg, LegOdometry, axis_rotation, quat_matrix, rpy_matrix
import xml.etree.ElementTree as ET


def leg_xml(leg, x, y):
    return f"""
  <joint name="{leg}_hip_joint" type="revolute"><parent link="base_link"/><child link="{leg}_hip"/>
    <origin xyz="{x} {y} 0"/><axis xyz="1 0 0"/></joint>
  <joint name="{leg}_thigh_joint" type="revolute"><parent link="{leg}_hip"/><child link="{leg}_thigh"/>
    <origin xyz="0 {0.1 if y > 0 else -0.1} 0"/><axis xyz="0 1 0"/></joint>
  <joint name="{leg}_calf_joint" type="revolute"><parent link="{leg}_thigh"/><child link="{leg}_calf"/>
    <origin xyz="0 0 -0.2"/><axis xyz="0 1 0"/></joint>
  <joint name="{leg}_foot_fixed" type="fixed"><parent link="{leg}_calf"/><child link="{leg}_foot"/>
    <origin xyz="0 0 -0.2"/></joint>"""


URDF = '<robot name="r">' + ''.join(leg_xml(leg, x, y) for leg, x, y in (
    ('FL', 0.2, 0.05), ('FR', 0.2, -0.05), ('RL', -0.2, 0.05), ('RR', -0.2, -0.05))) + '</robot>'
Q = (0.05, 0.8, -1.5)


def leg(name='FL'):
    return Leg(ET.fromstring(URDF), 'base_link', name + '_foot')


def sample(odo, v_body, gyro=(0.0, 0.0, 0.0), stance=LEGS, force=80.0, attitude=np.eye(3), forces=None):
    """Joint states of feet at rest in the world (the stance legs), the swing legs carrying no load."""
    joints = {}
    for name, l in odo.legs.items():
        foot, jac = l.kinematics(Q)
        dq = np.linalg.solve(jac, -(np.asarray(v_body) + np.cross(gyro, foot)))
        load = (forces or {}).get(name, force) if name in stance else 0.0
        tau = -jac.T @ (attitude.T @ np.array([0.0, 0.0, load]))
        if name not in stance:
            dq = dq + np.array([3.0, -2.0, 1.0])            # a swinging leg says nothing about the body
        joints.update({j: (q, d, t) for j, q, d, t in zip(l.joints, Q, dq, tau)})
    return joints


def test_chain_is_read_from_the_urdf():
    assert leg().joints == ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint']


def test_a_foot_without_a_chain_is_an_error():
    with pytest.raises(ValueError):
        Leg(ET.fromstring(URDF), 'base_link', 'nope_foot')


def test_straight_leg_position():
    foot, _ = leg().kinematics((0.0, 0.0, 0.0))
    assert foot == pytest.approx([0.2, 0.15, -0.4])


def test_jacobian_matches_finite_differences():
    l, q, eps = leg(), np.array(Q), 1e-6
    _, jac = l.kinematics(q)
    for i in range(3):
        d = np.zeros(3)
        d[i] = eps
        numeric = (l.kinematics(q + d)[0] - l.kinematics(q - d)[0]) / (2 * eps)
        assert jac[:, i] == pytest.approx(numeric, abs=1e-6)


def test_rotations():
    assert rpy_matrix(0.0, 0.0, np.pi / 2) @ [1, 0, 0] == pytest.approx([0, 1, 0], abs=1e-12)
    assert axis_rotation([0, 1, 0], np.pi / 2) @ [0, 0, 1] == pytest.approx([1, 0, 0], abs=1e-12)
    assert quat_matrix(np.cos(0.25), 0.0, 0.0, np.sin(0.25)) == pytest.approx(rpy_matrix(0.0, 0.0, 0.5), abs=1e-12)


def test_body_velocity_from_stance_feet():
    odo = LegOdometry(URDF)
    v, legs = odo.estimate(sample(odo, (0.3, -0.1, 0.02)), (0.0, 0.0, 0.0), np.eye(3))
    assert v == pytest.approx([0.3, -0.1, 0.02], abs=1e-9)
    assert all(x['stance'] for x in legs.values())


def test_body_rotation_is_taken_out():
    odo = LegOdometry(URDF)
    gyro = (0.1, -0.2, 0.5)
    v, _ = odo.estimate(sample(odo, (0.2, 0.0, 0.0), gyro), gyro, np.eye(3))
    assert v == pytest.approx([0.2, 0.0, 0.0], abs=1e-9)


def test_swing_legs_are_ignored():
    odo = LegOdometry(URDF)
    v, legs = odo.estimate(sample(odo, (0.25, 0.05, 0.0), stance=('FL', 'RR')), (0.0, 0.0, 0.0), np.eye(3))
    assert v == pytest.approx([0.25, 0.05, 0.0], abs=1e-9)
    assert [n for n, x in legs.items() if x['stance']] == ['FL', 'RR']


def test_force_is_taken_in_the_gravity_frame():
    odo = LegOdometry(URDF)
    attitude = rpy_matrix(0.0, 0.3, 0.0)
    _, legs = odo.estimate(sample(odo, (0.0, 0.0, 0.0), attitude=attitude), (0.0, 0.0, 0.0), attitude)
    assert legs['FL']['force_z'] == pytest.approx(80.0)


def test_no_stance_gives_no_velocity():
    odo = LegOdometry(URDF)
    v, _ = odo.estimate(sample(odo, (0.1, 0.0, 0.0), stance=()), (0.0, 0.0, 0.0), np.eye(3))
    assert v is None


def test_height_contact_takes_the_lowest_feet():
    odo = LegOdometry(URDF, contact='height', height_margin_m=0.01)
    joints = sample(odo, (0.1, 0.0, 0.0))
    joints['FR_calf_joint'] = (-2.0, joints['FR_calf_joint'][1], 0.0)         # foot pulled up
    _, legs = odo.estimate(joints, (0.0, 0.0, 0.0), np.eye(3))
    assert not legs['FR']['stance'] and legs['FL']['stance']


def test_weighted_mean_favours_the_loaded_foot():
    odo = LegOdometry(URDF, contact='weighted', force_threshold_n=40.0)
    joints = sample(odo, (0.2, 0.0, 0.0), stance=('FL', 'RR'), forces={'FL': 120.0, 'RR': 60.0})
    _, jac = odo.legs['RR'].kinematics(Q)
    slip = np.linalg.solve(jac, np.array([0.3, 0.0, 0.0]))    # RR foot slides: body speed off by -0.3
    for j, d in zip(odo.legs['RR'].joints, slip):
        joints[j] = (joints[j][0], joints[j][1] + d, joints[j][2])
    v, legs = odo.estimate(joints, (0.0, 0.0, 0.0), np.eye(3))
    assert legs['FL']['weight'] == pytest.approx(80.0) and legs['RR']['weight'] == pytest.approx(20.0)
    assert v[0] == pytest.approx(0.2 - 0.3 * 20.0 / 100.0)


def test_a_foot_under_the_threshold_counts_for_nothing():
    odo = LegOdometry(URDF, contact='weighted', force_threshold_n=40.0)
    v, legs = odo.estimate(sample(odo, (0.1, 0.0, 0.0), stance=('FL', 'RR'), forces={'RR': 30.0}), (0.0, 0.0, 0.0), np.eye(3))
    assert v == pytest.approx([0.1, 0.0, 0.0], abs=1e-9)
    assert not legs['RR']['stance']
