"""perception.true_pose (B4: ground truth seen from the true robot pose) and logic.scene_bins."""

import math
from types import SimpleNamespace as NS

import numpy as np

from perception.logic import BIN_CLASSES, scene_bins
from perception.true_pose import PoseHistory, matrix_to_quat, pose_matrix, quat_to_matrix, to_transform_stamped


def yaw_pose(x, y, z, yaw):
    return pose_matrix(NS(x=x, y=y, z=z), NS(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2)))


def test_quaternion_round_trip():
    for q in [(0, 0, 0, 1), (0, 0, math.sin(1.0), math.cos(1.0)), (0.1, -0.2, 0.3, 0.927)]:
        q = np.asarray(q) / np.linalg.norm(q)
        back = np.asarray(matrix_to_quat(quat_to_matrix(*q)))
        assert np.allclose(back, q) or np.allclose(back, -q)


def test_transform_stamped_matches_matrix():
    T = yaw_pose(1.0, 2.0, 0.3, 0.7)
    tf = to_transform_stamped(T, 'map', 'base_link')
    t, r = tf.transform.translation, tf.transform.rotation
    assert (t.x, t.y, t.z) == (1.0, 2.0, 0.3)
    assert math.isclose(math.atan2(2 * r.w * r.z, 1 - 2 * r.z * r.z), 0.7)


def test_history_nearest_and_gap():
    h = PoseHistory(keep_s=5.0)
    for k in range(10):
        h.add(k * 0.02, yaw_pose(k, 0, 0, 0))
    assert h.at(0.041)[0, 3] == 2                    # nearest stamp
    assert h.at()[0, 3] == 9                         # latest
    assert h.at(5.0) is None                         # too far from any sample
    h.add(100.0, yaw_pose(-1, 0, 0, 0))
    assert h.at(0.0) is None                         # older samples dropped (keep_s)


def test_history_resets_when_time_goes_back():
    h = PoseHistory()
    h.add(10.0, yaw_pose(1, 0, 0, 0))
    h.add(1.0, yaw_pose(2, 0, 0, 0))
    assert h.at()[0, 3] == 2 and len(h.times) == 1


def test_relative_pose_ignores_the_localization_estimate():
    """The camera <- map transform comes from the true pose: an item 1 m ahead of the robot is 1 m
    ahead whatever the estimated pose (the bug this fixes: TF map -> base_link was the estimate)."""
    T_map_base = yaw_pose(5.0, 3.0, 0.3, math.pi / 2)
    item_map = T_map_base @ np.array([1.0, 0.0, -0.3, 1.0])
    rel = np.linalg.inv(T_map_base) @ item_map
    assert np.allclose(rel[:3], (1.0, 0.0, -0.3))


def test_scene_bins_on_the_ground_with_rim_on_top():
    objects = {'bins': {'bin_recyclable': {'class': 'bin_recyclable', 'collision': [
        {'size': [0.60, 0.45, 0.02], 'pose': [0, 0, 0.01, 0, 0, 0]},
        {'size': [0.60, 0.02, 0.50], 'pose': [0, 0.215, 0.25, 0, 0, 0]}]}}}
    scene = {'bins': [{'model': 'bin_recyclable', 'bin_id': 'recyclable', 'pose': [1.5, 14.0, math.pi / 2]}]}
    [(cls, (x, y, z), yaw, size)] = scene_bins(scene, objects, lambda x, y: 0.2, 0.3)
    assert cls == 'bin_recyclable' and BIN_CLASSES[cls] == 'recyclable'
    assert math.isclose(x, 1.5) and math.isclose(y, 14.0) and math.isclose(yaw, math.pi / 2)
    assert np.allclose(size, (0.60, 0.45, 0.50))
    assert math.isclose(z + size[2] / 2, 0.2 + 0.50)  # rim = ground + 0.50 (as task.yaml bin_rim terrain)
