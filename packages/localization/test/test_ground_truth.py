"""Ground-truth localization: frames relabelled, pose carried over unchanged."""

import pytest

pytest.importorskip('nav_msgs')

from nav_msgs.msg import Odometry  # noqa: E402

from localization.ground_truth import odom_to_tf, relabel  # noqa: E402


def gt_msg():
    m = Odometry()
    m.header.frame_id, m.child_frame_id = 'map', 'base_link'
    m.header.stamp.sec = 12
    m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z = 4.0, 13.9, 0.3
    m.pose.pose.orientation.z, m.pose.pose.orientation.w = 0.6, 0.8
    m.twist.twist.linear.x = 0.25
    return m


def test_relabel_keeps_pose_and_twist():
    out = relabel(gt_msg(), 'odom', 'base_link')
    assert (out.header.frame_id, out.child_frame_id, out.header.stamp.sec) == ('odom', 'base_link', 12)
    assert out.pose.pose.position.x == 4.0 and out.pose.pose.orientation.z == 0.6
    assert out.twist.twist.linear.x == 0.25


def test_tf_matches_pose():
    tf = odom_to_tf(relabel(gt_msg(), 'odom', 'base_link'))
    assert (tf.header.frame_id, tf.child_frame_id) == ('odom', 'base_link')
    assert (tf.transform.translation.x, tf.transform.translation.y) == (4.0, 13.9)
    assert tf.transform.rotation.w == 0.8
