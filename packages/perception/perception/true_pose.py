"""Simulator ground truth seen from the robot (sim plumbing, not localization).

The ground-truth baselines stand in for on-board sensors (a perfect camera, a perfect check of
the held item) and for the attach plumbing. They must see the world relative to the robot as it
really is, not through the estimated TF map -> base_link, which carries the localization error
once localization isn't ground truth (B4, notes/briefs/B4-B6.md): the detector would put every
item at its true map position no matter how wrong the estimate is, and attach would look for
items in the wrong place.

TruePose keeps the recent /sim/ground_truth/pose messages (true map -> base_link) and composes
them with the robot-internal TF (base_link -> camera_link, gripper, ...), which is exact.
lookup(target) returns the true transform target <- map as a TransformStamped, a drop-in for
tf_buffer.lookup_transform(target, 'map', ...).
"""

import bisect

import numpy as np
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.time import Time
from tf2_ros import TransformException


def quat_to_matrix(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def matrix_to_quat(R):
    """Rotation matrix -> (x, y, z, w), w >= 0."""
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x = np.copysign(x, R[2, 1] - R[1, 2])
    y = np.copysign(y, R[0, 2] - R[2, 0])
    z = np.copysign(z, R[1, 0] - R[0, 1])
    return x, y, z, w


def pose_matrix(position, orientation):
    """4x4 from anything with x/y/z (position) and x/y/z/w (orientation)."""
    T = np.eye(4)
    T[:3, :3] = quat_to_matrix(orientation.x, orientation.y, orientation.z, orientation.w)
    T[:3, 3] = (position.x, position.y, position.z)
    return T


def transform_matrix(t):
    """4x4 from a geometry_msgs/Transform."""
    return pose_matrix(t.translation, t.rotation)


def to_transform_stamped(T, frame_id, child_frame_id, stamp=None):
    """4x4 (frame_id <- child_frame_id) -> TransformStamped."""
    tf = TransformStamped()
    tf.header.frame_id, tf.child_frame_id = frame_id, child_frame_id
    if stamp is not None:
        tf.header.stamp = stamp
    tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = (float(v) for v in T[:3, 3])
    q = matrix_to_quat(T[:3, :3])
    tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w = (
        float(v) for v in q)
    return tf


class PoseHistory:
    """Recent (stamp s, 4x4 map <- base) pairs; at(t) = the one nearest t (pure logic, unit-tested)."""

    def __init__(self, keep_s=5.0):
        self.keep_s = keep_s
        self.times, self.poses = [], []

    def add(self, t, T):
        if self.times and t < self.times[-1]:              # sim restarted / clock jumped back
            self.times, self.poses = [], []
        self.times.append(t)
        self.poses.append(T)
        cut = bisect.bisect_left(self.times, t - self.keep_s)
        if cut:
            del self.times[:cut], self.poses[:cut]

    def at(self, t=None, max_gap_s=0.1):
        """Pose nearest sim time t (None: the latest); None if empty or further than max_gap_s away."""
        if not self.times:
            return None
        if t is None:
            return self.poses[-1]
        k = bisect.bisect_left(self.times, t)
        best = min((i for i in (k - 1, k) if 0 <= i < len(self.times)), key=lambda i: abs(self.times[i] - t))
        return self.poses[best] if abs(self.times[best] - t) <= max_gap_s else None


class TruePose:
    """True map -> base_link from /sim/ground_truth/pose, composed with the robot's internal TF."""

    def __init__(self, node, tf_buffer, base_frame='base_link', map_frame='map'):
        self.tf_buffer, self.base_frame, self.map_frame = tf_buffer, base_frame, map_frame
        self.history = PoseHistory()
        node.create_subscription(Odometry, '/sim/ground_truth/pose', self.on_pose, 50)

    def on_pose(self, msg):
        s = msg.header.stamp
        self.history.add(s.sec + s.nanosec * 1e-9, pose_matrix(msg.pose.pose.position, msg.pose.pose.orientation))

    def map_from(self, frame, stamp=None):
        """True 4x4 map <- frame at stamp (builtin Time; None = latest). Raises TransformException."""
        T_map_base = self.history.at(None if stamp is None else stamp.sec + stamp.nanosec * 1e-9)
        if T_map_base is None:
            raise TransformException('no ground-truth robot pose (yet) near that time')
        if frame == self.base_frame:
            return T_map_base
        # the robot's own links: exact (URDF + joint states); the arm's latest pose, as before
        tf = self.tf_buffer.lookup_transform(self.base_frame, frame, Time())
        return T_map_base @ transform_matrix(tf.transform)

    def lookup(self, target, stamp=None, source=None):
        """True target <- source (default the map) as a TransformStamped, a drop-in for
        lookup_transform(target, source). Only a map source needs the true robot pose."""
        source = source or self.map_frame
        if source == target:
            return to_transform_stamped(np.eye(4), target, source, stamp)
        if source != self.map_frame:
            return self.tf_buffer.lookup_transform(target, source, Time())
        T = np.linalg.inv(self.map_from(target, stamp))
        return to_transform_stamped(T, target, self.map_frame, stamp)
