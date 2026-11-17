"""Leg odometry for walking controllers that publish none (rl_quadruped_controller, C1.1).

  /joint_states, /imu, /robot_description  ->  /leg_odom (nav_msgs/Odometry, twist in base_link)

Started only for a locomotion whose adapter has `state_source: imu`; the basic controller keeps its
own estimator. Stance-leg kinematics, the velocity measurement of legubiao's Estimator.cpp (and of
the MIT Cheetah linear filter): a foot on the ground doesn't move, so

  v_body = -(J(q) * dq + w x r_foot)     averaged over the stance feet, in base_link

The legs come from the URDF on /robot_description (nothing hard-coded) and are matched by joint
name. Contact (`contact`): the robot_lab controller has no foot force interfaces, so the vertical
ground force comes from the joint torques (F = -J^-T * tau, rotated by the IMU attitude). `weighted`
averages the feet by max(0, F - force_threshold_n), `effort` takes the feet above the threshold and
`height` the feet within a margin of the lowest one. Only the twist is meaningful: both EKFs fuse
vx, vy (and yaw rate); the pose stays at the identity.
"""
import math
import xml.etree.ElementTree as ET

import numpy as np

LEGS = ('FL', 'FR', 'RL', 'RR')


def rpy_matrix(roll, pitch, yaw):
    """URDF origin rotation: R = Rz(yaw) * Ry(pitch) * Rx(roll)."""
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch),
                              math.cos(yaw), math.sin(yaw))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_rotation(axis, angle):
    """Rodrigues' formula, axis a unit vector."""
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * k @ k


def quat_matrix(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def vector(element, name, default):
    return np.array([float(v) for v in element.get(name).split()]) if element is not None \
        and element.get(name) else np.array(default, dtype=float)


class Leg:
    """Joint chain from the base frame to a foot link, as parsed from the URDF."""

    def __init__(self, urdf_root, base_frame, foot_link):
        by_child = {j.find('child').get('link'): j for j in urdf_root.findall('joint')}
        chain, link = [], foot_link
        while link != base_frame:
            joint = by_child.get(link)
            if joint is None:
                raise ValueError(f'no joint chain from {foot_link} to {base_frame}')
            origin = joint.find('origin')
            moving = joint.get('type') in ('revolute', 'continuous')
            axis = vector(joint.find('axis'), 'xyz', [1.0, 0.0, 0.0]) if moving else None
            chain.append((joint.get('name') if moving else None, vector(origin, 'xyz', [0.0, 0.0, 0.0]),
                          rpy_matrix(*vector(origin, 'rpy', [0.0, 0.0, 0.0])), axis))
            link = joint.find('parent').get('link')
        self.chain = chain[::-1]
        self.joints = [name for name, _, _, axis in self.chain if axis is not None]

    def kinematics(self, q):
        """Foot position in the base frame and its 3 x n position Jacobian; q in the order of joints."""
        rot, pos, angles = np.eye(3), np.zeros(3), iter(q)
        axes, points = [], []
        for _, xyz, rpy, axis in self.chain:
            pos = pos + rot @ xyz
            rot = rot @ rpy
            if axis is not None:
                axes.append(rot @ axis)
                points.append(pos)
                rot = rot @ axis_rotation(axis, next(angles))
        return pos, np.column_stack([np.cross(a, pos - p) for a, p in zip(axes, points)])


class LegOdometry:
    """One estimate per joint sample; no ROS."""

    def __init__(self, urdf, base_frame='base_link', foot_suffix='_foot', contact='weighted',
                 force_threshold_n=40.0, height_margin_m=0.02):
        root = ET.fromstring(urdf)
        self.legs = {leg: Leg(root, base_frame, leg + foot_suffix) for leg in LEGS}
        self.contact, self.force_threshold_n, self.height_margin_m = contact, force_threshold_n, height_margin_m

    def estimate(self, joints, gyro, attitude):
        """joints: name -> (position, velocity, effort); gyro: body rate in base_link; attitude: base_link
        to the gravity-aligned world (3 x 3). Returns (body velocity or None, per-leg dict)."""
        gyro, legs = np.asarray(gyro, dtype=float), {}
        for name, leg in self.legs.items():
            q, dq, tau = np.array([joints[j] for j in leg.joints]).T
            foot, jac = leg.kinematics(q)
            try:
                force = -np.linalg.solve(jac.T, tau)            # ground force on the foot, base_link
            except np.linalg.LinAlgError:
                force = np.zeros(3)
            legs[name] = {'foot': foot, 'force_z': float((attitude @ force)[2]),
                          'height': float((attitude @ foot)[2]),
                          'velocity': -(jac @ dq + np.cross(gyro, foot))}
        lowest = min(v['height'] for v in legs.values())
        for v in legs.values():
            v['weight'] = v['height'] - lowest < self.height_margin_m if self.contact == 'height' \
                else max(0.0, v['force_z'] - self.force_threshold_n) if self.contact == 'weighted' \
                else v['force_z'] > self.force_threshold_n
            v['stance'] = v['weight'] > 0.0
        total = sum(float(v['weight']) for v in legs.values())
        if total <= 0.0:
            return None, legs
        return sum(float(v['weight']) * v['velocity'] for v in legs.values()) / total, legs


def main():
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
    from sensor_msgs.msg import Imu, JointState
    from std_msgs.msg import String

    class LegOdometryNode(Node):
        def __init__(self):
            super().__init__('leg_odometry')
            p = self.declare_parameter
            self.frames = p('odom_frame', 'odom').value, p('base_frame', 'base_link').value
            self.options = {'contact': p('contact', 'weighted').value,
                            'force_threshold_n': p('force_threshold_n', 40.0).value,
                            'height_margin_m': p('height_margin_m', 0.02).value}
            self.sample_dt = 1.0 / p('sample_hz', 200.0).value
            self.publish_dt = 1.0 / p('publish_hz', 50.0).value
            cov = p('twist_covariance_diag', [0.1, 0.1, 0.1, 0.1, 0.1, 0.05]).value
            self.no_stance_factor = p('no_stance_covariance_factor', 100.0).value
            self.covariance = np.diag(cov)
            self.odometry, self.imu, self.last_sample, self.last_publish = None, None, 0.0, 0.0
            self.sum, self.count, self.last = np.zeros(3), 0, np.zeros(3)
            self.pub = self.create_publisher(Odometry, '/leg_odom', 10)
            latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.create_subscription(String, '/robot_description', self.on_description, latched)
            self.create_subscription(Imu, '/imu', self.on_imu, qos_profile_sensor_data)
            self.create_subscription(JointState, '/joint_states', self.on_joints, 50)

        def on_description(self, msg):
            self.odometry = LegOdometry(msg.data, base_frame=self.frames[1], **self.options)
            self.get_logger().info('legs read from /robot_description')

        def on_imu(self, msg):
            self.imu = msg

        def on_joints(self, msg):
            if self.odometry is None or self.imu is None:
                return
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if stamp - self.last_sample >= self.sample_dt:
                self.last_sample = stamp
                g, o = self.imu.angular_velocity, self.imu.orientation
                velocity, _ = self.odometry.estimate(
                    {n: v for n, v in zip(msg.name, zip(msg.position, msg.velocity, msg.effort))},
                    (g.x, g.y, g.z), quat_matrix(o.w, o.x, o.y, o.z))
                if velocity is not None:
                    self.sum += velocity
                    self.count += 1
            if stamp - self.last_publish >= self.publish_dt:
                self.last_publish = stamp
                self.publish(msg.header.stamp)

        def publish(self, stamp):
            if self.count:
                self.last = self.sum / self.count
            out = Odometry()
            out.header.stamp, out.header.frame_id, out.child_frame_id = stamp, self.frames[0], self.frames[1]
            out.pose.pose.orientation.w = 1.0
            out.twist.twist.linear.x, out.twist.twist.linear.y, out.twist.twist.linear.z = map(float, self.last)
            out.twist.twist.angular.z = float(self.imu.angular_velocity.z)
            scale = 1.0 if self.count else self.no_stance_factor
            out.twist.covariance = list((self.covariance * scale).flatten())
            self.pub.publish(out)
            self.sum, self.count = np.zeros(3), 0

    rclpy.init()
    node = LegOdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
