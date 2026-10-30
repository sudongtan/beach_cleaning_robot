"""Arm test command (A0 DoD 4): run the arm through test_sequence in beach_description/config/arm_poses.yaml
(stock: rest -> reach ground in front -> rest -> over recyclable -> rest -> over residual -> rest;
custom: the same via an `up` waypoint, arm_poses_custom.yaml). Use the variant the robot was launched with.

  ros2 run bringup arm_test [--ros-args -p variant:=stock]   (default custom, like the launch files)

Sends each pose as a FollowJointTrajectory goal to arm_controller and gripper_controller and,
after each move, prints the largest joint error measured on /joint_states. Exit code 0 if every
move succeeded and settled within `tolerance` rad (m for the fingers).
"""
import os
import sys

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

ARM = ['arm_waist', 'arm_shoulder', 'arm_elbow', 'arm_forearm_roll', 'arm_wrist_angle', 'arm_wrist_rotate']
FINGERS = ['arm_left_finger', 'arm_right_finger']


def duration(sec):
    return Duration(sec=int(sec), nanosec=int((sec % 1) * 1e9))


class ArmTest(Node):
    def __init__(self):
        super().__init__('arm_test', parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.max_speed = self.declare_parameter('max_joint_speed', 1.0).value   # rad/s for planning move times
        self.min_time = self.declare_parameter('min_move_time', 2.0).value      # s
        self.tolerance = self.declare_parameter('tolerance', 0.05).value        # rad
        variant = self.declare_parameter('variant', 'custom').value
        name = 'arm_poses_custom.yaml' if variant == 'custom' else 'arm_poses.yaml'
        cfg_path = os.path.join(get_package_share_directory('beach_description'), 'config', name)
        self.get_logger().info(f'variant {variant}: {cfg_path}')
        cfg = yaml.safe_load(open(cfg_path))
        self.poses, self.sequence = cfg['poses'], cfg['test_sequence']
        self.arm = ActionClient(self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')
        self.gripper = ActionClient(self, FollowJointTrajectory, '/gripper_controller/follow_joint_trajectory')
        self.joints = {}
        self.create_subscription(JointState, '/joint_states', self.on_joints, 10)

    def on_joints(self, msg):
        self.joints.update(zip(msg.name, msg.position))

    def send(self, client, names, target, t):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = names
        goal.trajectory.points = [JointTrajectoryPoint(positions=[float(v) for v in target],
                                                       velocities=[0.0] * len(names),
                                                       time_from_start=duration(t))]
        fut = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut)
        handle = fut.result()
        if not handle.accepted:
            return None
        return handle.get_result_async()

    def run(self):
        for c, n in ((self.arm, 'arm_controller'), (self.gripper, 'gripper_controller')):
            if not c.wait_for_server(timeout_sec=30.0):
                self.get_logger().error(f'{n} action server not available')
                return False
        while rclpy.ok() and not all(j in self.joints for j in ARM + FINGERS):
            rclpy.spin_once(self, timeout_sec=0.1)

        ok = True
        for name in self.sequence:
            pose = self.poses[name]
            arm_target = pose['arm']
            fingers = pose.get('fingers', 0.037)
            finger_target = [fingers, -fingers]
            span = max(abs(a - self.joints[j]) for j, a in zip(ARM, arm_target))
            t = max(self.min_time, span / self.max_speed)
            futs = [self.send(self.arm, ARM, arm_target, t), self.send(self.gripper, FINGERS, finger_target, 1.0)]
            if None in futs:
                self.get_logger().error(f'{name}: goal rejected')
                return False
            codes = []
            for f in futs:
                rclpy.spin_until_future_complete(self, f)
                codes.append(f.result().result.error_code)
            # settle, then measure
            t_end = self.get_clock().now().nanoseconds * 1e-9 + 0.5
            while self.get_clock().now().nanoseconds * 1e-9 < t_end:
                rclpy.spin_once(self, timeout_sec=0.05)
            err = max(abs(a - self.joints[j]) for j, a in zip(ARM + FINGERS, arm_target + finger_target))
            worst = max(zip(ARM + FINGERS, arm_target + finger_target), key=lambda ja: abs(ja[1] - self.joints[ja[0]]))
            good = all(c == FollowJointTrajectory.Result.SUCCESSFUL for c in codes) and err <= self.tolerance
            ok &= good
            self.get_logger().info(f'{name:16s} move {t:4.1f} s  result codes {codes}  '
                                   f'max joint error {err:.3f} ({worst[0]})  {"OK" if good else "FAIL"}')
        return ok


def main():
    rclpy.init(args=sys.argv)
    node = ArmTest()
    ok = node.run()
    node.get_logger().info('arm test ' + ('PASSED' if ok else 'FAILED'))
    node.destroy_node()
    rclpy.try_shutdown()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
