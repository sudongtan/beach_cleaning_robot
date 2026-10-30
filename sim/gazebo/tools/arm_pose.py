#!/usr/bin/env python3
"""Move the arm (and fingers) to a named pose of the robot variant.

    arm_pose.py <pose> [--variant custom|stock] [--time 3.0] [--fingers OPENING_M] [--direct]
    arm_pose.py --joints q1,q2,q3,q4,q5,q6 [--fingers OPENING_M]

Poses come from beach_description/config/arm_poses.yaml (stock) or
arm_poses_custom.yaml (custom, the default). With the custom robot every move to a
named pose goes through the `up` waypoint first (direct moves pass < 1 cm from the
trunk and the basket with the long jaws; notes/logs/A0.md section 17), unless --direct.
Finger values are the left finger joint in m; "closed" is the joint's lower limit,
read from the robot model (stock 0.015, custom 0.010 = 2 cm jaw gap).
"""

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import rclpy
import yaml
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectoryPoint

REPO = Path(__file__).resolve().parents[3]
DESC = REPO / 'packages' / 'beach_description'
ARM = ['arm_waist', 'arm_shoulder', 'arm_elbow', 'arm_forearm_roll', 'arm_wrist_angle', 'arm_wrist_rotate']
FINGERS = ['arm_left_finger', 'arm_right_finger']
VARIANTS = ('custom', 'stock')


def poses_file(variant):
    return DESC / 'config' / ('arm_poses_custom.yaml' if variant == 'custom' else 'arm_poses.yaml')


def load_poses(variant):
    return yaml.safe_load(open(poses_file(variant)))['poses']


def finger_limits(variant):
    """(closed, open) left finger joint values from the robot model (xacro)."""
    urdf = subprocess.run(['xacro', str(DESC / 'urdf' / 'beach_robot.urdf.xacro'), f'variant:={variant}'],
                          check=True, capture_output=True, text=True).stdout
    for j in ET.fromstring(urdf).iter('joint'):
        if j.get('name') == 'arm_left_finger':
            lim = j.find('limit')
            return float(lim.get('lower')), float(lim.get('upper'))
    raise RuntimeError('arm_left_finger not found')


def send(node, controller, joints, positions, secs):
    client = ActionClient(node, FollowJointTrajectory, f'/{controller}/follow_joint_trajectory')
    if not client.wait_for_server(timeout_sec=10.0):
        sys.exit(f'{controller} action server not available')
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = joints
    pt = JointTrajectoryPoint(positions=[float(p) for p in positions])
    pt.time_from_start = Duration(sec=int(secs), nanosec=int((secs % 1) * 1e9))
    goal.trajectory.points = [pt]
    fut = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, fut)
    res = fut.result().get_result_async()
    rclpy.spin_until_future_complete(node, res)
    return res.result().result.error_code


def set_fingers(node, opening, secs=1.0):
    return send(node, 'gripper_controller', FINGERS, [opening, -opening], secs)


def move_arm(node, joints, secs=3.0):
    return send(node, 'arm_controller', ARM, joints, secs)


def move_to(node, variant, name, poses, secs=3.0, fingers=None, direct=False):
    """Named pose; custom robot via `up` (fingers are set at the target, not on the way)."""
    errs = []
    if variant == 'custom' and name != 'up' and not direct:
        errs.append(move_arm(node, poses['up']['arm'], secs))
    errs.append(move_arm(node, poses[name]['arm'], secs))
    errs.append(set_fingers(node, poses[name]['fingers'] if fingers is None else fingers))
    return max(errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pose', nargs='?')
    ap.add_argument('--variant', choices=VARIANTS, default='custom')
    ap.add_argument('--joints', help='six comma-separated arm joint angles instead of a named pose')
    ap.add_argument('--time', type=float, default=3.0)
    ap.add_argument('--fingers', type=float, help='left finger joint, m (default: the pose value)')
    ap.add_argument('--direct', action='store_true', help='custom: skip the `up` waypoint')
    a = ap.parse_args()
    rclpy.init()
    n = Node('arm_pose_tool')
    if a.joints:
        e = move_arm(n, [float(v) for v in a.joints.split(',')], a.time)
        if a.fingers is not None:
            e = max(e, set_fingers(n, a.fingers))
    else:
        e = move_to(n, a.variant, a.pose, load_poses(a.variant), a.time, a.fingers, a.direct)
    print(f'{a.pose or a.joints} ({a.variant}): error_code {e}')
    rclpy.try_shutdown()
    sys.exit(1 if e else 0)


if __name__ == '__main__':
    main()
