"""localization:=own_ekf (B4): our own planar EKF (localization/own_ekf.py) + the IMU relay.

  ros2 launch localization own_ekf.launch.py repo:=<repository root> seed:=<scene>
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node


def launch_setup(context):
    cfg = context.launch_configurations
    repo, seed = cfg['repo'] or os.environ.get('BEACH_REPO', ''), int(cfg['seed'])
    common = {'repo': repo, 'use_sim_time': True}
    return [Node(package='localization', executable='imu_relay', output='screen', parameters=[common]),
            Node(package='localization', executable='own_ekf', output='screen', parameters=[{**common, 'seed': seed}])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('repo', default_value=''),
        DeclareLaunchArgument('seed', default_value='0'),
        OpaqueFunction(function=launch_setup),
    ])
