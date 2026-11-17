"""localization:=ekf (B4): robot_localization dual EKF + navsat_transform, as Nav2's GPS tutorial.

  ros2 launch localization ekf.launch.py repo:=<repository root> seed:=<scene>

  /leg_odom + /localization/imu          -> ekf_local  -> /odom, TF odom -> base_link
  /gnss/fix + /localization/imu + /odom/global -> navsat_transform -> /localization/gnss_odom
  the above + /localization/gnss_odom    -> ekf_global -> /odom/global, TF map -> odom

The datum (map origin) is the scene's geo origin (bench/scenes.yaml `geo`, the simulator's
spherical_coordinates), so `map` is the scene frame; the global EKF starts at the scene's
robot_start pose (a known docking pose). Parameters: packages/localization/config/ekf.yaml.
"""
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node


def scene_of(repo, seed):
    scenes = yaml.safe_load(open(os.path.join(repo, 'bench', 'scenes.yaml')))['scenes']
    return next(s for s in scenes if s['seed'] == seed)


def initial_state(pose):
    x, y, yaw = pose[:3]
    return [float(x), float(y), 0.0, 0.0, 0.0, float(yaw)] + [0.0] * 9


def launch_setup(context):
    cfg = context.launch_configurations
    repo, seed = cfg['repo'] or os.environ.get('BEACH_REPO', ''), int(cfg['seed'])
    scene = scene_of(repo, seed)
    params = os.path.join(get_package_share_directory('localization'), 'config', 'ekf.yaml')
    sim = {'use_sim_time': True}
    geo = scene['geo']
    return [
        Node(package='localization', executable='imu_relay', output='screen', parameters=[{'repo': repo, **sim}]),
        Node(package='robot_localization', executable='ekf_node', name='ekf_local', output='screen',
             parameters=[params, sim], remappings=[('odometry/filtered', '/odom')]),
        Node(package='robot_localization', executable='ekf_node', name='ekf_global', output='screen',
             parameters=[params, sim, {'initial_state': initial_state(scene['robot_start']['pose'])}],
             remappings=[('odometry/filtered', '/odom/global')]),
        Node(package='robot_localization', executable='navsat_transform_node', name='navsat_transform',
             output='screen',
             parameters=[params, sim, {'datum': [float(geo['latitude_deg']), float(geo['longitude_deg']), 0.0]}],
             remappings=[('imu', '/localization/imu'), ('gps/fix', '/gnss/fix'),
                         ('odometry/filtered', '/odom/global'), ('odometry/gps', '/localization/gnss_odom')]),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('repo', default_value=''),
        DeclareLaunchArgument('seed', default_value='0'),
        OpaqueFunction(function=launch_setup),
    ])
