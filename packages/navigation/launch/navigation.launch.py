"""Nav2 for the beach robot (navigation:=nav2).

Our own launch instead of nav2_bringup's navigation_launch.py because the topic chain differs:
  controller_server / behavior_server -> /nav/cmd_vel_raw -> velocity_smoother -> /cmd_vel_nav
  -> safety supervisor -> /cmd_vel   (notes/architecture.md: only the supervisor publishes /cmd_vel)
and the NavigateToPose action is served as /nav/goal. No collision monitor (no range sensors yet;
safety is the supervisor's job). The static /map comes from the mapping node.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

NODES = ['controller_server', 'planner_server', 'behavior_server', 'bt_navigator', 'velocity_smoother']


def generate_launch_description():
    params = LaunchConfiguration('params_file')
    raw = ('cmd_vel', '/nav/cmd_vel_raw')
    nodes = [
        Node(package='nav2_controller', executable='controller_server', output='screen',
             parameters=[params], remappings=[raw]),
        Node(package='nav2_planner', executable='planner_server', output='screen', parameters=[params]),
        Node(package='nav2_behaviors', executable='behavior_server', output='screen',
             parameters=[params], remappings=[raw]),
        Node(package='nav2_bt_navigator', executable='bt_navigator', output='screen',
             parameters=[params], remappings=[('navigate_to_pose', '/nav/goal')]),
        Node(package='nav2_velocity_smoother', executable='velocity_smoother', output='screen',
             parameters=[params], remappings=[raw, ('cmd_vel_smoothed', '/cmd_vel_nav')]),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager', name='lifecycle_manager_navigation',
             output='screen', parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time'),
                                           'autostart': True, 'node_names': NODES}]),
    ]
    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=os.path.join(
            get_package_share_directory('navigation'), 'config', 'nav2_params.yaml')),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        *nodes,
    ])
