# A0 locomotion evaluation: legubiao basic_quadruped_controller in Gazebo with a custom xacro.
# Same steps as quadruped_ros2_control (dev) controllers/basic_quadruped_controller/launch/gazebo.launch.py
# (Apache-2.0), minus RViz, plus a `description_file` argument.
import os

import xacro
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

HERE = os.path.dirname(os.path.realpath(__file__))


def spawner(name):
    return Node(package='controller_manager', executable='spawner',
                arguments=[name, '--controller-manager', '/controller_manager'],
                parameters=[{'use_sim_time': True}])


def launch_setup(context):
    cfg = context.launch_configurations
    robot_description = xacro.process_file(
        cfg['description_file'], mappings={'GAZEBO': 'true', 'payload': cfg['payload']}).toxml()

    spawn = Node(package='ros_gz_sim', executable='create', output='screen',
                 arguments=['-topic', 'robot_description', '-name', 'go2', '-z', cfg['height']])
    jsb = spawner('joint_state_broadcaster')
    imu = spawner('imu_sensor_broadcaster')

    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [PathJoinSubstitution([FindPackageShare('ros_gz_sim'), 'launch', 'gz_sim.launch.py'])]),
            launch_arguments=[('gz_args', [f" -r -v 3 {'-s ' if cfg['headless'] == 'true' else ''}empty.sdf"])]),
        Node(package='ros_gz_bridge', executable='parameter_bridge',
             arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock']),
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': robot_description, 'publish_frequency': 20.0,
                          'use_tf_static': True, 'ignore_timestamp': True, 'use_sim_time': True}]),
        spawn,
        RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[imu, jsb])),
        RegisterEventHandler(OnProcessExit(target_action=jsb,
                                           on_exit=[spawner('basic_quadruped_controller')])),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('description_file', default_value=os.path.join(HERE, 'legubiao_payload.xacro')),
        DeclareLaunchArgument('payload', default_value='true'),
        DeclareLaunchArgument('height', default_value='0.5'),
        DeclareLaunchArgument('headless', default_value='false'),
        OpaqueFunction(function=launch_setup),
    ])
