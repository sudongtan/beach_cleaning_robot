"""Top-level simulation bring-up.

  ros2 launch bringup sim.launch.py sim:=gazebo world:=empty [headless:=false] [basket_load_kg:=0.0]
      [variant:=custom|stock]
  ros2 launch bringup sim.launch.py sim:=gazebo scene:=<scene_N.manifest.yaml>    (`make scene SEED=N`)
  ros2 launch bringup sim.launch.py sim:=gazebo scene:=<...> skeleton:=true        (`make run SEED=N`)

Starts the simulator bridge (world, robot, controllers) and the locomotion adapter
(/cmd_vel -> controller, controller odometry -> /leg_odom). Only sim:=gazebo exists so far.
"""
import os

import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

SIMS = ('gazebo',)
LOCOMOTION = {'gazebo': ('legubiao', 'legubiao_rl')}   # legubiao_rl: learned policy (HIMLoco)
CLOCK = '/sim/clock_throttled'   # 100 Hz sim-time clock for our nodes (sim_bridge_gazebo clock_throttle)


def launch_setup(context):
    cfg = context.launch_configurations
    sim, loco = cfg['sim'], cfg['locomotion']
    if sim not in SIMS:
        raise RuntimeError(f"sim:={sim} is not supported (available: {', '.join(SIMS)}; isaac comes in B0)")
    if loco not in LOCOMOTION[sim]:
        raise RuntimeError(f"locomotion:={loco} is not available for sim:={sim} ({', '.join(LOCOMOTION[sim])})")

    bridge = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('sim_bridge_gazebo'), 'launch', 'gazebo.launch.py')),
        launch_arguments={k: cfg[k] for k in ('world', 'scene', 'headless', 'basket_load_kg', 'variant',
                                              'locomotion')}.items())

    skeleton = cfg['skeleton'] == 'true'
    if skeleton and not cfg['scene']:
        raise RuntimeError('skeleton:=true needs scene:= (a scene manifest with its seed)')
    adapter = Node(
        package='locomotion', executable='legubiao_adapter', namespace='locomotion', output='screen',
        parameters=[os.path.join(get_package_share_directory('locomotion'), 'config', f'{loco}_adapter.yaml'),
                    # with the skeleton, localization owns odom -> base_link
                    {'use_sim_time': True, 'publish_tf': cfg['publish_odom_tf'] == 'true' and not skeleton}],
        remappings=[('/clock', CLOCK), ('imu', '/imu_sensor_broadcaster/imu')])
    nodes = [bridge, adapter]
    if skeleton:
        with open(cfg['scene']) as f:
            seed = yaml.safe_load(f)['seed']
        nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('bringup'), 'launch', 'skeleton.launch.py')),
            launch_arguments={'seed': str(seed), 'run_id': cfg['run_id'], 'clock_topic': CLOCK}.items()))
    if cfg['rviz'] == 'true':
        nodes.append(Node(package='rviz2', executable='rviz2', output='log',
                          arguments=['-d', os.path.join(get_package_share_directory('sim_bridge_gazebo'),
                                                        'rviz', 'sensors.rviz')],
                          parameters=[{'use_sim_time': True}]))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('sim', default_value='gazebo', description='gazebo (isaac: B0)'),
        DeclareLaunchArgument('world', default_value='empty'),
        DeclareLaunchArgument('scene', default_value='',
                              description='scene manifest (sim/gazebo/load_scene.py); overrides world'),
        DeclareLaunchArgument('headless', default_value='false'),
        DeclareLaunchArgument('basket_load_kg', default_value='0.0'),
        DeclareLaunchArgument('variant', default_value='',
                              description='custom | stock robot; empty = from the scene manifest, else custom'),
        DeclareLaunchArgument('locomotion', default_value='legubiao',
                              description='legubiao (basic controller) | legubiao_rl (HIMLoco policy)'),
        DeclareLaunchArgument('rviz', default_value='false', description='RViz with robot, TF, camera images'),
        DeclareLaunchArgument('skeleton', default_value='false',
                              description='true: also start the A2 skeleton (skeleton.launch.py) for the scene'),
        DeclareLaunchArgument('run_id', default_value='', description='task event log name (skeleton only)'),
        DeclareLaunchArgument('publish_odom_tf', default_value='true',
                              description='adapter publishes odom -> base_link (false once localization runs)'),
        OpaqueFunction(function=launch_setup),
    ])
