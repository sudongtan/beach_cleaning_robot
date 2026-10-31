"""Gazebo Harmonic bridge: world, robot spawn, robot_state_publisher, controllers,
topic bridge (sensors, /clock, ground-truth pose), GNSS covariance relay and, for
beach scenes, ground truth + attach/detach.

  ros2 launch sim_bridge_gazebo gazebo.launch.py [world:=empty] [headless:=false] [basket_load_kg:=0.0]
      [variant:=custom|stock]
  ros2 launch sim_bridge_gazebo gazebo.launch.py scene:=<scene_N.manifest.yaml>

`variant` (robot, notes/decisions.md "Two robot variants"): default `custom`; with a scene, the
manifest's variant (load_scene.py --variant) is used, and an explicit different value is an error.

`world` is a name in sim/gazebo/worlds/ (installed with this package) or a path to an .sdf.
`scene` is a manifest written by sim/gazebo/load_scene.py (`make scene` does it); it sets the
world, the spawn pose and the attachable items, and overrides world/x/y/z/yaw. The world then
starts paused: attach_server detaches the items (Gazebo attaches every DetachableJoint on
startup) and unpauses it.
"""
import os

import xacro
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

SHARE = get_package_share_directory('sim_bridge_gazebo')


def spawner(*controllers, ros_args=()):
    extra = [a for r in ros_args for a in ('--controller-ros-args', r)]
    # Beach scenes start paused until attach_server has detached the items, and controllers
    # only switch while the simulation steps: the default 5 s switch timeout lost that race with
    # the GUI open (joint_state_broadcaster stayed inactive -> no joint TFs). The service call
    # must outlast the switch timeout.
    return Node(package='controller_manager', executable='spawner', output='screen',
                arguments=[*controllers, '--controller-manager', '/controller_manager',
                           '--switch-timeout', '60', '--service-call-timeout', '70', *extra],
                parameters=[{'use_sim_time': True}])


# Gazebo locomotion implementation: legubiao's controller, with its own topics moved under
# /locomotion so only the `locomotion` adapter talks to it (it publishes /leg_odom and its TF).
LEG_CONTROLLER_REMAPS = ('-r odom:=/locomotion/controller_odom',
                         '-r /control_input:=/locomotion/control_input',
                         '-r /tf:=/locomotion/controller_tf')


VARIANTS = ('custom', 'stock')
DEFAULT_VARIANT = 'custom'


def launch_setup(context):
    cfg = context.launch_configurations
    variant = cfg['variant']
    pose = {k: cfg[k] for k in ('x', 'y', 'z', 'yaw')}
    attach_items = []
    scene = cfg['scene']
    if scene:
        with open(scene) as f:
            manifest = yaml.safe_load(f)
        scene_variant = manifest.get('variant', DEFAULT_VARIANT)
        if variant and variant != scene_variant:
            raise RuntimeError(f'variant:={variant} but the scene manifest was built for {scene_variant!r} '
                               f'(regenerate it: make scene VARIANT={variant})')
        variant = scene_variant
        world_path = manifest['world_sdf']
        pose = {k: str(manifest['spawn'][k]) for k in ('x', 'y', 'z', 'yaw')}
        attach_items = manifest['attach_items']
    else:
        world = cfg['world']
        world_path = world if world.endswith('.sdf') else os.path.join(SHARE, 'worlds', world + '.sdf')
    variant = variant or DEFAULT_VARIANT
    if variant not in VARIANTS:
        raise RuntimeError(f"variant:={variant} unknown (available: {', '.join(VARIANTS)})")
    if not os.path.isfile(world_path):
        raise RuntimeError(f'world not found: {world_path}')

    robot_description = xacro.process_file(
        os.path.join(SHARE, 'urdf', 'beach_robot_gazebo.urdf.xacro'),
        mappings={'basket_load_kg': cfg['basket_load_kg'], 'attach_items': ','.join(attach_items),
                  'variant': variant}).toxml()

    # Beach scenes start paused until attach_server has detached every item.
    run = '' if attach_items else '-r '
    gz_args = f"{run}-v 3 {'-s ' if cfg['headless'] == 'true' else ''}{world_path}"
    spawn = Node(package='ros_gz_sim', executable='create', output='screen',
                 arguments=['-topic', 'robot_description', '-name', 'beach_robot',
                            '-x', pose['x'], '-y', pose['y'], '-z', pose['z'], '-Y', pose['yaw']])
    broadcasters = spawner('joint_state_broadcaster', 'imu_sensor_broadcaster')

    scene_nodes = []
    if scene:
        scene_nodes = [
            # These two take sim time from Gazebo messages and don't need the 1 kHz /clock
            # (use_sim_time would cost each Python node ~25 % of a core).
            Node(package='sim_bridge_gazebo', executable='attach_server.py', output='screen',
                 parameters=[{'manifest': scene, 'unpause': bool(attach_items)}]),
            Node(package='sim_bridge_gazebo', executable='ground_truth.py', output='screen',
                 parameters=[{'manifest': scene}]),
        ]

    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
            launch_arguments={'gz_args': gz_args, 'on_exit_shutdown': 'true'}.items()),
        Node(package='ros_gz_bridge', executable='parameter_bridge', output='screen',
             parameters=[{'config_file': os.path.join(SHARE, 'config', 'bridge.yaml')}]),
        # /sim/clock_throttled: 100 Hz sim-time copy of /clock for our own nodes (CPU, decisions 2026-10-31)
        Node(package='sim_bridge_gazebo', executable='clock_throttle', output='screen'),
        # relays messages unchanged (stamps included): no clock needed
        Node(package='sim_bridge_gazebo', executable='gnss_relay.py', output='screen',
             parameters=[{'sensors_file': os.path.join(SHARE, 'config', 'sensors.yaml')}]),
        Node(package='robot_state_publisher', executable='robot_state_publisher', output='screen',
             parameters=[{'robot_description': robot_description, 'use_sim_time': True}]),
        *scene_nodes,
        spawn,
        RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[broadcasters])),
        RegisterEventHandler(OnProcessExit(
            target_action=broadcasters,
            on_exit=[spawner('basic_quadruped_controller', ros_args=LEG_CONTROLLER_REMAPS),
                     spawner('arm_controller', 'gripper_controller')])),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='empty'),
        DeclareLaunchArgument('scene', default_value='',
                              description='scene manifest from sim/gazebo/load_scene.py (overrides world and pose)'),
        DeclareLaunchArgument('headless', default_value='false', description='server only, no GUI'),
        DeclareLaunchArgument('basket_load_kg', default_value='0.0', description='fixed load in the basket'),
        DeclareLaunchArgument('variant', default_value='',
                              description='robot variant: custom (wide jaws, large basket; default) | '
                                          'stock (WidowX gripper, small basket). Empty = from the scene '
                                          'manifest, else custom'),
        DeclareLaunchArgument('x', default_value='0.0'),
        DeclareLaunchArgument('y', default_value='0.0'),
        DeclareLaunchArgument('z', default_value='0.45'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        OpaqueFunction(function=launch_setup),
    ])
