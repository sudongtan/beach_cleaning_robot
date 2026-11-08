"""A2 walking skeleton: every component, without a simulator.

  ros2 launch bringup skeleton.launch.py seed:=3 [repo:=<repository root>]
      [localization:=ground_truth] [mapping:=ground_truth] [perception:=ground_truth]
      [classifier:=ground_truth] [navigation:=nav2] [grasp_mode:=attach]

Starts localization, mapping, perception, classifier, world model, safety supervisor,
navigation (Nav2), grasping and task, all with use_sim_time. The simulator side (sim.launch.py)
provides /clock, /sim/ground_truth/*, /people, the controllers and /sim/attach; run it with
publish_odom_tf:=false so only localization owns odom -> base_link (A2b merges the two).
`repo` defaults to $BEACH_REPO (exported by the Makefile). Implementations are chosen with
launch arguments; each upgrade adds its name to IMPLEMENTATIONS (CLAUDE.md rules).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, SetRemap

# component -> {implementation: (package, executable)}
IMPLEMENTATIONS = {
    'localization': {'ground_truth': ('localization', 'ground_truth')},
    'mapping': {'ground_truth': ('mapping', 'ground_truth')},
    'perception': {'ground_truth': ('perception', 'ground_truth_detector')},
    'classifier': {'ground_truth': ('perception', 'ground_truth_classifier')},
}
NAVIGATION = ('nav2',)
GRASP_MODES = ('attach',)          # physics: not implemented yet (decisions 2026-10-30)


def launch_setup(context):
    cfg = context.launch_configurations
    repo = cfg['repo'] or os.environ.get('BEACH_REPO', '')
    if not repo or not os.path.isfile(os.path.join(repo, 'bench', 'scenes.yaml')):
        raise RuntimeError(f'repo:={repo!r} is not the repository root (set repo:= or BEACH_REPO)')
    seed = int(cfg['seed'])
    common = {'use_sim_time': True}
    scene = {'repo': repo, 'seed': seed, **common}

    nodes = []
    for comp, impls in IMPLEMENTATIONS.items():
        choice = cfg[comp]
        if choice not in impls:
            raise RuntimeError(f'{comp}:={choice} unknown (available: {", ".join(impls)})')
        pkg, exe = impls[choice]
        params = scene if comp != 'localization' else common
        nodes.append(Node(package=pkg, executable=exe, name=comp, output='screen', parameters=[params]))
    if cfg['navigation'] not in NAVIGATION:
        raise RuntimeError(f"navigation:={cfg['navigation']} unknown (available: {', '.join(NAVIGATION)})")
    if cfg['grasp_mode'] not in GRASP_MODES:
        raise RuntimeError(f"grasp_mode:={cfg['grasp_mode']} not available ({', '.join(GRASP_MODES)})")

    safety_cfg = os.path.join(get_package_share_directory('safety'), 'config', 'safety.yaml')
    nodes += [
        Node(package='world_model', executable='world_model', output='screen', parameters=[scene]),
        Node(package='safety', executable='supervisor', name='safety_supervisor', output='screen',
             parameters=[safety_cfg, common]),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('navigation'), 'launch', 'navigation.launch.py'))),
        Node(package='grasping', executable='grasping', output='screen',
             parameters=[{'repo': repo, 'grasp_mode': cfg['grasp_mode'], **common}]),
        Node(package='task', executable='task', output='screen',
             parameters=[{**scene, 'run_id': cfg['run_id'], 'locomotion': cfg['locomotion']}]),
    ]
    # all our nodes (and Nav2) take the 100 Hz sim-time clock from sim_bridge_gazebo's clock_throttle
    # instead of the 1 kHz /clock (CPU; decisions 2026-10-31). Empty clock_topic = /clock.
    if cfg['clock_topic']:
        return [GroupAction([SetRemap('/clock', cfg['clock_topic']), *nodes])]
    return nodes


def generate_launch_description():
    args = [DeclareLaunchArgument(c, default_value='ground_truth', description=f'{c} implementation')
            for c in IMPLEMENTATIONS]
    return LaunchDescription([
        DeclareLaunchArgument('seed', default_value='0', description='benchmark scene (bench/scenes.yaml)'),
        DeclareLaunchArgument('repo', default_value='', description='repository root (default: $BEACH_REPO)'),
        DeclareLaunchArgument('navigation', default_value='nav2'),
        DeclareLaunchArgument('grasp_mode', default_value='attach'),
        DeclareLaunchArgument('run_id', default_value='', description='event log name (default: time + seed)'),
        DeclareLaunchArgument('locomotion', default_value='legubiao_robot_lab',
                              description='walking controller (sim.launch.py); the task adapts its small moves'),
        DeclareLaunchArgument('clock_topic', default_value='',
                              description='clock topic for all nodes (sim.launch.py: /sim/clock_throttled)'),
        *args,
        OpaqueFunction(function=launch_setup),
    ])
