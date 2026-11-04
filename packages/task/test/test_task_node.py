"""Task node smoke test against fakes (own ROS domain): it plans the search for scene 3, sends
the first waypoints to a fake /nav/goal server only while safety is ok, and writes the event
log. Needs a sourced ROS environment (skipped otherwise)."""

import json
import os
import threading
import time
from pathlib import Path

import pytest

rclpy = pytest.importorskip('rclpy')
pytest.importorskip('nav2_msgs')
pytest.importorskip('beach_interfaces')
os.environ.setdefault('ROS_DOMAIN_ID', '66')
os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE'] = 'LOCALHOST'

from beach_interfaces.action import PickObject, PlaceObject  # noqa: E402
from beach_interfaces.msg import WorldModel  # noqa: E402
from beach_interfaces.srv import ClassifyItem, UpdateItem  # noqa: E402
from nav2_msgs.action import NavigateToPose  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from rclpy.action import ActionServer  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from task.node import TaskNode  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def test_task_node_searches_and_logs():
    rclpy.init()
    run_id = 'test_task_node'
    log = REPO / 'results' / 'runs' / f'{run_id}.jsonl'
    log.unlink(missing_ok=True)
    try:
        task = TaskNode(parameter_overrides=[Parameter('repo', value=str(REPO)), Parameter('seed', value=3),
                                             Parameter('run_id', value=run_id)])
        fake = rclpy.create_node('fakes')
        goals = []

        def execute(gh):
            goals.append((gh.request.pose.pose.position.x, gh.request.pose.pose.position.y))
            time.sleep(0.2)
            gh.succeed()
            return NavigateToPose.Result()
        ActionServer(fake, NavigateToPose, '/nav/goal', execute)
        # the node starts only once all its servers exist
        ActionServer(fake, PickObject, '/grasp/pick', lambda gh: PickObject.Result())
        ActionServer(fake, PlaceObject, '/grasp/place', lambda gh: PlaceObject.Result())
        fake.create_service(ClassifyItem, '/classify', lambda req, res: res)
        fake.create_service(UpdateItem, '/world_model/update_item', lambda req, res: res)
        odom_pub = fake.create_publisher(Odometry, '/odom/global', 10)
        wm_pub = fake.create_publisher(WorldModel, '/world_model', 10)
        safety_pub = fake.create_publisher(String, '/safety/state', LATCHED)
        states = []
        fake.create_subscription(String, '/task/state', lambda m: states.append(m.data), LATCHED)
        ex = MultiThreadedExecutor()
        ex.add_node(task)
        ex.add_node(fake)
        threading.Thread(target=ex.spin, daemon=True).start()

        def feed(seconds):
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                o = Odometry()
                o.pose.pose.position.x, o.pose.pose.position.y = 4.0, 13.946
                o.pose.pose.orientation.w = 1.0
                odom_pub.publish(o)
                wm_pub.publish(WorldModel())
                time.sleep(0.05)

        safety_pub.publish(String(data='stop'))
        feed(1.5)
        assert goals == []                                      # nothing starts while not ok
        safety_pub.publish(String(data='ok'))
        feed(2.5)
        assert len(goals) >= 2 and states and states[-1] == 'SEARCH'
        wp = task.task.waypoints
        assert goals[0] == pytest.approx(wp[0], abs=1e-6) and goals[1] == pytest.approx(wp[1], abs=1e-6)
        ex.shutdown()
        task.log_file.close()
        records = [json.loads(line) for line in log.read_text().splitlines()]
        assert records[0]['event'] == 'run_start' and records[0]['seed'] == 3
    finally:
        log.unlink(missing_ok=True)
        rclpy.try_shutdown()


def test_bin_rim_from_the_ground_under_the_bin():
    """B1: a bin on flat sand 0.20 m above map z 0 has its rim at 0.70; `fixed` (baseline A3) keeps 0.50."""
    import copy

    import yaml

    from task.node import bin_rims
    scene = copy.deepcopy(next(s for s in yaml.safe_load(open(REPO / 'bench' / 'scenes.yaml'))['scenes']
                               if s['seed'] == 3))
    t = scene['terrain']
    y = scene['bins'][0]['pose'][1]
    # the whole bin footprint in the flat wet zone, no roughness or dunes
    t.update(waterline_y=y - 1.0, wet_zone_width=2.0, wet_zone_height=0.20, roughness=[], dunes=[],
             z_min=-1.0, z_max=5.0)
    objects = yaml.safe_load(open(REPO / 'sim' / 'common' / 'objects.yaml'))
    fsm = yaml.safe_load(open(REPO / 'config' / 'task.yaml'))['task']['fsm']
    assert fsm['bin_rim'] == 'terrain'
    b = scene['bins'][0]
    x, by, rim, yaw = bin_rims(REPO, scene, objects, fsm)[b['bin_id']]
    assert (x, by, yaw) == tuple(b['pose'])
    assert rim == pytest.approx(0.70, abs=1e-6)
    assert bin_rims(REPO, scene, objects, dict(fsm, bin_rim='fixed'))[b['bin_id']][2] == pytest.approx(0.50)
    with pytest.raises(ValueError):
        bin_rims(REPO, scene, objects, dict(fsm, bin_rim='flat'))
