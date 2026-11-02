"""Grasping node against fake controllers, a fake /sim/attach, a fake world model and safety
state (own ROS domain). Needs a sourced ROS environment (skipped otherwise)."""

import os
import threading
import time
from pathlib import Path

import pytest

rclpy = pytest.importorskip('rclpy')
pytest.importorskip('beach_interfaces')
pytest.importorskip('pinocchio')
os.environ.setdefault('ROS_DOMAIN_ID', '65')
os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE'] = 'LOCALHOST'

from beach_interfaces.action import PickObject, PlaceObject  # noqa: E402
from beach_interfaces.msg import TrashItem, WorldModel  # noqa: E402
from beach_interfaces.srv import SimAttach  # noqa: E402
from control_msgs.action import FollowJointTrajectory  # noqa: E402
from geometry_msgs.msg import PoseStamped  # noqa: E402
from rclpy.action import ActionClient, ActionServer, CancelResponse  # noqa: E402
from rclpy.callback_groups import ReentrantCallbackGroup  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from grasping.arm import TIPS  # noqa: E402
from grasping.node import GraspingNode  # noqa: E402

REPO = str(Path(__file__).resolve().parents[3])
LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


class Fakes:

    def __init__(self, node):
        self.n = node
        self.arm_goals, self.cancelled, self.sim_calls = [], 0, []
        cb = ReentrantCallbackGroup()
        for name in ('arm_controller', 'gripper_controller'):
            ActionServer(node, FollowJointTrajectory, f'/{name}/follow_joint_trajectory',
                         self.execute(name), callback_group=cb, cancel_callback=lambda _: CancelResponse.ACCEPT)
        node.create_service(SimAttach, '/sim/attach', self.sim('attach'), callback_group=cb)
        node.create_service(SimAttach, '/sim/detach', self.sim('detach'), callback_group=cb)
        self.safety = node.create_publisher(String, '/safety/state', LATCHED)
        self.wm = node.create_publisher(WorldModel, '/world_model', 10)

    def execute(self, name):
        def cb(goal_handle):
            if name == 'arm_controller':
                self.arm_goals.append(list(goal_handle.request.trajectory.points[0].positions))
            end = time.monotonic() + 0.3
            while time.monotonic() < end:
                if goal_handle.is_cancel_requested:
                    self.cancelled += 1
                    goal_handle.canceled()
                    return FollowJointTrajectory.Result()
                time.sleep(0.01)
            goal_handle.succeed()
            return FollowJointTrajectory.Result(error_code=0)
        return cb

    def sim(self, kind):
        def cb(req, res):
            self.sim_calls.append((kind, req.object_id))
            res.success, res.message = True, 'ok'
            return res
        return cb


@pytest.fixture
def system():
    rclpy.init()
    grasping = GraspingNode(parameter_overrides=[Parameter('repo', value=REPO), Parameter('base_frame',
                                                                                          value='base_link')])
    fake_node = rclpy.create_node('fakes')
    fakes = Fakes(fake_node)
    ex = MultiThreadedExecutor()
    ex.add_node(grasping)
    ex.add_node(fake_node)
    t = threading.Thread(target=ex.spin, daemon=True)
    t.start()
    wm = WorldModel()
    wm.items.append(TrashItem(id='wm_000', scene_id='trash_03'))
    end = time.monotonic() + 3.0
    while time.monotonic() < end and grasping.scene_ids.get('wm_000') != 'trash_03':
        fakes.wm.publish(wm)
        time.sleep(0.05)
    yield grasping, fake_node, fakes
    ex.shutdown()
    grasping.destroy_node()
    fake_node.destroy_node()
    rclpy.try_shutdown()


def set_safety(grasping, fakes, state):
    """Publish a safety state and wait until the grasping node has it."""
    fakes.safety.publish(String(data=state))
    end = time.monotonic() + 3.0
    while time.monotonic() < end and grasping.safety_state != state:
        time.sleep(0.02)
    assert grasping.safety_state == state


def send(node, action, name, goal, timeout=60.0):
    client = ActionClient(node, action, name)
    assert client.wait_for_server(timeout_sec=5.0)
    fut = client.send_goal_async(goal)
    end = time.monotonic() + timeout
    while not fut.done() and time.monotonic() < end:
        time.sleep(0.02)
    res = fut.result().get_result_async()
    while not res.done() and time.monotonic() < end:
        time.sleep(0.02)
    return res.result().result


def pick_goal():
    g = PickObject.Goal(id='wm_000')
    g.pose = PoseStamped()
    g.pose.header.frame_id = 'base_link'
    g.pose.pose.position.x, g.pose.pose.position.z = 0.48, -0.285 + 0.033
    g.pose.pose.orientation.w = 1.0
    return g


def test_pick_and_store(system):
    grasping, node, fakes = system
    set_safety(grasping, fakes, 'ok')
    r = send(node, PickObject, '/grasp/pick', pick_goal())
    assert r.success, r.message
    assert fakes.sim_calls == [('attach', 'trash_03')]
    assert len(fakes.arm_goals) == 5                       # up, above, hover, lift, show
    r = send(node, PlaceObject, '/grasp/place',
             PlaceObject.Goal(target=PlaceObject.Goal.COMPARTMENT, item_id='wm_000', bin_id='recyclable'))
    assert r.success, r.message
    assert fakes.sim_calls[-1] == ('detach', 'trash_03')


def test_arm_holds_while_not_ok_and_resumes(system):
    grasping, node, fakes = system
    set_safety(grasping, fakes, 'slow')                      # someone within 3 m: no arm motion at all
    result = {}
    th = threading.Thread(target=lambda: result.update(r=send(node, PickObject, '/grasp/pick', pick_goal())))
    th.start()
    time.sleep(4.0)                                          # IK (~1.5 s) then waiting
    assert fakes.arm_goals == [] and fakes.sim_calls == []
    fakes.safety.publish(String(data='ok'))
    while len(fakes.arm_goals) < 3:                          # moving again
        time.sleep(0.01)
    fakes.safety.publish(String(data='stop'))                # person steps in mid-move
    time.sleep(1.0)
    n = len(fakes.arm_goals)
    assert fakes.cancelled >= 1
    time.sleep(1.0)
    assert len(fakes.arm_goals) == n                         # nothing new while stopped
    fakes.safety.publish(String(data='ok'))
    th.join(timeout=30.0)
    assert result['r'].success, result['r'].message
    assert fakes.sim_calls == [('attach', 'trash_03')]


def test_unreachable_pick_is_refused(system):
    grasping, node, fakes = system
    set_safety(grasping, fakes, 'ok')
    g = pick_goal()
    g.pose.pose.position.x = 1.2
    r = send(node, PickObject, '/grasp/pick', g)
    assert not r.success and r.message.startswith('outside_envelope') and fakes.arm_goals == []


def test_pick_hovers_over_the_item_height_from_the_goal(system):
    grasping, node, fakes = system
    set_safety(grasping, fakes, 'ok')
    g = pick_goal()
    g.size.x, g.size.y, g.size.z = 0.115, 0.066, 0.066
    r = send(node, PickObject, '/grasp/pick', g)
    assert r.success, r.message
    tips_z = grasping.arm.fk(fakes.arm_goals[2], TIPS)[0][2]          # hover
    top = g.pose.pose.position.z + g.size.z / 2
    assert tips_z == pytest.approx(top + grasping.cfg['hover_above_item_m'], abs=grasping.cfg['ik_tolerance_m'])
