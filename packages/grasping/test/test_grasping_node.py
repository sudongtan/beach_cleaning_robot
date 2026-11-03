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
        self.fail_goal = None                       # number of the arm goal that fails (1 = first)
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
            if name == 'arm_controller' and len(self.arm_goals) == self.fail_goal:
                return FollowJointTrajectory.Result(error_code=-5)   # GOAL_TOLERANCE_VIOLATED
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


def gt_objects(*objs):
    from vision_msgs.msg import Detection3D, Detection3DArray
    arr = Detection3DArray()
    arr.header.frame_id = 'base_link'
    for oid, (x, y, z) in objs:
        d = Detection3D(id=oid)
        d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.center.position.z = x, y, z
        d.bbox.center.orientation.w = 1.0
        arr.detections.append(d)
    return arr


def test_pick_refused_when_the_item_is_not_at_the_grasp_point(system):
    """A2b: a phantom item carried the scene id of an item already in the basket; attach mode
    attached it there, far from the jaws."""
    from vision_msgs.msg import Detection3DArray
    grasping, node, fakes = system
    set_safety(grasping, fakes, 'ok')
    pub = node.create_publisher(Detection3DArray, '/sim/ground_truth/objects', 10)
    end = time.monotonic() + 3.0
    while time.monotonic() < end and grasping.gt is None:
        pub.publish(gt_objects(('trash_03', (-0.2, 0.0, 0.05))))      # in the basket
        time.sleep(0.05)
    r = send(node, PickObject, '/grasp/pick', pick_goal())
    assert not r.success and r.message == 'no_sim_object'
    assert fakes.sim_calls == [] and fakes.arm_goals == []

    g = pick_goal()                                                   # really there: picked
    p = g.pose.pose.position
    grasping.gt = gt_objects(('trash_03', (p.x, p.y, p.z)))
    r = send(node, PickObject, '/grasp/pick', g)
    assert r.success, r.message
    assert fakes.sim_calls == [('attach', 'trash_03')]


def bin_goal():
    g = PlaceObject.Goal(target=PlaceObject.Goal.BIN, item_id='wm_000', bin_id='recyclable')
    g.pose = PoseStamped()
    g.pose.header.frame_id = 'base_link'
    g.pose.pose.position.x, g.pose.pose.position.z = 0.87, 0.5 - 0.285     # bin rim centre ahead
    g.pose.pose.orientation.w = 1.0
    return g


def test_bin_place_takes_the_item_from_inside_the_compartment(system):
    """A2b a2b_bin2: attaching from above the rim dragged the item through the basket wall."""
    from grasping.arm import compartment
    grasping, node, fakes = system
    set_safety(grasping, fakes, 'ok')
    c = compartment(grasping.arm, 'recyclable')
    item = ((c['x'][0] + c['x'][1]) / 2, (c['y'][0] + c['y'][1]) / 2, c['floor'] + 0.011)   # cigarette pack
    grasping.gt = gt_objects(('trash_03', item))
    grasping.gt.detections[0].bbox.size.x, grasping.gt.detections[0].bbox.size.y = 0.088, 0.055
    grasping.gt.detections[0].bbox.size.z = 0.022
    r = send(node, PlaceObject, '/grasp/place', bin_goal())
    assert r.success, r.message
    assert fakes.sim_calls == [('attach', 'trash_03'), ('detach', 'trash_03')]
    hover = grasping.arm.fk(fakes.arm_goals[3], TIPS)[0]                     # up, over, above, hover
    assert hover[2] == pytest.approx(item[2] + 0.011 + grasping.cfg['hover_above_item_m'], abs=0.006)
    assert hover[:2] == pytest.approx(item[:2], abs=0.01)
    release = fakes.arm_goals[-3]
    assert abs(release[0]) <= 1.0 + 1e-6                                     # front side: no half-turn swing


def test_failed_bin_place_releases_the_item(system):
    from grasping.arm import compartment
    grasping, node, fakes = system
    set_safety(grasping, fakes, 'ok')
    c = compartment(grasping.arm, 'recyclable')
    grasping.gt = gt_objects(('trash_03', ((c['x'][0] + c['x'][1]) / 2, 0.07, c['floor'] + 0.03)))
    grasping.gt.detections[0].bbox.size.z = 0.06
    fakes.fail_goal = 5                         # up, over, above, hover, then the lift out fails
    r = send(node, PlaceObject, '/grasp/place', bin_goal())
    assert not r.success and 'item released' in r.message
    assert fakes.sim_calls == [('attach', 'trash_03'), ('detach', 'trash_03')]


def test_bin_place_refused_when_the_item_is_not_in_the_basket(system):
    """A2b a2b_full1: the tin had left the basket; the unload tried an item 20 m behind the robot.
    A distinct reason lets the task free the compartment and look for the item again."""
    grasping, node, fakes = system
    set_safety(grasping, fakes, 'ok')
    grasping.gt = gt_objects(('trash_03', (-20.392, -2.813, -0.528)))
    r = send(node, PlaceObject, '/grasp/place', bin_goal())
    assert not r.success and r.message.startswith('not_in_basket')
    assert fakes.sim_calls == [] and fakes.arm_goals == []


def hold_at_show_pose(grasping, size, yaw=0.6):
    """TF of the jaw tips at the show pose, and ground truth of an item hanging below them."""
    import numpy as np
    from geometry_msgs.msg import TransformStamped
    t, R = grasping.arm.pose(grasping.cfg['show_pose'], TIPS)
    tf = TransformStamped()
    tf.header.frame_id, tf.child_frame_id = 'base_link', TIPS
    tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = map(float, t)
    w = np.sqrt(max(1e-12, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    tf.transform.rotation.w = w
    tf.transform.rotation.x = (R[2, 1] - R[1, 2]) / (4 * w)
    tf.transform.rotation.y = (R[0, 2] - R[2, 0]) / (4 * w)
    tf.transform.rotation.z = (R[1, 0] - R[0, 1]) / (4 * w)
    grasping.tf_buffer.set_transform_static(tf, 'test')
    c = t + R @ np.array([size[2] / 2 + grasping.cfg['hover_above_item_m'], 0.0, 0.0])   # tool x = down
    grasping.gt = gt_objects(('trash_03', tuple(c)))
    d = grasping.gt.detections[0]
    d.bbox.size.x, d.bbox.size.y, d.bbox.size.z = size
    d.bbox.center.orientation.z, d.bbox.center.orientation.w = np.sin(yaw / 2), np.cos(yaw / 2)
    return c


def store_goal():
    return PlaceObject.Goal(target=PlaceObject.Goal.COMPARTMENT, item_id='wm_000', bin_id='recyclable')


def test_store_lowers_the_item_to_a_spot_in_the_compartment(system):
    grasping, node, fakes = system
    set_safety(grasping, fakes, 'ok')
    hold_at_show_pose(grasping, (0.21, 0.065, 0.065))                 # PET bottle
    r = send(node, PlaceObject, '/grasp/place', store_goal())
    assert r.success, r.message
    over = grasping.arm.fk(grasping.arm.poses['over_recyclable']['arm'], TIPS)[0]
    low = grasping.arm.fk(fakes.arm_goals[3], TIPS)[0]                 # up, over, above, lower
    assert low[2] < over[2] - 0.05
    assert fakes.sim_calls == [('detach', 'trash_03')]


def test_store_refused_when_the_compartment_is_full(system):
    """Still held, nothing moved: the task puts the item back and empties the basket first."""
    from grasping.arm import compartment
    grasping, node, fakes = system
    set_safety(grasping, fakes, 'ok')
    hold_at_show_pose(grasping, (0.11, 0.075, 0.075))                 # tin
    c = compartment(grasping.arm, 'recyclable')
    pile = gt_objects(('trash_05', (sum(c['x']) / 2, sum(c['y']) / 2, c['rim'] - 0.06))).detections[0]
    pile.bbox.size.x, pile.bbox.size.y, pile.bbox.size.z = 0.34, 0.15, 0.08
    grasping.gt.detections.append(pile)
    r = send(node, PlaceObject, '/grasp/place', store_goal())
    assert not r.success and r.message.startswith('compartment_full')
    assert fakes.sim_calls == [] and fakes.arm_goals == []


def test_store_without_ground_truth_releases_from_the_fixed_pose(system):
    grasping, node, fakes = system
    set_safety(grasping, fakes, 'ok')
    grasping.gt = gt_objects(('trash_03', (0.5, 0.0, 0.0)))           # no TF of the jaw tips
    r = send(node, PlaceObject, '/grasp/place', store_goal())
    assert r.success, r.message
    assert fakes.arm_goals[1] == pytest.approx(grasping.arm.poses['over_recyclable']['arm'])
    assert fakes.sim_calls == [('detach', 'trash_03')]


def test_bin_place_of_an_item_already_in_the_bin_succeeds_without_moving(system):
    """A2b a2b_dev1: a bottle lying on the jar came out with it and fell into the bin."""
    grasping, node, fakes = system
    set_safety(grasping, fakes, 'ok')
    grasping.gt = gt_objects(('trash_03', (0.811, 0.034, -0.250)))     # bin rim centre at x 0.87
    r = send(node, PlaceObject, '/grasp/place', bin_goal())
    assert r.success and r.message == 'already_in_bin'
    assert fakes.sim_calls == [] and fakes.arm_goals == []
