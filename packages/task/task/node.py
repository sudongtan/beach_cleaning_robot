"""Task node: runs task.fsm on ROS (notes/architecture.md).

  in:  /odom/global (robot pose), /world_model, /safety/state, /camera/rgb (for /classify)
  out: /nav/goal (NavigateToPose), /grasp/pick, /grasp/place, /classify,
       /world_model/update_item, /cmd_vel_nav (final approach), /task/state
  log: results/runs/<run_id>.jsonl, one JSON object per line (first line: run metadata)

The search path comes from the scene's geofence and the same /map grid the mapping node
publishes (mapping.ground_truth.load). Bin positions come from bench/scenes.yaml (bins are
static map features, like the obstacles in /map). Parameters: repo, seed, run_id.
"""

import json
import math
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import rclpy
import yaml
from beach_interfaces.action import PickObject, PlaceObject
from beach_interfaces.msg import WorldModel
from beach_interfaces.srv import ClassifyItem, UpdateItem
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import Image
from std_msgs.msg import String

from mapping.ground_truth import load as load_map
from mapping.grid import lethal_at
from task.fsm import Capacity, ItemView, Snapshot, Task
from task.search import lane_spacing, plan

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
SUCCEEDED = 4                                     # action_msgs/GoalStatus.STATUS_SUCCEEDED
TARGETS = {'compartment': PlaceObject.Goal.COMPARTMENT, 'bin': PlaceObject.Goal.BIN,
           'ground': PlaceObject.Goal.GROUND}


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def pose_stamped(frame, x, y, z=0.0, yaw=0.0):
    p = PoseStamped()
    p.header.frame_id = frame
    p.pose.position.x, p.pose.position.y, p.pose.position.z = float(x), float(y), float(z)
    p.pose.orientation.z, p.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    return p


class ActionHandle:
    """A goal sent to an action server; done() once its result is in, result() mapped by `convert`."""

    def __init__(self, client, goal, convert, feedback=None):
        self.convert, self.value, self.goal_handle = convert, None, None
        self.event = threading.Event()
        if not client.wait_for_server(timeout_sec=2.0):
            self.value = convert(None)
            self.event.set()
            return
        client.send_goal_async(goal, feedback_callback=feedback).add_done_callback(self.on_sent)

    def on_sent(self, fut):
        self.goal_handle = fut.result()
        if not self.goal_handle.accepted:
            self.value = self.convert(None)
            self.event.set()
            return
        self.goal_handle.get_result_async().add_done_callback(self.on_result)

    def on_result(self, fut):
        self.value = self.convert(fut.result())
        self.event.set()

    def done(self):
        return self.event.is_set()

    def result(self):
        return self.value

    def cancel(self):
        if self.goal_handle is not None and not self.done():
            self.goal_handle.cancel_goal_async()


class ServiceHandle:

    def __init__(self, client, request, convert):
        self.convert, self.value, self.event = convert, None, threading.Event()
        if not client.wait_for_service(timeout_sec=2.0):
            self.value = convert(None)
            self.event.set()
            return
        client.call_async(request).add_done_callback(self.on_done)

    def on_done(self, fut):
        self.value = self.convert(fut.result())
        self.event.set()

    def done(self):
        return self.event.is_set()

    def result(self):
        return self.value


class TaskNode(Node):

    def __init__(self, **kwargs):
        super().__init__('task', **kwargs)
        repo = self.declare_parameter('repo', '').value
        if not repo:
            raise RuntimeError('parameter repo (repository root) is required')
        repo = Path(repo)
        self.seed = self.declare_parameter('seed', 0).value
        run_id = self.declare_parameter('run_id', '').value or datetime.now(timezone.utc).strftime(
            f'%Y%m%dT%H%M%SZ_seed{self.seed}')
        cfg = yaml.safe_load(open(repo / 'config' / 'task.yaml'))['task']
        grasp = yaml.safe_load(open(repo / 'config' / 'grasping.yaml'))['grasping']
        wmc = yaml.safe_load(open(repo / 'config' / 'world_model.yaml'))['world_model']
        limits = yaml.safe_load(open(repo / 'config' / 'robot_limits.yaml'))['robot_limits']
        sorting = yaml.safe_load(open(repo / 'config' / 'sorting_rules.yaml'))['sorting']
        camera = yaml.safe_load(open(repo / 'config' / 'sensors.yaml'))['camera']
        objects = yaml.safe_load(open(repo / 'sim' / 'common' / 'objects.yaml'))
        scene, occ, res = load_map(str(repo), self.seed)
        s = cfg['search']
        spacing = lane_spacing(camera['horizontal_fov_rad'], s['view_distance_m'], s['overlap'])
        waypoints = plan(scene['geofence']['polygon'], scene['robot_start']['pose'][:2], spacing,
                         s['edge_margin_m'], s['waypoint_spacing_m'], lambda x, y: lethal_at(occ, res, x, y))
        bins = {}
        for b in scene['bins']:
            rim = max(p['pose'][2] + p['size'][2] / 2 for p in objects['bins'][b['model']]['collision'])
            bins[b['bin_id']] = (b['pose'][0], b['pose'][1], rim, b['pose'][2])
        self.task = Task(self, cfg['fsm'], grasp,
                         Capacity(wmc['max_items_per_compartment'], limits['basket_load_max_kg'],
                                  wmc['default_item_mass_kg']),
                         waypoints, bins, sorting['fallback_bin'], sorting['confidence_threshold'])

        self.robot = None
        self.items, self.compartments, self.load = {}, {}, 0.0
        self.safety = 'stop'
        self.image = Image()
        self.nav_handle = None
        self.lock = threading.Lock()
        self.log_path = repo / 'results' / 'runs' / f'{run_id}.jsonl'
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.log_path, 'a')
        self.log({'event': 'run_start', 'run_id': run_id, 'seed': self.seed, 'waypoints': len(waypoints),
                  'wall_time': datetime.now(timezone.utc).isoformat()})

        io = ReentrantCallbackGroup()
        self.nav = ActionClient(self, NavigateToPose, '/nav/goal', callback_group=io)
        self.pick_client = ActionClient(self, PickObject, '/grasp/pick', callback_group=io)
        self.place_client = ActionClient(self, PlaceObject, '/grasp/place', callback_group=io)
        self.classify_client = self.create_client(ClassifyItem, '/classify', callback_group=io)
        self.update_client = self.create_client(UpdateItem, '/world_model/update_item', callback_group=io)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_nav', 10)
        self.state_pub = self.create_publisher(String, '/task/state', LATCHED)
        self.create_subscription(Odometry, '/odom/global', self.on_odom, 10, callback_group=io)
        self.create_subscription(WorldModel, '/world_model', self.on_world_model, 10, callback_group=io)
        self.create_subscription(String, '/safety/state', self.on_safety, LATCHED, callback_group=io)
        self.create_subscription(Image, '/camera/rgb', self.on_image, 1, callback_group=io)
        self.create_timer(1.0 / cfg['fsm']['rate_hz'], self.tick, callback_group=MutuallyExclusiveCallbackGroup())
        self.last_state = None
        self.started = False

    # --- inputs ---------------------------------------------------------------------------
    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_odom(self, msg):
        p = msg.pose.pose
        self.robot = (p.position.x, p.position.y, yaw_of(p.orientation))

    def on_world_model(self, msg):
        with self.lock:
            fresh = {}
            for it in msg.items:
                p = it.pose.pose.position
                old = self.items.get(it.id)
                view = ItemView(it.id, it.object_class, p.x, p.y, p.z, it.status, it.bin_id)
                if old is not None and getattr(old, 'override_until', 0) > self.now() and old.status != it.status:
                    view = old                    # our own update hasn't reached /world_model yet
                fresh[it.id] = view
            self.items = fresh
            self.compartments = {'recyclable': list(msg.compartment_recyclable),
                                 'residual': list(msg.compartment_residual)}
            self.load = msg.basket_load_kg

    def on_safety(self, msg):
        self.safety = msg.data

    def on_image(self, msg):
        self.image = msg

    # --- io for task.fsm --------------------------------------------------------------------
    def nav_to(self, x, y, yaw):
        goal = NavigateToPose.Goal(pose=pose_stamped('map', x, y, yaw=yaw))
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        self.nav_handle = ActionHandle(self.nav, goal, lambda r: r is not None and r.status == SUCCEEDED)
        return self.nav_handle

    def cancel_nav(self):
        if self.nav_handle is not None:
            self.nav_handle.cancel()

    def pick(self, item_id, pos):
        goal = PickObject.Goal(id=item_id, pose=pose_stamped('map', *pos))
        return ActionHandle(self.pick_client, goal,
                            lambda r: (False, 'no_server') if r is None else (r.result.success, r.result.message))

    def place(self, target, item_id, bin_id, pose):
        goal = PlaceObject.Goal(target=TARGETS[target], item_id=item_id, bin_id=bin_id)
        if pose is not None:
            x, y, z, yaw = pose
            goal.pose = pose_stamped('map', x, y, z, yaw)
        return ActionHandle(self.place_client, goal,
                            lambda r: (False, 'no_server') if r is None else (r.result.success, r.result.message))

    def classify(self):
        return ServiceHandle(self.classify_client, ClassifyItem.Request(image=self.image),
                             lambda r: ('', '', 0.0) if r is None else (r.material, r.bin_id, r.confidence))

    def update(self, item_id, status, material='', bin_id='', skip_reason=''):
        req = UpdateItem.Request(id=item_id, status=status, material=material, bin_id=bin_id,
                                 skip_reason=skip_reason)
        h = ServiceHandle(self.update_client, req, lambda r: r is not None and r.success)
        end = time.monotonic() + 5.0
        while not h.done() and time.monotonic() < end:
            time.sleep(0.01)
        ok = bool(h.result())
        if ok:
            with self.lock:                       # apply locally until /world_model shows it
                it = self.items.get(item_id)
                if it is not None:
                    it.status, it.bin_id = status, bin_id
                    it.override_until = self.now() + 3.0
                    for c in self.compartments.values():
                        if item_id in c:
                            c.remove(item_id)
                    if status == 1 and bin_id in self.compartments:
                        self.compartments[bin_id].append(item_id)
        self.log({'event': 'update', 'item': item_id, 'status': status, 'ok': ok, 'material': material,
                  'bin': bin_id, 'reason': skip_reason})
        return ok

    def cmd_vel(self, vx, vy, wz):
        cmd = Twist()
        cmd.linear.x, cmd.linear.y, cmd.angular.z = float(vx), float(vy), float(wz)
        self.cmd_pub.publish(cmd)

    def log(self, record):
        record = dict(record)
        record.setdefault('t', round(self.now(), 2))
        self.log_file.write(json.dumps(record) + '\n')
        self.log_file.flush()

    # --- loop -----------------------------------------------------------------------------
    def servers_ready(self):
        return (self.nav.server_is_ready() and self.pick_client.server_is_ready()
                and self.place_client.server_is_ready() and self.classify_client.service_is_ready()
                and self.update_client.service_is_ready())

    def tick(self):
        if self.robot is None or self.task.state == 'DONE':
            return
        if not self.started:                      # a missing server would otherwise fail every goal
            if not self.servers_ready():
                return
            self.started = True
            self.log({'event': 'servers_ready'})
        with self.lock:
            snap = Snapshot(self.now(), self.robot, dict(self.items),
                            {k: list(v) for k, v in self.compartments.items()}, self.load, self.safety == 'ok')
        self.task.tick(snap)
        state = f'{self.task.state} {self.task.item}'.strip()
        if state != self.last_state:
            self.state_pub.publish(String(data=state))
            self.last_state = state


def main():
    rclpy.init()
    node = TaskNode()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.log_file.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
