"""Grasping node: /grasp/pick and /grasp/place (notes/architecture.md), grasp_mode:=attach.

The robot stands still while the arm moves (the task guarantees that). Every arm step
waits while /safety/state != ok and a running move is cancelled (the controller holds
position) as soon as the state leaves ok (arm rule, decisions 2026-10-30). Plans come
from grasping.arm (IK), step order from grasping.sequence.

The stop on not ok is config `safety_stop`: `hold_reference` (default) preempts the move with a
goal to the controller's reference position at that instant (/arm_controller/controller_state),
`cancel` cancels the goal. A cancel makes the controller hold the *measured* position
(joint_trajectory_controller set_hold_position), so a PD joint lagging its reference under load
gives way by that lag (B4 question 9, b4_robot_lab s5 arm violation).

Attach mode needs the simulator's object id: the world model's scene_id for the item,
else the /sim/ground_truth/objects entry nearest the jaw tips (attach mode is itself a
simulation shortcut), placed relative to the robot with the true robot pose
(perception.true_pose), never the estimated TF map -> base_link. Physics mode is not implemented yet (A2 runs in attach mode).
With config `collision` set (B5), every planned move is checked against the robot body, the basket,
the items in it, the bin and the held item (grasping.collision, ground truth of the items and the
bin pose of the goal); a failing plan is replanned (next slot, IK restart, lower release) or refused,
and every rejection is logged. null = off: the plans of 9ac4816.
Parameters: repo, grasp_mode. Settings: config/grasping.yaml, config/robot_limits.yaml.
"""

import math
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from beach_interfaces.action import PickObject, PlaceObject
from beach_interfaces.msg import WorldModel
from beach_interfaces.srv import SimAttach
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTrajectoryControllerState
from geometry_msgs.msg import TransformStamped
from rclpy.action import ActionClient, ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from std_msgs.msg import String
from tf2_geometry_msgs import do_transform_pose
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint
from vision_msgs.msg import Detection3DArray

from perception.true_pose import TruePose
from grasping.arm import (ARM, TIPS, Arm, bin_release_target, in_basket, near, plan_pick, plan_store,
                          plan_unload, release_clearances)
from grasping.collision import Collision, State, bin_boxes, describe, reason
from grasping.sequence import (Runner, Step, bin_steps, ground_steps, pick_steps, recovery_steps,
                               store_steps)

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
FINGERS = ['arm_left_finger', 'arm_right_finger']


def duration(secs):
    return Duration(sec=int(secs), nanosec=int((secs % 1) * 1e9))


def quat_matrix(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class RosIo:
    """The io interface of grasping.sequence.Runner on top of ROS (called from action threads).

    Timeouts run on sim time (the node clock): at RTF 0.2-0.3 a 2.5 s arm move takes ~10 s of
    wall time, and a wall-clock limit expired just before the controller succeeded. A wall-time
    cap (WALL_CAP x the sim limit) still ends a wait if the simulation stops.
    """

    WALL_CAP = 20.0

    def __init__(self, node):
        self.n = node

    def now(self):
        return self.n.get_clock().now().nanoseconds * 1e-9

    def sleep(self, s):
        time.sleep(s)

    def safety_ok(self):
        return self.n.safety_state == 'ok'

    def _wait(self, fut, timeout):
        end = time.monotonic() + timeout
        while not fut.done() and time.monotonic() < end:
            time.sleep(0.01)
        return fut.done()

    def _trajectory(self, client, joints, positions, secs, watch_safety):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = joints
        goal.trajectory.points = [JointTrajectoryPoint(positions=[float(p) for p in positions],
                                                       time_from_start=duration(secs))]
        if not client.wait_for_server(timeout_sec=5.0):
            return 'failed'
        sent = client.send_goal_async(goal)
        if not self._wait(sent, 5.0) or not sent.result().accepted:
            return 'failed'
        handle = sent.result()
        result = handle.get_result_async()
        limit = secs + 10.0
        end, wall_end = self.now() + limit, time.monotonic() + self.WALL_CAP * limit
        while not result.done():
            if watch_safety and not self.safety_ok():
                self.stop(client, joints, handle)
                return 'interrupted'
            if self.now() > end or time.monotonic() > wall_end:
                self.n.get_logger().warn(f'{client._action_name}: no result after {limit:.1f} s sim time')
                return 'failed'
            time.sleep(0.02)
        return 'ok' if result.result().result.error_code == 0 else 'failed'

    def stop(self, client, joints, handle):
        """Stop a running move: hold the controller reference (if fresh), else cancel the goal."""
        ref, stamp = self.n.arm_reference
        if self.n.cfg['safety_stop'] == 'hold_reference' and ref and time.monotonic() - stamp < 1.0 \
                and all(j in ref for j in joints):
            goal = FollowJointTrajectory.Goal()
            goal.trajectory.joint_names = joints
            goal.trajectory.points = [JointTrajectoryPoint(positions=[ref[j] for j in joints],
                                                           velocities=[0.0] * len(joints),
                                                           time_from_start=duration(self.n.cfg['stop_hold_s']))]
            sent = client.send_goal_async(goal)           # preempts the running goal
            if self._wait(sent, 2.0) and sent.result().accepted:
                self.n.get_logger().info('safety stop: holding the arm at the controller reference')
                return
            self.n.get_logger().warn('hold goal not accepted; cancelling the move')
        self._wait(handle.cancel_goal_async(), 2.0)

    def move_arm(self, q, secs):
        # the goal has positions only, so the controller moves each joint at |dq| / secs; stretch the move
        # so that no joint is faster than max_joint_speed_radps (from the controller reference, if fresh)
        vmax = self.n.cfg['max_joint_speed_radps']
        ref, stamp = self.n.arm_reference
        if vmax > 0 and time.monotonic() - stamp < 1.0 and all(j in ref for j in ARM):
            secs = max(secs, max(abs(float(p) - ref[j]) for j, p in zip(ARM, q)) / vmax)
        return self._trajectory(self.n.arm_client, ARM, q, secs, watch_safety=True)

    def fingers(self, value, secs):
        return self._trajectory(self.n.gripper_client, FINGERS, [value, -value], secs, watch_safety=False) == 'ok'

    def _sim(self, client, object_id):
        if not object_id or not client.wait_for_service(timeout_sec=5.0):
            return False
        fut = client.call_async(SimAttach.Request(object_id=object_id))
        if not self._wait(fut, 10.0):
            return False
        self.n.get_logger().info(f'{client.srv_name} {object_id}: {fut.result().message}')
        return fut.result().success

    def attach(self, object_id):
        return self._sim(self.n.attach_client, object_id)

    def detach(self, object_id):
        return self._sim(self.n.detach_client, object_id)


class GraspingNode(Node):

    def __init__(self, **kwargs):
        super().__init__('grasping', **kwargs)
        repo = self.declare_parameter('repo', '').value
        if not repo:
            raise RuntimeError('parameter repo (repository root) is required')
        repo = Path(repo)
        self.mode = self.declare_parameter('grasp_mode', 'attach').value
        self.base_frame = self.declare_parameter('base_frame', 'base_link').value
        self.cfg = yaml.safe_load(open(repo / 'config' / 'grasping.yaml'))['grasping']
        self.limits = yaml.safe_load(open(repo / 'config' / 'robot_limits.yaml'))['robot_limits']
        objects = yaml.safe_load(open(repo / 'sim' / 'common' / 'objects.yaml'))
        self.bin_models = objects['bins']
        bin_boxes = objects['bins']['bin_recyclable']['collision']
        self.bin_half = (max(abs(b['pose'][0]) + b['size'][0] / 2 for b in bin_boxes),
                         max(abs(b['pose'][1]) + b['size'][1] / 2 for b in bin_boxes))
        self.arm = Arm(self.limits['variant'], self.cfg['ik_restarts'], self.cfg['ik_tolerance_m'])
        self.runner = Runner(RosIo(self), self.cfg['move_time_s'], self.cfg['finger_time_s'],
                             self.cfg['safety_hold_timeout_s'])
        col = self.cfg.get('collision')
        self.collision = Collision.from_config(self.arm, col)
        if self.collision:
            self.check_named_moves()
        self.safety_state = 'stop'                 # not ok until the supervisor says so
        self.arm_reference = ({}, 0.0)             # arm controller reference {joint: rad}, wall stamp
        self.scene_ids = {}
        self.gt = None
        self.picks = {}                            # item id -> pick plan (for putting it back)
        self.busy = threading.Lock()

        cb = ReentrantCallbackGroup()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # ground-truth objects relative to the robot: the true robot pose, not the estimated TF (B4)
        self.truth = TruePose(self, self.tf_buffer)
        self.arm_client = ActionClient(self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory',
                                       callback_group=cb)
        self.gripper_client = ActionClient(self, FollowJointTrajectory,
                                           '/gripper_controller/follow_joint_trajectory', callback_group=cb)
        self.attach_client = self.create_client(SimAttach, '/sim/attach', callback_group=cb)
        self.detach_client = self.create_client(SimAttach, '/sim/detach', callback_group=cb)
        self.create_subscription(String, '/safety/state', self.on_safety, LATCHED, callback_group=cb)
        self.create_subscription(JointTrajectoryControllerState, '/arm_controller/controller_state',
                                 self.on_arm_state, 10, callback_group=cb)
        self.create_subscription(WorldModel, '/world_model', self.on_world_model, 10, callback_group=cb)
        self.create_subscription(Detection3DArray, '/sim/ground_truth/objects', self.on_gt, 10, callback_group=cb)
        ActionServer(self, PickObject, '/grasp/pick', self.execute_pick, callback_group=cb,
                     cancel_callback=lambda _: CancelResponse.ACCEPT)
        ActionServer(self, PlaceObject, '/grasp/place', self.execute_place, callback_group=cb,
                     cancel_callback=lambda _: CancelResponse.ACCEPT)

    def on_safety(self, msg):
        self.safety_state = msg.data

    def on_arm_state(self, msg):
        self.arm_reference = (dict(zip(msg.joint_names, msg.reference.positions)), time.monotonic())

    def on_world_model(self, msg):
        self.scene_ids = {it.id: it.scene_id for it in msg.items}

    def on_gt(self, msg):
        self.gt = msg

    def to_base(self, pose_stamped):
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, pose_stamped.header.frame_id, Time())
        except TransformException as e:
            self.get_logger().warn(f'no transform to {self.base_frame}: {e}')
            return None
        p = do_transform_pose(pose_stamped.pose, tf)
        return p

    def sim_object(self, item_id, near_base=None):
        """Simulator object id for attach mode: the item's scene id, else the ground-truth object
        nearest the jaw tips. For a pick (near_base given) the object must be within 0.15 m of the
        tips: a phantom item carried the scene id of an item already in the basket, and attaching
        it there dragged it out through the basket wall (A2b)."""
        scene_id = self.scene_ids.get(item_id, '')
        if near_base is None:
            return scene_id
        if self.gt is None:
            return scene_id
        try:
            tf = self.truth.lookup(self.base_frame, source=self.gt.header.frame_id)
        except TransformException:
            return scene_id
        best, best_d = '', 0.15
        for d in self.gt.detections:
            if scene_id and d.id != scene_id:
                continue
            p = do_transform_pose(d.bbox.center, tf).position
            dist = math.dist((p.x, p.y, p.z), near_base)
            if dist < best_d:
                best, best_d = d.id, dist
        return best

    def run(self, goal_handle, steps, result):
        def feedback(phase):
            msg = (PickObject.Feedback() if isinstance(goal_handle.request, PickObject.Goal)
                   else PlaceObject.Feedback())
            msg.phase = phase
            goal_handle.publish_feedback(msg)
        ok, message = self.runner.run(steps, feedback, lambda: goal_handle.is_cancel_requested)
        if not ok and self.runner.held:
            # never walk on with an item attached (A2b a2b_bin2: the next try found it "already
            # attached", and the recovery dragged it along); it drops where it is
            if self.runner.io.detach(self.runner.held):
                self.runner.held = None
            message += '; item released'
        if not ok and not goal_handle.is_cancel_requested:
            # never leave the arm where a failure stopped it: the robot walks next (best effort)
            rec_ok, rec_msg = self.runner.run(recovery_steps(self.arm.poses), feedback)
            message += '' if rec_ok else f'; recovery to rest failed: {rec_msg}'
        result.success, result.message = ok, message
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
        elif ok:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        self.get_logger().info(f'{type(goal_handle.request).__name__}: {message}')
        return result

    def refuse(self, goal_handle, result, message):
        self.get_logger().warn(message)
        result.success, result.message = False, message
        goal_handle.abort()
        return result

    def start_q(self):
        """Arm pose a sequence starts from: the controller reference if fresh, else `rest`."""
        ref, stamp = self.arm_reference
        if time.monotonic() - stamp < 1.0 and all(j in ref for j in ARM):
            return np.array([ref[j] for j in ARM])
        return np.asarray(self.arm.poses['rest']['arm'], dtype=float)

    def check(self, plan_name, steps, start, state, until=None):
        """Collision check of a step list (until: stop before the first step with this phase).
        Returns (failure text or None, end pose, end state); rejections are logged."""
        if until is not None:
            steps = steps[:next(i for i, st in enumerate(steps) if st.phase == until)]
        t = time.monotonic()
        failure, q, end = self.collision.check_steps(steps, start, state)
        if failure is not None:
            self.get_logger().info(f'collision reject {plan_name}: {describe(failure)} '
                                   f'({(time.monotonic() - t) * 1000:.0f} ms)')
            return reason(failure), q, end
        return None, q, end

    def check_named_moves(self):
        """Moves between the named poses with an empty basket and nothing held (B5.2); a failure is a bug."""
        poses, st = self.arm.poses, State()
        moves = [('rest', 'up'), ('up', 'rest'), ('up', 'show'), ('show', 'up'),
                 ('up', 'over_recyclable'), ('over_recyclable', 'up'), ('up', 'over_residual'), ('over_residual', 'up')]
        q = {**{k: v['arm'] for k, v in poses.items()}, 'show': self.cfg['show_pose']}
        for a, b in moves:
            hit = self.collision.check_move(q[a], q[b], st)
            d = self.collision.min_move_distance(q[a], q[b], st)
            text = f'named move {a} -> {b}: min {d[0] * 1000:.1f} mm ({d[1][0]}/{d[1][1]})'
            if hit is None:
                self.get_logger().info(text)
            else:
                self.get_logger().error(text + ' below collision.clearance_m')

    def basket_boxes(self):
        """Items lying in the basket as obstacle boxes {sim id: {'centre', 'R', 'size'}} in base_link
        (ground truth); {} if unknown."""
        if self.gt is None:
            return {}
        try:
            tf = self.truth.lookup(self.base_frame, source=self.gt.header.frame_id)
        except TransformException:
            return {}
        boxes = {}
        for d in self.gt.detections:
            p = do_transform_pose(d.bbox.center, tf)
            c = np.array([p.position.x, p.position.y, p.position.z])
            if in_basket(self.arm, c):
                o = p.orientation
                boxes[d.id] = {'centre': c, 'R': quat_matrix(o.x, o.y, o.z, o.w),
                               'size': (d.bbox.size.x, d.bbox.size.y, d.bbox.size.z)}
        return boxes

    def execute_pick(self, goal_handle):
        g, result = goal_handle.request, PickObject.Result()
        if self.mode != 'attach':
            return self.refuse(goal_handle, result, f'grasp_mode {self.mode} not implemented (A2 uses attach)')
        if not self.busy.acquire(blocking=False):
            return self.refuse(goal_handle, result, 'busy')
        try:
            pose = self.to_base(g.pose)
            if pose is None:
                return self.refuse(goal_handle, result, 'no_transform')
            c = (pose.position.x, pose.position.y, pose.position.z)
            height = g.size.z if g.size.z > 0 else self.cfg['assumed_item_height_m']
            check = None
            if self.collision:
                o = pose.orientation
                items = self.basket_boxes()
                if g.size.z > 0:                          # the item to be picked: no obstacle until attached
                    items['target'] = {'centre': np.array(c), 'R': quat_matrix(o.x, o.y, o.z, o.w),
                                       'size': (g.size.x, g.size.y, g.size.z)}
                state, start = State(items, excluded={'target'}), self.start_q()

                def check(plan):
                    steps = pick_steps(plan, self.arm.poses, self.cfg['show_pose'], self.arm.finger_open,
                                       self.arm.finger_closed, 'target')
                    return self.check('pick', steps, start, state)[0]
            ok, why, plan = plan_pick(self.arm, c, height, self.cfg, check)
            if not ok:
                return self.refuse(goal_handle, result, f'{why} (item at x {c[0]:.3f} y {c[1]:.3f} z {c[2]:.3f} '
                                                        f'in {self.base_frame})')
            hover_tips = self.arm.fk(plan['hover'], TIPS)[0]
            obj = self.sim_object(g.id, tuple(hover_tips))
            if not obj:
                return self.refuse(goal_handle, result, 'no_sim_object')
            self.picks[g.id] = plan
            steps = pick_steps(plan, self.arm.poses, self.cfg['show_pose'], self.arm.finger_open,
                               self.arm.finger_closed, obj)
            return self.run(goal_handle, steps, result)
        finally:
            self.busy.release()

    def execute_place(self, goal_handle):
        g, result = goal_handle.request, PlaceObject.Result()
        if self.mode != 'attach':
            return self.refuse(goal_handle, result, f'grasp_mode {self.mode} not implemented (A2 uses attach)')
        if not self.busy.acquire(blocking=False):
            return self.refuse(goal_handle, result, 'busy')
        try:
            obj = self.sim_object(g.item_id)
            if not obj:
                return self.refuse(goal_handle, result, 'no_sim_object')
            if g.target == PlaceObject.Goal.COMPARTMENT:
                plan, why = None, 'no ground truth of the basket'
                scene = self.basket_scene(obj)
                if scene is not None:
                    check = None
                    if self.collision:
                        state = State(self.basket_boxes(), held=dict(scene[0], id=obj))
                        state.items.pop(obj, None)
                        start = self.start_q()

                        def check(plan):
                            steps = store_steps(self.arm.poses, g.bin_id, self.arm.finger_open, obj, plan)
                            return self.check('store slot', steps, start, state)[0]
                    ok, why, plan, info = plan_store(self.arm, g.bin_id, scene[0], scene[1], self.cfg, check)
                    self.get_logger().info(
                        f"store {obj}: {why or 'ok'} slot {np.round(info.get('slot', ()), 3).tolist()} pile "
                        f"{info.get('pile', float('nan')):.3f} drop {info.get('drop', float('nan')):.3f}; in basket "
                        + ', '.join(f"({o['centre'][0]:.3f} {o['centre'][1]:.3f} top {o['top']:.3f})" for o in scene[1])
                        + ''.join(f"; rejected slot {np.round(xy, 3).tolist()}: {w}" for xy, w in info.get('rejected', [])))
                    if why == 'compartment_full':                 # still held: the task puts it back
                        return self.refuse(goal_handle, result, f"compartment_full (pile {info['pile']:.3f})")
                steps = store_steps(self.arm.poses, g.bin_id, self.arm.finger_open, obj, plan)
                if not plan:
                    if self.collision:                            # the held item is unknown: arm and basket only
                        state = State(self.basket_boxes())
                        state.items.pop(obj, None)
                        failure = self.check('store fixed', steps, self.start_q(), state)[0]
                        if failure:
                            return self.refuse(goal_handle, result, f'compartment_full ({why}; {failure})')
                    self.get_logger().warn(f'store {obj}: {why}; release from the fixed pose')
            elif g.target == PlaceObject.Goal.GROUND:
                if g.item_id not in self.picks:
                    return self.refuse(goal_handle, result, 'no_pick_plan')
                steps = ground_steps(self.picks.pop(g.item_id), self.arm.poses, self.arm.finger_open, obj)
                if self.collision:
                    scene = self.basket_scene(obj)
                    state = State(self.basket_boxes(), held=None if scene is None else dict(scene[0], id=obj))
                    state.items.pop(obj, None)
                    why = self.check('ground', steps, self.start_q(), state)[0]
                    if why:                                       # let go where it is (in front of the robot)
                        steps = [Step('detach', obj, 'release')] + recovery_steps(self.arm.poses)
                        self.get_logger().warn(f'put back {obj}: {why}; released at the current pose')
            elif g.target == PlaceObject.Goal.BIN:
                item = self.gt_item(obj)
                if item is None:
                    return self.refuse(goal_handle, result, 'item_not_found')
                centre, height = item
                if not in_basket(self.arm, centre) and self.in_bin(g.pose, centre):
                    # carried along with the item below it (a2b_dev1: a bottle lying on the jar);
                    # compartments hold one bin's items, so it is where it belongs
                    self.get_logger().info(f'place {obj}: already_in_bin')
                    result.success, result.message = True, 'already_in_bin'
                    goal_handle.succeed()
                    return result
                if not in_basket(self.arm, centre):
                    return self.refuse(goal_handle, result, f'not_in_basket (item at x {centre[0]:.3f} '
                                                            f'y {centre[1]:.3f} z {centre[2]:.3f})')
                check, release_check = None, None
                if self.collision:
                    state = State(self.basket_boxes(), excluded={obj}, bins=self.bin_obstacles(g.pose, g.bin_id))
                    start, after = self.start_q(), {}

                    def check(plan):
                        steps = bin_steps(self.arm.poses, g.bin_id, plan, plan['above'], self.arm.finger_open,
                                          self.arm.finger_closed, obj)
                        why, q_end, st_end = self.check('unload', steps, start, state, until='over_bin')
                        after.update(q=q_end, state=st_end)
                        return why

                    def release_check(q):
                        steps = bin_steps(self.arm.poses, g.bin_id, plan, q, self.arm.finger_open,
                                          self.arm.finger_closed, obj)
                        k = next(i for i, st in enumerate(steps) if st.phase == 'over_bin')
                        return self.check('bin release', steps[k:], after['q'], after['state'])[0]
                ok, why, plan = plan_unload(self.arm, g.bin_id, centre, height, self.cfg, check)
                if not ok:
                    return self.refuse(goal_handle, result, f'unload {why} (item at x {centre[0]:.3f} '
                                                            f'y {centre[1]:.3f} z {centre[2]:.3f})')
                q = self.bin_release(g.pose, height + self.cfg['hover_above_item_m'], release_check)
                if q is None:
                    return self.refuse(goal_handle, result, 'bin_unreachable')
                steps = bin_steps(self.arm.poses, g.bin_id, plan, q, self.arm.finger_open,
                                  self.arm.finger_closed, obj)
            else:
                return self.refuse(goal_handle, result, f'unknown target {g.target}')
            return self.run(goal_handle, steps, result)
        finally:
            self.busy.release()

    def basket_scene(self, object_id):
        """For plan_store, from ground truth: the held item in the jaw-tips frame ({'p', 'R', 'size'})
        and the items lying in the basket ({'centre', 'half', 'top'} in base_link); None if unknown."""
        if self.gt is None:
            return None
        try:
            tips = self.tf_buffer.lookup_transform(self.base_frame, TIPS, Time())
            tf = self.truth.lookup(self.base_frame, source=self.gt.header.frame_id)
        except TransformException:
            return None
        r = tips.transform.rotation
        R_bt = quat_matrix(r.x, r.y, r.z, r.w)
        t_bt = np.array([tips.transform.translation.x, tips.transform.translation.y, tips.transform.translation.z])
        held, others = None, []
        for d in self.gt.detections:
            p = do_transform_pose(d.bbox.center, tf)
            c = np.array([p.position.x, p.position.y, p.position.z])
            o = p.orientation
            R = quat_matrix(o.x, o.y, o.z, o.w)
            size = (d.bbox.size.x, d.bbox.size.y, d.bbox.size.z)
            if d.id == object_id:
                held = {'p': R_bt.T @ (c - t_bt), 'R': R_bt.T @ R, 'size': size}
            elif in_basket(self.arm, c):
                ext = np.abs(R) @ np.asarray(size)
                others.append({'centre': c, 'half': ext[:2] / 2, 'top': c[2] + ext[2] / 2})
        return None if held is None else (held, others)

    def gt_item(self, object_id):
        """(bbox centre, vertical extent) of a simulator object in base_link (attach mode: the
        item lying in a compartment, from ground truth), or None."""
        d = next((d for d in (self.gt.detections if self.gt else []) if d.id == object_id), None)
        if d is None:
            return None
        try:
            tf = self.truth.lookup(self.base_frame, source=self.gt.header.frame_id)
        except TransformException:
            return None
        p = do_transform_pose(d.bbox.center, tf)
        q = p.orientation
        # third row of the item's rotation in base_link: how much of each bbox axis is vertical
        r2 = (2 * (q.x * q.z - q.w * q.y), 2 * (q.y * q.z + q.w * q.x), 1 - 2 * (q.x * q.x + q.y * q.y))
        size = (d.bbox.size.x, d.bbox.size.y, d.bbox.size.z)
        height = sum(abs(r) * s for r, s in zip(r2, size))
        return (p.position.x, p.position.y, p.position.z), height

    def in_bin(self, pose_stamped, p):
        """Is point p (base_link) inside the bin whose opening centre (at the rim) is pose_stamped?"""
        pose = self.to_base(pose_stamped)
        if pose is None:
            return False
        q = pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        dx, dy = p[0] - pose.position.x, p[1] - pose.position.y
        u, v = math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy
        return abs(u) <= self.bin_half[0] and abs(v) <= self.bin_half[1] and p[2] < pose.position.z

    def bin_obstacles(self, pose_stamped, bin_id):
        """The goal bin's collision boxes in base_link (its pose: opening centre at the rim); [] if unknown."""
        pose = self.to_base(pose_stamped)
        model = self.bin_models.get(f'bin_{bin_id}')
        if pose is None or model is None:
            return []
        q = pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return bin_boxes(model, (pose.position.x, pose.position.y, pose.position.z), yaw)

    def bin_release(self, pose_stamped, hang=0.0, check=None):
        """Joint target with the jaw tips over the bin opening (pose: opening centre at the rim),
        raised by `hang` (how far the held item reaches below the tips), on the front side of `up`;
        clearance_above_rim_m above the rim, or lower down to min_clearance_above_rim_m if out of reach
        (or if check(q) -> failure text rejects the higher ones; collision check, B5)."""
        pose = self.to_base(pose_stamped)
        if pose is None:
            return None
        q = pose.orientation
        rel_yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        depth = abs(math.cos(rel_yaw)) * self.bin_half[0] + abs(math.sin(rel_yaw)) * self.bin_half[1]
        c = (pose.position.x, pose.position.y, pose.position.z)
        up = self.arm.poses['up']['arm']
        bd = self.limits['bin_drop']
        for clearance in release_clearances(bd):
            target = bin_release_target(c, c[0] - depth, bd, hang, clearance)
            ok, arm_q, err, tilt = self.arm.ik(target, bd['max_tilt_deg'], seed_q=up,
                                               bounds=near(up, self.cfg['release_joint_spread_rad']))
            if ok and check is not None and check(np.asarray(arm_q)):
                continue
            if ok:
                if clearance < bd['clearance_above_rim_m']:
                    self.get_logger().info(f'release {clearance:.2f} m above the rim (hang {hang:.3f})')
                return np.asarray(arm_q)
        return None


def main():
    rclpy.init()
    node = GraspingNode()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
