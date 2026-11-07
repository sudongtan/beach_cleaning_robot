#!/usr/bin/env python3
"""Arm motion after a cancelled move (B4 question 9: arm rule after a safety cancel).

Moves the arm from `up` to each target (one-point trajectory, --move-s like grasping's
move_time_s), cancels the goal at each fraction of the move (as grasping does when
/safety/state leaves ok), and records /joint_states for --after-s. Reports the arm rule
of bench/recording.py from the cancel time on: the largest change of any arm joint within
arm_window_s (bench/bench.yaml), and the deviation from the pose at the cancel. A hold
with no move first gives the stand-sway floor.

    LOCO=legubiao_robot_lab TOOL=cancel_test.py turn_walk_test.sh 3 [--move-s 2.5] [--fractions 0.3,0.5,0.8]

--mode hold_ref preempts the move with a goal to the controller's reference position instead
(/arm_controller/controller_state, reached after --hold-s).

--item <id> attaches a scene object first (the gripper loaded, as in a PUT_BACK).

Prints one line per trial and a RESULT line (worst change vs arm_change_rad).
"""
import argparse
import sys
from collections import deque
from pathlib import Path

import rclpy
import yaml
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTrajectoryControllerState
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arm_pose  # noqa: E402
from beach_interfaces.srv import SimAttach  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
ARM = ['arm_waist', 'arm_shoulder', 'arm_elbow', 'arm_forearm_roll', 'arm_wrist_angle', 'arm_wrist_rotate']
UP = [0.0, 0.0, -0.8, 0.0, 1.6, 0.0]
# IK-like pick/put-back poses in front of the robot (forearm_roll and wrist_rotate non-zero)
TARGETS = {'front_left': [0.5, 1.2, -0.15, 0.6, 0.6, 0.4],
           'front_right': [-0.5, 1.25, -0.2, -0.6, 0.5, -0.4]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--move-s', type=float, default=2.5)
    ap.add_argument('--fractions', default='0.3,0.5,0.8')
    ap.add_argument('--after-s', type=float, default=2.0)
    ap.add_argument('--mode', choices=('cancel', 'hold_ref'), default='cancel',
                    help='cancel: bare goal cancel (JTC holds the measured position); hold_ref: preempt with '
                         'a goal to the controller reference at the stop instant')
    ap.add_argument('--item', help='attach this scene object first (robot standing 0.49 m behind it, GOTO)')
    ap.add_argument('--hold-s', type=float, default=0.1, help='hold_ref: time_from_start of the hold point')
    a, _ = ap.parse_known_args()      # turn_walk_test.sh passes its own args too
    rule = yaml.safe_load(open(REPO / 'bench' / 'bench.yaml'))['recorder']
    limit, window = rule['arm_change_rad'], rule['arm_window_s']
    rclpy.init()
    n = Node('cancel_test', parameter_overrides=[Parameter('use_sim_time', value=True)])
    st = {'q': None, 'log': None, 'ref': None}

    def on_js(m):
        d = dict(zip(m.name, m.position))
        if all(j in d for j in ARM):
            st['q'] = [d[j] for j in ARM]
            if st['log'] is not None:
                st['log'].append((now(), st['q']))
    n.create_subscription(JointState, '/joint_states', on_js, 50)
    n.create_subscription(JointTrajectoryControllerState, '/arm_controller/controller_state',
                          lambda m: st.update(ref=dict(zip(m.joint_names, m.reference.positions))), 10)
    client = ActionClient(n, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')

    def now():
        return n.get_clock().now().nanoseconds * 1e-9

    def spin(s):
        t0 = now()
        while now() - t0 < s:
            rclpy.spin_once(n, timeout_sec=0.01)

    def send(q, secs):
        g = FollowJointTrajectory.Goal()
        g.trajectory.joint_names = ARM
        g.trajectory.points = [JointTrajectoryPoint(
            positions=[float(v) for v in q], time_from_start=Duration(sec=int(secs), nanosec=int(secs % 1 * 1e9)))]
        f = client.send_goal_async(g)
        rclpy.spin_until_future_complete(n, f)
        return f.result()

    def move(q, secs):
        h = send(q, secs)
        r = h.get_result_async()
        rclpy.spin_until_future_complete(n, r)
        spin(0.5)

    def measure(t0):
        """Arm rule from t0 on over the logged samples: (max change in a window, joint, max deviation)."""
        log = [(t, q) for t, q in st['log'] if t >= t0]
        ref, win, worst, dev = log[0][1], deque(), (0.0, ''), 0.0
        for t, q in log:
            win.append((t, q))
            while t - win[0][0] > window:
                win.popleft()
            for _, p in win:
                d = [abs(x - y) for x, y in zip(q, p)]
                if max(d) > worst[0]:
                    worst = (max(d), ARM[d.index(max(d))])
            dev = max(dev, max(abs(x - y) for x, y in zip(q, ref)))
        return worst[0], worst[1], dev

    client.wait_for_server(timeout_sec=30.0)
    while st['q'] is None:
        rclpy.spin_once(n, timeout_sec=0.1)
    if a.item:
        poses = arm_pose.load_poses('custom')
        closed, _ = arm_pose.finger_limits('custom')
        arm_pose.move_to(n, 'custom', 'ground_front', poses)
        att = n.create_client(SimAttach, '/sim/attach')
        att.wait_for_service(timeout_sec=10.0)
        fut = att.call_async(SimAttach.Request(object_id=a.item))
        rclpy.spin_until_future_complete(n, fut)
        print(f'attach {a.item}: {fut.result().message}', flush=True)
        arm_pose.set_fingers(n, closed)
    move(UP, 3.0)
    st['log'] = []
    t0 = now()
    spin(a.after_s)
    ch, j, dev = measure(t0)
    print(f'hold (no move): change {ch:.4f} rad ({j}), deviation {dev:.4f}', flush=True)
    worst = (ch, 'hold')
    for name, q in TARGETS.items():
        for f in (float(v) for v in a.fractions.split(',')):
            st['log'] = []
            h = send(q, a.move_s)
            spin(f * a.move_s)
            tc = now()
            if a.mode == 'cancel':
                rclpy.spin_until_future_complete(n, h.cancel_goal_async())
            else:
                send([st['ref'][j] for j in ARM], a.hold_s)
            spin(a.after_s)
            ch, j, dev = measure(tc)
            status = h.get_result_async()
            rclpy.spin_until_future_complete(n, status)
            code = status.result().status
            print(f'{name} {a.mode} at {f:.1f}: change {ch:.4f} rad ({j}), deviation {dev:.4f}, '
                  f'{"canceled" if code == GoalStatus.STATUS_CANCELED else "status " + str(code)}', flush=True)
            if ch > worst[0]:
                worst = (ch, f'{name}@{f:.1f}')
            move(UP, 3.0)
    print(f'RESULT worst change {worst[0]:.4f} rad at {worst[1]} (rule {limit} rad / {window} s): '
          f'{"PASS" if worst[0] <= limit else "FAIL"}', flush=True)
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
