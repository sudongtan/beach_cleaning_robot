#!/usr/bin/env python3
"""A0 locomotion evaluation: scripted drive + ground-truth check.

Drives the robot through a fixed sequence (timed in sim time), reads the true pose
from Gazebo (/world/<world>/dynamic_pose/info) and the stack's /odom, and reports per
phase: displacement, achieved speed, yaw rate, odom error, max tilt, min height, falls.

  python3 drive_test.py --stack legubiao --world empty   [--out result.json]
  python3 drive_test.py --stack champ    --world default
  python3 drive_test.py --stack cmd_vel  --world empty --model beach_robot --odom-topic /leg_odom

cmd_vel: our interface (bringup sim.launch.py): publishes /cmd_vel only and waits until the
robot has stood up by itself (base above 0.25 m for 3 s).

legubiao takes control_input_msgs/Inputs (dev FSM: 2 = down then stand, 4 = trot; ly/rx in [-1, 1]
scaled to +-0.4 m/s and +-0.5 rad/s). champ takes /cmd_vel directly.
"""
import argparse
import json
import math
import threading

import rclpy
from geometry_msgs.msg import Twist
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node as GzNode
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter

# (name, duration s, vx m/s, wz rad/s); legubiao gets stand/trot mode commands first
PHASES = [
    ('settle', 5.0, 0.0, 0.0),
    ('forward', 10.0, 0.3, 0.0),
    ('stop1', 3.0, 0.0, 0.0),
    ('turn_left', 6.0, 0.0, 0.5),
    ('stop2', 3.0, 0.0, 0.0),
    ('hold_60s', 60.0, 0.0, 0.0),
]
FALL_TILT = 0.6   # rad roll or pitch
FALL_Z = 0.15     # m base height


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def tilt_of(q):
    roll = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
    return max(abs(roll), abs(pitch))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class DriveTest(Node):
    def __init__(self, args):
        super().__init__('a0_drive_test', parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.args = args
        self.lock = threading.Lock()
        self.gt = None    # (x, y, z, yaw, tilt)
        self.odom = None  # (x, y, yaw)
        self.create_subscription(Odometry, args.odom_topic, self.on_odom, 10)
        if args.stack == 'legubiao':
            from control_input_msgs.msg import Inputs
            self.Inputs = Inputs
            self.pub = self.create_publisher(Inputs, '/control_input', 10)
        else:
            self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.gz = GzNode()
        self.gz.subscribe(Pose_V, f'/world/{args.world}/dynamic_pose/info', self.on_gz_pose)

    def on_gz_pose(self, msg):
        for p in msg.pose:
            if p.name == self.args.model:
                o = p.orientation
                with self.lock:
                    self.gt = (p.position.x, p.position.y, p.position.z, yaw_of(o), tilt_of(o))

    def on_odom(self, msg):
        p = msg.pose.pose
        with self.lock:
            self.odom = (p.position.x, p.position.y, yaw_of(p.orientation))

    def send(self, vx, wz, command=0):
        if self.args.stack == 'legubiao':
            m = self.Inputs()
            m.command = command
            m.ly = float(vx / 0.4)
            m.rx = float(-wz / 0.5)
            self.pub.publish(m)
        else:
            t = Twist()
            t.linear.x = float(vx)
            t.angular.z = float(wz)
            self.pub.publish(t)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def run_for(self, duration, vx, wz, command=0, track=None):
        t0 = self.now()
        while rclpy.ok() and self.now() - t0 < duration:
            # mode commands are edge-triggered: send once, then back to 0
            self.send(vx, wz, command if self.now() - t0 < 0.5 else 0)
            rclpy.spin_once(self, timeout_sec=0.05)
            if track is not None:
                with self.lock:
                    gt = self.gt
                if gt:
                    track['max_tilt'] = max(track['max_tilt'], gt[4])
                    track['min_z'] = min(track['min_z'], gt[2])
                    # accumulate yaw so turns beyond 180 deg are not folded back
                    track['yaw_sum'] += wrap(gt[3] - track['yaw_prev'])
                    track['yaw_prev'] = gt[3]

    def snapshot(self):
        with self.lock:
            return self.gt, self.odom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stack', choices=['legubiao', 'champ', 'cmd_vel'], required=True)
    ap.add_argument('--odom-topic', default='/odom')
    ap.add_argument('--world', required=True)
    ap.add_argument('--model', default='go2')
    ap.add_argument('--out')
    args = ap.parse_args()

    rclpy.init()
    node = DriveTest(args)
    node.get_logger().info('waiting for sim clock and ground-truth pose ...')
    while rclpy.ok() and (node.now() == 0.0 or node.snapshot()[0] is None):
        rclpy.spin_once(node, timeout_sec=0.1)

    if args.stack == 'legubiao':
        # dev-branch FSM: passive -2-> fixed down -2-> fixed stand -4-> trotting
        node.get_logger().info('fixed down (2), fixed stand (2), trot (4)')
        node.run_for(3.0, 0.0, 0.0, command=2)
        node.run_for(4.0, 0.0, 0.0, command=2)
        node.run_for(3.0, 0.0, 0.0, command=4)

    if args.stack == 'cmd_vel':
        node.get_logger().info('waiting for the robot to stand up by itself ...')
        t_up = None
        while rclpy.ok():
            node.run_for(0.2, 0.0, 0.0)
            up = node.snapshot()[0][2] > 0.25
            t_up = (t_up or node.now()) if up else None
            if t_up and node.now() - t_up > 3.0:
                break
        node.get_logger().info(f'standing, base z = {node.snapshot()[0][2]:.3f} m')

    gt0, odom0 = node.snapshot()
    results, fell = [], False
    for name, dur, vx, wz in PHASES:
        node.get_logger().info(f'{name}: {dur:.0f} s, vx={vx}, wz={wz}')
        g_a, o_a = node.snapshot()
        track = {'max_tilt': g_a[4], 'min_z': g_a[2], 'yaw_sum': 0.0, 'yaw_prev': g_a[3]}
        node.run_for(dur, vx, wz, track=track)
        g_b, o_b = node.snapshot()
        dx, dy = g_b[0] - g_a[0], g_b[1] - g_a[1]
        fwd = dx * math.cos(g_a[3]) + dy * math.sin(g_a[3])
        r = {
            'phase': name, 'duration_s': dur, 'cmd_vx': vx, 'cmd_wz': wz,
            'gt_forward_m': round(fwd, 3), 'gt_dist_m': round(math.hypot(dx, dy), 3),
            'gt_vx': round(fwd / dur, 3), 'gt_wz': round(track['yaw_sum'] / dur, 3),
            'max_tilt_deg': round(math.degrees(track['max_tilt']), 1), 'min_z_m': round(track['min_z'], 3),
        }
        if o_a and o_b:
            r['odom_dist_m'] = round(math.hypot(o_b[0] - o_a[0], o_b[1] - o_a[1]), 3)
            r['odom_dyaw_deg'] = round(math.degrees(wrap(o_b[2] - o_a[2])), 1)
        r['gt_dyaw_deg'] = round(math.degrees(track['yaw_sum']), 1)
        r['fell'] = track['max_tilt'] > FALL_TILT or track['min_z'] < FALL_Z
        results.append(r)
        print(json.dumps(r), flush=True)
        if r['fell']:
            fell = True
            node.get_logger().error(f'FELL during {name}')
            break
    node.send(0.0, 0.0)

    gt1, odom1 = node.snapshot()
    summary = {'stack': args.stack, 'fell': fell, 'phases': results}
    if odom0 and odom1:
        gt_path = math.hypot(gt1[0] - gt0[0], gt1[1] - gt0[1])
        od_path = math.hypot(odom1[0] - odom0[0], odom1[1] - odom0[1])
        summary['net_displacement_gt_m'] = round(gt_path, 3)
        summary['net_displacement_odom_m'] = round(od_path, 3)
    print('SUMMARY', json.dumps({k: v for k, v in summary.items() if k != 'phases'}), flush=True)
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(summary, f, indent=2)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
