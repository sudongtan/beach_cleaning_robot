#!/usr/bin/env python3
"""Benchmark metrics recorder: one per run, started by bench/runner.py after the stand-up.
Subscribes to the frozen topics (notes/architecture.md), feeds bench/recording.py and writes its
snapshot to <out> every `flush_s` (atomic rename), and once more on SIGINT / SIGTERM.

  recorder.py <out.json> <seed> --ros-args -p use_sim_time:=true -r /clock:=/sim/clock_throttled
"""
import json
import os
import sys
import threading
import time
from pathlib import Path

import rclpy
import yaml
from rclpy.clock import Clock
from geometry_msgs.msg import PolygonStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.experimental import EventsExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from vision_msgs.msg import Detection3DArray

from beach_interfaces.msg import WorldModel

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from recording import Recording  # noqa: E402

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def stamp(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


class Recorder(Node):
    def __init__(self, out, seed):
        super().__init__('bench_recorder')
        bench = yaml.safe_load(open(REPO / 'bench' / 'bench.yaml'))
        cfg = dict(bench['recorder'], stall_move_m=bench['bench']['limits']['stall_move_m'])
        safety = yaml.safe_load(open(REPO / 'config' / 'safety.yaml'))['safety_supervisor']['ros__parameters']
        scene = next(s for s in yaml.safe_load(open(REPO / 'bench' / 'scenes.yaml'))['scenes'] if s['seed'] == seed)
        self.rec = Recording(cfg, safety['stop_distance_m'], scene['geofence']['polygon'])
        self.out = Path(out)
        self.lock = threading.Lock()
        sub = self.create_subscription
        sub(Odometry, '/sim/ground_truth/pose', self.on_pose, 50)
        sub(Odometry, '/odom/global', self.on_odom_global, 10)
        sub(Twist, '/cmd_vel', lambda m: self.rec.on_cmd(m.linear.x, m.linear.y, m.angular.z), 10)
        sub(Detection3DArray, '/people', self.on_people, 10)
        sub(String, '/safety/state', lambda m: self.rec.on_safety(m.data), LATCHED)
        sub(String, '/task/state', lambda m: self.rec.on_task(m.data, time.time()), LATCHED)
        # /joint_states comes at the controller rate (~470 Hz wall): take it raw and decode every
        # joints_every-th message (the arm rule compares with the pose when the state left ok)
        self.joints_every, self.joints_n = bench['recorder']['joints_every'], 0
        sub(JointState, '/joint_states', self.on_joints, 50, raw=True)
        sub(PolygonStamped, '/geofence', lambda m: self.rec.on_geofence([[p.x, p.y] for p in m.polygon.points]),
            LATCHED)
        sub(WorldModel, '/world_model', self.on_world_model, 5)
        sub(Detection3DArray, '/sim/ground_truth/objects', self.on_objects, 5)
        self.wm_next = 0.0
        self.create_timer(bench['recorder']['flush_s'], self.flush, clock=Clock())   # wall timer

    def on_pose(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self.rec.on_pose(stamp(m.header), (p.x, p.y, p.z), (q.x, q.y, q.z, q.w), time.time())

    def on_odom_global(self, m):
        p = m.pose.pose.position
        self.rec.on_odom_global(stamp(m.header), p.x, p.y)

    def on_people(self, m):
        self.rec.on_people(stamp(m.header), [(d.bbox.center.position.x, d.bbox.center.position.y)
                                             for d in m.detections])

    def on_joints(self, raw):
        self.joints_n += 1
        if self.joints_n % self.joints_every:
            return
        m = deserialize_message(raw, JointState)
        self.rec.on_joints(stamp(m.header), dict(zip(m.name, m.position)))

    def on_world_model(self, m):
        now = time.monotonic()
        if now < self.wm_next:                         # the coverage grid is big; 1 Hz is plenty
            return
        self.wm_next = now + 1.0
        items = [{'id': i.id, 'scene_id': i.scene_id, 'object_class': i.object_class, 'material': i.material,
                  'bin_id': i.bin_id, 'status': int(i.status), 'skip_reason': i.skip_reason} for i in m.items]
        self.rec.on_world_model(items, list(m.coverage.data))

    def on_objects(self, m):
        self.rec.on_objects(stamp(m.header), {d.id: [round(d.bbox.center.position.x, 4),
                                                     round(d.bbox.center.position.y, 4),
                                                     round(d.bbox.center.position.z, 4)] for d in m.detections})

    def flush(self, final=False):
        with self.lock:
            snap = self.rec.snapshot(time.time(), final)
            tmp = self.out.with_suffix('.tmp')
            tmp.write_text(json.dumps(snap))
            os.replace(tmp, self.out)


def main():
    out, seed = sys.argv[1], int(sys.argv[2])
    rclpy.init(args=sys.argv)                      # SIGINT / SIGTERM shut the context down -> final flush
    node = Recorder(out, seed)
    # EventsExecutor: the default executor rebuilds its wait set per message (~70 % of a core at the
    # ~600 msg/s here, mostly /joint_states)
    executor = EventsExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.flush(final=True)
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
