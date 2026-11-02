"""Waits for a full run: prints task / safety state changes and the pose every 120 s; stops on DONE or FELL,
or when the task state and the robot position (0.5 m) haven't changed for `stall` s (wall).
  watch_until.py <limit_s> <stall_s>"""
import sys, time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String
from nav_msgs.msg import Odometry
L = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
rclpy.init(); n = Node('watch2'); st = {'pose': None, 'state': '', 'changed': time.monotonic()}
t0 = time.monotonic()
def p(*a): print(f'[{time.monotonic()-t0:6.0f}s]', *a, flush=True)
def on_task(m):
    st['state'] = m.data; st['changed'] = time.monotonic(); p('task:', m.data)
n.create_subscription(String, '/task/state', on_task, L)
n.create_subscription(String, '/safety/state', lambda m: p('safety:', m.data), L)
st['anchor'] = None
def on_pose(m):
    st['pose'] = m
    q = m.pose.pose.position
    if st['anchor'] is None or ((q.x - st['anchor'][0])**2 + (q.y - st['anchor'][1])**2) ** 0.5 > 0.5:
        st['anchor'] = (q.x, q.y); st['changed'] = max(st['changed'], time.monotonic())   # moving = progress
n.create_subscription(Odometry, '/sim/ground_truth/pose', on_pose, 10)
limit, stall = float(sys.argv[1]), float(sys.argv[2])
last = 0
while time.monotonic() - t0 < limit:
    rclpy.spin_once(n, timeout_sec=0.2)
    if st['state'].startswith(('DONE', 'FELL')):
        p(st['state']); break
    if time.monotonic() - st['changed'] > stall:
        p(f'STALL: no task state change and no movement for {stall:.0f} s'); break
    if time.monotonic() - last > 120 and st['pose'] is not None:
        last = time.monotonic(); q = st['pose'].pose.pose.position
        p(f"sim t {st['pose'].header.stamp.sec}s pose ({q.x:.2f}, {q.y:.2f}, z {q.z:.2f})")
