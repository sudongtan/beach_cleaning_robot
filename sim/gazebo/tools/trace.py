"""Dense trace every 0.5 s wall: ground-truth pose, height, tilt / roll / pitch, arm joints, /cmd_vel_nav, /cmd_vel,
safety and task state, and every ground-truth object within 0.8 m of base_link (robot frame x/y,
map z: shows items leaving the basket).  trace.py <duration_s>"""
import math
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from vision_msgs.msg import Detection3DArray

L = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
rclpy.init()
n = Node('trace')
st = {'pose': None, 'nav': None, 'out': None, 'safety': '', 'task': '', 'arm': None, 'objs': None}
n.create_subscription(Odometry, '/sim/ground_truth/pose', lambda m: st.__setitem__('pose', m), 10)
n.create_subscription(Twist, '/cmd_vel_nav', lambda m: st.__setitem__('nav', m), 10)
n.create_subscription(Twist, '/cmd_vel', lambda m: st.__setitem__('out', m), 10)
n.create_subscription(String, '/safety/state', lambda m: st.__setitem__('safety', m.data), L)
n.create_subscription(String, '/task/state', lambda m: st.__setitem__('task', m.data), L)
ARM = ['arm_waist', 'arm_shoulder', 'arm_elbow', 'arm_forearm_roll', 'arm_wrist_angle', 'arm_wrist_rotate']
def on_js(m):
    d = dict(zip(m.name, m.position))
    if all(a in d for a in ARM):
        st['arm'] = [d[a] for a in ARM]
n.create_subscription(JointState, '/joint_states', on_js, 10)
n.create_subscription(Detection3DArray, '/sim/ground_truth/objects', lambda m: st.__setitem__('objs', m), 10)
end = time.monotonic() + float(sys.argv[1])
last = 0


def tw(m):
    return '-' if m is None else f'{m.linear.x:+.2f},{m.linear.y:+.2f},{m.angular.z:+.2f}'


while time.monotonic() < end:
    rclpy.spin_once(n, timeout_sec=0.05)
    if time.monotonic() - last > 0.5 and st['pose'] is not None:
        last = time.monotonic()
        p = st['pose'].pose.pose
        q = p.orientation
        tilt = math.degrees(math.acos(max(-1, min(1, 1 - 2 * (q.x * q.x + q.y * q.y)))))
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        roll = math.degrees(math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y)))
        pitch = math.degrees(math.asin(max(-1, min(1, 2 * (q.w * q.y - q.z * q.x)))))
        arm = ' '.join(f'{v:+.2f}' for v in st['arm']) if st['arm'] else '-'
        near = []
        for d in (st['objs'].detections if st['objs'] else []):
            o = d.bbox.center.position
            dx, dy = o.x - p.position.x, o.y - p.position.y
            if math.hypot(dx, dy) < 0.8:
                c, s_ = math.cos(yaw), math.sin(yaw)
                near.append(f'{d.id}({c * dx + s_ * dy:+.2f},{-s_ * dx + c * dy:+.2f},z{o.z:.2f})')
        print(f"t {st['pose'].header.stamp.sec:4d} pos {p.position.x:6.2f} {p.position.y:6.2f} z {p.position.z:5.2f} "
              f"yaw {math.degrees(yaw):5.0f} tilt {tilt:4.1f} roll {roll:+5.1f} pitch {pitch:+5.1f} arm [{arm}] nav {tw(st['nav'])} out {tw(st['out'])} "
              f"{st['safety']:5s} {st['task']} near {' '.join(near) or '-'}", flush=True)
