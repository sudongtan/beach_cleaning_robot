#!/usr/bin/env python3
"""A1b/A1c task 7 check: attach and detach one item on command.

Expects the robot standing next to the item (e.g. after goto.py). Sequence:
gripper down (ground_front), /sim/attach, close the jaws, lift (up), over each
compartment in turn (the item is detached over the last one), back to rest.
Prints the item's ground-truth pose after each step and checks the error cases.
With the custom robot every move goes through the `up` waypoint (arm_pose.py).

    attach_test.py <item_id> [--variant custom|stock] [--show q1,...,q6 --image out.png]

--show: after lifting, hold the item at these arm joints (e.g. in front of the head
camera) and save a /camera/rgb frame to --image.
"""

import argparse
import sys
import time
from pathlib import Path

import rclpy
from beach_interfaces.srv import SimAttach
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection3DArray

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arm_pose  # noqa: E402


class T(Node):
    def __init__(self):
        super().__init__('attach_test')
        self.objs = None
        self.img = None
        self.base = None
        self.create_subscription(Odometry, '/sim/ground_truth/pose', lambda m: setattr(self, 'base', m), 10)
        self.create_subscription(Detection3DArray, '/sim/ground_truth/objects', self.cb, 10)
        self.create_subscription(Image, '/camera/rgb', lambda m: setattr(self, 'img', m), 10)
        self.att = self.create_client(SimAttach, '/sim/attach')
        self.det = self.create_client(SimAttach, '/sim/detach')

    def cb(self, m):
        self.objs = m

    def item(self, oid):
        self.objs = None
        while self.objs is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        for d in self.objs.detections:
            if d.id == oid:
                return d.results[0].pose.pose.position
        return None

    def call(self, client, oid):
        client.wait_for_service(timeout_sec=10.0)
        fut = client.call_async(SimAttach.Request(object_id=oid))
        rclpy.spin_until_future_complete(self, fut)
        return fut.result()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('item')
    ap.add_argument('--variant', choices=arm_pose.VARIANTS, default='custom')
    ap.add_argument('--show', help='arm joints to hold the lifted item at (comma-separated)')
    ap.add_argument('--image', default='held_item.png')
    a = ap.parse_args()
    poses = arm_pose.load_poses(a.variant)
    closed, _ = arm_pose.finger_limits(a.variant)
    rclpy.init()
    n = T()

    def show(label):
        p = n.item(a.item)
        print(f'{label:36s} {a.item} at ({p.x:7.3f}, {p.y:7.3f}, {p.z:6.3f})', flush=True)
        return p

    def move(name, fingers=None):
        e = arm_pose.move_to(n, a.variant, name, poses, fingers=fingers)
        if e:
            print(f'  arm move {name}: error code {e}')

    p0 = show('start')
    move('ground_front')
    show('gripper down (ground_front)')
    r = n.call(n.att, a.item)
    print(f'/sim/attach -> success={r.success} "{r.message}"')
    arm_pose.set_fingers(n, closed)                       # close the jaws (visual in attach mode)
    move('up', fingers=closed)
    p1 = show('lifted (up)')
    if a.show:
        arm_pose.move_arm(n, [float(v) for v in a.show.split(',')])
        time.sleep(1.0)
        show('held in front of the camera')
        n.img = None
        while n.img is None:
            rclpy.spin_once(n, timeout_sec=0.1)
        import numpy as np
        from PIL import Image as PILImage
        PILImage.fromarray(np.frombuffer(bytes(n.img.data), np.uint8).reshape(n.img.height, n.img.width, 3)).save(a.image)
        print(f'saved camera frame {a.image}')
        move('up', fingers=closed)
    move('over_residual', fingers=closed)
    show('over the residual compartment')
    move('over_recyclable', fingers=closed)
    p2 = show('over the recyclable compartment')
    r = n.call(n.det, a.item)
    print(f'/sim/detach -> success={r.success} "{r.message}"')
    time.sleep(2.0)
    p3 = show('2 s after detach')
    move('rest')
    p4 = show('arm back at rest')
    for oid, client, label in (('nonexistent', n.att, 'attach unknown id'),
                               (a.item, n.det, 'detach when already detached')):
        r = n.call(client, oid)
        print(f'{label}: success={r.success} "{r.message}"')
    import math
    b = n.base.pose.pose
    yaw = math.atan2(2 * (b.orientation.w * b.orientation.z + b.orientation.x * b.orientation.y),
                     1 - 2 * (b.orientation.y ** 2 + b.orientation.z ** 2))
    dx, dy = p4.x - b.position.x, p4.y - b.position.y
    bx, by = math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy
    print(f'RESULT ({a.variant}) lifted {p1.z - p0.z:+.3f} m, moved '
          f'{((p2.x - p0.x) ** 2 + (p2.y - p0.y) ** 2) ** 0.5:.2f} m while attached; final item position in '
          f'base_link: x {bx:+.3f} y {by:+.3f} z {p4.z - b.position.z:+.3f} m '
          f'(recyclable compartment = +y half of the basket)')
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
