#!/usr/bin/env python3
"""Ground truth from Gazebo, on simulator-agnostic topics (notes/architecture.md).

  /sim/ground_truth/objects  vision_msgs/Detection3DArray, map frame: every trash
                             item and distractor (id from scenes.yaml, class =
                             material or distractor type), from Gazebo's pose stream
  /people                    vision_msgs/Detection3DArray, map frame: the walking
                             person, or an empty array if the scene has none

(/sim/ground_truth/pose comes from Gazebo's OdometryPublisher via ros_gz_bridge.)

Gazebo doesn't publish the actor's pose, so /people is evaluated from the
person's timeline in the manifest (the same samples the actor's waypoints were
made from) at the current sim time, with linear interpolation.

Everything is driven by Gazebo's pose stream, whose header carries the sim
time, so this node doesn't subscribe to the 1 kHz /clock (run it without
use_sim_time: stamps are set from the Gazebo time, not the node clock).

Parameters:
  manifest (str)          scene_<seed>.manifest.yaml from sim/gazebo/load_scene.py
  objects_rate_hz (float) default 10
  people_rate_hz (float)  default 20
"""

import bisect
import math

import rclpy
import yaml
from geometry_msgs.msg import Pose
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node as GzNode
from builtin_interfaces.msg import Time
from rclpy.node import Node
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose


def yaw_quat(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def rotate(q, v):
    """Rotate vector v by quaternion q = (x, y, z, w)."""
    x, y, z, w = q
    vx, vy, vz = v
    tx, ty, tz = 2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)
    return (vx + w * tx + y * tz - z * ty, vy + w * ty + z * tx - x * tz, vz + w * tz + x * ty - y * tx)


def detection(det_id, cls, pos, quat, size, stamp, frame):
    """Detection3D whose bbox centre is half the height above the object's base."""
    d = Detection3D()
    d.header.stamp, d.header.frame_id = stamp, frame
    d.id = det_id
    p = Pose()
    p.position.x, p.position.y, p.position.z = pos
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = quat
    h = ObjectHypothesisWithPose()
    h.hypothesis.class_id, h.hypothesis.score = cls, 1.0
    h.pose.pose = p
    d.results.append(h)
    off = rotate(quat, (0.0, 0.0, size[2] / 2.0))
    d.bbox.center.position.x = pos[0] + off[0]
    d.bbox.center.position.y = pos[1] + off[1]
    d.bbox.center.position.z = pos[2] + off[2]
    d.bbox.center.orientation = p.orientation
    d.bbox.size.x, d.bbox.size.y, d.bbox.size.z = size
    return d


class GroundTruth(Node):
    def __init__(self):
        super().__init__('ground_truth')
        with open(self.declare_parameter('manifest', '').value) as f:
            m = yaml.safe_load(f)
        self.objects = {o['id']: o for o in m['objects']}
        self.person = m['person']
        if self.person:
            tl = self.person['timeline']
            self.times = [s[0] for s in tl]
            self.period = self.times[-1]

        self.obj_period = 1.0 / self.declare_parameter('objects_rate_hz', 10.0).value
        self.people_period = 1.0 / self.declare_parameter('people_rate_hz', 20.0).value
        self.next_obj = self.next_people = 0.0
        self.pub_obj = self.create_publisher(Detection3DArray, '/sim/ground_truth/objects', 10)
        self.pub_people = self.create_publisher(Detection3DArray, '/people', 10)
        self.gz = GzNode()
        self.gz.subscribe(Pose_V, f"/world/{m['world_name']}/pose/info", self.on_poses)

    def on_poses(self, msg):
        """Called from a gz-transport thread for every pose update (sim time in the header)."""
        t = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
        stamp = Time(sec=msg.header.stamp.sec, nanosec=msg.header.stamp.nsec)
        if t < self.next_obj and t < self.next_people:
            return
        if t >= self.next_obj:
            self.next_obj = t + self.obj_period
            self.publish_objects(msg, stamp)
        if t >= self.next_people:
            self.next_people = t + self.people_period
            self.publish_people(t, stamp)

    def publish_objects(self, msg, stamp):
        arr = Detection3DArray()
        arr.header.stamp, arr.header.frame_id = stamp, 'map'
        seen = set()
        for p in msg.pose:
            o = self.objects.get(p.name)
            if o is None or p.name in seen:
                continue
            seen.add(p.name)
            pos = (p.position.x, p.position.y, p.position.z)
            quat = (p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)
            arr.detections.append(detection(p.name, o['class'], pos, quat, o['size'], stamp, 'map'))
        self.pub_obj.publish(arr)

    def person_pose(self, t):
        tl = self.person['timeline']
        if self.person['loop'] and self.period > 0:
            t = t % self.period
        if t >= self.times[-1]:
            return tl[-1][1:]
        k = max(0, bisect.bisect_right(self.times, t) - 1)
        t0, t1 = self.times[k], self.times[k + 1]
        f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        a, b = tl[k], tl[k + 1]
        dyaw = math.atan2(math.sin(b[4] - a[4]), math.cos(b[4] - a[4]))
        return [a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2]),
                a[3] + f * (b[3] - a[3]), a[4] + f * dyaw]

    def publish_people(self, t, stamp):
        arr = Detection3DArray()
        arr.header.stamp, arr.header.frame_id = stamp, 'map'
        if self.person:
            x, y, ground_z, yaw = self.person_pose(t)
            arr.detections.append(detection(self.person['id'], 'person', (x, y, ground_z),
                                            yaw_quat(yaw), self.person['size'], stamp, 'map'))
        self.pub_people.publish(arr)


def main():
    rclpy.init()
    node = GroundTruth()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
