"""World model node against fake detections and TFs (own ROS domain). Needs a sourced ROS
environment with beach_interfaces built (skipped otherwise)."""

import os
import time
from pathlib import Path

import numpy as np
import pytest

rclpy = pytest.importorskip('rclpy')
pytest.importorskip('beach_interfaces')
os.environ.setdefault('ROS_DOMAIN_ID', '64')
os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE'] = 'LOCALHOST'

from beach_interfaces.msg import TrashItem, WorldModel  # noqa: E402
from beach_interfaces.srv import UpdateItem  # noqa: E402
from geometry_msgs.msg import TransformStamped  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from tf2_ros import StaticTransformBroadcaster  # noqa: E402
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose  # noqa: E402

from world_model.node import WorldModelNode  # noqa: E402

REPO = str(Path(__file__).resolve().parents[3])


def tf(child, xyz):
    t = TransformStamped()
    t.header.frame_id, t.child_frame_id = 'map', child
    t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = xyz
    t.transform.rotation.w = 1.0
    return t


def detections(objs):
    arr = Detection3DArray()
    arr.header.frame_id = 'camera_link'
    for oid, cls, (x, y, z) in objs:
        d = Detection3D()
        d.id = oid
        d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.center.position.z = x, y, z
        d.bbox.center.orientation.w = 1.0
        h = ObjectHypothesisWithPose()
        h.hypothesis.class_id, h.hypothesis.score = cls, 1.0
        d.results.append(h)
        arr.detections.append(d)
    return arr


def test_world_model_node():
    rclpy.init()
    try:
        node = WorldModelNode(parameter_overrides=[Parameter('repo', value=REPO), Parameter('seed', value=3)])
        fake = rclpy.create_node('fake')
        stamp = fake.get_clock().now().to_msg()
        cam, grip = tf('camera_link', (10.0, 15.0, 0.34)), tf('arm_ee_gripper_link', (10.4, 15.0, 0.5))
        cam.header.stamp = grip.header.stamp = stamp
        StaticTransformBroadcaster(fake).sendTransform([cam, grip])
        pub = fake.create_publisher(Detection3DArray, '/detections', 10)
        got = []
        fake.create_subscription(WorldModel, '/world_model', got.append, 10)
        client = fake.create_client(UpdateItem, '/world_model/update_item')
        ex = SingleThreadedExecutor()
        ex.add_node(node)
        ex.add_node(fake)
        rng = np.random.default_rng(1)
        end = time.monotonic() + 2.5
        while time.monotonic() < end:
            noise = rng.normal(0, 0.05, 2)
            pub.publish(detections([('trash_04', 'trash', (3.0 + noise[0], noise[1], -0.3)),
                                    ('', 'trash', (0.4, 0.0, 0.16)),            # the held item at the gripper
                                    ('', 'trash', (0.0, -11.0, -0.3))]))        # map (10, 4): outside the geofence
            ex.spin_once(timeout_sec=0.05)
        last = got[-1]
        assert len(last.items) == 1
        it = last.items[0]
        assert it.scene_id == 'trash_04' and it.status == TrashItem.SEEN and it.times_seen > 10
        assert abs(it.pose.pose.position.x - 13.0) < 0.05 and abs(it.pose.pose.position.y - 15.0) < 0.05
        assert sum(v == 100 for v in last.coverage.data) > 0

        req = UpdateItem.Request(id=it.id, status=TrashItem.COLLECTED, material='glass', bin_id='recyclable')
        fut = client.call_async(req)
        end = time.monotonic() + 3.0
        while time.monotonic() < end and not fut.done():
            ex.spin_once(timeout_sec=0.05)
        assert fut.result().success
        while time.monotonic() < end + 1.0 and not (got[-1].compartment_recyclable == [it.id]):
            ex.spin_once(timeout_sec=0.05)
        assert got[-1].compartment_recyclable == [it.id]
        assert got[-1].basket_load_kg == pytest.approx(0.114)                  # seed 3 trash_04 is the glass jar
    finally:
        rclpy.try_shutdown()
