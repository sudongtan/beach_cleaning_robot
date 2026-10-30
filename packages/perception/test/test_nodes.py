"""Detector and classifier nodes against fake ground truth and fake TFs (own ROS domain).
Needs a sourced ROS environment with beach_interfaces built (skipped otherwise)."""

import os
import time
from pathlib import Path

import pytest

rclpy = pytest.importorskip('rclpy')
pytest.importorskip('beach_interfaces')
os.environ.setdefault('ROS_DOMAIN_ID', '63')
os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE'] = 'LOCALHOST'

from geometry_msgs.msg import TransformStamped  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from tf2_ros import StaticTransformBroadcaster  # noqa: E402
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose  # noqa: E402

from beach_interfaces.srv import ClassifyItem  # noqa: E402
from perception.ground_truth_classifier import GroundTruthClassifier  # noqa: E402
from perception.ground_truth_detector import GroundTruthDetector  # noqa: E402

REPO = str(Path(__file__).resolve().parents[3])


def gt(objs):
    arr = Detection3DArray()
    arr.header.frame_id = 'map'
    for oid, cls, (x, y, z) in objs:
        d = Detection3D()
        d.id = oid
        d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.center.position.z = x, y, z
        d.bbox.center.orientation.w = 1.0
        h = ObjectHypothesisWithPose()
        h.hypothesis.class_id = cls
        d.results.append(h)
        arr.detections.append(d)
    return arr


def static_tf(node, child, xyz):
    tf = TransformStamped()
    tf.header.frame_id, tf.child_frame_id = 'map', child
    tf.header.stamp = node.get_clock().now().to_msg()
    tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = xyz
    tf.transform.rotation.w = 1.0
    return tf


@pytest.fixture
def ros():
    rclpy.init()
    yield
    rclpy.try_shutdown()


def spin_until(ex, cond, seconds=3.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end and not cond():
        ex.spin_once(timeout_sec=0.05)
    return cond()


def test_detector_reports_only_what_the_camera_sees(ros):
    det = GroundTruthDetector(parameter_overrides=[Parameter('repo', value=REPO)])
    fake = rclpy.create_node('fake')
    StaticTransformBroadcaster(fake).sendTransform(static_tf(fake, 'camera_link', (1.0, 0.0, 0.5)))
    pub = fake.create_publisher(Detection3DArray, '/sim/ground_truth/objects', 10)
    got = []
    fake.create_subscription(Detection3DArray, '/detections', got.append, 10)
    ex = SingleThreadedExecutor()
    ex.add_node(det)
    ex.add_node(fake)
    objs = gt([('trash_00', 'pet', (4.0, 0.0, 0.0)),        # 3 m ahead, below the camera: seen
               ('distractor_00', 'shell', (3.0, 0.6, 0.0)),  # ahead, to the left: seen
               ('trash_01', 'metal', (-2.0, 0.0, 0.0)),      # behind: not seen
               ('trash_02', 'glass', (20.0, 0.0, 0.0)),      # beyond depth_max: not seen
               ('trash_03', 'residual', (2.0, 5.0, 0.0))])   # far outside the horizontal FOV
    assert spin_until(ex, lambda: (pub.publish(objs), bool(got) and got[-1].detections)[1])
    last = got[-1]
    assert last.header.frame_id == 'camera_link'
    found = {d.id: d for d in last.detections}
    assert set(found) == {'trash_00', 'distractor_00'}
    assert found['trash_00'].results[0].hypothesis.class_id == 'trash'
    assert found['distractor_00'].results[0].hypothesis.class_id == 'distractor'
    p = found['trash_00'].bbox.center.position
    assert (p.x, p.y, p.z) == pytest.approx((3.0, 0.0, -0.5))       # relative to the camera


def test_classifier_uses_the_object_at_the_gripper(ros):
    cls = GroundTruthClassifier(parameter_overrides=[Parameter('repo', value=REPO)])
    fake = rclpy.create_node('fake')
    StaticTransformBroadcaster(fake).sendTransform(static_tf(fake, 'arm_ee_gripper_link', (1.0, 1.0, 0.3)))
    pub = fake.create_publisher(Detection3DArray, '/sim/ground_truth/objects', 10)
    client = fake.create_client(ClassifyItem, '/classify')
    ex = SingleThreadedExecutor()
    ex.add_node(cls)
    ex.add_node(fake)
    assert client.wait_for_service(timeout_sec=3.0)

    def classify(objs):
        want = [d.id for d in objs.detections]

        def delivered():
            pub.publish(objs)
            return cls.objects is not None and [d.id for d in cls.objects.detections] == want
        assert spin_until(ex, delivered)
        fut = client.call_async(ClassifyItem.Request())
        assert spin_until(ex, fut.done)
        return fut.result()

    r = classify(gt([('trash_05', 'pet', (1.0, 1.0, 0.22)), ('trash_06', 'glass', (1.5, 1.0, 0.0))]))
    assert (r.material, r.recyclable, r.bin_id, r.confidence) == ('pet', True, 'recyclable', 1.0)
    r = classify(gt([('trash_06', 'glass', (1.5, 1.0, 0.0))]))           # nothing within the radius
    assert (r.bin_id, r.confidence) == ('residual', 0.0)
    r = classify(gt([('distractor_02', 'shell', (1.0, 1.05, 0.25))]))
    assert (r.material, r.bin_id) == ('distractor', '')
