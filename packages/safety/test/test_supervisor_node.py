"""The node against fake inputs (own ROS domain, wall time): a person walks up to the robot
and away again -> /safety/state ok -> slow -> stop -> slow -> ok, /cmd_vel zero while stopped;
/estop latches. Needs a sourced ROS environment (skipped otherwise)."""

import os
import time
from pathlib import Path

import pytest
import yaml

rclpy = pytest.importorskip('rclpy')
os.environ.setdefault('ROS_DOMAIN_ID', '61')
os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE'] = 'LOCALHOST'

from geometry_msgs.msg import Point32, PolygonStamped, TransformStamped, Twist  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile  # noqa: E402
from std_msgs.msg import Bool, String  # noqa: E402
from tf2_ros import StaticTransformBroadcaster  # noqa: E402
from vision_msgs.msg import Detection3D, Detection3DArray  # noqa: E402

from safety.supervisor_node import SupervisorNode  # noqa: E402

CFG = yaml.safe_load(open(Path(__file__).resolve().parents[3] / 'config' / 'safety.yaml'))
PARAMS = CFG['safety_supervisor']['ros__parameters']
LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def test_person_approach_and_estop():
    rclpy.init()
    try:
        sup = SupervisorNode(parameter_overrides=[Parameter(k, value=v) for k, v in PARAMS.items()])
        fake = rclpy.create_node('fake_inputs')
        states, cmds = [], []
        fake.create_subscription(String, '/safety/state', lambda m: states.append(m.data), LATCHED)
        fake.create_subscription(Twist, '/cmd_vel', lambda m: cmds.append((time.monotonic(), m.linear.x)), 10)
        cmd_pub = fake.create_publisher(Twist, '/cmd_vel_nav', 10)
        people_pub = fake.create_publisher(Detection3DArray, '/people', 10)
        estop_pub = fake.create_publisher(Bool, '/estop', LATCHED)
        fence = PolygonStamped()
        fence.header.frame_id = 'map'
        fence.polygon.points = [Point32(x=x, y=y) for x, y in ((-50.0, -50.0), (50.0, -50.0), (50.0, 50.0), (-50.0, 50.0))]
        fake.create_publisher(PolygonStamped, '/geofence', LATCHED).publish(fence)
        tf = TransformStamped()
        tf.header.frame_id, tf.child_frame_id = 'map', 'base_link'
        tf.transform.rotation.w = 1.0
        tf.header.stamp = fake.get_clock().now().to_msg()
        StaticTransformBroadcaster(fake).sendTransform(tf)
        ex = SingleThreadedExecutor()
        ex.add_node(sup)
        ex.add_node(fake)

        def run(seconds, distance):
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                cmd = Twist()
                cmd.linear.x = 0.3
                cmd_pub.publish(cmd)
                arr = Detection3DArray()
                arr.header.frame_id = 'map'
                if distance is not None:
                    d = Detection3D()
                    d.bbox.center.position.x = float(distance)
                    arr.detections.append(d)
                people_pub.publish(arr)
                ex.spin_once(timeout_sec=0.02)

        run(2.0, 6.0)          # startup stop -> ok
        assert states[-1] == 'ok' and cmds[-1][1] == pytest.approx(0.3)
        run(0.5, 2.5)          # slow zone
        assert states[-1] == 'slow' and cmds[-1][1] == pytest.approx(0.3 * PARAMS['slow_factor'])
        run(0.5, 1.0)          # stop zone
        assert states[-1] == 'stop' and cmds[-1][1] == 0.0
        t_stop = time.monotonic()
        run(0.5, 5.0)          # walked away, but the hold time hasn't passed
        assert states[-1] == 'stop' and all(v == 0.0 for t, v in cmds if t > t_stop)
        run(2.0, 5.0)          # released
        assert states[-1] == 'ok'
        estop_pub.publish(Bool(data=True))
        run(0.5, 20.0)
        assert states[-1] == 'estop' and cmds[-1][1] == 0.0
        estop_pub.publish(Bool(data=False))
        run(0.5, 20.0)
        assert states[-1] == 'ok'
        # silence on /people: fail-safe stop
        end = time.monotonic() + 1.0
        while time.monotonic() < end:
            cmd_pub.publish(Twist())
            ex.spin_once(timeout_sec=0.02)
        assert states[-1] == 'stop'
        assert states[:1] == ['stop'] and 'slow' in states
    finally:
        rclpy.try_shutdown()


def test_to_frame_rotated_robot():
    """Robot at map (2, 0) facing +y; a person at map (2, 1) is 1 m straight ahead of it."""
    import math
    from geometry_msgs.msg import Transform

    from safety.supervisor_node import to_frame
    tf = Transform()                       # base_link <- map: inverse of the robot pose
    tf.rotation.z, tf.rotation.w = math.sin(-math.pi / 4), math.cos(-math.pi / 4)
    tf.translation.y = 2.0
    assert to_frame(tf, (2.0, 1.0, 0.0)) == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)
