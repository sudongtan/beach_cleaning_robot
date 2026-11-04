"""Safety supervisor node: the only publisher of /cmd_vel (notes/architecture.md).

  /cmd_vel_nav (Twist), /people (Detection3DArray), /estop (Bool, latched),
  /geofence (PolygonStamped, latched), robot pose from TF map -> base_link
    -> /cmd_vel (Twist), /safety/state (String: ok / slow / stop / estop)

Rules and hysteresis live in safety.supervisor (unit-tested); this node only converts
messages. People positions are transformed into base_link with TF; if that fails the
message is dropped, so a missing transform ends up as stale /people -> stop.
Thresholds: config/safety.yaml.
"""

import math

import rclpy
from geometry_msgs.msg import PolygonStamped, Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection3DArray

from safety.supervisor import Params, Supervisor

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def to_frame(tf, p):
    """Point p (x, y, z) through geometry_msgs/Transform tf."""
    q, t = tf.rotation, tf.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    r = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
    return [sum(r[i][j] * p[j] for j in range(3)) + (t.x, t.y, t.z)[i] for i in range(3)]


class SupervisorNode(Node):

    def __init__(self, **kwargs):
        super().__init__('safety_supervisor', **kwargs)
        names = Params.__dataclass_fields__
        values = {n: self.declare_parameter(n, getattr(Params, n)).value for n in names}
        self.sup = Supervisor(Params.from_dict(values))
        self.base_frame = self.declare_parameter('base_frame', 'base_link').value
        self.map_frame = self.declare_parameter('map_frame', 'map').value
        self.pose_timeout = self.declare_parameter('pose_timeout_s', 0.5).value
        rate = self.declare_parameter('rate_hz', 20.0).value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.state_pub = self.create_publisher(String, '/safety/state', LATCHED)
        self.create_subscription(Twist, '/cmd_vel_nav', self.on_cmd, 10)
        self.create_subscription(Detection3DArray, '/people', self.on_people, 10)
        self.create_subscription(Bool, '/estop', self.on_estop, LATCHED)
        self.create_subscription(PolygonStamped, '/geofence', self.on_geofence, LATCHED)
        self.create_timer(1.0 / rate, self.tick)
        self.last_state = None

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_cmd(self, msg):
        self.sup.on_cmd(self.now(), msg.linear.x, msg.linear.y, msg.angular.z)

    def on_estop(self, msg):
        self.sup.on_estop(msg.data)

    def on_people(self, msg):
        dists = []
        if msg.detections:
            try:
                tf = self.tf_buffer.lookup_transform(self.base_frame, msg.header.frame_id, Time()).transform
            except TransformException as e:
                self.get_logger().warn(f'/people dropped, no transform: {e}', throttle_duration_sec=2.0)
                return
            for d in msg.detections:
                c = d.bbox.center.position
                x, y, _ = to_frame(tf, (c.x, c.y, c.z))
                dists.append(math.hypot(x, y))
        self.sup.on_people(self.now(), dists)

    def on_geofence(self, msg):
        self.sup.on_geofence([(p.x, p.y) for p in msg.polygon.points])

    def robot_pose(self):
        """(x, y, yaw) of base_link in map from TF, or None if missing or older than pose_timeout_s."""
        try:
            tf = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, Time())
        except TransformException:
            return None
        stamp = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
        if stamp > 0 and self.now() - stamp > self.pose_timeout:             # stamp 0: static transform
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        return t.x, t.y, math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    def tick(self):
        self.sup.on_pose(self.robot_pose(), self.now())
        state, (vx, vy, wz) = self.sup.step(self.now())
        cmd = Twist()
        cmd.linear.x, cmd.linear.y, cmd.angular.z = vx, vy, wz
        self.cmd_pub.publish(cmd)
        if state != self.last_state:
            self.state_pub.publish(String(data=state))
            self.get_logger().info(f'safety state: {state}')
            self.last_state = state


def main():
    rclpy.init()
    node = SupervisorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
