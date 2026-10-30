"""Ground-truth localization baseline (localization:=ground_truth).

  /sim/ground_truth/pose (Odometry, map -> base_link)
    -> /odom/global (map frame), /odom (odom frame), TF odom -> base_link,
       static TF map -> odom (identity)

The same topics and TFs the A4 dual EKF will publish (notes/architecture.md,
"Localization"). With map -> odom the identity, the odom pose equals the true pose.
"""

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster


def relabel(msg, frame_id, child_frame_id):
    """Copy of an Odometry message with new frame ids (pose, twist and covariances unchanged)."""
    out = Odometry()
    out.header.stamp = msg.header.stamp
    out.header.frame_id = frame_id
    out.child_frame_id = child_frame_id
    out.pose = msg.pose
    out.twist = msg.twist
    return out


def odom_to_tf(msg):
    tf = TransformStamped()
    tf.header = msg.header
    tf.child_frame_id = msg.child_frame_id
    p, q = msg.pose.pose.position, msg.pose.pose.orientation
    tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = p.x, p.y, p.z
    tf.transform.rotation = q
    return tf


class GroundTruthLocalization(Node):

    def __init__(self, **kwargs):
        super().__init__('localization', **kwargs)
        p = self.declare_parameter
        self.map_frame = p('map_frame', 'map').value
        self.odom_frame = p('odom_frame', 'odom').value
        self.base_frame = p('base_frame', 'base_link').value
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.global_pub = self.create_publisher(Odometry, '/odom/global', 10)
        self.tf = TransformBroadcaster(self)
        identity = TransformStamped()
        identity.header.stamp = self.get_clock().now().to_msg()
        identity.header.frame_id, identity.child_frame_id = self.map_frame, self.odom_frame
        identity.transform.rotation.w = 1.0
        self.static_tf = StaticTransformBroadcaster(self)
        self.static_tf.sendTransform(identity)
        self.create_subscription(Odometry, '/sim/ground_truth/pose', self.on_pose, 10)

    def on_pose(self, msg):
        self.global_pub.publish(relabel(msg, self.map_frame, self.base_frame))
        odom = relabel(msg, self.odom_frame, self.base_frame)
        self.odom_pub.publish(odom)
        self.tf.sendTransform(odom_to_tf(odom))


def main():
    rclpy.init()
    node = GroundTruthLocalization()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
