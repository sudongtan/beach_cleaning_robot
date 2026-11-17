"""Own EKF localization (localization:=own_ekf, B4): two localization.ekf_core.PlanarEkf filters.

  /leg_odom (body twist) + /localization/imu (yaw, yaw rate)        -> local filter -> /odom, TF odom -> base_link
  the same + /gnss/fix (local ENU about the scene's datum, gated)   -> global filter -> /odom/global, TF map -> odom

Same inputs and outputs as localization:=ekf (robot_localization). The local filter starts at the
odom origin; the global one at the scene's robot_start (as ekf.launch.py). The GNSS antenna's lever
arm comes from TF base_link -> gnss_link. The global filter also estimates the leg odometry's speed
scale (GNSS makes it observable); the local filter uses that estimate. Parameters: repo, seed; settings:
config/localization.yaml `own_ekf`.
"""

import math
from pathlib import Path

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Imu, NavSatFix
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener

from localization.ekf_core import LocalEnu, PlanarEkf


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def stamp_s(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def compose_map_odom(map_pose, odom_pose):
    """map -> odom = map_T_base * inverse(odom_T_base), planar (x, y, yaw) tuples."""
    mx, my, myaw = map_pose
    ox, oy, oyaw = odom_pose
    yaw = math.atan2(math.sin(myaw - oyaw), math.cos(myaw - oyaw))
    c, s = math.cos(yaw), math.sin(yaw)
    return mx - (c * ox - s * oy), my - (s * ox + c * oy), yaw


class OwnEkf(Node):

    def __init__(self, **kwargs):
        super().__init__('own_ekf', **kwargs)
        repo = Path(self.declare_parameter('repo', '').value)
        seed = self.declare_parameter('seed', 0).value
        cfg = yaml.safe_load(open(repo / 'config' / 'localization.yaml'))['localization']['own_ekf']
        scene = next(s for s in yaml.safe_load(open(repo / 'bench' / 'scenes.yaml'))['scenes'] if s['seed'] == seed)
        geo = scene['geo']
        self.enu = LocalEnu(geo['latitude_deg'], geo['longitude_deg'], geo.get('elevation_m', 0.0))
        self.cfg = cfg
        start = scene['robot_start']['pose']
        sc = cfg['odom_scale']
        self.local = PlanarEkf(cfg['process_noise'], x0=(0.0, 0.0, start[2]), p0=cfg['initial_stddev_local'],
                               gate_chi2=cfg['gnss_gate_chi2'], scale0=sc['initial'])
        self.glob = PlanarEkf(cfg['process_noise'], x0=tuple(start[:3]), p0=cfg['initial_stddev_global'],
                              gate_chi2=cfg['gnss_gate_chi2'], estimate_scale=sc['estimate'], scale0=sc['initial'],
                              scale_std=sc['initial_stddev'], scale_noise=sc['noise'])
        self.lever = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf = TransformBroadcaster(self)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.global_pub = self.create_publisher(Odometry, '/odom/global', 10)
        self.create_subscription(Odometry, '/leg_odom', self.on_leg_odom, 20)
        self.create_subscription(Imu, '/localization/imu', self.on_imu, 50)
        self.create_subscription(NavSatFix, '/gnss/fix', self.on_fix, 10)
        self.create_timer(1.0 / cfg['publish_hz'], self.publish)
        self.create_timer(10.0, self.report)

    def on_leg_odom(self, msg):
        t, v = stamp_s(msg.header.stamp), msg.twist.twist
        cov = msg.twist.covariance
        var = [max(cov[0], 1e-4), max(cov[7], 1e-4), max(cov[35], 1e-4)]
        for f in (self.local, self.glob):
            f.update_body_velocity(t, (v.linear.x, v.linear.y, v.angular.z), var)

    def on_imu(self, msg):
        t = stamp_s(msg.header.stamp)
        yaw_var = max(msg.orientation_covariance[8], 1e-6)
        wz_var = max(msg.angular_velocity_covariance[8], 1e-8)
        for f in (self.local, self.glob):
            f.update_imu(t, yaw_of(msg.orientation), msg.angular_velocity.z, yaw_var, wz_var)

    def on_fix(self, msg):
        if msg.status.status < 0:
            return
        if self.lever is None:
            try:
                tr = self.tf_buffer.lookup_transform('base_link', msg.header.frame_id or 'gnss_link', Time())
                self.lever = (tr.transform.translation.x, tr.transform.translation.y)
            except TransformException:
                return
        e, n, _ = self.enu(msg.latitude, msg.longitude, msg.altitude)
        cov = msg.position_covariance
        self.glob.update_gnss(stamp_s(msg.header.stamp), e, n, max(cov[0], 1e-4), max(cov[4], 1e-4), self.lever)
        self.local.s[6] = self.glob.scale()        # the odom filter uses the calibrated leg odometry scale

    def odometry(self, f, frame):
        o = Odometry()
        sec = f.t or 0.0
        o.header.stamp.sec, o.header.stamp.nanosec = int(sec), int((sec % 1) * 1e9)
        o.header.frame_id, o.child_frame_id = frame, 'base_link'
        x, y, yaw, vx, vy, wz = f.s[:6]
        o.pose.pose.position.x, o.pose.pose.position.y = float(x), float(y)
        o.pose.pose.orientation.z, o.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        cov = np.zeros((6, 6))
        cov[np.ix_([0, 1, 5], [0, 1, 5])] = f.P[:3, :3]
        o.pose.covariance = [float(v) for v in cov.flatten()]
        o.twist.twist.linear.x, o.twist.twist.linear.y, o.twist.twist.angular.z = float(vx), float(vy), float(wz)
        return o

    def publish(self):
        if self.local.t is None or self.glob.t is None:
            return
        odom, glob = self.odometry(self.local, 'odom'), self.odometry(self.glob, 'map')
        self.odom_pub.publish(odom)
        self.global_pub.publish(glob)
        stamp = odom.header.stamp
        tfs = []
        for parent, child, (x, y, yaw) in (('odom', 'base_link', self.local.pose()),
                                           ('map', 'odom', compose_map_odom(self.glob.pose(), self.local.pose()))):
            tf = TransformStamped()
            tf.header.stamp, tf.header.frame_id, tf.child_frame_id = stamp, parent, child
            tf.transform.translation.x, tf.transform.translation.y = float(x), float(y)
            tf.transform.rotation.z, tf.transform.rotation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            tfs.append(tf)
        self.tf.sendTransform(tfs)

    def report(self):
        self.get_logger().info(f'gnss used {self.glob.gnss_used}, gated out {self.glob.gnss_rejected}, '
                               f'leg odometry scale {self.glob.scale():.3f}')


def main():
    rclpy.init()
    node = OwnEkf()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
