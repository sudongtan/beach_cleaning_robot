"""Adapter between our locomotion interface and legubiao's quadruped controllers
(basic_quadruped_controller; rl_quadruped_controller with `state_source: imu`).

  /cmd_vel (geometry_msgs/Twist)  ->  control_input_msgs/Inputs on `control_input`
  controller odometry on `controller_odom`  ->  /leg_odom (nav_msgs/Odometry, odom -> base_link)
                                                 [+ TF odom -> base_link if publish_tf]

At startup it waits for the controller's odometry (= controller active), then walks the
controller's FSM from passive to walking (`standup_sequence`; basic: 2, 2, 4 = trotting).
Walking with zero velocity is "standing". Without /cmd_vel for `cmd_timeout` seconds it
commands zero velocity.

`state_source: imu` is for a controller without odometry (rl_quadruped_controller): ready =
IMU messages on `imu` and a subscriber on `control_input`; tilt from the IMU orientation;
no /leg_odom.

Fall: if the body tilt (from the controller's own orientation estimate or the IMU) stays above
`fall_tilt_deg` for `fall_hold_s`, it sends PASSIVE (command 1: legs limp) and ignores
/cmd_vel from then on. A fallen robot kept walking its legs and crawled out of the geofence
(A2b run 5); there is no automatic stand-up.

Yaw trim (`yaw_trim_ki` > 0, needs `state_source: imu`): the robot_lab policy turns at about
+0.17 rad/s with a zero command in Gazebo (B4 log), so the adapter adds the integral of
(commanded - measured yaw rate, IMU gyro z) to the yaw command, limited to +-`yaw_trim_max`.
Off (0) by default.

The controller publishes its twist in the odom frame; /leg_odom carries it in base_link
(nav_msgs/Odometry convention). Simulator-agnostic: talks to the controller over ROS only.
"""

import math

import rclpy
from control_input_msgs.msg import Inputs
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster


def clamp(v, lo=-1.0, hi=1.0):
    return max(lo, min(hi, v))


def sticks(vx, vy, wz, max_v, limit=1.0):
    """Twist -> (ly, lx, rx) sticks for the controller: vx = max_vx*ly, vy = -max_vy*lx,
    wz = -max_wz*rx, each stick clamped to +-limit. Returns the sticks and whether one was clamped."""
    raw = (vx / max_v[0], -vy / max_v[1], -wz / max_v[2])
    out = tuple(clamp(r, -limit, limit) for r in raw)
    return out, out != raw


def yaw_trim_step(trim, wz_cmd, wz_meas, dt, ki, limit):
    """One integrator step of the yaw trim: trim += ki * (wz_cmd - wz_meas) * dt, kept in +-limit."""
    return clamp(trim + ki * (wz_cmd - wz_meas) * dt, -limit, limit)


def tilt_deg(q):
    """Angle between the body z axis and world up, for orientation quaternion q (x, y, z, w)."""
    x, y, z, w = q
    return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))


def rotate_to_body(q, v):
    """Rotate world vector v into the body frame given orientation quaternion q (x, y, z, w)."""
    x, y, z, w = q
    # R^T v for unit quaternion: v' = q* v q
    r = [[1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)],
         [2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)],
         [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)]]
    return [sum(r[i][j] * v[j] for j in range(3)) for i in range(3)]


class LegubiaoAdapter(Node):

    def __init__(self):
        super().__init__('legubiao_adapter')
        p = self.declare_parameter
        self.max_vx = p('max_vx', 0.4).value
        self.max_vy = p('max_vy', 0.3).value
        self.max_wz = p('max_wz', 0.5).value
        # largest stick value sent (1.0 = a real stick); the robot_lab policy needs more (its yaml)
        self.stick_limit = p('stick_limit', 1.0).value
        rate = p('publish_rate', 50.0).value
        self.cmd_timeout = p('cmd_timeout', 0.5).value
        self.seq = list(p('standup_sequence', [2, 2, 4]).value)
        self.waits = list(p('standup_waits', [3.0, 4.0, 2.0]).value)
        self.standup_delay = p('standup_delay', 2.0).value
        self.odom_frame = p('odom_frame', 'odom').value
        self.base_frame = p('base_frame', 'base_link').value
        self.publish_tf = p('publish_tf', True).value
        self.fall_tilt = p('fall_tilt_deg', 30.0).value
        self.fall_hold = p('fall_hold_s', 0.3).value
        self.state_source = p('state_source', 'controller_odom').value   # controller_odom | imu
        self.yaw_trim_ki = p('yaw_trim_ki', 0.0).value       # 1/s; 0 = off
        self.yaw_trim_max = p('yaw_trim_max', 0.4).value     # rad/s
        self.yaw_trim = 0.0
        self.gyro_z = 0.0
        self.trim_time = None
        self.tilt = 0.0
        self.tilted_since = None
        self.fallen = False

        self.inputs_pub = self.create_publisher(Inputs, 'control_input', 10)
        self.odom_pub = self.create_publisher(Odometry, '/leg_odom', 10)
        self.tf = TransformBroadcaster(self) if self.publish_tf else None
        self.create_subscription(Twist, '/cmd_vel', self.on_cmd_vel, 10)
        if self.state_source == 'imu':
            self.create_subscription(Imu, 'imu', self.on_imu, 10)
        else:
            self.create_subscription(Odometry, 'controller_odom', self.on_controller_odom, 10)

        self.cmd = Twist()
        self.cmd_time = None
        self.first_odom_time = None
        self.step = -1          # index into the stand-up sequence; len(seq) = done
        self.step_time = None
        self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(f'waiting for the controller ({self.state_source}) ...')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_cmd_vel(self, msg):
        self.cmd = msg
        self.cmd_time = self.now()

    def on_imu(self, msg):
        # ready once the controller listens too (it subscribes when configured, the IMU
        # broadcaster runs before the leg controller is spawned)
        if self.first_odom_time is None and self.inputs_pub.get_subscription_count() > 0:
            self.first_odom_time = self.now()
        q = msg.orientation
        self.tilt = tilt_deg((q.x, q.y, q.z, q.w))
        self.gyro_z = msg.angular_velocity.z

    def on_controller_odom(self, msg):
        if self.first_odom_time is None:
            self.first_odom_time = self.now()
        q = msg.pose.pose.orientation
        self.tilt = tilt_deg((q.x, q.y, q.z, q.w))
        out = Odometry()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.odom_frame
        out.child_frame_id = self.base_frame
        out.pose = msg.pose
        lin = msg.twist.twist.linear
        vb = rotate_to_body((q.x, q.y, q.z, q.w), (lin.x, lin.y, lin.z))
        out.twist.twist.linear.x, out.twist.twist.linear.y, out.twist.twist.linear.z = vb
        out.twist.twist.angular = msg.twist.twist.angular  # already body frame (gyro)
        out.twist.covariance = msg.twist.covariance
        self.odom_pub.publish(out)
        if self.tf is not None:
            t = TransformStamped()
            t.header = out.header
            t.child_frame_id = self.base_frame
            p = msg.pose.pose.position
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = p.x, p.y, p.z
            t.transform.rotation = q
            self.tf.sendTransform(t)

    def tick(self):
        now = self.now()
        msg = Inputs()
        if self.fallen:
            self.inputs_pub.publish(msg)               # zero inputs; the controller stays passive
            return
        if self.step >= len(self.seq):                 # only once standing (stand-up starts lying down)
            if self.tilt > self.fall_tilt:
                self.tilted_since = now if self.tilted_since is None else self.tilted_since
                if now - self.tilted_since >= self.fall_hold:
                    self.fallen = True
                    msg.command = 1                    # PASSIVE (StateTrotting/StateRL::checkChange)
                    self.inputs_pub.publish(msg)
                    self.get_logger().error(f'FALL: tilt {self.tilt:.0f} deg; controller set to passive, '
                                            '/cmd_vel ignored from now on')
                    return
            else:
                self.tilted_since = None
        # stand-up sequence: mode commands are edge-triggered, sent for one tick each
        if self.step < len(self.seq):
            if self.first_odom_time is None or now - self.first_odom_time < self.standup_delay:
                self.inputs_pub.publish(msg)
                return
            if self.step < 0 or now - self.step_time >= self.waits[self.step]:
                self.step += 1
                self.step_time = now
                if self.step < len(self.seq):
                    msg.command = int(self.seq[self.step])
                    self.get_logger().info(f'stand-up: command {msg.command}')
                else:
                    self.get_logger().info('stand-up done, following /cmd_vel')
            self.inputs_pub.publish(msg)
            return

        vx = vy = wz = 0.0
        if self.cmd_time is not None and now - self.cmd_time <= self.cmd_timeout:
            vx, vy, wz = self.cmd.linear.x, self.cmd.linear.y, self.cmd.angular.z
        if self.yaw_trim_ki > 0.0:
            if self.trim_time is not None:
                self.yaw_trim = yaw_trim_step(self.yaw_trim, wz, self.gyro_z, now - self.trim_time,
                                              self.yaw_trim_ki, self.yaw_trim_max)
            self.trim_time = now
            wz += self.yaw_trim
        # controller: vx = max_vx*ly, vy = -max_vy*lx, wz = -max_wz*rx (StateTrotting.cpp;
        # StateRL.cpp: the same with max 1)
        (msg.ly, msg.lx, msg.rx), clamped = sticks(
            vx, vy, wz, (self.max_vx, self.max_vy, self.max_wz), self.stick_limit)
        if clamped:
            self.get_logger().warn('cmd_vel beyond the controller limits, clamped',
                                   throttle_duration_sec=5.0)
        self.inputs_pub.publish(msg)


def main():
    rclpy.init()
    node = LegubiaoAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
