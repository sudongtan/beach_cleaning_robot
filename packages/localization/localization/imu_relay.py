"""IMU relay for the EKFs (localization:=ekf | own_ekf, B4).

  /imu -> /localization/imu, the same message with a diagonal orientation covariance from
  config/localization.yaml (Gazebo's IMU publishes a zero orientation covariance, which
  robot_localization can't weight). Parameter: repo.
"""

from pathlib import Path

import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import Imu


def with_covariance(msg, stddev):
    msg.orientation_covariance = [0.0] * 9
    for i, s in enumerate(stddev):
        msg.orientation_covariance[4 * i] = float(s) ** 2
    return msg


class ImuRelay(Node):

    def __init__(self, **kwargs):
        super().__init__('imu_relay', **kwargs)
        repo = Path(self.declare_parameter('repo', '').value)
        cfg = yaml.safe_load(open(repo / 'config' / 'localization.yaml'))['localization']['imu_relay']
        self.stddev = cfg['orientation_stddev_rad']
        self.pub = self.create_publisher(Imu, '/localization/imu', 20)
        self.create_subscription(Imu, '/imu', lambda m: self.pub.publish(with_covariance(m, self.stddev)), 20)


def main():
    rclpy.init()
    node = ImuRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
