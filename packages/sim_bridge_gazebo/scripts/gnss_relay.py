#!/usr/bin/env python3
"""/gnss/fix_raw -> /gnss/fix with a known diagonal covariance.

The ros_gz NavSat conversion leaves position_covariance at zero, which
robot_localization (A4) can't use. Values come from config/sensors.yaml. Gazebo
applies the horizontal noise in degrees (one model for lat and lon), so east
noise is the north noise times cos(latitude).

Parameter: sensors_file (str) path to sensors.yaml.
"""

import math

import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix


class GnssRelay(Node):
    def __init__(self):
        super().__init__('gnss_relay')
        with open(self.declare_parameter('sensors_file', '').value) as f:
            g = yaml.safe_load(f)['gnss']
        self.sigma_n = g['horizontal_noise_stddev_m']
        self.sigma_u = g['vertical_noise_stddev_m']
        self.cov_type = g['covariance_type']
        self.pub = self.create_publisher(NavSatFix, '/gnss/fix', 10)
        self.create_subscription(NavSatFix, '/gnss/fix_raw', self.cb, 10)

    def cb(self, msg):
        sigma_e = self.sigma_n * math.cos(math.radians(msg.latitude))
        msg.position_covariance = [sigma_e ** 2, 0.0, 0.0,
                                   0.0, self.sigma_n ** 2, 0.0,
                                   0.0, 0.0, self.sigma_u ** 2]      # ENU, m^2
        msg.position_covariance_type = self.cov_type
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = GnssRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
