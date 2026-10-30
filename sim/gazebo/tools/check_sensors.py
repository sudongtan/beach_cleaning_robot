#!/usr/bin/env python3
"""A1b DoD 3-4 check while a scene is up and the robot stands still:
GNSS error vs ground truth and its covariance, IMU sanity, camera frames saved
as PNG (rgb + depth) for a look.

    check_sensors.py [--fixes 100] [--out DIR] [--seed N]
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import rclpy
import yaml
from nav_msgs.msg import Odometry
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, Imu, NavSatFix

REPO = Path(__file__).resolve().parents[3]

# WGS84
A_EQ = 6378137.0
E2 = 6.69437999014e-3


def enu_from_geo(lat, lon, lat0, lon0):
    """Local east/north (m) of (lat, lon) from (lat0, lon0); small-area approximation."""
    phi = math.radians(lat0)
    rn = A_EQ / math.sqrt(1 - E2 * math.sin(phi) ** 2)            # prime vertical radius
    rm = rn * (1 - E2) / (1 - E2 * math.sin(phi) ** 2)            # meridional radius
    return math.radians(lon - lon0) * rn * math.cos(phi), math.radians(lat - lat0) * rm


class C(Node):
    def __init__(self):
        super().__init__('check_sensors')
        self.fix, self.gt, self.imu, self.info, self.rgb, self.depth = [], None, [], None, None, None
        self.create_subscription(NavSatFix, '/gnss/fix', lambda m: self.fix.append((m, self.gt)), 10)
        self.create_subscription(Odometry, '/sim/ground_truth/pose', self.on_gt, 10)
        self.create_subscription(Imu, '/imu', lambda m: self.imu.append(m), qos_profile_sensor_data)
        self.create_subscription(CameraInfo, '/camera/info', lambda m: setattr(self, 'info', m), 10)
        self.create_subscription(Image, '/camera/rgb', lambda m: setattr(self, 'rgb', m), 10)
        self.create_subscription(Image, '/camera/depth', lambda m: setattr(self, 'depth', m), 10)

    def on_gt(self, m):
        self.gt = m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fixes', type=int, default=100)
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--out', default=str(REPO / 'sim' / 'gazebo' / 'worlds' / 'generated' / 'tools'))
    a = ap.parse_args()
    geo = [s for s in yaml.safe_load(open(REPO / 'bench' / 'scenes.yaml'))['scenes'] if s['seed'] == a.seed][0]['geo']
    rclpy.init()
    n = C()
    while len(n.fix) < a.fixes or n.rgb is None or n.depth is None or n.info is None:
        rclpy.spin_once(n, timeout_sec=0.1)

    errs = []
    for m, gt in n.fix:
        if gt is None:
            continue
        e, nn = enu_from_geo(m.latitude, m.longitude, geo['latitude_deg'], geo['longitude_deg'])
        p = gt.pose.pose.position          # base_link; gnss_link is 3 cm forward, ~8 cm up: ignored
        errs.append((e - p.x, nn - p.y, m.altitude - p.z))
    E = np.array(errs)
    last = n.fix[-1][0]
    cov = np.array(last.position_covariance).reshape(3, 3)
    print(f'GNSS: {len(E)} fixes, frame {last.header.frame_id}, status {last.status.status}')
    print(f'  error east  mean {E[:, 0].mean():+.2f} m  std {E[:, 0].std():.2f} m   (covariance sigma {math.sqrt(cov[0, 0]):.2f})')
    print(f'  error north mean {E[:, 1].mean():+.2f} m  std {E[:, 1].std():.2f} m   (covariance sigma {math.sqrt(cov[1, 1]):.2f})')
    print(f'  error up    mean {E[:, 2].mean():+.2f} m  std {E[:, 2].std():.2f} m   (covariance sigma {math.sqrt(cov[2, 2]):.2f})')
    h = np.hypot(E[:, 0], E[:, 1])
    print(f'  horizontal error: RMS {math.sqrt((h ** 2).mean()):.2f} m, 95 % {np.percentile(h, 95):.2f} m, max {h.max():.2f} m')
    print(f'  covariance type {last.position_covariance_type}, diagonal {np.diag(cov).round(3).tolist()}')

    acc = np.array([[m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z] for m in n.imu[-200:]])
    gyr = np.array([[m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z] for m in n.imu[-200:]])
    print(f'IMU: {len(n.imu)} msgs, frame {n.imu[-1].header.frame_id}, |accel| mean {np.linalg.norm(acc, axis=1).mean():.2f} m/s^2, '
          f'gyro std {gyr.std(axis=0).round(4).tolist()} rad/s')

    out = Path(a.out)
    rgb = np.frombuffer(bytes(n.rgb.data), np.uint8).reshape(n.rgb.height, n.rgb.width, 3)
    PILImage.fromarray(rgb).save(out / 'camera_rgb.png')
    d = np.frombuffer(bytes(n.depth.data), np.float32).reshape(n.depth.height, n.depth.width)
    finite = np.isfinite(d)
    dd = np.where(finite, d, 0.0)
    PILImage.fromarray((np.clip(dd / 10.0, 0, 1) * 255).astype(np.uint8)).save(out / 'camera_depth.png')
    print(f'camera: rgb {n.rgb.width}x{n.rgb.height} {n.rgb.encoding} frame {n.rgb.header.frame_id}; '
          f'depth {n.depth.encoding}, {finite.mean() * 100:.0f} % finite, range {dd[finite].min():.2f}-{dd[finite].max():.2f} m; '
          f'info fx={n.info.k[0]:.1f} frame {n.info.header.frame_id}')
    print(f'saved {out}/camera_rgb.png, camera_depth.png')
    rclpy.try_shutdown()


if __name__ == '__main__':
    sys.exit(main())
