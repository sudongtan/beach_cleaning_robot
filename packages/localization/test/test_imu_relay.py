"""IMU relay (B4): orientation covariance from config, everything else unchanged."""

import pytest

pytest.importorskip('sensor_msgs')
from sensor_msgs.msg import Imu  # noqa: E402

from localization.imu_relay import with_covariance  # noqa: E402
from localization.own_ekf import compose_map_odom  # noqa: E402


def test_covariance_diagonal_from_stddev():
    m = Imu()
    m.orientation.w = 1.0
    m.angular_velocity.z = 0.3
    out = with_covariance(m, [0.01, 0.01, 0.035])
    assert list(out.orientation_covariance) == pytest.approx([1e-4, 0, 0, 0, 1e-4, 0, 0, 0, 0.035 ** 2])
    assert out.angular_velocity.z == 0.3 and out.orientation.w == 1.0


def test_map_odom_composition():
    """map -> odom such that (map -> odom) * (odom -> base) = (map -> base)."""
    import math
    mo = compose_map_odom((5.0, 3.0, 1.0), (1.0, 2.0, 0.5))
    x, y, yaw = mo
    c, s = math.cos(yaw), math.sin(yaw)
    assert (x + c * 1.0 - s * 2.0, y + s * 1.0 + c * 2.0) == pytest.approx((5.0, 3.0))
    assert yaw == pytest.approx(0.5)
