"""Own planar EKF core (localization/ekf_core.py, B4)."""

import math

import numpy as np

from localization.ekf_core import LocalEnu, PlanarEkf, wrap

Q = [1e-3, 1e-3, 1e-3, 0.5, 0.5, 0.3]


def test_local_enu_east_and_north():
    enu = LocalEnu(41.3784, 2.1925)
    e, n, u = enu(41.3784, 2.1925 + 25.0 / (111320.0 * math.cos(math.radians(41.3784))))
    assert abs(e - 25.0) < 0.1 and abs(n) < 1e-3 and abs(u) < 0.01
    e, n, _ = enu(41.3784 + 20.0 / 111034.0, 2.1925)       # ~111.03 km per degree of latitude at 41.4 N
    assert abs(n - 20.0) < 0.05 and abs(e) < 1e-3


def drive(ekf, seconds, v=(0.3, 0.0, 0.0), dt=0.02, imu_yaw=None, gnss=None, rng=None):
    """Feed body velocity (and IMU, GNSS) for a robot driving at v from the filter's start."""
    truth = list(ekf.pose())
    t = ekf.t or 0.0
    for k in range(int(seconds / dt)):
        t += dt
        c, s = math.cos(truth[2]), math.sin(truth[2])
        truth = [truth[0] + (c * v[0] - s * v[1]) * dt, truth[1] + (s * v[0] + c * v[1]) * dt,
                 wrap(truth[2] + v[2] * dt)]
        ekf.update_body_velocity(t, v, (0.01, 0.01, 0.01))
        ekf.update_imu(t, truth[2], v[2], 0.035 ** 2, 1e-6)
        if gnss is not None and k % 5 == 0:
            e, n = truth[0] + rng.normal(0, gnss), truth[1] + rng.normal(0, gnss)
            ekf.update_gnss(t, e, n, gnss ** 2, gnss ** 2)
    return truth


def test_dead_reckoning_straight_and_turning():
    ekf = PlanarEkf(Q, x0=(1.0, 2.0, 0.0))
    truth = drive(ekf, 10.0, v=(0.3, 0.0, 0.0))
    assert np.allclose(ekf.pose(), truth, atol=0.02)
    truth = drive(ekf, 5.0, v=(0.2, 0.0, 0.3))
    assert np.allclose(ekf.pose()[:2], truth[:2], atol=0.05) and abs(wrap(ekf.pose()[2] - truth[2])) < 0.02


def test_gnss_pulls_a_wrong_start_in():
    rng = np.random.default_rng(0)
    ekf = PlanarEkf(Q, x0=(5.0, 5.0, 0.0), p0=(5.0, 5.0, 0.1))     # true start (0, 0)
    ekf.s[:2] = (5.0, 5.0)
    truth = [0.0, 0.0, 0.0]
    ekf.s[:3] = (5.0, 5.0, 0.0)
    t = 0.0
    for k in range(3000):
        t += 0.02
        ekf.update_body_velocity(t, (0.0, 0.0, 0.0), (0.01, 0.01, 0.01))
        ekf.update_imu(t, 0.0, 0.0, 0.035 ** 2, 1e-6)
        if k % 5 == 0:
            ekf.update_gnss(t, truth[0] + rng.normal(0, 1.5), truth[1] + rng.normal(0, 1.5), 2.25, 2.25)
    assert math.hypot(*ekf.pose()[:2]) < 0.5


def test_gnss_outlier_is_gated_out():
    ekf = PlanarEkf(Q, p0=(0.1, 0.1, 0.01))
    ekf.update_body_velocity(0.0, (0, 0, 0), (0.01, 0.01, 0.01))
    assert not ekf.update_gnss(0.1, 30.0, 0.0, 2.25, 2.25)          # 30 m off, sigma 1.5 m
    assert ekf.gnss_rejected == 1 and math.hypot(*ekf.pose()[:2]) < 1e-6
    assert ekf.update_gnss(0.2, 1.0, 0.0, 2.25, 2.25)


def test_lever_arm_is_applied():
    """The antenna 0.2 m ahead of base_link, robot facing north at (0, 0): fixes at (0, 0.2)."""
    ekf = PlanarEkf(Q, x0=(0.0, 0.0, math.pi / 2), p0=(1.0, 1.0, 0.001))
    for k in range(200):
        ekf.update_imu(k * 0.05, math.pi / 2, 0.0, 1e-6, 1e-6)
        ekf.update_gnss(k * 0.05, 0.0, 0.2, 0.01, 0.01, lever=(0.2, 0.0))
    assert np.allclose(ekf.pose()[:2], (0.0, 0.0), atol=0.02)


def test_noisy_gnss_averages_down():
    rng = np.random.default_rng(1)
    ekf = PlanarEkf(Q, x0=(0.0, 0.0, 0.0), p0=(0.5, 0.5, 0.05))
    errs = []
    for _ in range(6):
        truth = drive(ekf, 10.0, v=(0.3, 0.0, 0.05), gnss=1.5, rng=rng)
        errs.append(math.dist(ekf.pose()[:2], truth[:2]))
    assert max(errs) < 1.0 and np.mean(errs) < 0.6
