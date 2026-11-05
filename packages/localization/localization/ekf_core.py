"""Own planar EKF (localization:=own_ekf, B4): pure numpy, no ROS.

State s = [x, y, yaw, vx, vy, wz, k]: pose in the world frame (odom or map), velocity in the body frame
(x forward, y left), k = the leg odometry's speed scale (leg speed = k * true speed; measured 0.87-0.94
in Gazebo, B4 log). Constant-velocity prediction; k is a slow random walk, estimated only when
estimate_scale is on (it needs GNSS to be observable: map filter only). Measurements:
  - leg odometry: body velocity (k vx, k vy, wz)
  - IMU: yaw (world-referenced, ENU) and yaw rate
  - GNSS: the antenna's x, y in the world frame (local ENU about the datum), with the antenna's
    lever arm from base_link; rejected when its Mahalanobis distance exceeds the gate (chi-square,
    2 dof), so a single bad fix can't pull the estimate
The same class runs twice: without GNSS for odom (smooth, drifts), with GNSS for map.
Geodetic -> local ENU: WGS84 -> ECEF -> ENU at the datum (exact; no UTM grid convergence).
"""

import math

import numpy as np

# WGS84
A = 6378137.0
F = 1 / 298.257223563
E2 = F * (2 - F)


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def ecef(lat_deg, lon_deg, alt):
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    n = A / math.sqrt(1 - E2 * math.sin(lat) ** 2)
    return np.array([(n + alt) * math.cos(lat) * math.cos(lon),
                     (n + alt) * math.cos(lat) * math.sin(lon),
                     (n * (1 - E2) + alt) * math.sin(lat)])


class LocalEnu:
    """Geodetic (deg, deg, m) -> east, north, up (m) about a datum."""

    def __init__(self, lat_deg, lon_deg, alt=0.0):
        self.origin = ecef(lat_deg, lon_deg, alt)
        lat, lon = math.radians(lat_deg), math.radians(lon_deg)
        self.R = np.array([[-math.sin(lon), math.cos(lon), 0.0],
                           [-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)],
                           [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)]])

    def __call__(self, lat_deg, lon_deg, alt=0.0):
        return self.R @ (ecef(lat_deg, lon_deg, alt) - self.origin)


class PlanarEkf:

    def __init__(self, process_noise, x0=(0.0, 0.0, 0.0), p0=(1.0, 1.0, 0.1), gate_chi2=9.21,
                 estimate_scale=False, scale0=1.0, scale_std=0.1, scale_noise=1e-6):
        """process_noise: 6 spectral densities (per second) for x, y, yaw, vx, vy, wz;
        x0 = (x, y, yaw) start; p0 = their standard deviations; the scale k starts at scale0 (std
        scale_std, random walk scale_noise per s) and stays fixed unless estimate_scale."""
        self.q = np.append(np.asarray(process_noise, dtype=float), scale_noise if estimate_scale else 0.0)
        self.s = np.array([x0[0], x0[1], x0[2], 0.0, 0.0, 0.0, scale0])
        self.P = np.diag([p0[0] ** 2, p0[1] ** 2, p0[2] ** 2, 1.0, 1.0, 1.0,
                          scale_std ** 2 if estimate_scale else 0.0])
        self.t = None
        self.gate = gate_chi2
        self.gnss_rejected = 0
        self.gnss_used = 0

    def predict(self, t):
        if self.t is None:
            self.t = t
            return
        dt = t - self.t
        if dt <= 0:
            return                                # late message: update at the current state
        x, y, yaw, vx, vy, wz, k = self.s
        c, s = math.cos(yaw), math.sin(yaw)
        self.s = np.array([x + (c * vx - s * vy) * dt, y + (s * vx + c * vy) * dt, wrap(yaw + wz * dt), vx, vy, wz, k])
        F = np.eye(7)
        F[0, 2], F[0, 3], F[0, 4] = (-s * vx - c * vy) * dt, c * dt, -s * dt
        F[1, 2], F[1, 3], F[1, 4] = (c * vx - s * vy) * dt, s * dt, c * dt
        F[2, 5] = dt
        self.P = F @ self.P @ F.T + np.diag(self.q * dt)
        self.t = t

    def _update(self, r, H, R, angle_rows=()):
        for i in angle_rows:
            r[i] = wrap(r[i])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.s = self.s + K @ r
        self.s[2] = wrap(self.s[2])
        I_KH = np.eye(7) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T       # Joseph form: stays symmetric positive
        return float(r @ np.linalg.solve(S, r))

    def update_body_velocity(self, t, v, var):
        """Leg odometry: (vx, vy, wz) in the body frame, variances var (3); measures (k vx, k vy, wz)."""
        self.predict(t)
        vx, vy, k = self.s[3], self.s[4], self.s[6]
        H = np.zeros((3, 7))
        H[0, 3], H[1, 4], H[2, 5] = k, k, 1.0
        H[0, 6], H[1, 6] = vx, vy
        h = np.array([k * vx, k * vy, self.s[5]])
        self._update(np.asarray(v, float) - h, H, np.diag(var))

    def update_imu(self, t, yaw, wz, yaw_var, wz_var):
        self.predict(t)
        H = np.zeros((2, 7))
        H[0, 2] = H[1, 5] = 1.0
        self._update(np.array([yaw - self.s[2], wz - self.s[5]]), H, np.diag([yaw_var, wz_var]), angle_rows=(0,))

    def update_gnss(self, t, east, north, var_e, var_n, lever=(0.0, 0.0)):
        """Antenna position in the world frame; lever = antenna (x, y) in base_link. False if gated out."""
        self.predict(t)
        x, y, yaw = self.s[:3]
        c, s = math.cos(yaw), math.sin(yaw)
        lx, ly = lever
        h = np.array([x + c * lx - s * ly, y + s * lx + c * ly])
        H = np.zeros((2, 7))
        H[0, 0] = H[1, 1] = 1.0
        H[0, 2], H[1, 2] = -s * lx - c * ly, c * lx - s * ly
        r = np.array([east, north]) - h
        R = np.diag([var_e, var_n])
        d2 = float(r @ np.linalg.solve(H @ self.P @ H.T + R, r))
        if d2 > self.gate:
            self.gnss_rejected += 1
            return False
        self._update(r, H, R)
        self.gnss_used += 1
        return True

    def pose(self):
        return tuple(self.s[:3])

    def scale(self):
        return float(self.s[6])
