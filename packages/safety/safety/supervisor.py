"""Safety rules as plain Python (no ROS), so every rule is unit-tested.

State, in priority order:
  estop  /estop true seen; latched until an explicit /estop false
  stop   person within stop_distance, or /people stale (fail-safe), or no /people yet
  slow   person within slow_distance
  ok     otherwise
The stop threshold is speed-dependent (speed and separation monitoring, ISO/TS 15066 with
the approach speed K of ISO 13855): stop when the person is within
  stop_distance + extra + K * (reaction + T_stop(v)) + v * reaction + v^2 / (2 decel),
T_stop(v) = max(0, v - still) / decel, so the robot is at rest (below `still`) before a person
walking at K reaches stop_distance (decision 2026-11-04). v is the speed the robot would have
at the level: the larger of the speed measured from the pose over speed_window_s and the
/cmd_vel_nav speed (scaled by slow_factor for slow). So slow is allowed beyond the threshold
at the slow speed, ok beyond slow_distance and the threshold at full speed; the levels don't
depend on the supervisor's own output (no stop / slow oscillation).
Leaving stop or slow needs the person beyond the threshold + release_margin for
release_hold_s (hysteresis). Output velocity: zero in estop/stop or when /cmd_vel_nav is
stale, scaled by slow_factor in slow, passed through in ok.
Geofence (any state): within geofence_margin of the polygon's edge, or outside it, the
outward velocity component is removed; no geofence or no robot pose -> zero linear velocity
(turning stays allowed). The geofence never changes the state.
"""

import math
from dataclasses import dataclass

OK, SLOW, STOP, ESTOP = 'ok', 'slow', 'stop', 'estop'
ZERO = (0.0, 0.0, 0.0)
_RANK = {OK: 0, SLOW: 1, STOP: 2}


@dataclass
class Params:
    slow_distance_m: float = 3.0
    stop_distance_m: float = 1.5
    slow_factor: float = 0.3
    release_margin_m: float = 0.3
    release_hold_s: float = 1.0
    people_timeout_s: float = 0.5
    cmd_timeout_s: float = 0.5
    geofence_margin_m: float = 0.3
    # speed-dependent stop threshold; all zero = the plain stop_distance_m
    approach_speed_mps: float = 0.0
    reaction_time_s: float = 0.0
    robot_decel_mps2: float = 0.0
    still_speed_mps: float = 0.0
    separation_extra_m: float = 0.0
    speed_window_s: float = 0.5

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: float(v) for k, v in d.items() if k in cls.__dataclass_fields__})


class Supervisor:

    def __init__(self, params):
        self.p = params
        self.estop = False
        self.people_t = None       # time of the last /people message
        self.nearest = None        # nearest person distance in it (None = nobody)
        self.cmd_t = None
        self.cmd = ZERO            # (vx, vy, wz)
        self.level = STOP          # ok / slow / stop, with hysteresis
        self.clear_since = None    # since when the person has been far enough to step down
        self.fence = None          # [(x, y)] in map
        self.pose = None           # (x, y, yaw) of base_link in map, None = unknown
        self.track = []            # [(t, x, y)] over speed_window_s, for the measured speed

    def on_geofence(self, polygon):
        self.fence = [tuple(p) for p in polygon] if len(polygon) >= 3 else None

    def on_pose(self, pose, t=None):
        """Current robot pose, or None when it isn't known (or too old); t: its time (for the speed)."""
        self.pose = pose
        if pose is None or t is None:
            return
        self.track.append((t, pose[0], pose[1]))
        while len(self.track) > 2 and t - self.track[1][0] >= self.p.speed_window_s:
            self.track.pop(0)

    def speed(self):
        """Measured robot planar speed over the window (0 without a pose track)."""
        if len(self.track) >= 2:
            (t0, x0, y0), (t1, x1, y1) = self.track[0], self.track[-1]
            if t1 - t0 > 1e-6:
                return math.hypot(x1 - x0, y1 - y0) / (t1 - t0)
        return 0.0

    def limits(self, t):
        """(stop below, ok from): person distances for the levels, at the speed each level allows."""
        fresh = self.cmd_t is not None and t - self.cmd_t <= self.p.cmd_timeout_s
        v_cmd = math.hypot(self.cmd[0], self.cmd[1]) if fresh else 0.0
        v = self.speed()
        stop = self.stop_threshold(max(v, v_cmd * self.p.slow_factor))
        return stop, max(self.p.slow_distance_m, self.stop_threshold(max(v, v_cmd)))

    def stop_threshold(self, v):
        """Person distance below which a robot at speed v must stop (see the module doc)."""
        p = self.p
        d = p.stop_distance_m + p.separation_extra_m
        if p.approach_speed_mps <= 0:
            return d
        t_stop = max(0.0, v - p.still_speed_mps) / p.robot_decel_mps2 if p.robot_decel_mps2 > 0 else 0.0
        braking = v * v / (2 * p.robot_decel_mps2) if p.robot_decel_mps2 > 0 else 0.0
        return d + p.approach_speed_mps * (p.reaction_time_s + t_stop) + v * p.reaction_time_s + braking

    def on_estop(self, active):
        self.estop = bool(active)

    def on_people(self, t, distances):
        self.people_t = t
        self.nearest = min(distances) if distances else None

    def on_cmd(self, t, vx, vy, wz):
        self.cmd_t, self.cmd = t, (vx, vy, wz)

    def _raw_level(self, t):
        """Level from the current inputs alone, without hysteresis."""
        if self.people_t is None or t - self.people_t > self.p.people_timeout_s:
            return STOP
        if self.nearest is None:
            return OK
        stop, ok = self.limits(t)
        return STOP if self.nearest < stop else SLOW if self.nearest < ok else OK

    def _release_distance(self, t):
        stop, ok = self.limits(t)
        return (stop if self.level == STOP else ok) + self.p.release_margin_m

    def step(self, t):
        """Advance to time t; returns (state, (vx, vy, wz)) to publish."""
        raw = self._raw_level(t)
        if _RANK[raw] >= _RANK[self.level]:
            self.level, self.clear_since = raw, None          # escalate at once
        else:
            stale = self.people_t is None or t - self.people_t > self.p.people_timeout_s
            far = not stale and (self.nearest is None or self.nearest > self._release_distance(t))
            if not far:
                self.clear_since = None
            elif self.clear_since is None:
                self.clear_since = t
            elif t - self.clear_since >= self.p.release_hold_s:
                self.level, self.clear_since = raw, None
        if self.estop:
            return ESTOP, ZERO
        if self.level == STOP:
            return STOP, ZERO
        if self.cmd_t is None or t - self.cmd_t > self.p.cmd_timeout_s:
            return self.level, ZERO
        cmd = tuple(v * self.p.slow_factor for v in self.cmd) if self.level == SLOW else self.cmd
        return self.level, self.fenced(cmd)

    def fenced(self, cmd):
        vx, vy, wz = cmd
        if self.fence is None or self.pose is None:
            return 0.0, 0.0, wz
        vx, vy = geofence_filter((vx, vy), self.pose, self.fence, self.p.geofence_margin_m)
        return vx, vy, wz


def inside_polygon(x, y, poly):
    inside = False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def closest_on_segment(px, py, a, b):
    (ax, ay), (bx, by) = a, b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return ax + t * dx, ay + t * dy


def geofence_filter(v_body, pose, poly, margin):
    """Remove the outward part of a body-frame velocity near or outside the polygon."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    vx, vy = c * v_body[0] - s * v_body[1], s * v_body[0] + c * v_body[1]      # into map
    inside = inside_polygon(x, y, poly)
    edges = list(zip(poly, poly[1:] + poly[:1]))
    near = []
    for a, b in edges:
        qx, qy = closest_on_segment(x, y, a, b)
        d = math.hypot(qx - x, qy - y)
        near.append((d, qx, qy, a, b))
    if not inside:
        near = [min(near)]                                  # outside: only the nearest edge
    for d, qx, qy, a, b in near:
        if inside and d >= margin:
            continue
        if d > 1e-9:
            nx, ny = ((qx - x) / d, (qy - y) / d) if inside else ((x - qx) / d, (y - qy) / d)
        else:                                               # on the edge: use the edge normal
            ex, ey = b[0] - a[0], b[1] - a[1]
            L = math.hypot(ex, ey)
            nx, ny = ey / L, -ex / L
            if inside_polygon(x + 0.01 * nx, y + 0.01 * ny, poly):
                nx, ny = -nx, -ny
        out = vx * nx + vy * ny
        if out > 0:
            vx, vy = vx - out * nx, vy - out * ny
    return c * vx + s * vy, -s * vx + c * vy                                    # back to body
