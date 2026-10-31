"""Safety rules as plain Python (no ROS), so every rule is unit-tested.

State, in priority order:
  estop  /estop true seen; latched until an explicit /estop false
  stop   person within stop_distance, or /people stale (fail-safe), or no /people yet
  slow   person within slow_distance
  ok     otherwise
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

    def on_geofence(self, polygon):
        self.fence = [tuple(p) for p in polygon] if len(polygon) >= 3 else None

    def on_pose(self, pose):
        """Current robot pose, or None when it isn't known (or too old)."""
        self.pose = pose

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
        if self.nearest is None or self.nearest >= self.p.slow_distance_m:
            return OK
        return STOP if self.nearest < self.p.stop_distance_m else SLOW

    def _release_distance(self):
        d = self.p.stop_distance_m if self.level == STOP else self.p.slow_distance_m
        return d + self.p.release_margin_m

    def step(self, t):
        """Advance to time t; returns (state, (vx, vy, wz)) to publish."""
        raw = self._raw_level(t)
        if _RANK[raw] >= _RANK[self.level]:
            self.level, self.clear_since = raw, None          # escalate at once
        else:
            stale = self.people_t is None or t - self.people_t > self.p.people_timeout_s
            far = not stale and (self.nearest is None or self.nearest > self._release_distance())
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
