"""Safety rules as plain Python (no ROS), so every rule is unit-tested.

State, in priority order:
  estop  /estop true seen; latched until an explicit /estop false
  stop   person within stop_distance, or /people stale (fail-safe), or no /people yet
  slow   person within slow_distance
  ok     otherwise
Leaving stop or slow needs the person beyond the threshold + release_margin for
release_hold_s (hysteresis). Output velocity: zero in estop/stop or when /cmd_vel_nav is
stale, scaled by slow_factor in slow, passed through in ok.
"""

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
        if self.level == SLOW:
            return SLOW, tuple(v * self.p.slow_factor for v in self.cmd)
        return OK, self.cmd
