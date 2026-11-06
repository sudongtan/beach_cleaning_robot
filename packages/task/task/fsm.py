"""Cleaning task state machine as plain Python (no ROS). The node supplies `io` and calls
tick() at a fixed rate with a Snapshot; tests use a fake io.

SEARCH -> APPROACH -> PICK -> CLASSIFY -> STORE (or PUT_BACK) -> SEARCH ...
       -> (basket full, or search finished) GO_TO_BIN -> PLACE (one item per call; then a sideways
          step to the bin beside it and PLACE there (bin_placement: step), or back out to
          bin_standoff_m, since Nav2 won't start facing the bin) -> GO_TO_BIN (next bin) -> SEARCH
       -> DONE when the search is finished, no trash is left and the basket is empty.
       -> FELL (terminal) as soon as the body tilt exceeds fall_tilt_deg.
Before any Nav2 goal (nav_start: escape): unless the robot can turn on the spot without its footprint's
outline touching a lethal map cell, the task first gets to such a pose (Escape), then sends the goal.

io interface (actions return handles with done() and result()):
  nav_to(x, y, yaw), cancel_nav()          result: bool
  pick(item_id, (x, y, z, yaw), size)      result: (success, message)   size: bbox (x, y, z)
  classify()                               result: (material, bin_id, confidence)
  place(target, item_id, bin_id, (x, y, z, yaw))  target: 'compartment' | 'bin' | 'ground'
                                           result: (success, message)
  update(item_id, status, material='', bin_id='', skip_reason='') -> bool    (world model)
  cmd_vel(vx, vy, wz)                      final approach / bin step / leaving a bin, through the safety supervisor
  log(record)                              event log (dict)

While /safety/state != ok no new action starts and the final approach sends zero velocity
(the supervisor and the grasping node stop motion themselves).
"""

import math
from dataclasses import dataclass, field

SEEN, COLLECTED, PLACED, SKIPPED = 0, 1, 2, 3
COMPARTMENTS = ('recyclable', 'residual')


@dataclass
class ItemView:
    id: str
    object_class: str
    x: float
    y: float
    z: float
    status: int = SEEN
    bin_id: str = ''
    yaw: float = 0.0
    size: tuple = (0.0, 0.0, 0.0)   # bbox (x, y, z = height); zeros = unknown


@dataclass
class Snapshot:
    t: float
    robot: tuple                    # (x, y, yaw) in map
    items: dict                     # id -> ItemView
    compartments: dict              # bin id -> [item ids]
    load_kg: float = 0.0
    safety_ok: bool = True
    tilt_deg: float = 0.0           # body tilt from vertical


@dataclass
class Capacity:
    max_items_per_compartment: int
    max_load_kg: float
    default_item_mass_kg: float

    def can_store(self, snap, bin_id):
        return (len(snap.compartments.get(bin_id, [])) < self.max_items_per_compartment
                and snap.load_kg + self.default_item_mass_kg <= self.max_load_kg + 1e-9)

    def full(self, snap):
        return not any(self.can_store(snap, c) for c in COMPARTMENTS)


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def min_planar(vx, vy, vmin):
    """Scale a planar command (vx, vy) that is non-zero but slower than vmin up to vmin, same
    direction (vmin 0 = off). For walking controllers that don't move on small commands."""
    v = math.hypot(vx, vy)
    if v == 0.0 or v >= vmin:
        return vx, vy
    return vx * vmin / v, vy * vmin / v


def to_robot(robot, x, y):
    """Map point -> robot frame (x forward, y left)."""
    rx, ry, ryaw = robot
    dx, dy = x - rx, y - ry
    c, s = math.cos(ryaw), math.sin(ryaw)
    return c * dx + s * dy, -s * dx + c * dy


class Escape:
    """Pending Nav2 goal behind an escape (Task.escape_tick drives it): not done while escaping,
    then done / result / cancel of the Nav2 goal itself."""

    def __init__(self, task, goal, start):
        self.task, self.goal, self.inner = task, goal, None
        self.last, self.travelled, self.stage = tuple(start), 0.0, 0     # 0 trail, 1 straight, 2 turn, 3 back
        self.target, self.best, self.since = None, None, 0.0    # progress towards the current trail pose

    def track(self, robot):
        self.travelled += math.dist(self.last[:2], robot[:2])
        self.last = tuple(robot)

    def stalled(self, target, dist, heading_err, t, cfg):
        """No progress towards target (distance by progress_min_m or heading by progress_min_rad) for
        progress_timeout_s; a new target restarts the clock."""
        if target != self.target or self.best is None:
            self.target, self.best, self.since = target, (dist, heading_err), t
            return False
        if dist < self.best[0] - cfg['progress_min_m'] or heading_err < self.best[1] - cfg['progress_min_rad']:
            self.best, self.since = (min(dist, self.best[0]), min(heading_err, self.best[1])), t
            return False
        return t - self.since > cfg['progress_timeout_s']

    def done(self):
        return self.inner is not None and self.inner.done()

    def result(self):
        return self.inner.result()

    def cancel(self):
        if self.inner is None:
            self.task.stop()


@dataclass
class Task:
    io: object
    cfg: dict                       # config/task.yaml task.fsm
    grasp: dict                     # config/grasping.yaml grasping (reach envelope)
    capacity: Capacity
    waypoints: list                 # search path [(x, y)]
    bins: dict                      # bin id -> (x, y, rim_z, yaw): front of the bin is its +x
    fallback_bin: str = 'residual'
    confidence_threshold: float = 0.6
    allowed: object = None          # allowed(x, y): may the robot go for an item there (free map cell
                                    # inside the geofence)? None = everywhere
    footprint: list = None          # Nav2's footprint polygon [(x, y)] in the robot frame; None = no escape
    state: str = 'SEARCH'
    wp: int = 0
    item: str = ''
    sub: str = ''                   # step inside a state
    pending: object = None
    attempts: dict = field(default_factory=dict)
    nav_fails: int = 0
    item_since: float = 0.0
    settle_since: float = None
    nav_since: float = 0.0
    material: str = ''
    bin_id: str = ''
    unload: list = field(default_factory=list)   # (bin id, item id) still to place
    at_bin: str = ''                             # the bin GO_TO_BIN / PLACE is working at
    leave_since: float = 0.0
    step_to: str = ''                            # PLACE step: the bin beside at_bin the robot steps to
    stuck: set = field(default_factory=set)      # items that could not be placed: left in the basket
    cmd: tuple = (0.0, 0.0, 0.0)                 # last own /cmd_vel_nav command (rate-limited) and its time
    cmd_t: float = None
    best_dist: float = None                      # final approach: closest distance so far ...
    best_since: float = 0.0                      # ... and since when
    trail: list = field(default_factory=list)    # own moves since the last Nav2 goal: robot poses
    bin_approach_fails: int = 0                  # bin final approaches without progress in a row

    # --- helpers ------------------------------------------------------------------------
    def go(self, state, snap, item=None, sub=''):
        self.state, self.sub, self.pending = state, sub, None
        if item is not None:
            if item != self.item:
                self.item_since = snap.t
                self.nav_fails = 0
                it = snap.items.get(item)
                if it is not None:                       # where the target is, for debugging runs
                    self.io.log({'t': round(snap.t, 2), 'event': 'target', 'item': item,
                                 'x': round(it.x, 3), 'y': round(it.y, 3), 'z': round(it.z, 3),
                                 'robot': [round(v, 3) for v in snap.robot]})
            self.item = item
        self.io.log({'t': round(snap.t, 2), 'event': 'state', 'state': state, 'item': self.item})

    def stop(self):
        """Zero command now (arrived, paused, fallen)."""
        self.io.cmd_vel(0.0, 0.0, 0.0)
        self.cmd, self.cmd_t = (0.0, 0.0, 0.0), None

    def move(self, snap, vx, vy, wz):
        """Command (vx, vy, wz), changed from the last own command by at most final_approach
        max_accel per second (like Nav2's velocity smoother; Nav2's commands are already smoothed)."""
        dt = 0.1 if self.cmd_t is None else min(max(snap.t - self.cmd_t, 0.0), 0.5)
        vx, vy = min_planar(vx, vy, self.cfg['final_approach'].get('min_speed_mps', 0.0))
        lim = self.cfg['final_approach']['max_accel']
        self.cmd = tuple(c + max(-a * dt, min(a * dt, w - c)) for c, w, a in zip(self.cmd, (vx, vy, wz), lim))
        self.cmd_t = snap.t
        self.io.cmd_vel(*self.cmd)
        if not isinstance(self.pending, Escape):
            self.record(snap.robot)

    def record(self, pose):
        """Add an own-move pose to the trail. Back near an older pose (a back-out after going in), the
        trail is cut back to it, so retracing never goes forward again (towards a bin or item)."""
        step = self.cfg['escape']['trail_step_m']
        for i, p in enumerate(self.trail):
            if math.dist(p[:2], pose[:2]) < step:
                del self.trail[i + 1:]
                return
        self.trail.append(tuple(pose))

    def result(self):
        """Result of the pending action once done, else None."""
        if self.pending is None or not self.pending.done():
            return None
        r, self.pending = self.pending.result(), None
        return r

    def skip(self, snap, reason):
        self.io.update(self.item, SKIPPED, skip_reason=reason)
        self.io.log({'t': round(snap.t, 2), 'event': 'skipped', 'item': self.item, 'reason': reason})
        self.stop()
        self.item = ''
        self.best_dist = None
        self.go('SEARCH', snap)

    def nav(self, snap, x, y, yaw):
        self.nav_since = snap.t
        if self.cfg['nav_start'] == 'escape' and self.footprint and not self.turn_free(snap.robot):
            self.io.log({'t': round(snap.t, 2), 'event': 'escape', 'robot': [round(v, 3) for v in snap.robot],
                         'trail': len(self.trail)})
            self.pending = Escape(self, (x, y, yaw), snap.robot)
            return
        self.trail = []
        self.pending = self.io.nav_to(x, y, yaw)

    def outline_clear(self, pose):
        """Is the footprint's outline at pose (x, y, yaw), grown by outline_margin_m, on free map cells
        only (as Nav2's footprintCost: the polygon's edges, a cell off the map counts as lethal)?"""
        if self.allowed is None:
            return True
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        step, m = self.cfg['escape']['outline_step_m'], self.cfg['escape']['outline_margin_m']
        pts = [(px + math.copysign(m, px), py + math.copysign(m, py)) for px, py in self.footprint]
        for (ax, ay), (bx, by) in zip(pts, pts[1:] + pts[:1]):
            n = max(1, math.ceil(math.hypot(bx - ax, by - ay) / step))
            for k in range(n):
                px, py = ax + (bx - ax) * k / n, ay + (by - ay) * k / n
                if not self.allowed(x + c * px - s * py, y + s * px + c * py):
                    return False
        return True

    def turn_free(self, pose):
        """Can the robot turn on the spot at pose: outline clear at every heading (turn_step_deg apart)?"""
        step = math.radians(self.cfg['escape']['turn_step_deg'])
        return all(self.outline_clear((pose[0], pose[1], k * step)) for k in range(int(round(2 * math.pi / step))))

    def straight_to_turn_free(self, robot):
        """Nearest turn-free pose straight ahead or behind, or None. On the way the outline may only touch
        lethal cells where it touches them now (getting out of them), never again once clear."""
        x, y, yaw = robot
        c, s = math.cos(yaw), math.sin(yaw)
        ec = self.cfg['escape']
        n = int(ec['straight_max_m'] / ec['trail_step_m'])
        start_clear = self.outline_clear(robot)
        blocked, was_clear = {1: False, -1: False}, {1: start_clear, -1: start_clear}
        for k in range(1, n + 1):
            for sign in (1, -1):
                if blocked[sign]:
                    continue
                d = sign * k * ec['trail_step_m']
                p = (x + c * d, y + s * d, yaw)
                if not self.outline_clear(p):
                    blocked[sign] = was_clear[sign]
                    continue
                was_clear[sign] = True
                if self.turn_free(p):
                    return p
        return None

    def escape_tick(self, snap):
        """Drive the pending Escape until the robot is turn-free: back along the trail (newest pose first,
        its heading); then straight to the nearest turn-free pose; then (best effort) turn to the nearest
        clear heading and back straight out back_m; at most max_m in all. Then send the Nav2 goal."""
        e, fa, ec = self.pending, self.cfg['final_approach'], self.cfg['escape']
        if not snap.safety_ok:
            self.stop()
            self.best_since = e.since = snap.t
            return
        e.track(snap.robot)
        free = self.turn_free(snap.robot)
        while self.trail and self.reached(snap.robot, self.trail[-1]):
            self.trail.pop()
        if not free and not self.trail:
            x, y, yaw = snap.robot
            if e.stage == 0:
                e.stage = 1
                p = self.straight_to_turn_free(snap.robot)
                if p is not None:
                    self.trail.append(p)
            if not self.trail and e.stage == 1 and not self.outline_clear(snap.robot):
                e.stage = 2
                turn = self.clear_heading(snap.robot)
                if turn is not None:
                    self.trail.append((x, y, turn))
            if not self.trail and e.stage == 2 and not self.outline_clear(snap.robot):
                e.stage = 3
                self.trail.append((x - ec['back_m'] * math.cos(yaw), y - ec['back_m'] * math.sin(yaw), yaw))
        stalled = False
        if not free and self.trail:
            tx, ty, tyaw = self.trail[-1]
            stalled = e.stalled(self.trail[-1], math.dist((tx, ty), snap.robot[:2]),
                                abs(wrap(tyaw - snap.robot[2])), snap.t, {**fa, **ec})
        if free or not self.trail or e.travelled > ec['max_m'] or stalled:
            self.stop()
            self.io.log({'t': round(snap.t, 2), 'event': 'escape_done', 'turn_free': free, 'stalled': stalled,
                         'clear': self.outline_clear(snap.robot), 'moved_m': round(e.travelled, 3),
                         'stage': e.stage, 'robot': [round(v, 3) for v in snap.robot]})
            self.trail = []
            e.inner = self.io.nav_to(*e.goal)
            self.nav_since = snap.t
            return
        tx, ty, tyaw = self.trail[-1]
        x, y = to_robot(snap.robot, tx, ty)
        clip = lambda v, m: max(-m, min(m, v))         # noqa: E731
        self.move(snap, clip(fa['gain'] * x, fa['max_vx']), clip(fa['gain'] * y, fa['max_vy']),
                  clip(fa['gain'] * wrap(tyaw - snap.robot[2]), fa['max_wz']))

    def reached(self, robot, pose):
        return (math.dist(pose[:2], robot[:2]) <= self.cfg['final_approach']['tolerance_m']
                and abs(wrap(pose[2] - robot[2])) <= self.cfg['escape']['heading_tol_rad'])

    def clear_heading(self, robot):
        """The heading nearest the robot's with a clear outline at its position (turn_step_deg steps,
        up to max_turn_deg each way; one step further when that is clear too, as a margin), or None."""
        ec = self.cfg['escape']
        x, y, yaw = robot
        step = math.radians(ec['turn_step_deg'])
        for k in range(1, int(ec['max_turn_deg'] // ec['turn_step_deg']) + 1):
            for sign in (1, -1):
                h = wrap(yaw + sign * k * step)
                if self.outline_clear((x, y, h)):
                    further = wrap(h + sign * step)
                    return further if self.outline_clear((x, y, further)) else h
        return None

    def nearest_trash(self, snap):
        rx, ry, _ = snap.robot
        todo = [it for it in snap.items.values() if it.status == SEEN and it.object_class == 'trash'
                and (self.allowed is None or self.allowed(it.x, it.y))]
        return min(todo, key=lambda it: math.hypot(it.x - rx, it.y - ry), default=None)

    def basket_items(self, snap):
        """(bin id, item id) in the basket that can still be placed."""
        return [(c, i) for c in COMPARTMENTS for i in snap.compartments.get(c, []) if i not in self.stuck]

    # --- main loop ----------------------------------------------------------------------
    def tick(self, snap):
        if self.state in ('DONE', 'FELL'):
            return
        if snap.tilt_deg > self.cfg['fall_tilt_deg']:
            if self.pending is not None:
                self.io.cancel_nav()
                if hasattr(self.pending, 'cancel'):        # a grasp action: stop the arm too
                    self.pending.cancel()
                self.pending = None
            self.stop()
            self.io.log({'t': round(snap.t, 2), 'event': 'fell', 'tilt_deg': round(snap.tilt_deg, 1),
                         'robot': [round(v, 3) for v in snap.robot], 'state': self.state, 'sub': self.sub,
                         'item': self.item})
            return self.go('FELL', snap)
        if isinstance(self.pending, Escape) and self.pending.inner is None:
            return self.escape_tick(snap)
        if not snap.safety_ok and self.pending is None:
            if (self.state, self.sub) in (('APPROACH', 'final'), ('GO_TO_BIN', 'final'), ('PLACE', 'leave'),
                                          ('PLACE', 'step')):
                self.stop()
                self.best_since = snap.t                    # a safety pause is not "no progress"

            return                                      # pause: start nothing new
        if self.item and self.state in ('APPROACH',) and snap.t - self.item_since > self.cfg['item_timeout_s']:
            if self.pending is not None:
                self.io.cancel_nav()
                self.pending = None
            return self.skip(snap, 'no_progress')
        getattr(self, 'state_' + self.state.lower())(snap)

    def state_search(self, snap):
        if self.pending is not None:                    # walking to a waypoint
            item = self.nearest_trash(snap) if not self.capacity.full(snap) else None
            if item is not None:
                self.io.cancel_nav()
                self.pending = None
                return self.go('APPROACH', snap, item.id, 'nav')
            r = self.result()
            if r is None and snap.t - self.nav_since > self.cfg['nav_timeout_s']:
                self.io.cancel_nav()
                self.pending, r = None, False
            if r is not None:
                self.wp += 1                            # reached or unreachable: next waypoint
            return
        full = self.capacity.full(snap)
        if self.basket_items(snap) and full:
            return self.go('GO_TO_BIN', snap)
        item = self.nearest_trash(snap) if not full else None   # full and nothing placeable: only search
        if item is not None:
            return self.go('APPROACH', snap, item.id, 'nav')
        if self.wp < len(self.waypoints):
            x, y = self.waypoints[self.wp]
            rx, ry, _ = snap.robot
            return self.nav(snap, x, y, math.atan2(y - ry, x - rx))
        if self.basket_items(snap):
            return self.go('GO_TO_BIN', snap)
        self.io.log({'t': round(snap.t, 2), 'event': 'done'})
        self.go('DONE', snap)

    def state_approach(self, snap):
        it = snap.items.get(self.item)
        if it is None or it.status != SEEN:
            return self.go('SEARCH', snap)
        if self.sub == 'nav':
            if self.pending is None:
                rx, ry, _ = snap.robot
                d = math.hypot(it.x - rx, it.y - ry)
                yaw = math.atan2(it.y - ry, it.x - rx)
                if d <= self.cfg['standoff_m'] + 0.3:   # already close: go straight to the final approach
                    self.sub = 'final'
                    return
                return self.nav(snap, it.x - self.cfg['standoff_m'] * math.cos(yaw),
                                it.y - self.cfg['standoff_m'] * math.sin(yaw), yaw)
            r = self.result()
            if r is None and snap.t - self.nav_since > self.cfg['nav_timeout_s']:
                self.io.cancel_nav()
                self.pending, r = None, False
            if r is True:
                self.sub = 'final'
            elif r is False:
                self.nav_fails += 1
                if self.nav_fails > self.cfg['nav_retries']:
                    return self.skip(snap, 'unreachable')
            return
        # final approach on the item position relative to the robot
        fa = self.cfg['final_approach']
        x, y = to_robot(snap.robot, it.x, it.y)
        dist = math.hypot(x - fa['target_x_m'], y)
        if self.best_dist is None or dist < self.best_dist - fa['progress_min_m']:
            self.best_dist, self.best_since = dist, snap.t
        elif snap.t - self.best_since > fa['progress_timeout_s']:
            self.best_dist = None
            return self.skip(snap, 'no_progress')
        # stop near the middle of the envelope: grasping checks it in 3D (body pitch shifts x)
        if abs(x - fa['target_x_m']) <= fa['tolerance_m'] and abs(y) <= fa['tolerance_m']:
            self.stop()
            if self.settle_since is None:
                self.settle_since = snap.t
            if snap.t - self.settle_since >= fa['settle_s']:
                self.settle_since = None
                self.best_dist = None
                self.go('PICK', snap)
            return
        self.settle_since = None
        clip = lambda v, m: max(-m, min(m, v))         # noqa: E731
        bearing = math.atan2(y, x)
        self.move(snap, clip(fa['gain'] * (x - fa['target_x_m']), fa['max_vx']),
                  clip(fa['gain'] * y, fa['max_vy']),
                  clip(fa['gain'] * bearing, fa['max_wz']) if x > 0.2 else 0.0)

    def state_pick(self, snap):
        it = snap.items.get(self.item)
        if self.pending is None:
            if it is None or it.status != SEEN:
                return self.go('SEARCH', snap)
            self.pending = self.io.pick(self.item, (it.x, it.y, it.z, it.yaw), it.size)
            return
        r = self.result()
        if r is None:
            return
        ok, msg = r
        self.io.log({'t': round(snap.t, 2), 'event': 'pick', 'item': self.item, 'success': ok, 'message': msg})
        if ok:
            return self.go('CLASSIFY', snap)
        self.attempts[self.item] = self.attempts.get(self.item, 0) + 1
        if self.attempts[self.item] >= self.cfg['grasp_retries']:
            return self.skip(snap, 'grasp_failed')
        self.go('APPROACH', snap, self.item, 'final')   # re-approach, then retry

    def state_classify(self, snap):
        if self.pending is None:
            self.pending = self.io.classify()
            return
        r = self.result()
        if r is None:
            return
        material, bin_id, conf = r
        self.io.log({'t': round(snap.t, 2), 'event': 'classify', 'item': self.item, 'material': material,
                     'bin': bin_id, 'confidence': conf})
        if material == 'distractor':
            return self.go('PUT_BACK', snap, sub='distractor')
        if conf < self.confidence_threshold or bin_id not in COMPARTMENTS:
            bin_id = self.fallback_bin                  # never risk the recycling bin
        if not self.capacity.can_store(snap, bin_id):
            return self.go('PUT_BACK', snap, sub='full')
        self.material, self.bin_id = material, bin_id
        self.go('STORE', snap)

    def state_store(self, snap):
        if self.pending is None:
            self.pending = self.io.place('compartment', self.item, self.bin_id, None)
            return
        r = self.result()
        if r is None:
            return
        ok, msg = r
        self.io.log({'t': round(snap.t, 2), 'event': 'store', 'item': self.item, 'bin': self.bin_id,
                     'success': ok, 'message': msg})
        if ok:
            self.io.update(self.item, COLLECTED, self.material, self.bin_id)
        elif msg.startswith('compartment_full'):
            # no room left that keeps it below the rim (still held): as a full count in CLASSIFY
            return self.go('PUT_BACK', snap, self.item, 'full')
        else:
            self.io.update(self.item, SKIPPED, skip_reason='store_failed')
        self.item = ''
        self.go('SEARCH', snap)

    def state_put_back(self, snap):
        if self.pending is None:
            self.pending = self.io.place('ground', self.item, '', None)
            return
        r = self.result()
        if r is None:
            return
        ok, _ = r
        if self.sub == 'distractor':
            self.io.update(self.item, SKIPPED, skip_reason='distractor')
        elif self.sub == 'full' and not self.basket_items(snap):
            self.io.update(self.item, SKIPPED, skip_reason='compartment_full')   # full of unplaceable items
        elif not ok:
            self.io.update(self.item, SKIPPED, skip_reason='put_back_failed')
        self.item = ''
        # its compartment was full: empty the basket first, then the item is picked again
        self.go('GO_TO_BIN' if self.sub == 'full' and self.basket_items(snap) else 'SEARCH', snap)

    def state_go_to_bin(self, snap):
        if self.sub == 'final':
            return self.bin_final_approach(snap)
        if self.pending is None:
            self.unload = self.basket_items(snap)
            if not self.unload:
                return self.go('SEARCH', snap)
            self.at_bin = self.unload[0][0]
            bx, by, _, byaw = self.bins[self.at_bin]
            d = self.cfg['bin_standoff_m']
            return self.nav(snap, bx + d * math.cos(byaw), by + d * math.sin(byaw), wrap(byaw + math.pi))
        r = self.result()
        if r is None and snap.t - self.nav_since > self.cfg['nav_timeout_s']:
            self.io.cancel_nav()
            self.pending, r = None, False
        if r is True:
            self.nav_fails = 0
            self.sub, self.best_dist = 'final', None
        elif r is False:
            self.nav_fails += 1
            self.io.log({'t': round(snap.t, 2), 'event': 'bin_nav_failed', 'bin': self.at_bin})
            if self.nav_fails > self.cfg['nav_retries']:
                stuck = [i for b, i in self.unload if b == self.at_bin]
                self.stuck.update(stuck)
                self.io.log({'t': round(snap.t, 2), 'event': 'left_in_basket', 'items': stuck})
                self.nav_fails = 0
                self.go('GO_TO_BIN', snap)

    def bin_final_approach(self, snap):
        """Straight in towards the bin until base_link is bin_place_distance_m from its centre."""
        fa = self.cfg['final_approach']
        bx, by, _, _ = self.bins[self.at_bin]
        x, y = to_robot(snap.robot, bx, by)
        err = x - self.cfg['bin_place_distance_m']
        if abs(err) <= 0.03 and abs(y) <= 0.05:
            self.stop()
            if self.settle_since is None:
                self.settle_since = snap.t
            if snap.t - self.settle_since >= fa['settle_s']:
                self.settle_since, self.best_dist, self.bin_approach_fails = None, None, 0
                self.go('PLACE', snap)
            return
        # no progress (a controller that doesn't make the small move): Nav2 to the standoff again,
        # after nav_retries leave this bin's items in the basket
        dist = math.hypot(err, y)
        if self.best_dist is None or dist < self.best_dist - fa['progress_min_m']:
            self.best_dist, self.best_since = dist, snap.t
        elif snap.t - self.best_since > fa['progress_timeout_s']:
            self.stop()
            self.best_dist, self.settle_since = None, None
            self.bin_approach_fails += 1
            self.io.log({'t': round(snap.t, 2), 'event': 'bin_approach_failed', 'bin': self.at_bin,
                         'robot': [round(v, 3) for v in snap.robot]})
            if self.bin_approach_fails > self.cfg['nav_retries']:
                stuck = [i for b, i in self.unload if b == self.at_bin]
                self.stuck.update(stuck)
                self.io.log({'t': round(snap.t, 2), 'event': 'left_in_basket', 'items': stuck})
                self.bin_approach_fails = 0
            return self.go('GO_TO_BIN', snap)
        self.settle_since = None
        clip = lambda v, m: max(-m, min(m, v))         # noqa: E731
        self.move(snap, clip(fa['gain'] * err, fa['max_vx']), clip(fa['gain'] * y, fa['max_vy']),
                  clip(fa['gain'] * math.atan2(y, x), fa['max_wz']))

    def state_place(self, snap):
        if self.sub == 'leave':
            return self.leave_bin(snap)
        if self.sub == 'step':
            return self.bin_step(snap)
        if self.pending is None:
            here = [u for u in self.unload if u[0] == self.at_bin]
            if not here:
                self.step_to = self.bin_beside(snap)
                if self.step_to:
                    self.best_dist = None
                    self.io.log({'t': round(snap.t, 2), 'event': 'bin_step', 'from': self.at_bin, 'to': self.step_to,
                                 'robot': [round(v, 3) for v in snap.robot]})
                    return self.go('PLACE', snap, sub='step')
                self.leave_since = snap.t
                return self.go('PLACE', snap, sub='leave')
            bin_id, iid = here[0]
            bx, by, rim, byaw = self.bins[bin_id]
            self.item = iid
            self.pending = self.io.place('bin', iid, bin_id, (bx, by, rim, byaw))
            return
        r = self.result()
        if r is None:
            return
        ok, msg = r
        bin_id = next(b for b, i in self.unload if i == self.item)
        self.io.log({'t': round(snap.t, 2), 'event': 'place', 'item': self.item, 'bin': bin_id,
                     'success': ok, 'message': msg})
        if ok:
            self.io.update(self.item, PLACED, bin_id=bin_id)
        elif msg.startswith('not_in_basket'):
            # it fell out (a2b_full1): free its compartment slot and look for it again; a detection
            # near where it was picked merges into it, elsewhere it is a new item
            self.io.update(self.item, SEEN)
            self.io.log({'t': round(snap.t, 2), 'event': 'lost_from_basket', 'item': self.item})
            self.unload = [u for u in self.unload if u[1] != self.item]
            self.item = ''
            return
        else:
            self.attempts[self.item] = self.attempts.get(self.item, 0) + 1
            if self.attempts[self.item] < self.cfg['grasp_retries']:
                return                                  # retry the same item
        self.unload = [u for u in self.unload if u[1] != self.item]
        if not ok:
            self.stuck.add(self.item)
            self.io.log({'t': round(snap.t, 2), 'event': 'left_in_basket', 'items': [self.item]})
        self.item = ''

    def bin_beside(self, snap):
        """With bin_placement `step`: the next bin still to unload into, if it stands beside at_bin
        (same yaw, offset only along at_bin's front), else ''."""
        if self.cfg['bin_placement'] != 'step':
            return ''
        bs = self.cfg['bin_step']
        bx, by, _, byaw = self.bins[self.at_bin]
        for b, _ in self.unload:
            if b == self.at_bin:
                continue
            ox, oy, _, oyaw = self.bins[b]
            along = math.cos(byaw) * (ox - bx) + math.sin(byaw) * (oy - by)
            side = -math.sin(byaw) * (ox - bx) + math.cos(byaw) * (oy - by)
            if (abs(wrap(oyaw - byaw)) < 0.1 and abs(along) < bs['along_tolerance_m']
                    and abs(side) <= bs['max_lateral_m']):
                return b
            return ''
        return ''

    def bin_step(self, snap):
        """Straight sideways from the place pose of at_bin to that of step_to, heading held (no turn
        and no back-out next to the bins), then the bin's final approach and PLACE. No progress
        -> back out and let Nav2 walk (as bin_placement `walk`)."""
        fa = self.cfg['final_approach']
        bx, by, _, byaw = self.bins[self.step_to]
        d = self.cfg['bin_place_distance_m']
        x, y = to_robot(snap.robot, bx + d * math.cos(byaw), by + d * math.sin(byaw))
        heading = wrap(byaw + math.pi - snap.robot[2])
        dist = math.hypot(x, y)
        if dist <= fa['tolerance_m']:
            self.stop()
            self.io.log({'t': round(snap.t, 2), 'event': 'bin_step_done', 'to': self.step_to,
                         'robot': [round(v, 3) for v in snap.robot]})
            self.at_bin, self.best_dist = self.step_to, None
            return self.go('GO_TO_BIN', snap, sub='final')
        if self.best_dist is None or dist < self.best_dist - fa['progress_min_m']:
            self.best_dist, self.best_since = dist, snap.t
        elif snap.t - self.best_since > fa['progress_timeout_s']:
            self.stop()
            self.io.log({'t': round(snap.t, 2), 'event': 'bin_step_failed', 'to': self.step_to,
                         'robot': [round(v, 3) for v in snap.robot]})
            self.best_dist = None
            self.leave_since = snap.t
            return self.go('PLACE', snap, sub='leave')
        clip = lambda v, m: max(-m, min(m, v))         # noqa: E731
        self.move(snap, clip(fa['gain'] * x, fa['max_vx']), clip(fa['gain'] * y, self.cfg['bin_step']['max_vy']),
                  clip(fa['gain'] * heading, fa['max_wz']))

    def leave_bin(self, snap):
        """Back straight out until base_link is bin_standoff_m from the bin centre, then GO_TO_BIN
        (next bin, or back to SEARCH): at the place distance the footprint is inside the bin's
        inflation and Nav2's controller refuses to move ("collision ahead"; reversing is off)."""
        fa = self.cfg['final_approach']
        bx, by, _, _ = self.bins[self.at_bin]
        x, _ = to_robot(snap.robot, bx, by)
        err = x - self.cfg['bin_standoff_m']            # negative while too close
        if err >= -fa['tolerance_m'] or snap.t - self.leave_since > self.cfg['bin_leave_timeout_s']:
            self.stop()
            return self.go('GO_TO_BIN', snap)
        self.move(snap, max(-fa['max_vx'], fa['gain'] * err), 0.0, 0.0)
