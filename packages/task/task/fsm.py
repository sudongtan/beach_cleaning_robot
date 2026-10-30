"""Cleaning task state machine as plain Python (no ROS). The node supplies `io` and calls
tick() at a fixed rate with a Snapshot; tests use a fake io.

SEARCH -> APPROACH -> PICK -> CLASSIFY -> STORE (or PUT_BACK) -> SEARCH ...
       -> (basket full, or search finished) GO_TO_BIN -> PLACE (one item per call) -> SEARCH
       -> DONE when the search is finished, no trash is left and the basket is empty.

io interface (actions return handles with done() and result()):
  nav_to(x, y, yaw), cancel_nav()          result: bool
  pick(item_id, (x, y, z))                 result: (success, message)
  classify()                               result: (material, bin_id, confidence)
  place(target, item_id, bin_id, (x, y, z, yaw))  target: 'compartment' | 'bin' | 'ground'
                                           result: (success, message)
  update(item_id, status, material='', bin_id='', skip_reason='') -> bool    (world model)
  cmd_vel(vx, vy, wz)                      final approach, through the safety supervisor
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


@dataclass
class Snapshot:
    t: float
    robot: tuple                    # (x, y, yaw) in map
    items: dict                     # id -> ItemView
    compartments: dict              # bin id -> [item ids]
    load_kg: float = 0.0
    safety_ok: bool = True


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


def to_robot(robot, x, y):
    """Map point -> robot frame (x forward, y left)."""
    rx, ry, ryaw = robot
    dx, dy = x - rx, y - ry
    c, s = math.cos(ryaw), math.sin(ryaw)
    return c * dx + s * dy, -s * dx + c * dy


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
    stuck: set = field(default_factory=set)      # items that could not be placed: left in the basket

    # --- helpers ------------------------------------------------------------------------
    def go(self, state, snap, item=None, sub=''):
        self.state, self.sub, self.pending = state, sub, None
        if item is not None:
            if item != self.item:
                self.item_since = snap.t
                self.nav_fails = 0
            self.item = item
        self.io.log({'t': round(snap.t, 2), 'event': 'state', 'state': state, 'item': self.item})

    def result(self):
        """Result of the pending action once done, else None."""
        if self.pending is None or not self.pending.done():
            return None
        r, self.pending = self.pending.result(), None
        return r

    def skip(self, snap, reason):
        self.io.update(self.item, SKIPPED, skip_reason=reason)
        self.io.log({'t': round(snap.t, 2), 'event': 'skipped', 'item': self.item, 'reason': reason})
        self.io.cmd_vel(0.0, 0.0, 0.0)
        self.item = ''
        self.go('SEARCH', snap)

    def nav(self, snap, x, y, yaw):
        self.pending = self.io.nav_to(x, y, yaw)
        self.nav_since = snap.t

    def nearest_trash(self, snap):
        rx, ry, _ = snap.robot
        todo = [it for it in snap.items.values() if it.status == SEEN and it.object_class == 'trash']
        return min(todo, key=lambda it: math.hypot(it.x - rx, it.y - ry), default=None)

    def basket_items(self, snap):
        """(bin id, item id) in the basket that can still be placed."""
        return [(c, i) for c in COMPARTMENTS for i in snap.compartments.get(c, []) if i not in self.stuck]

    # --- main loop ----------------------------------------------------------------------
    def tick(self, snap):
        if self.state == 'DONE':
            return
        if not snap.safety_ok and self.pending is None:
            if self.state == 'APPROACH' and self.sub == 'final':
                self.io.cmd_vel(0.0, 0.0, 0.0)
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
        lo, hi = self.grasp['reach_x_m']
        if lo <= x <= hi and abs(y) <= self.grasp['reach_y_max_m']:
            self.io.cmd_vel(0.0, 0.0, 0.0)
            if self.settle_since is None:
                self.settle_since = snap.t
            if snap.t - self.settle_since >= fa['settle_s']:
                self.settle_since = None
                self.go('PICK', snap)
            return
        self.settle_since = None
        clip = lambda v, m: max(-m, min(m, v))         # noqa: E731
        bearing = math.atan2(y, x)
        self.io.cmd_vel(clip(fa['gain'] * (x - fa['target_x_m']), fa['max_vx']),
                        clip(fa['gain'] * y, fa['max_vy']),
                        clip(fa['gain'] * bearing, fa['max_wz']) if x > 0.2 else 0.0)

    def state_pick(self, snap):
        it = snap.items.get(self.item)
        if self.pending is None:
            if it is None or it.status != SEEN:
                return self.go('SEARCH', snap)
            self.pending = self.io.pick(self.item, (it.x, it.y, it.z))
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
        if self.pending is None:
            self.unload = self.basket_items(snap)
            if not self.unload:
                return self.go('SEARCH', snap)
            bx, by, _, byaw = self.bins[self.unload[0][0]]
            d = self.cfg['bin_standoff_m']
            return self.nav(snap, bx + d * math.cos(byaw), by + d * math.sin(byaw), wrap(byaw + math.pi))
        r = self.result()
        if r is None and snap.t - self.nav_since > self.cfg['nav_timeout_s']:
            self.io.cancel_nav()
            self.pending, r = None, False
        if r is True:
            self.nav_fails = 0
            self.go('PLACE', snap)
        elif r is False:
            self.nav_fails += 1
            self.io.log({'t': round(snap.t, 2), 'event': 'bin_nav_failed', 'bin': self.unload[0][0]})
            if self.nav_fails > self.cfg['nav_retries']:
                stuck = [i for b, i in self.unload if b == self.unload[0][0]]
                self.stuck.update(stuck)
                self.io.log({'t': round(snap.t, 2), 'event': 'left_in_basket', 'items': stuck})
                self.nav_fails = 0
                self.go('GO_TO_BIN', snap)

    def state_place(self, snap):
        if self.pending is None:
            here = [u for u in self.unload if u[0] == self.unload[0][0]] if self.unload else []
            if not here:
                return self.go('GO_TO_BIN', snap)       # next bin, or back to SEARCH when empty
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
        else:
            self.attempts[self.item] = self.attempts.get(self.item, 0) + 1
            if self.attempts[self.item] < self.cfg['grasp_retries']:
                return                                  # retry the same item
        self.unload = [u for u in self.unload if u[1] != self.item]
        if not ok:
            self.stuck.add(self.item)
            self.io.log({'t': round(snap.t, 2), 'event': 'left_in_basket', 'items': [self.item]})
        self.item = ''
