"""Task state machine against a fake world (no ROS): full runs, distractors, grasp retries,
unreachable items, safety pauses, a full basket, low confidence, failed placing, timeouts."""

import math
from pathlib import Path

import pytest
import yaml

from task.fsm import COLLECTED, PLACED, SEEN, SKIPPED, Capacity, ItemView, Snapshot, Task

REPO = Path(__file__).resolve().parents[3]
CFG = yaml.safe_load(open(REPO / 'config' / 'task.yaml'))['task']['fsm']
GRASP = yaml.safe_load(open(REPO / 'config' / 'grasping.yaml'))['grasping']
BINS = {'recyclable': (1.5, 13.3, 0.5, 0.0), 'residual': (1.5, 14.5, 0.5, 0.0)}
DT = 1.0 / CFG['rate_hz']


class Handle:
    def __init__(self, world, ticks, value):
        self.world, self.at, self.value = world, world.tick_no + ticks, value
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def done(self):
        return self.world.tick_no >= self.at

    def result(self):
        return self.value


class FakeWorld:
    """Robot, items and world model; `io` of the task. Items are discovered within 5 m."""

    def __init__(self, items, materials, pick_fail=None, nav_fail=None, place_fail=(), confidence=None, lost=(),
                 store_full=()):
        self.tick_no, self.t = 0, 0.0
        self.robot = [4.0, 14.0, 0.0]
        self.truth = {k: dict(v) for k, v in items.items()}        # id -> {cls, x, y}
        self.materials = materials
        self.known = {}
        self.compartments = {'recyclable': [], 'residual': []}
        self.pick_fail = dict(pick_fail or {})                      # id -> failures before success
        self.nav_fail = set(nav_fail or ())                         # item ids whose stand-off can't be reached
        self.place_fail = set(place_fail)
        self.lost = set(lost)                                       # fall out of the basket once, back where picked
        self.picked_at = {}
        self.store_full = set(store_full)                           # compartment_full once on storing
        self.confidence = confidence or {}
        self.held = None
        self.safe = lambda t: True
        self.log_records, self.calls_while_unsafe = [], 0
        self.cmd = (0.0, 0.0, 0.0)
        self.cmds = []                                              # (t, command) of every cmd_vel call
        self.target_item = None
        self.nav_from_bin = 0                                       # Nav2 goals sent while facing a bin up close

    def bin_dist(self, bin_id):
        bx, by, _, _ = BINS[bin_id]
        return math.hypot(self.robot[0] - bx, self.robot[1] - by)

    # --- io -----------------------------------------------------------------------------
    def _call(self):
        if not self.safe(self.t):
            self.calls_while_unsafe += 1

    def nav_to(self, x, y, yaw):
        self._call()
        if any(self.bin_dist(b) < CFG['bin_standoff_m'] - 0.05 for b in BINS):
            self.nav_from_bin += 1                                  # Nav2 refuses: lethal cells ahead
            return Handle(self, 5, False)
        near = [i for i in self.nav_fail if i in self.known and
                math.hypot(self.known[i].x - x, self.known[i].y - y) < 1.0]
        if near:
            return Handle(self, 5, False)
        self.robot = [x, y, yaw]                                    # arrives when the handle is done
        return Handle(self, 5, True)

    def cancel_nav(self):
        pass

    def pick(self, iid, pose, size):
        self._call()
        if self.pick_fail.get(iid, 0) > 0:
            self.pick_fail[iid] -= 1
            return Handle(self, 3, (False, 'attach_failed'))
        self.held = iid
        return Handle(self, 3, (True, 'ok'))

    def classify(self):
        self._call()
        m = self.materials[self.held]
        if m == 'distractor':
            return Handle(self, 2, ('distractor', '', 1.0))
        conf = self.confidence.get(self.held, 1.0)
        return Handle(self, 2, (m, 'residual' if m == 'residual' else 'recyclable', conf))

    def place(self, target, iid, bin_id, pose):
        self._call()
        if target == 'compartment' and iid in self.store_full:
            self.store_full.discard(iid)
            return Handle(self, 3, (False, 'compartment_full (pile 0.120)'))
        if target == 'bin' and iid in self.lost:
            self.lost.discard(iid)
            return Handle(self, 3, (False, 'not_in_basket (item at x -20.392 y -2.813 z -0.528)'))
        if target == 'bin' and iid in self.place_fail:
            return Handle(self, 3, (False, 'attach_failed'))
        if target == 'bin' and self.bin_dist(bin_id) > CFG['bin_place_distance_m'] + 0.1:
            return Handle(self, 3, (False, 'bin_unreachable'))
        if target == 'ground':
            self.held = None
        return Handle(self, 3, (True, 'ok'))

    def update(self, iid, status, material='', bin_id='', skip_reason=''):
        it = self.known[iid]
        if status == COLLECTED:
            self.compartments[bin_id].append(iid)
            it.bin_id = bin_id
            self.picked_at[iid] = self.truth.pop(iid, None)
        if status == SEEN and it.status == COLLECTED:
            self.compartments[it.bin_id].remove(iid)
            it.bin_id = ''
            self.truth[iid] = self.picked_at[iid]                  # lies where it was picked
        if status == PLACED:
            self.compartments[it.bin_id].remove(iid)
            it.bin_id = bin_id
        it.status = status
        it.skip_reason = skip_reason
        return True

    def cmd_vel(self, vx, vy, wz):
        if (vx, vy, wz) != (0.0, 0.0, 0.0) and not self.safe(self.t):
            self.calls_while_unsafe += 1
        self.cmd = (vx, vy, wz)
        self.cmds.append((self.t, (vx, vy, wz)))

    def log(self, rec):
        self.log_records.append(rec)

    # --- simulation -----------------------------------------------------------------------
    def snapshot(self):
        x, y, yaw = self.robot
        for iid, v in self.truth.items():
            if iid not in self.known and math.hypot(v['x'] - x, v['y'] - y) < 5.0:
                self.known[iid] = ItemView(iid, v['cls'], v['x'], v['y'], 0.03)
        return Snapshot(self.t, tuple(self.robot), self.known,
                        {k: list(v) for k, v in self.compartments.items()},
                        0.05 * sum(len(v) for v in self.compartments.values()), self.safe(self.t))

    def step(self):
        vx, vy, wz = self.cmd if self.safe(self.t) else (0.0, 0.0, 0.0)
        x, y, yaw = self.robot
        self.robot = [x + (vx * math.cos(yaw) - vy * math.sin(yaw)) * DT,
                      y + (vx * math.sin(yaw) + vy * math.cos(yaw)) * DT, yaw + wz * DT]
        self.tick_no += 1
        self.t += DT


def make(world, waypoints, capacity=Capacity(5, 2.0, 0.2), cfg=None, allowed=None):
    return Task(world, cfg or CFG, GRASP, capacity, waypoints, BINS, allowed=allowed)


def run(task, world, max_ticks=20000):
    while task.state != 'DONE' and world.tick_no < max_ticks:
        task.tick(world.snapshot())
        world.step()
    return task


ITEMS = {'a': {'cls': 'trash', 'x': 8.0, 'y': 14.5}, 'b': {'cls': 'trash', 'x': 15.0, 'y': 13.0},
         'c': {'cls': 'trash', 'x': 22.0, 'y': 16.0}, 'd': {'cls': 'distractor', 'x': 10.0, 'y': 14.0}}
MATS = {'a': 'pet', 'b': 'residual', 'c': 'glass', 'd': 'distractor'}
PATH = [(6.0, 14.0), (12.0, 14.0), (18.0, 14.0), (24.0, 14.0)]


def statuses(world):
    return {k: (v.status, v.bin_id) for k, v in world.known.items()}


def test_full_run_collects_sorts_and_places_everything():
    w = FakeWorld(ITEMS, MATS)
    t = run(make(w, PATH), w)
    assert t.state == 'DONE'
    s = statuses(w)
    assert s['a'] == (PLACED, 'recyclable') and s['c'] == (PLACED, 'recyclable') and s['b'] == (PLACED, 'residual')
    assert s['d'][0] == SEEN                                    # a known distractor is never picked
    assert w.compartments == {'recyclable': [], 'residual': []}
    events = [r['event'] for r in w.log_records]
    assert 'done' in events and events.count('place') == 3


def test_misdetected_distractor_is_put_back_and_skipped():
    items = dict(ITEMS, d={'cls': 'trash', 'x': 10.0, 'y': 14.0})      # detector calls it trash
    w = FakeWorld(items, MATS)
    run(make(w, PATH), w)
    assert w.known['d'].status == SKIPPED and w.known['d'].skip_reason == 'distractor'
    assert 'd' not in w.compartments['recyclable'] + w.compartments['residual']


def test_grasp_retried_then_succeeds():
    w = FakeWorld(ITEMS, MATS, pick_fail={'a': CFG['grasp_retries'] - 1})
    run(make(w, PATH), w)
    assert w.known['a'].status == PLACED


def test_grasp_fails_too_often_and_is_skipped():
    w = FakeWorld(ITEMS, MATS, pick_fail={'a': 99})
    t = run(make(w, PATH), w)
    assert t.state == 'DONE'
    assert (w.known['a'].status, w.known['a'].skip_reason) == (SKIPPED, 'grasp_failed')
    picks = [r for r in w.log_records if r['event'] == 'pick' and r['item'] == 'a']
    assert len(picks) == CFG['grasp_retries']


def test_unreachable_item_is_skipped():
    w = FakeWorld(ITEMS, MATS, nav_fail={'b'})
    t = run(make(w, PATH), w)
    assert t.state == 'DONE' and (w.known['b'].status, w.known['b'].skip_reason) == (SKIPPED, 'unreachable')


def test_nothing_starts_while_unsafe():
    w = FakeWorld(ITEMS, MATS)
    w.safe = lambda t: not (3.0 <= t < 30.0 or 60.0 <= t < 90.0)
    t = run(make(w, PATH), w)
    assert t.state == 'DONE' and w.calls_while_unsafe == 0
    assert all(w.known[i].status == PLACED for i in 'abc')


def test_full_basket_goes_to_the_bins_in_between():
    w = FakeWorld(ITEMS, MATS)
    t = run(make(w, PATH, capacity=Capacity(1, 2.0, 0.05)), w)
    assert t.state == 'DONE' and all(w.known[i].status == PLACED for i in 'abc')
    states = [r['state'] for r in w.log_records if r['event'] == 'state']
    assert states.count('GO_TO_BIN') >= 2                     # more than one trip


def test_low_confidence_goes_to_residual():
    w = FakeWorld(ITEMS, MATS, confidence={'a': 0.3})
    run(make(w, PATH), w)
    assert w.known['a'].bin_id == 'residual'


def test_failed_placing_leaves_the_item_and_finishes():
    w = FakeWorld(ITEMS, MATS, place_fail={'a'})
    t = run(make(w, PATH), w)
    assert t.state == 'DONE'
    assert w.known['a'].status == COLLECTED and 'a' in t.stuck
    assert all(w.known[i].status == PLACED for i in 'bc')
    assert any(r['event'] == 'left_in_basket' for r in w.log_records)


def test_final_approach_gets_the_item_into_reach():
    w = FakeWorld({'a': {'cls': 'trash', 'x': 8.0, 'y': 14.5}}, {'a': 'pet'})
    t = make(w, PATH)
    while t.state != 'PICK' and w.tick_no < 5000:
        t.tick(w.snapshot())
        w.step()
    assert t.state == 'PICK'
    x, y, yaw = w.robot
    c, s = math.cos(yaw), math.sin(yaw)
    rx, ry = c * (8.0 - x) + s * (14.5 - y), -s * (8.0 - x) + c * (14.5 - y)
    assert GRASP['reach_x_m'][0] <= rx <= GRASP['reach_x_m'][1] and abs(ry) <= GRASP['reach_y_max_m']
    assert w.cmd == (0.0, 0.0, 0.0)                            # standing still for the pick


def test_no_progress_times_out():
    cfg = dict(CFG, item_timeout_s=5.0)
    w = FakeWorld({'a': {'cls': 'trash', 'x': 8.0, 'y': 14.5}}, {'a': 'pet'})
    w.cmd_vel = lambda vx, vy, wz: None                        # the robot never moves on its own
    w.nav_to = lambda x, y, yaw: Handle(w, 1, True)            # "arrives" without moving
    t = run(make(w, PATH, cfg=cfg), w, max_ticks=3000)
    assert (w.known['a'].status, w.known['a'].skip_reason) == (SKIPPED, 'no_progress')


def test_empty_beach_finishes_after_the_search():
    w = FakeWorld({}, {})
    t = run(make(w, PATH), w)
    assert t.state == 'DONE' and t.wp == len(PATH)


@pytest.mark.parametrize('seed', range(5))
def test_random_scenes_always_finish(seed):
    import random
    rnd = random.Random(seed)
    items = {f'i{k}': {'cls': rnd.choice(['trash', 'trash', 'distractor']), 'x': rnd.uniform(5, 25),
                       'y': rnd.uniform(10, 18)} for k in range(12)}
    mats = {k: (rnd.choice(['pet', 'glass', 'residual', 'metal']) if v['cls'] == 'trash' else 'distractor')
            for k, v in items.items()}
    w = FakeWorld(items, mats, pick_fail={'i0': 99, 'i3': 1}, place_fail={'i5'})
    t = run(make(w, PATH, capacity=Capacity(2, 2.0, 0.05)), w, max_ticks=60000)
    assert t.state == 'DONE'
    for k, v in w.known.items():
        if v.status == PLACED:
            assert v.bin_id == ('residual' if mats[k] == 'residual' else 'recyclable')


def test_compartment_full_of_stuck_items_skips_instead_of_looping():
    items = {'a': {'cls': 'trash', 'x': 8.0, 'y': 14.5}, 'b': {'cls': 'trash', 'x': 15.0, 'y': 13.0}}
    w = FakeWorld(items, {'a': 'pet', 'b': 'glass'}, place_fail={'a'})
    t = run(make(w, PATH, capacity=Capacity(1, 2.0, 0.05)), w)
    assert t.state == 'DONE' and 'a' in t.stuck
    assert (w.known['b'].status, w.known['b'].skip_reason) == (SKIPPED, 'compartment_full')


def test_bin_approach_ends_at_the_place_distance():
    w = FakeWorld({'a': {'cls': 'trash', 'x': 8.0, 'y': 14.5}}, {'a': 'pet'})
    t = make(w, [])
    while t.state != 'PLACE' and w.tick_no < 10000:
        t.tick(w.snapshot())
        w.step()
    assert t.state == 'PLACE'
    bx, by, _, _ = BINS['recyclable']
    assert math.hypot(w.robot[0] - bx, w.robot[1] - by) == pytest.approx(CFG['bin_place_distance_m'], abs=0.04)
    assert w.cmd == (0.0, 0.0, 0.0)


def test_items_outside_the_allowed_area_are_never_targeted():
    items = dict(ITEMS, e={'cls': 'trash', 'x': 20.0, 'y': 11.0})         # seen, but in a forbidden area
    w = FakeWorld(items, dict(MATS, e='pet'))
    t = run(make(w, PATH, allowed=lambda x, y: y > 12.0), w)
    assert t.state == 'DONE' and w.known['e'].status == SEEN
    assert not any(r.get('item') == 'e' for r in w.log_records if r['event'] in ('target', 'pick'))


class PhantomWorld(FakeWorld):
    """An item whose map position moves with the robot (a localization/TF artefact): it stays
    1 m ahead-left of the robot whatever the robot does."""

    def snapshot(self):
        snap = super().snapshot()
        if 'ghost' in self.known:
            x, y, yaw = self.robot
            g = self.known['ghost']
            g.x, g.y = x + math.cos(yaw + 0.5), y + math.sin(yaw + 0.5)
        return snap


def test_phantom_item_is_given_up_quickly_and_the_robot_stays_put():
    w = PhantomWorld({'ghost': {'cls': 'trash', 'x': 5.0, 'y': 14.5}}, {'ghost': 'pet'})
    t = make(w, PATH)
    start = None
    while w.tick_no < 3000 and w.known.get('ghost', None) is None or (w.known['ghost'].status == SEEN
                                                                         and w.tick_no < 3000):
        t.tick(w.snapshot())
        if t.state == 'APPROACH' and t.sub == 'final' and start is None:
            start = (w.t, tuple(w.robot))
        w.step()
    g = w.known['ghost']
    assert (g.status, g.skip_reason) == (SKIPPED, 'no_progress')
    fa = CFG['final_approach']
    assert w.t - start[0] <= fa['progress_timeout_s'] + 2.0
    assert math.dist(w.robot[:2], start[1][:2]) <= fa['max_vx'] * (fa['progress_timeout_s'] + 2.0) + 0.01


def test_safety_pause_during_final_approach_is_not_no_progress():
    w = FakeWorld({'a': {'cls': 'trash', 'x': 8.0, 'y': 14.5}}, {'a': 'pet'})
    t = make(w, PATH)
    while not (t.state == 'APPROACH' and t.sub == 'final') and w.tick_no < 5000:
        t.tick(w.snapshot())
        w.step()
    pause_from = w.t
    w.safe = lambda now: not (pause_from <= now < pause_from + 3 * CFG['final_approach']['progress_timeout_s'])
    run(t, w)
    assert w.known['a'].status == PLACED


def test_fall_stops_everything_and_ends_the_run():
    w = FakeWorld(ITEMS, MATS)
    t = make(w, PATH)
    fell_at = None
    while t.state not in ('DONE', 'FELL') and w.tick_no < 20000:
        snap = w.snapshot()
        if w.known.get('a') is not None and w.known['a'].status == COLLECTED and fell_at is None:
            fell_at = w.t                               # fall right after the first store
        if fell_at is not None:
            snap.tilt_deg = 95.0
        t.tick(snap)
        w.step()
    assert t.state == 'FELL' and w.cmd == (0.0, 0.0, 0.0)
    calls = len(w.log_records)
    for _ in range(50):                                 # nothing happens any more
        snap = w.snapshot()
        snap.tilt_deg = 0.0                             # even if it "looks" upright again
        t.tick(snap)
        w.step()
    assert t.state == 'FELL' and len(w.log_records) == calls
    assert [r for r in w.log_records if r['event'] == 'fell'][0]['tilt_deg'] == 95.0


def test_fall_during_an_arm_action_cancels_it():
    w = FakeWorld(ITEMS, MATS)
    t = make(w, PATH)
    while t.state != 'PICK' or t.pending is None:
        t.tick(w.snapshot())
        w.step()
    handle = t.pending
    snap = w.snapshot()
    snap.tilt_deg = 40.0
    t.tick(snap)
    assert t.state == 'FELL' and handle.cancelled


def test_first_bin_fails_then_the_robot_walks_to_the_second():
    """A2b a2b_bin2: after the recyclable item failed, the residual one was placed from the recyclable
    bin's spot (bin_unreachable). The fake world refuses placing from further than the place distance."""
    w = FakeWorld(ITEMS, MATS, place_fail={'a'})
    t = make(w, PATH)
    placed_from = {}
    real_place = w.place

    def place(target, iid, bin_id, pose):
        if target == 'bin':
            placed_from[iid] = (bin_id, w.bin_dist(bin_id))
        return real_place(target, iid, bin_id, pose)
    w.place = place
    run(t, w)
    assert t.state == 'DONE' and w.known['b'].status == PLACED and w.known['b'].bin_id == 'residual'
    assert placed_from['b'][0] == 'residual'
    assert placed_from['b'][1] == pytest.approx(CFG['bin_place_distance_m'], abs=0.04)


def test_robot_backs_out_of_the_bin_before_navigating():
    """A2b a2b_bin2: facing the bin at the place distance, Nav2 saw 'collision ahead' and never moved."""
    w = FakeWorld(ITEMS, MATS)
    t = run(make(w, PATH, capacity=Capacity(1, 2.0, 0.05)), w)
    assert t.state == 'DONE' and all(w.known[i].status == PLACED for i in 'abc')
    assert w.nav_from_bin == 0                               # every Nav2 goal from outside the bin zone


def test_safety_pause_while_backing_out_stops_the_robot():
    w = FakeWorld({'a': {'cls': 'trash', 'x': 8.0, 'y': 14.5}}, {'a': 'pet'})
    t = make(w, [])
    while not (t.state == 'PLACE' and t.sub == 'leave') and w.tick_no < 10000:
        t.tick(w.snapshot())
        w.step()
    assert t.sub == 'leave'
    t.tick(w.snapshot())
    assert w.cmd[0] < 0.0                                     # backing out
    w.safe = lambda now: False
    t.tick(w.snapshot())
    assert w.cmd == (0.0, 0.0, 0.0) and w.calls_while_unsafe == 0


def test_item_not_in_the_basket_frees_its_slot_and_is_picked_again():
    """A2b a2b_full1: the tin fell back out of the basket; three unload attempts on an item 20 m
    away, and it kept using a compartment slot."""
    w = FakeWorld(ITEMS, MATS, lost={'a'})
    t = run(make(w, PATH), w)
    assert t.state == 'DONE'
    assert statuses(w)['a'] == (PLACED, 'recyclable')               # found again, placed on the next trip
    places = [r for r in w.log_records if r['event'] == 'place' and r['item'] == 'a']
    assert [r['success'] for r in places] == [False, True]          # no retries on a missing item
    assert any(r['event'] == 'lost_from_basket' and r['item'] == 'a' for r in w.log_records)
    assert not any(r['event'] == 'left_in_basket' for r in w.log_records)
    assert w.compartments == {'recyclable': [], 'residual': []}


def test_store_refused_as_full_puts_the_item_back_and_empties_the_basket_first():
    """Grasping found no spot that keeps the item below the rim (the count limit isn't reached)."""
    w = FakeWorld(ITEMS, MATS, store_full={'c'})
    t = run(make(w, PATH), w)
    assert t.state == 'DONE'
    s = statuses(w)
    assert s['a'] == (PLACED, 'recyclable') and s['c'] == (PLACED, 'recyclable')
    ev = [(r['event'], r.get('item')) for r in w.log_records]
    i = ev.index(('store', 'c'))
    states = [r['state'] for r in w.log_records[i:] if r['event'] == 'state']
    assert states[:2] == ['PUT_BACK', 'GO_TO_BIN']


def test_own_commands_change_no_faster_than_the_smoother_limits():
    """A2b falls: steps in the turn rate tip the robot; the task's final approaches and backing out
    are rate-limited like Nav2's velocity smoother (immediate zero only to stop)."""
    w = FakeWorld(ITEMS, MATS)
    run(make(w, PATH), w)
    lim = CFG['final_approach']['max_accel']
    assert len(w.cmds) > 50
    for (t0, a), (t1, b) in zip(w.cmds, w.cmds[1:]):
        if b == (0.0, 0.0, 0.0):
            continue
        dt = max(t1 - t0, DT)
        for k in range(3):
            assert abs(b[k] - a[k]) <= lim[k] * min(dt, 0.5) + 1e-9, (t1, a, b)
