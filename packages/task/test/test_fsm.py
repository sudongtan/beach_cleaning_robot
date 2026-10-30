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

    def done(self):
        return self.world.tick_no >= self.at

    def result(self):
        return self.value


class FakeWorld:
    """Robot, items and world model; `io` of the task. Items are discovered within 5 m."""

    def __init__(self, items, materials, pick_fail=None, nav_fail=None, place_fail=(), confidence=None):
        self.tick_no, self.t = 0, 0.0
        self.robot = [4.0, 14.0, 0.0]
        self.truth = {k: dict(v) for k, v in items.items()}        # id -> {cls, x, y}
        self.materials = materials
        self.known = {}
        self.compartments = {'recyclable': [], 'residual': []}
        self.pick_fail = dict(pick_fail or {})                      # id -> failures before success
        self.nav_fail = set(nav_fail or ())                         # item ids whose stand-off can't be reached
        self.place_fail = set(place_fail)
        self.confidence = confidence or {}
        self.held = None
        self.safe = lambda t: True
        self.log_records, self.calls_while_unsafe = [], 0
        self.cmd = (0.0, 0.0, 0.0)
        self.target_item = None

    # --- io -----------------------------------------------------------------------------
    def _call(self):
        if not self.safe(self.t):
            self.calls_while_unsafe += 1

    def nav_to(self, x, y, yaw):
        self._call()
        near = [i for i in self.nav_fail if i in self.known and
                math.hypot(self.known[i].x - x, self.known[i].y - y) < 1.0]
        if near:
            return Handle(self, 5, False)
        self.robot = [x, y, yaw]                                    # arrives when the handle is done
        return Handle(self, 5, True)

    def cancel_nav(self):
        pass

    def pick(self, iid, pos):
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
        if target == 'bin' and iid in self.place_fail:
            return Handle(self, 3, (False, 'attach_failed'))
        if target == 'ground':
            self.held = None
        return Handle(self, 3, (True, 'ok'))

    def update(self, iid, status, material='', bin_id='', skip_reason=''):
        it = self.known[iid]
        if status == COLLECTED:
            self.compartments[bin_id].append(iid)
            it.bin_id = bin_id
            self.truth.pop(iid, None)
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


def make(world, waypoints, capacity=Capacity(5, 2.0, 0.2), cfg=None):
    return Task(world, cfg or CFG, GRASP, capacity, waypoints, BINS)


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
