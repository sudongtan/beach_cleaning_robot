"""Unit tests for the benchmark metrics (bench/metrics.py), the run recording (bench/recording.py) and
the runner's end decision (bench/runner.py decide_end), on small hand-made inputs."""
import math
import sys
from pathlib import Path

import pytest
import yaml

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
import metrics  # noqa: E402
from recording import Recording, coverage_share, duplicates, inside_polygon, tilt_deg  # noqa: E402
from runner import decide_end, parse_seeds  # noqa: E402

CFG = yaml.safe_load(open(BENCH / 'bench.yaml'))
MCFG, RCFG = CFG['metrics'], dict(CFG['recorder'], stall_move_m=0.5)
LIM = CFG['bench']['limits']
RULES = {'pet': 'recyclable', 'metal': 'recyclable', 'glass': 'recyclable', 'residual': 'residual'}
HALF, RIM = MCFG['bin_inner_half'], MCFG['bin_rim_m']


# --- bin and basket -------------------------------------------------------------------------
def test_in_bin_inner_box_and_rim():
    b = (2.0, 3.0, 0.0, 0.1)                        # x, y, yaw, base z
    assert metrics.in_bin((2.0, 3.0, 0.2), b, HALF, RIM)
    assert metrics.in_bin((2.27, 3.2, 0.2), b, HALF, RIM)
    assert not metrics.in_bin((2.29, 3.0, 0.2), b, HALF, RIM)      # past the inner half length
    assert not metrics.in_bin((2.0, 3.21, 0.2), b, HALF, RIM)      # past the inner half width
    assert not metrics.in_bin((2.0, 3.0, 0.61), b, HALF, RIM)      # above the rim (base 0.1 + 0.5)
    assert metrics.in_bin((2.0, 3.0, 0.59), b, HALF, RIM)


def test_in_bin_turned():
    b = (0.0, 0.0, math.pi / 2, 0.0)                # long side along y
    assert metrics.in_bin((0.0, 0.27, 0.1), b, HALF, RIM)
    assert not metrics.in_bin((0.27, 0.0, 0.1), b, HALF, RIM)


def test_in_basket():
    robot = (5.0, 5.0, 0.3, math.pi / 2)
    assert metrics.in_basket((5.0, 4.8, 0.35), robot, MCFG['basket_half'], MCFG['basket_below_base_m'])
    assert not metrics.in_basket((5.0, 5.0, 0.05), robot, MCFG['basket_half'], MCFG['basket_below_base_m'])
    assert not metrics.in_basket((5.4, 5.0, 0.35), robot, MCFG['basket_half'], MCFG['basket_below_base_m'])


# --- sorting, contamination, false pickups ----------------------------------------------------
SCENE = {
    'trash': [{'id': 'trash_00', 'material': 'pet'}, {'id': 'trash_01', 'material': 'residual'},
              {'id': 'trash_02', 'material': 'glass'}, {'id': 'trash_03', 'material': 'residual'},
              {'id': 'trash_04', 'material': 'metal'}],
    'distractors': [{'id': 'distractor_00'}, {'id': 'distractor_01'}],
}
BINS = {'recyclable': (1.0, 1.0, 0.0, 0.0), 'residual': (1.0, 2.0, 0.0, 0.0)}


def test_sorting_perfect():
    objects = {'trash_00': [1.0, 1.0, 0.1], 'trash_01': [1.0, 2.0, 0.1], 'trash_02': [1.1, 1.0, 0.1],
               'trash_03': [0.9, 2.1, 0.1], 'trash_04': [1.0, 0.9, 0.1],
               'distractor_00': [5, 5, 0.1], 'distractor_01': [6, 6, 0.1]}
    where = metrics.locate(objects, None, BINS, MCFG)
    s = metrics.sorting(SCENE, where, RULES, 2.0)
    assert s['items_collected'] == 5 and s['items_total'] == 5
    assert s['sorting_accuracy'] == 1.0
    assert s['contamination_count'] == 0 and s['contamination_rate'] == 0.0
    assert s['distractors_binned_or_basket'] == []


def test_sorting_contamination_weighted_and_basket():
    robot = (8.0, 8.0, 0.3, 0.0)
    objects = {'trash_00': [1.0, 2.0, 0.1],          # pet in residual: wrong, not contamination
               'trash_01': [1.0, 1.0, 0.1],          # residual in recyclable: contamination
               'trash_02': [1.0, 1.1, 0.1],          # glass in recyclable: right
               'trash_03': [8.0, 8.0, 0.35],         # in the basket
               'trash_04': [3.0, 3.0, 0.05],         # on the ground
               'distractor_00': [1.0, 2.1, 0.1],     # distractor in a bin
               'distractor_01': [8.1, 8.0, 0.4]}     # distractor in the basket
    where = metrics.locate(objects, robot, BINS, MCFG)
    s = metrics.sorting(SCENE, where, RULES, 2.0)
    assert s['items_collected'] == 3
    assert s['sorting_correct'] == 1
    assert s['sorting_accuracy'] == pytest.approx(1 / 3)
    assert s['contamination_count'] == 1
    assert s['contamination_rate'] == pytest.approx(0.5)      # 1 of the 2 items in the recyclable bin
    assert s['recyclable_in_residual'] == 1
    assert s['weighted_sorting_errors'] == 3.0
    assert s['items_in_basket'] == 1
    assert s['distractors_binned_or_basket'] == ['distractor_00', 'distractor_01']
    assert s['placement']['trash_04'] == 'ground'


def test_sorting_nothing_binned():
    s = metrics.sorting(SCENE, {}, RULES, 2.0)
    assert s['items_collected'] == 0 and s['sorting_accuracy'] is None and s['contamination_rate'] is None
    assert len(s['items_missing_from_ground_truth']) == 5


def test_events_grasp_confusion_false_pickup():
    wm_ids = {'wm_000': 'trash_00', 'wm_001': 'trash_01', 'wm_002': 'distractor_00'}
    events = [
        {'event': 'pick', 'item': 'wm_000', 'success': True},
        {'event': 'classify', 'item': 'wm_000', 'material': 'pet'},
        {'event': 'pick', 'item': 'wm_001', 'success': False},
        {'event': 'pick', 'item': 'wm_001', 'success': True},
        {'event': 'classify', 'item': 'wm_001', 'material': ''},       # nothing found -> unknown
        {'event': 'pick', 'item': 'wm_002', 'success': True},           # a distractor picked
        {'event': 'skipped', 'item': 'wm_009', 'reason': 'no_progress'},
        {'event': 'done'},
    ]
    e = metrics.event_metrics(events, wm_ids, SCENE)
    assert (e['picks'], e['picks_ok']) == (4, 3) and e['grasp_success_rate'] == 0.75
    assert e['picked_distractors'] == ['distractor_00']
    assert e['confusion']['pet'] == {'pet': 1} and e['confusion']['residual'] == {'unknown': 1}
    assert e['classification_accuracy'] == 0.5
    assert e['skipped'] == {'no_progress': 1} and e['task_done']


# --- recording: coverage, duplicates, safety rules ---------------------------------------------
def test_coverage_ignores_outside_cells():
    assert coverage_share([100, 0, -1, -1, 100, 0]) == 0.5
    assert coverage_share([-1, -1]) is None


def test_duplicates_count_extra_entries():
    items = [{'scene_id': 'trash_01'}, {'scene_id': 'trash_01'}, {'scene_id': 'trash_01'},
             {'scene_id': 'trash_02'}, {'scene_id': ''}, {'scene_id': ''}]
    assert duplicates(items) == {'trash_01': 2}


def test_inside_polygon_and_tilt():
    sq = [[0, 0], [2, 0], [2, 2], [0, 2]]
    assert inside_polygon(1, 1, sq) and not inside_polygon(3, 1, sq)
    assert tilt_deg((0, 0, 0, 1)) == pytest.approx(0)
    s = math.sin(math.radians(40) / 2)
    assert tilt_deg((s, 0, 0, math.cos(math.radians(40) / 2))) == pytest.approx(40)


def walk(rec, t0, t1, x0, vx, y=5.0, dt=0.02):
    t = t0
    while t <= t1 + 1e-9:
        rec.on_pose(t, (x0 + vx * (t - t0), y, 0.3), (0, 0, 0, 1), wall=100 + t)
        t += dt


def test_distance_ignores_jitter():
    rec = Recording(RCFG, 1.5)
    for k in range(200):                              # 5 mm back and forth: no distance
        rec.on_pose(k * 0.02, (1.0 + 0.005 * (k % 2), 1.0, 0.3), (0, 0, 0, 1), wall=k)
    assert rec.distance == 0.0
    walk(rec, 4.0, 14.0, 1.0, 0.2, y=1.0)
    assert rec.distance == pytest.approx(2.0, abs=0.02)


def test_person_rule_moving_vs_stopped():
    rec = Recording(RCFG, 1.5)
    rec.on_safety('stop')
    rec.on_people(0.0, [(6.0, 5.0)])
    rec.on_cmd(0, 0, 0)
    for k in range(100):                              # standing still 1 m from the person: no violation
        rec.on_people(k * 0.02, [(6.0, 5.0)])
        rec.on_pose(k * 0.02, (5.0, 5.0, 0.3), (0, 0, 0, 1), wall=k)
    assert rec.person_viol.items == [] and rec.min_person['d'] == pytest.approx(1.0)
    rec.on_cmd(0.1, 0, 0)                              # a non-zero /cmd_vel while close: violation
    rec.on_pose(2.02, (5.0, 5.0, 0.3), (0, 0, 0, 1), wall=200)
    assert len(rec.person_viol.items) == 1 and rec.person_viol.items[0]['by_cmd']


def test_person_rule_by_speed_and_stale_people():
    rec = Recording(RCFG, 1.5)
    rec.on_people(0.0, [(5.5, 5.0)])
    walk(rec, 0.0, 0.45, 5.0, 0.2)                    # 0.2 m/s close to the person, cmd zero
    assert any(e['by_speed'] for e in rec.person_viol.items)
    rec2 = Recording(RCFG, 1.5)
    rec2.on_people(0.0, [(5.5, 5.0)])
    walk(rec2, 1.0, 2.0, 5.0, 0.2)                    # /people older than 0.5 s: not checked
    assert rec2.person_viol.items == []


def test_arm_rule():
    names = RCFG['arm_joints']
    rec = Recording(RCFG, 1.5)
    rec.on_safety('ok')
    rec.on_joints(0.0, dict(zip(names, [0.0] * 6)))
    rec.on_joints(0.1, dict(zip(names, [0.5] * 6)))   # moves while ok: fine
    rec.on_safety('slow')
    rec.on_joints(0.2, dict(zip(names, [0.52] * 6)))  # 0.02 rad: fine
    assert rec.arm_viol.items == []
    rec.on_joints(0.3, dict(zip(names, [0.5, 0.5, 0.56, 0.5, 0.5, 0.5])))   # 0.04 within 0.1 s
    assert len(rec.arm_viol.items) == 1 and rec.arm_viol.items[0]['joint'] == 'arm_elbow'
    rec.on_safety('ok')
    rec.on_joints(0.4, dict(zip(names, [1.0] * 6)))
    assert len(rec.arm_viol.items) == 1


def test_arm_slow_creep_is_reported_not_counted():
    """b2 reading: 0.038 rad over 5 s (A3 dev scene: held arm in slow) is no violation, but the
    change since the state switch is reported."""
    names = RCFG['arm_joints']
    rec = Recording(RCFG, 1.5)
    rec.on_safety('ok')
    rec.on_joints(0.0, dict(zip(names, [0.0] * 6)))
    rec.on_safety('slow')
    for k in range(501):
        rec.on_joints(0.01 * k, dict(zip(names, [0.0, 0.0, 0.0, 0.038 * k / 500, 0.0, 0.0])))
    assert rec.arm_viol.items == []
    assert rec.arm_max_dev == pytest.approx(0.038)
    assert rec.arm_max_at['joint'] == 'arm_forearm_roll'


def test_geofence_time_outside():
    rec = Recording(RCFG, 1.5, geofence=[[0, 0], [10, 0], [10, 10], [0, 10]])
    walk(rec, 0.0, 2.0, 9.0, 1.0, dt=0.1)             # leaves at x 10 (t 1.0)
    assert rec.outside_s == pytest.approx(1.0, abs=0.11)
    assert len(rec.outside.items) == 1


def test_safety_split_at_fall():
    rec = {'fall_t': 100.0, 'person_violations': [{'t_start': 50, 't_end': 51}],
           'arm_violations': [{'t_start': 99.0, 't_end': 99.5}],
           'outside_geofence': [{'t_start': 90.0, 't_end': 120.0}],
           'min_person': None, 'arm_max_change_not_ok_rad': 0.1, 'safety_counts': {}}
    s = metrics.safety(rec, 2.0)
    assert s['violations'] == 2                       # person at 50, geofence from 90; arm at 99 is the fall
    assert len(s['after_fall']['arm']) == 1
    assert s['outside_geofence_s'] == pytest.approx(8.0)


# --- runner end decision --------------------------------------------------------------------
def snap(**kw):
    s = {'task_state': 'SEARCH', 'wall': 1000.0, 'last_pose_wall': 1000.0, 'sim_t': 500.0,
         'progress_wall': 1000.0}
    s.update(kw)
    return s


@pytest.mark.parametrize('flags,s,alive,now,expect', [
    ({}, snap(), True, 1001.0, None),
    ({'done': True}, snap(), True, 1001.0, 'done'),
    ({}, snap(task_state='DONE'), True, 1001.0, 'done'),
    ({'fall_line': True}, snap(), True, 1001.0, 'fell'),
    ({}, snap(task_state='FELL (tilt 33)'), True, 1001.0, 'fell'),
    ({}, snap(), False, 1001.0, 'dead_stack'),
    ({'dead_line': 'process has died'}, snap(), True, 1001.0, 'dead_stack'),
    ({}, snap(last_pose_wall=800.0), True, 1001.0, 'dead_stack'),
    ({}, snap(wall=800.0), True, 1001.0, 'dead_stack'),
    ({}, snap(sim_t=3600.0), True, 1001.0, 'sim_cap'),
    ({}, snap(progress_wall=300.0), True, 1001.0, 'stall'),
    ({'done': True}, snap(progress_wall=300.0), False, 1001.0, 'done'),     # done wins
])
def test_decide_end(flags, s, alive, now, expect):
    assert decide_end(now, 900.0, 950.0, s, flags, LIM, alive) == expect


def test_decide_end_wall_cap_and_no_recorder():
    s = snap(wall=13000.0, last_pose_wall=13000.0, progress_wall=13000.0)
    assert decide_end(13001.0, 0.0, 10.0, s, {}, LIM, True) == 'wall_cap'
    assert decide_end(100.0, 0.0, 10.0, None, {}, LIM, True) is None
    assert decide_end(200.0, 0.0, 10.0, None, {}, LIM, True) == 'dead_stack'


def test_parse_seeds():
    assert parse_seeds('all', [0, 1]) == [0, 1]
    assert parse_seeds('0-3', []) == [0, 1, 2, 3]
    assert parse_seeds('101', []) == [101]
    assert parse_seeds('2,5-6', []) == [2, 5, 6]
