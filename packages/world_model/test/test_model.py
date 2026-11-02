"""World model: de-duplication with noisy positions, status transitions, capacity, coverage."""

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from perception.logic import Fov
from world_model.model import COLLECTED, PLACED, SEEN, SKIPPED, Coverage, WorldModel

REPO = Path(__file__).resolve().parents[3]
CFG = yaml.safe_load(open(REPO / 'config' / 'world_model.yaml'))['world_model']
LIMITS = yaml.safe_load(open(REPO / 'config' / 'robot_limits.yaml'))['robot_limits']
CAMERA = yaml.safe_load(open(REPO / 'config' / 'sensors.yaml'))['camera']


def wm(masses=None):
    return WorldModel(CFG['dedup_radius_m'], CFG['max_items_per_compartment'], LIMITS['basket_load_max_kg'],
                      CFG['default_item_mass_kg'], masses or {})


def test_noisy_detections_of_one_item_merge():
    m = wm()
    rng = np.random.default_rng(0)
    ids = {m.add_detection(k * 0.1, 'trash', 5 + rng.normal(0, 0.05), 3 + rng.normal(0, 0.05), 0.0, 1.0)
           for k in range(50)}
    assert len(ids) == 1
    it = m.items[ids.pop()]
    assert it.times_seen == 50
    assert math.hypot(it.x - 5, it.y - 3) < 0.03            # averaged


def test_items_apart_stay_separate():
    m = wm()
    a = m.add_detection(0, 'trash', 5.0, 3.0, 0, 1.0)
    b = m.add_detection(0, 'trash', 5.0 + CFG['dedup_radius_m'] + 0.05, 3.0, 0, 1.0)
    assert a != b and len(m.items) == 2


def test_different_class_is_a_different_item():
    m = wm()
    assert m.add_detection(0, 'trash', 5, 3, 0, 1) != m.add_detection(0, 'distractor', 5.05, 3, 0, 1)


def test_detection_merges_into_the_nearest_item():
    m = wm()
    a = m.add_detection(0, 'trash', 5.0, 3.0, 0, 1)
    b = m.add_detection(0, 'trash', 5.5, 3.0, 0, 1)
    assert m.add_detection(1, 'trash', 5.4, 3.0, 0, 1) == b and m.items[a].times_seen == 1


def test_collected_item_no_longer_matches():
    m = wm()
    a = m.add_detection(0, 'trash', 5, 3, 0, 1)
    assert m.update(a, COLLECTED, 'pet', 'recyclable') == (True, 'ok')
    assert m.add_detection(1, 'trash', 5, 3, 0, 1) != a      # something new lies there


def test_skipped_item_still_matches():
    m = wm()
    a = m.add_detection(0, 'distractor', 5, 3, 0, 1)
    m.update(a, SKIPPED, skip_reason='distractor')
    assert m.add_detection(1, 'distractor', 5.1, 3, 0, 1) == a and len(m.items) == 1


def test_status_transitions():
    m = wm()
    a = m.add_detection(0, 'trash', 5, 3, 0, 1)
    assert not m.update(a, PLACED, bin_id='recyclable')[0]   # can't skip the basket
    assert m.update(a, COLLECTED, 'glass', 'recyclable')[0]
    assert m.compartments['recyclable'] == [a]
    assert m.update(a, PLACED, bin_id='recyclable')[0]
    assert m.compartments['recyclable'] == [] and m.items[a].bin_id == 'recyclable'
    assert not m.update(a, SEEN)[0]                          # placed is final
    assert not m.update('wm_999', SEEN)[0]


def test_dropped_item_goes_back_to_seen():
    m = wm()
    a = m.add_detection(0, 'trash', 5, 3, 0, 1)
    m.update(a, COLLECTED, 'pet', 'residual')
    assert m.update(a, SEEN)[0]
    assert m.compartments['residual'] == [] and m.items[a].bin_id == '' and m.remaining()[0].id == a


def test_compartment_count_limit():
    m = wm()
    n = CFG['max_items_per_compartment']
    ids = [m.add_detection(0, 'trash', k * 1.0, 0, 0, 1, scene_id=f's{k}') for k in range(n + 1)]
    for k in range(n):
        m.items[ids[k]].mass_kg = 0.01
        assert m.update(ids[k], COLLECTED, 'pet', 'recyclable')[0]
    m.items[ids[n]].mass_kg = 0.01
    assert not m.update(ids[n], COLLECTED, 'pet', 'recyclable')[0]
    assert m.update(ids[n], COLLECTED, 'residual', 'residual')[0]   # the other compartment has room
    assert not m.basket_full()


def test_load_limit_with_ground_truth_masses():
    heavy = {f's{k}': 0.2 for k in range(20)}
    m = wm(heavy)
    stored = 0
    for k in range(20):
        a = m.add_detection(0, 'trash', k * 1.0, 0, 0, 1, scene_id=f's{k}')
        if m.update(a, COLLECTED, 'glass', 'recyclable' if k % 2 else 'residual')[0]:
            stored += 1
    assert m.load_kg() <= LIMITS['basket_load_max_kg'] + 1e-9
    assert stored == min(2 * CFG['max_items_per_compartment'], int(LIMITS['basket_load_max_kg'] / 0.2 + 1e-9))
    assert m.basket_full()


def test_unknown_compartment_rejected():
    m = wm()
    a = m.add_detection(0, 'trash', 5, 3, 0, 1)
    assert not m.update(a, COLLECTED, 'pet', 'blue_bin')[0]


def test_coverage_marks_the_ground_in_view():
    fov = Fov(CAMERA)
    w, h, res = 20.0, 10.0, CFG['coverage_resolution_m']
    ny, nx = int(h / res), int(w / res)
    inside = np.ones((ny, nx), dtype=bool)
    inside[:, :4] = False                                     # a strip outside the geofence
    cov = Coverage(w, h, res, inside, np.zeros((ny, nx)))
    # camera 0.34 m above flat ground at (5, 5) looking along +x, pitched 20 deg down
    p = math.radians(20)
    R = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    cov.mark(R, (5.0, 5.0, 0.34), fov)

    def cell(x, y):
        return cov.grid[int(y / res), int(x / res)]
    assert cell(7.0, 5.0) == 100                              # 2 m ahead
    assert cell(4.0, 5.0) == 0                                # behind
    assert cell(5.1, 5.0) == 0                                # under the camera, below the view
    assert cell(7.0, 9.5) == 0                                # outside the horizontal FOV
    assert cell(1.0, 5.0) == -1
    assert 0 < cov.fraction() < 0.5


def test_coverage_never_marks_outside_the_geofence():
    fov = Fov(CAMERA)
    inside = np.zeros((20, 40), dtype=bool)
    cov = Coverage(20.0, 10.0, 0.5, inside, np.zeros((20, 40)))
    cov.mark(np.eye(3), (5.0, 5.0, 0.34), fov)
    assert (cov.grid == -1).all() and cov.fraction() == 0.0


def test_position_averaged_yaw_and_size_from_latest():
    m = wm()
    a = m.add_detection(0, 'trash', 5.0, 3.0, 0.03, 1.0, yaw=0.2, size=(0.11, 0.066, 0.066))
    assert m.add_detection(1, 'trash', 5.1, 3.0, 0.03, 1.0, yaw=0.4, size=(0.12, 0.07, 0.06)) == a
    it = m.items[a]
    assert it.x == pytest.approx(5.05) and it.yaw == 0.4 and it.size == (0.12, 0.07, 0.06)


def test_size_unknown_by_default():
    m = wm()
    assert m.items[m.add_detection(0, 'trash', 5, 3, 0, 1)].size == (0.0, 0.0, 0.0)
