"""Search pattern on the ten benchmark scenes with their real /map grids."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from mapping.grid import build, inside_polygon, lethal_at
from task.search import lane_spacing, lanes, line_segments, plan

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'sim' / 'common'))
import terrain as terrain_mod  # noqa: E402

SCENES = yaml.safe_load(open(REPO / 'bench' / 'scenes.yaml'))['scenes']
OBJECTS = yaml.safe_load(open(REPO / 'sim' / 'common' / 'objects.yaml'))
TRAV = yaml.safe_load(open(REPO / 'config' / 'terrain.yaml'))['traversability']
MAPCFG = yaml.safe_load(open(REPO / 'config' / 'mapping.yaml'))['mapping']
CAMERA = yaml.safe_load(open(REPO / 'config' / 'sensors.yaml'))['camera']
S = yaml.safe_load(open(REPO / 'config' / 'task.yaml'))['task']['search']
RES = TRAV['grid_resolution_m']
SPACING = lane_spacing(CAMERA['horizontal_fov_rad'], S['view_distance_m'], S['overlap'])


@pytest.fixture(scope='module', params=[s['seed'] for s in SCENES])
def scene_plan(request):
    s = next(x for x in SCENES if x['seed'] == request.param)
    t = terrain_mod.traversability(s, TRAV['max_slope_deg'], RES)
    occ = build(s, OBJECTS, t['occupancy'], RES, MAPCFG['obstacle_models'], MAPCFG['obstacle_margin_m'],
                MAPCFG['lethal_outside_geofence'])
    path = plan(s['geofence']['polygon'], s['robot_start']['pose'][:2], SPACING, S['edge_margin_m'],
                S['waypoint_spacing_m'], lambda x, y: lethal_at(occ, RES, x, y))
    return s, occ, path


def test_spacing_is_footprint_width_with_overlap():
    width = 2 * S['view_distance_m'] * math.tan(CAMERA['horizontal_fov_rad'] / 2)
    assert SPACING == pytest.approx(width * (1 - S['overlap']))
    assert 1.0 < SPACING < width


def test_waypoints_inside_geofence_and_free(scene_plan):
    s, occ, path = scene_plan
    assert path
    X = np.array([[p[0] for p in path]])
    Y = np.array([[p[1] for p in path]])
    assert inside_polygon(X, Y, s['geofence']['polygon']).all()
    assert not any(lethal_at(occ, RES, x, y) for x, y in path)


def test_lanes_cover_the_whole_area(scene_plan):
    """Every point of the geofence (inset by the margin) is within half a lane spacing of a lane."""
    s, occ, path = scene_plan
    poly = s['geofence']['polygon']
    lane_ys = sorted({round(p[1], 6) for p in path})
    ys = [p[1] for p in poly]
    for y in np.linspace(min(ys) + S['edge_margin_m'], max(ys) - S['edge_margin_m'], 200):
        assert min(abs(y - ly) for ly in lane_ys) <= SPACING / 2 + 1e-6


def test_starts_near_the_robot_and_alternates(scene_plan):
    s, occ, path = scene_plan
    start = s['robot_start']['pose'][:2]
    corners = [(min(p[0] for p in path), min(p[1] for p in path)), (max(p[0] for p in path), max(p[1] for p in path))]
    nearest_end = min(math.dist(start, (cx, cy)) for cx in (corners[0][0], corners[1][0])
                      for cy in (corners[0][1], corners[1][1]))
    assert math.dist(start, path[0]) <= nearest_end + S['waypoint_spacing_m']
    rows = {}
    for x, y in path:
        rows.setdefault(round(y, 6), []).append(x)
    dirs = [np.sign(xs[-1] - xs[0]) for xs in rows.values() if len(xs) > 1]
    assert all(a != b for a, b in zip(dirs, dirs[1:]))


def test_waypoint_spacing(scene_plan):
    s, occ, path = scene_plan
    rows = {}
    for x, y in path:
        rows.setdefault(round(y, 6), []).append(x)
    for xs in rows.values():
        gaps = np.abs(np.diff(xs))
        # dropped lethal waypoints can double a gap, never more
        assert (gaps <= 2 * S['waypoint_spacing_m'] + 1e-6).all()


def test_concave_polygon_gives_two_segments():
    u = [[0, 0], [10, 0], [10, 10], [6, 10], [6, 4], [4, 4], [4, 10], [0, 10]]
    assert line_segments(u, 7.0) == [(0.0, 4.0), (6.0, 10.0)]
    assert len(plan(u, (1, 1), 2.0, 0.5, 5.0)) > 0


def test_too_small_polygon():
    assert lanes([[0, 0], [1, 0], [1, 0.5], [0, 0.5]], 2.0, 0.5) == []
