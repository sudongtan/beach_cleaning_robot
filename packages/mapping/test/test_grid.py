"""Mapping grid: obstacles, geofence and terrain on the real benchmark scenes."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from mapping.grid import FREE, LETHAL, build, inside_polygon, lethal_at, primitive_mask

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'sim' / 'common'))
import terrain as terrain_mod  # noqa: E402

SCENES = yaml.safe_load(open(REPO / 'bench' / 'scenes.yaml'))['scenes']
OBJECTS = yaml.safe_load(open(REPO / 'sim' / 'common' / 'objects.yaml'))
TRAV = yaml.safe_load(open(REPO / 'config' / 'terrain.yaml'))['traversability']
CFG = yaml.safe_load(open(REPO / 'config' / 'mapping.yaml'))['mapping']
RES = TRAV['grid_resolution_m']


def grid(scene):
    t = terrain_mod.traversability(scene, TRAV['max_slope_deg'], RES)
    return build(scene, OBJECTS, t['occupancy'], RES, CFG['obstacle_models'], CFG['obstacle_margin_m'],
                 CFG['lethal_outside_geofence'])


@pytest.fixture(scope='module', params=[s['seed'] for s in SCENES])
def scene_grid(request):
    s = next(x for x in SCENES if x['seed'] == request.param)
    return s, grid(s)


def test_shape_covers_the_area(scene_grid):
    s, occ = scene_grid
    w, h = s['area']['size']
    assert occ.shape == (round(h / RES), round(w / RES))


def test_robot_start_is_free(scene_grid):
    s, occ = scene_grid
    x, y, _ = s['robot_start']['pose']
    assert not lethal_at(occ, RES, x, y)


def test_bins_are_lethal_and_reachable_from_the_front(scene_grid):
    """Bin centres are lethal; 0.7 m in front of each (+x, towards the beach) is free: base_link there puts
    the robot's head ~5 cm from the bin wall, where the arm reaches over the rim (decisions 2026-10-13)."""
    s, occ = scene_grid
    for b in s['bins']:
        x, y, yaw = b['pose']
        assert lethal_at(occ, RES, x, y)
        assert not lethal_at(occ, RES, x + math.cos(yaw) * 0.7, y + math.sin(yaw) * 0.7)


def test_large_distractors_are_lethal(scene_grid):
    s, occ = scene_grid
    for d in s['distractors']:
        x, y, _ = d['pose']
        if d['model'] in CFG['obstacle_models']:
            assert lethal_at(occ, RES, x, y), d['id']


def test_outside_geofence_is_lethal(scene_grid):
    s, occ = scene_grid
    poly = s['geofence']['polygon']
    ys = [p[1] for p in poly]
    assert lethal_at(occ, RES, 20.0, min(ys) - 0.3)        # water side
    assert lethal_at(occ, RES, 0.1, 15.0)                  # west edge


def test_most_of_the_geofence_is_free(scene_grid):
    """A sanity bound: obstacles and slopes don't block the cleaning area."""
    s, occ = scene_grid
    w, h = s['area']['size']
    xs = (np.arange(occ.shape[1]) + 0.5) * RES
    ys = (np.arange(occ.shape[0]) + 0.5) * RES
    X, Y = np.meshgrid(xs, ys)
    inside = inside_polygon(X, Y, s['geofence']['polygon'])
    assert (occ[inside] == FREE).mean() > 0.9


def test_trash_is_on_free_cells(scene_grid):
    s, occ = scene_grid
    for t in s['trash']:
        assert not lethal_at(occ, RES, t['pose'][0], t['pose'][1]), t['id']


def test_rotated_box_footprint():
    X, Y = np.meshgrid(np.arange(-1, 1.001, 0.05), np.arange(-1, 1.001, 0.05))
    prim = {'type': 'box', 'size': [0.9, 0.08, 0.08], 'pose': [0, 0, 0.04, 0, 0, 0]}
    m = primitive_mask(X, Y, (0.0, 0.0, math.pi / 2), prim, 0.0)     # log turned north-south
    assert m[np.isclose(Y, 0.4) & np.isclose(X, 0.0)].all()
    assert not m[np.isclose(Y, 0.0) & np.isclose(X, 0.4)].any()


def test_polygon():
    X, Y = np.meshgrid([0.5, 1.5, 2.5], [0.5])
    assert inside_polygon(X, Y, [[0, 0], [2, 0], [2, 2], [0, 2]]).tolist() == [[True, True, False]]
