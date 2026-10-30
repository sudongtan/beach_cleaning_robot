"""Perception baseline logic with the project's camera and sorting configs."""

import math
from pathlib import Path

import pytest
import yaml

from perception.logic import DISTRACTOR, TRASH, Fov, nearest, object_class, sort, vertical_fov

REPO = Path(__file__).resolve().parents[3]
CAMERA = yaml.safe_load(open(REPO / 'config' / 'sensors.yaml'))['camera']
SORTING = yaml.safe_load(open(REPO / 'config' / 'sorting_rules.yaml'))['sorting']
OBJECTS = yaml.safe_load(open(REPO / 'sim' / 'common' / 'objects.yaml'))
MATERIALS = {s['material'] for s in OBJECTS['trash'].values()}
FOV = Fov(CAMERA)


def test_vertical_fov_of_640x480():
    v = vertical_fov(1.2, 640, 480)
    assert math.tan(v / 2) == pytest.approx(math.tan(0.6) * 0.75)


def test_straight_ahead_in_range():
    assert FOV.sees(3.0, 0.0, 0.0)


def test_behind_too_close_too_far():
    assert not FOV.sees(-1.0, 0.0, 0.0)
    assert not FOV.sees(CAMERA['depth_min_m'] / 2, 0.0, 0.0)
    assert not FOV.sees(CAMERA['depth_max_m'] + 0.1, 0.0, 0.0)


def test_horizontal_edges():
    h = CAMERA['horizontal_fov_rad'] / 2
    assert FOV.sees(2.0, 2.0 * math.tan(h - 0.01), 0.0)
    assert not FOV.sees(2.0, 2.0 * math.tan(h + 0.01), 0.0)
    assert not FOV.sees(2.0, -2.0 * math.tan(h + 0.01), 0.0)


def test_vertical_edges():
    v = vertical_fov(CAMERA['horizontal_fov_rad'], CAMERA['width'], CAMERA['height']) / 2
    assert FOV.sees(2.0, 0.0, -2.0 * math.tan(v - 0.01))
    assert not FOV.sees(2.0, 0.0, -2.0 * math.tan(v + 0.01))


def test_every_material_is_trash_every_distractor_type_is_not():
    for m in MATERIALS:
        assert object_class(m, MATERIALS) == TRASH
    for d in OBJECTS['distractors'].values():
        assert object_class(d['class'], MATERIALS) == DISTRACTOR


def test_nearest_within_radius():
    objs = [('a', 'pet', (1.0, 0.0, 0.0)), ('b', 'metal', (1.1, 0.0, 0.0)), ('c', 'glass', (3.0, 0.0, 0.0))]
    assert nearest(objs, (1.08, 0.0, 0.0), 0.15)[0] == 'b'
    assert nearest(objs, (2.0, 0.0, 0.0), 0.15) is None
    assert nearest([], (0, 0, 0), 1.0) is None


def test_sorting_rules():
    assert sort('pet', 1.0, SORTING, MATERIALS) == ('pet', True, 'recyclable')
    assert sort('glass', 1.0, SORTING, MATERIALS) == ('glass', True, 'recyclable')
    assert sort('residual', 1.0, SORTING, MATERIALS) == ('residual', False, 'residual')


def test_low_confidence_or_unknown_goes_to_residual():
    assert sort('pet', SORTING['confidence_threshold'] - 0.01, SORTING, MATERIALS)[2] == 'residual'
    assert sort('', 0.0, SORTING, MATERIALS)[2] == 'residual'


def test_distractor_has_no_bin():
    assert sort('shell', 1.0, SORTING, MATERIALS) == (DISTRACTOR, False, '')


def test_every_material_has_a_rule():
    assert MATERIALS <= set(SORTING['rules'])
