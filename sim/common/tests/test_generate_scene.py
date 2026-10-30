"""Tests for the scene generator, the terrain exports and the shared assets.

    pytest sim/common
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

COMMON = Path(__file__).resolve().parents[1]
REPO = COMMON.parents[1]
sys.path.insert(0, str(COMMON))
import generate_scene as gen  # noqa: E402
import scene_schema  # noqa: E402
import terrain as T  # noqa: E402

SEEDS = list(range(10))


@pytest.fixture(scope='module')
def inp():
    return gen.load_inputs()


@pytest.fixture(scope='module')
def doc(inp):
    return gen.generate(SEEDS, inp)


def footprint(objects, kind, model):
    return objects[kind][model]['footprint_radius']


def inside(poly, x, y):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs) <= x <= max(xs) and min(ys) <= y <= max(ys)   # generator makes rectangles


# Determinism -----------------------------------------------------------------

def test_byte_identical_twice(inp):
    assert gen.dump(gen.generate(SEEDS, inp)) == gen.dump(gen.generate(SEEDS, inp))


def test_scene_depends_only_on_its_seed(inp, doc):
    alone = gen.generate([3], inp)['scenes'][0]
    assert alone == doc['scenes'][3]


def test_committed_scenes_yaml_is_up_to_date(doc):
    committed = (REPO / 'bench' / 'scenes.yaml').read_text()
    assert committed == gen.dump(doc), 'bench/scenes.yaml is stale: run `make generate`'


# Schema ----------------------------------------------------------------------

def test_schema_valid(doc, inp):
    assert scene_schema.validate(doc, inp['objects']) == []


def test_schema_catches_errors(doc, inp):
    bad = yaml.safe_load(gen.dump(doc))
    del bad['scenes'][0]['trash'][0]['pose']
    bad['scenes'][1]['bins'][0]['bin_id'] = 'recyclable'
    bad['scenes'][1]['bins'][1]['bin_id'] = 'recyclable'
    errs = scene_schema.validate(bad, inp['objects'])
    assert any('trash' in e and 'pose' in e for e in errs)
    assert any('one recyclable' in e for e in errs)


# Content constraints ---------------------------------------------------------

def test_counts_in_range(doc, inp):
    p = inp['params']['counts']
    for s in doc['scenes']:
        assert p['trash'][0] <= len(s['trash']) <= p['trash'][1]
        assert p['distractors'][0] <= len(s['distractors']) <= p['distractors'][1]
        assert len(s['bins']) == 2


def test_every_material_in_every_scene(doc):
    for s in doc['scenes']:
        assert {t['material'] for t in s['trash']} == set(gen.MATERIALS)


def test_person_in_half_the_scenes(doc):
    assert [s['seed'] for s in doc['scenes'] if s['person']] == [1, 3, 5, 7, 9]


def test_no_overlaps(doc, inp):
    o = inp['objects']
    for s in doc['scenes']:
        items = [(b['pose'], footprint(o, 'bins', b['model'])) for b in s['bins']]
        items += [(t['pose'], footprint(o, 'trash', t['model'])) for t in s['trash']]
        items += [(d['pose'], footprint(o, 'distractors', d['model'])) for d in s['distractors']]
        items += [(lm['pose'], footprint(o, 'landmarks', lm['model'])) for lm in s['landmarks']
                  if lm['model'] == 'lifeguard_tower']
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (a, ra), (b, rb) = items[i], items[j]
                d = math.hypot(a[0] - b[0], a[1] - b[1])
                assert d >= ra + rb, f'seed {s["seed"]}: overlap {a} {b}'


def test_trash_and_distractors_inside_geofence(doc):
    for s in doc['scenes']:
        poly = s['geofence']['polygon']
        for it in s['trash'] + s['distractors']:
            assert inside(poly, it['pose'][0], it['pose'][1]), (s['seed'], it['id'])


def test_geofence_inside_area_and_out_of_water(doc):
    for s in doc['scenes']:
        w, h = s['area']['size']
        for x, y in s['geofence']['polygon']:
            assert 0 <= x <= w and 0 <= y <= h
            assert y > s['terrain']['waterline_y']


def test_nothing_in_the_water(doc):
    for s in doc['scenes']:
        wl = s['terrain']['waterline_y']
        for it in s['trash'] + s['distractors'] + s['bins']:
            assert it['pose'][1] > wl
        assert s['robot_start']['pose'][1] > wl


def test_bins_and_items_on_gentle_slopes(doc, inp):
    lim = inp['traversability']['placement_max_slope_deg']
    for s in doc['scenes']:
        for it in s['bins'] + s['trash'] + s['distractors']:
            assert T.slope_deg_at(s['terrain'], it['pose'][0], it['pose'][1]) <= lim
        x, y, _ = s['robot_start']['pose']
        assert T.slope_deg_at(s['terrain'], x, y) <= lim


def test_geofence_is_walkable(doc, inp):
    lim = inp['traversability']['max_slope_deg']
    for s in doc['scenes']:
        (x0, y0), _, (x1, y1), _ = s['geofence']['polygon']
        X, Y = np.meshgrid(np.arange(x0, x1, 0.1), np.arange(y0, y1, 0.1))
        assert T.slope_deg_at(s['terrain'], X, Y).max() <= lim + 0.1, s['seed']


def test_person_path_crosses_the_cleaning_area(doc):
    for s in doc['scenes']:
        if not s['person']:
            continue
        poly = s['geofence']['polygon']
        wps = s['person']['waypoints']
        hits = 0
        for (x0, y0), (x1, y1) in zip(wps, wps[1:] + wps[:1]):
            pts = [(x0 + f * (x1 - x0), y0 + f * (y1 - y0)) for f in np.linspace(0, 1, 200)]
            hits += any(inside(poly, x, y) for x, y in pts)
        assert hits >= 1


# Terrain exports -------------------------------------------------------------

def test_heightmap_png(tmp_path, doc):
    t = doc['scenes'][0]['terrain']
    T.write_heightmap_png(t, tmp_path / 'hm.png')
    img = np.array(Image.open(tmp_path / 'hm.png'))
    n = t['heightmap']['pixels']
    assert img.shape == (n, n) and (n - 1) & (n - 2) == 0      # 2^k + 1
    assert img.min() == 0 and img.max() == 65535               # full range, see write_heightmap_png
    lo, hi = T.heightmap_range(t)
    h = T.height_grid(t)
    back = img.astype(float) / 65535.0 * (hi - lo) + lo
    assert np.allclose(np.flipud(back), h, atol=1e-3)           # row 0 of the image is north


def test_traversability_marks_water_lethal(doc):
    s = doc['scenes'][0]
    g = T.traversability(s, 10.0, 0.25)
    assert g['occupancy'].shape == (120, 160)
    assert (g['occupancy'][g['water']] == 100).all()
    assert (g['occupancy'] == 0).mean() > 0.6


def test_wet_zone_is_flat(doc):
    t = doc['scenes'][0]['terrain']
    y = t['waterline_y'] + t['wet_zone_width'] / 2
    hs = T.height_at(t, np.linspace(0, 40, 50), np.full(50, y))
    assert np.allclose(hs, t['wet_zone_height'])


def test_terrain_obj(tmp_path, doc):
    t = dict(doc['scenes'][0]['terrain'])
    t['heightmap'] = dict(t['heightmap'], pixels=9)
    T.write_obj_mesh(t, tmp_path / 't.obj')
    lines = (tmp_path / 't.obj').read_text().splitlines()
    assert sum(l.startswith('v ') for l in lines) == 81
    assert sum(l.startswith('f ') for l in lines) == 2 * 8 * 8


# Shared assets ---------------------------------------------------------------

def test_every_model_has_a_mesh_and_a_class(inp):
    o = inp['objects']
    for kind in ('trash', 'distractors', 'bins', 'landmarks'):
        for name, spec in o[kind].items():
            assert (COMMON / spec['mesh']).is_file(), name
            cls = spec.get('material', spec.get('class'))
            assert cls in o['classes'], name
    ids = list(o['classes'].values())
    assert len(ids) == len(set(ids))


def test_trash_is_liftable(inp):
    for name, spec in inp['objects']['trash'].items():
        assert 0 < spec['mass'] <= 0.5, name     # D1-class arm payload ~0.5 kg (scope.md)
