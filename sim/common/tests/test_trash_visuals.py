"""B5 trash visuals: visuals only, so every item's visual meshes must match its collision box (the
cross-phase tests and grasp results were made with those sizes), UVs must be valid, and the meshes
small (RTF)."""

from pathlib import Path

import numpy as np
import pytest
import yaml

import trash_visuals as tv

COMMON = Path(__file__).resolve().parents[1]
OBJECTS = yaml.safe_load(open(COMMON / 'objects.yaml'))
BUILT = tv.build()


def collision_bounds(spec):
    lo = np.min([np.subtract(p['pose'][:3], np.divide(p['size'], 2)) for p in spec['collision']], axis=0)
    hi = np.max([np.add(p['pose'][:3], np.divide(p['size'], 2)) for p in spec['collision']], axis=0)
    return lo, hi


@pytest.mark.parametrize('name', sorted(tv.ITEMS))
def test_visual_matches_the_collision_box(name):
    parts = [BUILT[name]['print']] + ([BUILT[name]['body']] if 'body' in BUILT[name] else [])
    lo = np.min([p.bounds()[0] for p in parts], axis=0)
    hi = np.max([p.bounds()[1] for p in parts], axis=0)
    clo, chi = collision_bounds(OBJECTS['trash'][name])
    assert np.allclose(lo, clo, atol=0.002) and np.allclose(hi, chi, atol=0.002), (name, lo, hi, clo, chi)


@pytest.mark.parametrize('name', sorted(tv.ITEMS))
def test_uvs_valid_and_mesh_small(name):
    for key in ('print', 'body'):
        m = BUILT[name].get(key)
        if m is None:
            continue
        for v, uv, f, _ in m.parts:
            assert uv.shape == (len(v), 2) and uv.min() >= 0 and uv.max() <= 1
            assert f.min() >= 0 and f.max() < len(v)
        assert m.triangles() <= 2000, (name, key, m.triangles())
    assert BUILT[name]['texture'].size == (tv.TEX, tv.TEX)


def test_every_trash_item_has_visuals_in_objects_yaml():
    for name, spec in OBJECTS['trash'].items():
        vis = spec.get('visuals')
        assert vis, name
        for v in vis:
            assert (COMMON / v['mesh']).is_file(), v['mesh']
            assert ('texture' in v) != ('color' in v), v
            if 'texture' in v:
                assert (COMMON / v['texture']).is_file(), v['texture']
