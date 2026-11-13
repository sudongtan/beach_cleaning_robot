"""Mac tests for the pure render-check helpers (notes/briefs/D0_prep.md §3 item 9)."""

import numpy as np
import pytest
import yaml
from PIL import Image

import render_checks as rc
from conftest import ISAAC, open_stage

CFG = yaml.safe_load((ISAAC / 'render.yaml').read_text())['render_check']


def _img(kind, h=48, w=64):
    rng = np.random.default_rng(1)
    if kind == 'black':
        return np.zeros((h, w, 3), np.uint8)
    if kind == 'nearly_black':
        return rng.integers(0, 4, (h, w, 3)).astype(np.uint8)
    if kind == 'uniform':
        return np.full((h, w, 3), 140, np.uint8)
    if kind == 'white':
        return np.full((h, w, 3), 255, np.uint8)
    # a plausible frame: sky over sand with noise
    a = np.zeros((h, w, 3), np.float64)
    a[: h // 2] = (158, 204, 242)
    a[h // 2:] = (194, 178, 128)
    return np.clip(a + rng.normal(0, 8, a.shape), 0, 255).astype(np.uint8)


@pytest.mark.parametrize('kind,ok,black,uniform', [
    ('black', False, True, True),
    ('nearly_black', False, True, True),
    ('uniform', False, False, True),
    ('white', False, False, True),
    ('normal', True, False, False),
])
def test_image_check(kind, ok, black, uniform):
    c = rc.image_check(_img(kind), CFG['min_mean'], CFG['min_std'])
    assert c['ok'] is ok
    assert c['not_black'] is (not black)
    assert c['not_uniform'] is (not uniform)


def test_image_check_ignores_alpha():
    a = np.dstack([_img('black'), np.full((48, 64), 255, np.uint8)])
    assert not rc.image_check(a, CFG['min_mean'], CFG['min_std'])['not_black']


def test_find_and_load_images(tmp_path):
    d = tmp_path / 'top' / 'sub'
    d.mkdir(parents=True)
    for k in (0, 1):
        Image.fromarray(_img('normal')).save(d / f'rgb_{k:04d}.png')
    Image.fromarray(_img('black')).save(d / 'semantic_0000.png')
    found = rc.find_images(tmp_path, 'top')
    assert [p.name for p in found] == ['rgb_0000.png', 'rgb_0001.png']
    assert rc.load_image(found[-1]).shape == (48, 64, 3)
    assert rc.find_images(tmp_path, 'oblique') == []


def test_trash_motion():
    before = {'a': (0, 0, 0.105), 'b': (1, 1, 0.205)}
    settled = {'a': (0, 0, 0.100), 'b': (1, 1, 0.2001)}
    m = rc.trash_motion(before, settled, CFG['max_trash_move_m'])
    assert m['rests'] and m['updated'] and m['max_move_m'] == pytest.approx(0.005, abs=1e-6)
    flew = rc.trash_motion(before, {'a': (0, 0, 0.5), 'b': (1, 1, 0.2)}, 0.02)
    assert not flew['rests']
    stale = rc.trash_motion(before, dict(before), 0.02)
    assert stale['rests'] and not stale['updated']
    nan = rc.trash_motion(before, {'a': (0, 0, float('nan')), 'b': (1, 1, 0.2)}, 0.02)
    assert not nan['rests']
    lost = rc.trash_motion(before, {'a': (0, 0, 0.1)}, 0.02)
    assert not lost['rests'] and lost['missing'] == ['b']


def test_scan_log():
    text = ('2026-11-14 [Info] fine\n[Error] [omni.physx] triangle mesh collision on dynamic body\n'
            'Failed to resolve asset ../x.png\nconvexHull approximation used\n[Error] other\n')
    m = rc.scan_log(text, CFG['log_patterns'])
    assert m['error']['count'] == 2
    assert m['triangle mesh']['count'] == 1
    assert m['convexHull']['count'] == 1
    assert m['failed to resolve']['count'] == 1


def test_newest_log(tmp_path):
    import os
    import time
    a, b = tmp_path / 'Kit' / 'a.log', tmp_path / 'Kit' / 'b.log'
    a.parent.mkdir()
    a.write_text('a')
    b.write_text('b')
    now = time.time()
    os.utime(a, (now - 100, now - 100))
    assert rc.newest_log([tmp_path, tmp_path / 'missing']) == b
    assert rc.newest_log([tmp_path], since=now + 10) is None


def test_evaluate():
    good = rc.image_check(_img('normal'), CFG['min_mean'], CFG['min_std'])
    bad = rc.image_check(_img('black'), CFG['min_mean'], CFG['min_std'])
    rest = rc.trash_motion({'a': (0, 0, 0.1)}, {'a': (0, 0, 0.096)}, 0.02)
    clean = rc.scan_log('[Error] unrelated', CFG['log_patterns'])
    fail = CFG['fail_log_patterns']
    assert rc.evaluate({'top': good}, rest, clean, fail)['ok']        # 'error' alone doesn't fail
    v = rc.evaluate({'top': good, 'oblique': bad, 'robot_eye': None}, rest, clean, fail)
    assert not v['ok'] and len(v['failures']) == 2
    assert not rc.evaluate({'top': good}, None, clean, fail)['ok']
    hull = rc.scan_log('convexHull fallback', CFG['log_patterns'])
    assert not rc.evaluate({'top': good}, rest, hull, fail)['ok']
    assert not rc.evaluate({}, rest, clean, fail)['ok']


def test_scene_cameras_and_trash(built, scene_of):
    stage = open_stage(built(0)[0])
    cams = rc.scene_cameras(stage)
    assert [c[0] for c in cams] == ['top', 'oblique', 'robot_eye']
    assert dict((c[0], c[2]) for c in cams)['robot_eye'] == (640, 480)
    assert all(c[1].startswith('/World/Cameras/') for c in cams)
    pos = rc.trash_positions(stage)
    scene = scene_of(0)
    assert set(pos) == {it['id'] for it in scene['trash']}
    for it in scene['trash']:
        assert pos[it['id']][:2] == pytest.approx(tuple(it['pose'][:2]), abs=1e-6)
