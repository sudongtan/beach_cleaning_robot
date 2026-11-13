"""Run sim/isaac/render_check.py on the Mac against stub Omniverse modules.

Not a render test: it catches Python errors in the GPU script's glue (call order, paths,
json output, exit code) before any GPU time. The stubs open the real .usda with pxr, write
one PNG per writer on each orchestrator.step, and advance a fake timeline that drops the
trash by `drop` metres.
"""

import json
import runpy
import sys
import types

import numpy as np
import pytest
from PIL import Image
from pxr import Gf, Usd, UsdGeom

from conftest import ISAAC


def _install_stubs(monkeypatch, image_kind='normal', drop=0.004, open_ok=True):
    state = {'stage': None, 'writers': [], 'time': 0.0, 'playing': False, 'calls': []}

    class App:
        def __init__(self, launch_config=None):
            state['calls'].append(('app', launch_config))

        def update(self):
            if state['playing']:
                state['time'] += 1.0 / 60.0
                if state['time'] > 0.5 and not state.get('dropped'):
                    for p in state['stage'].GetPrimAtPath('/World/Trash').GetChildren():
                        op = UsdGeom.Xformable(p).GetOrderedXformOps()[0]
                        v = op.Get()
                        op.Set(Gf.Vec3d(v[0], v[1], v[2] - drop))
                    state['dropped'] = True

        def close(self):
            state['calls'].append(('close',))

    isaacsim = types.ModuleType('isaacsim')
    isaacsim.SimulationApp = App
    stage_mod = types.ModuleType('isaacsim.core.experimental.utils.stage')

    def open_stage(path):
        state['stage'] = Usd.Stage.Open(path)
        return open_ok, state['stage']
    stage_mod.open_stage = open_stage
    stage_mod.is_stage_loading = lambda: False

    carb = types.ModuleType('carb')
    carb_settings = types.ModuleType('carb.settings')
    settings = {}
    carb_settings.get_settings = lambda: types.SimpleNamespace(set=settings.__setitem__)
    carb.settings = carb_settings

    rep = types.ModuleType('omni.replicator.core')

    class Backend:
        def initialize(self, output_dir):
            self.dir = output_dir

    class Writer:
        def initialize(self, backend, rgb):
            assert rgb
            self.backend, self.n = backend, 0

        def attach(self, rp):
            self.rp = rp
            state['writers'].append(self)

        def detach(self):
            state['writers'].remove(self)

    class RP:
        def __init__(self, path, res, name):
            assert state['stage'].GetPrimAtPath(path), path
            self.res, self.name = res, name

        def destroy(self):
            pass

    def step(rt_subframes=1):
        from pathlib import Path
        for w in state['writers']:
            w_, h_ = w.rp.res
            rng = np.random.default_rng(w.n)
            img = (np.zeros((h_, w_, 3), np.uint8) if image_kind == 'black'
                   else rng.integers(0, 255, (h_, w_, 3)).astype(np.uint8))
            d = Path(w.backend.dir)
            d.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img).save(d / f'rgb_{w.n:04d}.png')
            w.n += 1

    rep.orchestrator = types.SimpleNamespace(set_capture_on_play=lambda v: None, step=step,
                                             wait_until_complete=lambda: None)
    rep.create = types.SimpleNamespace(render_product=lambda path, res, name: RP(path, res, name))
    rep.backends = types.SimpleNamespace(get=lambda n: Backend())
    rep.writers = types.SimpleNamespace(get=lambda n: Writer())

    class Timeline:
        def play(self):
            state['playing'] = True

        def pause(self):
            state['playing'] = False

        def stop(self):
            state['playing'] = False

        def get_current_time(self):
            return state['time']

    timeline = types.ModuleType('omni.timeline')
    timeline.get_timeline_interface = lambda: Timeline()
    omni = types.ModuleType('omni')
    omni.replicator = types.ModuleType('omni.replicator')
    omni.replicator.core = rep
    omni.timeline = timeline
    for name, mod in {'isaacsim': isaacsim, 'isaacsim.core': types.ModuleType('isaacsim.core'),
                      'isaacsim.core.experimental': types.ModuleType('isaacsim.core.experimental'),
                      'isaacsim.core.experimental.utils': types.ModuleType('isaacsim.core.experimental.utils'),
                      'isaacsim.core.experimental.utils.stage': stage_mod,
                      'carb': carb, 'carb.settings': carb_settings, 'omni': omni,
                      'omni.replicator': omni.replicator, 'omni.replicator.core': rep,
                      'omni.timeline': timeline}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return state, settings


def _run(monkeypatch, usd, out):
    code = {}

    def fake_exit(c):
        code['c'] = c
        raise SystemExit(c)
    monkeypatch.setattr('os._exit', fake_exit)
    monkeypatch.setattr(sys, 'argv', ['render_check.py', '--usd', str(usd), '--out', str(out)])
    with pytest.raises(SystemExit):
        runpy.run_path(str(ISAAC / 'render_check.py'), run_name='__main__')
    return code['c'], json.loads((out / 'checks.json').read_text()), json.loads((out / 'timing.json').read_text())


def test_render_check_passes(monkeypatch, built, tmp_path):
    state, settings = _install_stubs(monkeypatch)
    out = tmp_path / 'scene_0'
    code, checks, timing = _run(monkeypatch, built(0)[0], out)
    assert code == 0, checks.get('failures')
    assert checks['ok'] and checks['cameras'] == ['top', 'oblique', 'robot_eye']
    assert checks['images']['top']['shape'] == [960, 1280, 3]
    assert checks['images']['top']['expected_shape'] == checks['images']['top']['shape']
    assert checks['physics']['rests'] and checks['physics']['updated']
    assert checks['physics']['sim_seconds'] >= 2.0
    assert state['calls'][0] == ('app', {'headless': True}) and state['calls'][-1] == ('close',)
    assert settings['/physics/updateToUsd'] is True
    for k in ('app_start_s', 'stage_open_s', 'first_frame_s', 'captures_s', 'physics_wall_s', 'total_s'):
        assert k in timing


def test_render_check_fails_on_black(monkeypatch, built, tmp_path):
    _install_stubs(monkeypatch, image_kind='black')
    code, checks, _ = _run(monkeypatch, built(0)[0], tmp_path / 's')
    assert code == 1 and not checks['ok']
    assert any('black' in f for f in checks['failures'])


def test_render_check_fails_when_trash_flies(monkeypatch, built, tmp_path):
    _install_stubs(monkeypatch, drop=-0.5)
    code, checks, _ = _run(monkeypatch, built(0)[0], tmp_path / 's')
    assert code == 1 and not checks['physics']['rests']


def test_render_check_fails_when_positions_stale(monkeypatch, built, tmp_path):
    _install_stubs(monkeypatch, drop=0.0)
    code, checks, _ = _run(monkeypatch, built(0)[0], tmp_path / 's')
    assert code == 1 and not checks['physics']['updated']


def test_render_check_reports_open_failure(monkeypatch, built, tmp_path):
    _install_stubs(monkeypatch, open_ok=False)
    code, checks, _ = _run(monkeypatch, built(0)[0], tmp_path / 's')
    assert code == 1 and 'open_stage failed' in checks['exception']
