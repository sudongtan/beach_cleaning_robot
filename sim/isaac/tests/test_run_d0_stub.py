"""Run sim/isaac/run_d0.py on the Mac against stub Isaac modules backed by the fake simulator.

Not a physics test: it catches Python errors in the GPU script's glue (prim lookup on the composed
stage, array shapes, joint mapping, result.json, the exit path) before any GPU time. The stubs mirror
the 6.2 APIs run_d0.py uses (tag v6.2.0): Articulation / RigidPrim return warp-like arrays with
.numpy() and shape (1, n); SimulationManager.step(steps=n); app.play; stage.open_stage -> (ok, stage).
"""

import json
import runpy
import sys
import types

import numpy as np
import pytest
from pxr import Usd, UsdGeom

from conftest import ISAAC

pytest.importorskip('pinocchio')
from test_d0 import CFG, URDF, FakeIO, LAPTOP, fake_import  # noqa: E402  (skips without the laptop files)

import compose  # noqa: E402


class Arr:
    def __init__(self, a):
        self.a = np.asarray(a, dtype=np.float32)

    def numpy(self):
        return self.a


def install(monkeypatch, world, stage_path, calls):
    isaacsim = types.ModuleType('isaacsim')

    class App:
        def __init__(self, cfg):
            calls.append(('app', cfg))

        def update(self):
            pass

        def close(self):
            calls.append(('close',))
            raise SystemExit(0)          # like Isaac's: close() exits 0 itself
    isaacsim.SimulationApp = App

    app_mod = types.ModuleType('isaacsim.core.experimental.utils.app')
    app_mod.play = lambda **k: calls.append(('play',))
    stage_mod = types.ModuleType('isaacsim.core.experimental.utils.stage')
    stage_mod.open_stage = lambda p: (True, Usd.Stage.Open(p))
    stage_mod.is_stage_loading = lambda: False

    prims = types.ModuleType('isaacsim.core.experimental.prims')

    class Articulation:
        def __init__(self, path):
            calls.append(('articulation', path))
            self.dof_names = list(world.dof_names)

        def get_dof_positions(self):
            return Arr(world.q()[None])

        def get_dof_velocities(self):
            return Arr(world.dq()[None])

        def set_dof_positions(self, q):
            assert q.shape == (1, len(world.dof_names))
            world.set_q(q[0])

        def set_dof_position_targets(self, q):
            assert q.shape == (1, len(world.dof_names))
            world.set_targets(q[0])

        def get_world_poses(self):                  # the articulation root (wxyz)
            p, q = world.body_pose()
            return Arr(np.asarray(p)[None]), Arr(np.asarray(q)[None])

        def set_world_poses(self, p, q):
            assert p.shape == (1, 3) and q.shape == (1, 4)
            calls.append(('reset_root',))

        def set_dof_velocities(self, v):
            assert v.shape == (1, len(world.dof_names))

        def get_dof_gains(self):
            return Arr(np.ones((1, len(world.dof_names)))), Arr(np.ones((1, len(world.dof_names))))

    class RigidPrim:
        def __init__(self, path):
            self.path = path

        def _pose(self):
            name = self.path.rsplit('/', 1)[-1]
            if name == CFG['robot']['body_link']:
                return world.body_pose()
            if name == CFG['robot']['gripper_link']:
                return world.link_pose(name)
            return world.item_pose(name)

        def get_world_poses(self):
            p, q = self._pose()
            return Arr(np.asarray(p)[None]), Arr(np.asarray(q)[None])

        def get_velocities(self):
            return Arr(np.zeros((1, 3))), Arr(np.zeros((1, 3)))

        def set_world_poses(self, p, q):
            assert p.shape == (1, 3) and q.shape == (1, 4)
            world.place_item(self.path.rsplit('/', 1)[-1], p[0], q[0])

        def set_velocities(self, lin, ang):
            assert lin.shape == ang.shape == (1, 3)
    prims.Articulation, prims.RigidPrim = Articulation, RigidPrim

    sm_mod = types.ModuleType('isaacsim.core.simulation_manager')

    class SimulationManager:
        dt = None

        @classmethod
        def set_physics_dt(cls, dt):
            assert not any(c == ('play',) for c in calls), 'set_physics_dt after play (6.2 raises)'
            cls.dt = dt

        @classmethod
        def get_physics_dt(cls):
            return cls.dt

        @classmethod
        def step(cls, *, steps=1):
            world.step(steps)
    sm_mod.SimulationManager = SimulationManager

    mods = {'isaacsim': isaacsim, 'isaacsim.core': types.ModuleType('isaacsim.core'),
            'isaacsim.core.experimental': types.ModuleType('isaacsim.core.experimental'),
            'isaacsim.core.experimental.utils': types.ModuleType('isaacsim.core.experimental.utils'),
            'isaacsim.core.experimental.utils.app': app_mod, 'isaacsim.core.experimental.utils.stage': stage_mod,
            'isaacsim.core.experimental.prims': prims, 'isaacsim.core.simulation_manager': sm_mod}
    for k, v in mods.items():
        monkeypatch.setitem(sys.modules, k, v)


def _run(monkeypatch, tmp_path, mode, extra=()):
    robot = tmp_path / 'robot.usda'
    fake_import(robot)
    stage, meta = compose.build('flat', robot, tmp_path / f'stage_{mode}.usda', pick=(mode == 'pick'))
    x, y, z, yaw = meta['spawn_base_link']
    world = FakeIO((x, y, z, yaw))
    st = Usd.Stage.Open(str(stage))
    for n in meta['pick_items']:              # the fake world starts with the items where compose parked them
        m = UsdGeom.Xformable(st.GetPrimAtPath(f'/World/PickItems/{n}')).ComputeLocalToWorldTransform(0)
        world.place_item(n, np.array(m.ExtractTranslation()), np.array([1.0, 0, 0, 0]))
    calls = []
    install(monkeypatch, world, stage, calls)
    out = tmp_path / f'out_{mode}'
    monkeypatch.setattr(sys, 'argv', ['run_d0.py', '--stage', str(stage), '--mode', mode, '--out', str(out), *extra])
    monkeypatch.setattr('os._exit', lambda c: (_ for _ in ()).throw(SystemExit(c)))
    with pytest.raises(SystemExit):
        runpy.run_path(str(ISAAC / 'run_d0.py'), run_name='__main__')
    return json.loads((out / 'result.json').read_text()), calls, out


def test_run_d0_walk_glue(monkeypatch, tmp_path):
    res, calls, out = _run(monkeypatch, tmp_path, 'walk')
    assert 'exception' not in res, res.get('exception')
    assert res['paths']['articulation_root'] == '/World/Robot'
    assert res['paths']['body'] == '/World/Robot/trunk'
    assert res['physics_dt_reported'] == pytest.approx(1 / 200)
    assert [p['name'] for p in res['walk']['phases']] == ['stand', 'walk_0.2', 'turn_0.3', 'stop']
    assert res['ok'] is False and any('walk_0.2' in r for r in res['reasons'])   # the fake doesn't walk
    assert (out / 'trace.npy').is_file()
    assert calls[0] == ('app', {'headless': True}) and calls[-1] == ('close',)


def test_run_d0_pick_glue(monkeypatch, tmp_path):
    res, calls, out = _run(monkeypatch, tmp_path, 'pick', ['--cases', '0,4'])
    assert 'exception' not in res, res.get('exception')
    assert [p['case'] for p in res['picks']] == [0, 1]
    assert res['ok'], res['reasons']
    for p in res['picks']:
        assert p['tips_item_top_at_attach_m'] < CFG['pick']['pass']['max_tips_item_top_m']
        assert p['fall_phase'] is None and p['legs'] == 'policy'
    assert ('reset_root',) in calls                    # the second case starts from the reset pose
    assert res['reset_errors'] is None or 'set_dof_velocities' not in res['reset_errors']
