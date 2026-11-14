# REQUIRES_GPU
"""D0b / D0c in Isaac Sim 6.2: stand + walk on a composed stage, or the attach-mode picks.

    /isaac-sim/python.sh sim/isaac/run_d0.py --stage OUT.usda --mode walk|pick --out DIR
        [--gyro api|fd] [--frames]

The logic is in d0_runner.py (tested on the Mac with a fake simulator); this file is only the Isaac
side (IsaacIO). APIs from tag v6.2.0 of github.com/isaac-sim/IsaacSim: SimulationApp first;
isaacsim.core.experimental.utils.stage.open_stage; SimulationManager.set_physics_dt (before play)
and SimulationManager.step(steps=n) (standalone_examples/api/isaacsim.core.experimental.api/
simulation_callbacks.py); isaacsim.core.experimental.utils.app.play; isaacsim.core.experimental.prims
Articulation (dof_names, get/set_dof_positions, set_dof_position_targets, get_dof_gains; radians)
and RigidPrim (get_world_poses -> wxyz, get_velocities, set_world_poses, set_velocities).
Writes DIR/result.json (judged from the file: SimulationApp.close() exits 0 itself, D0 prep).
--frames saves an image sequence (Replicator); it may advance rendering only, but a run with frames
is never the one judged (gpu_session_d0.sh runs the checks without it).
"""

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

T0 = time.time()
from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({'headless': True})

import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from isaacsim.core.experimental.prims import Articulation, RigidPrim  # noqa: E402
from pxr import Usd, UsdPhysics  # noqa: E402

try:
    from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
except ImportError:                                                  # module path differs
    from isaacsim.core.simulation_manager.impl.simulation_manager import SimulationManager  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'common'))
import d0_runner  # noqa: E402
import robot_model  # noqa: E402
import walk  # noqa: E402


def np_(a):
    return a.numpy() if hasattr(a, 'numpy') else np.asarray(a)


class IsaacIO:
    def __init__(self, stage, robot_root, body, links, items, physics_dt, frames_dir=None, camera=None):
        self.stage = stage
        self.art = Articulation(robot_root)
        self.body = RigidPrim(body)
        self.links = {k: RigidPrim(v) for k, v in links.items()}
        self.items = {k: RigidPrim(v) for k, v in items.items()}
        self.item_paths = items
        self.dt = physics_dt
        self.n_steps = 0
        self.dof_names = list(self.art.dof_names)
        self.frames = None
        if frames_dir and camera:
            self._frames_setup(frames_dir, camera)

    def _frames_setup(self, frames_dir, camera):
        try:
            import omni.replicator.core as rep
            rep.orchestrator.set_capture_on_play(False)
            self.rp = rep.create.render_product(camera, (640, 480), name='view')
            backend = rep.backends.get('DiskBackend')
            backend.initialize(output_dir=str(frames_dir))
            self.writer = rep.writers.get('BasicWriter')
            self.writer.initialize(backend=backend, rgb=True)
            self.writer.attach(self.rp)
            self.frames = rep
            self.frame_errors = 0
        except Exception:
            self.frames = None
            self.frames_error = traceback.format_exc()

    def q(self):
        return np_(self.art.get_dof_positions())[0]

    def dq(self):
        return np_(self.art.get_dof_velocities())[0]

    def set_q(self, q):
        self.art.set_dof_positions(np.asarray(q, dtype=np.float32)[None, :])

    def set_targets(self, q):
        self.art.set_dof_position_targets(np.asarray(q, dtype=np.float32)[None, :])

    def body_pose(self):
        p, q = self.body.get_world_poses()
        return np_(p)[0].astype(float), np_(q)[0].astype(float)

    def body_ang_vel_world(self):
        lin, ang = self.body.get_velocities()
        return np_(ang)[0].astype(float)

    def link_pose(self, name):
        p, q = self.links[name].get_world_poses()
        return np_(p)[0].astype(float), np_(q)[0].astype(float)

    def item_pose(self, name):
        p, q = self.items[name].get_world_poses()
        return np_(p)[0].astype(float), np_(q)[0].astype(float)

    def place_item(self, name, xyz, wxyz):
        self.items[name].set_world_poses(np.asarray(xyz, np.float32)[None, :], np.asarray(wxyz, np.float32)[None, :])
        self.items[name].set_velocities(np.zeros((1, 3), np.float32), np.zeros((1, 3), np.float32))

    def hold_item(self, name, xyz, wxyz):
        self.place_item(name, xyz, wxyz)

    def set_item_kinematic(self, name, on):
        prim = self.stage.GetPrimAtPath(self.item_paths[name])
        UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr().Set(bool(on))

    def step(self, n):
        SimulationManager.step(steps=int(n))
        self.n_steps += int(n)

    def time(self):
        return self.n_steps * self.dt

    def capture(self, tag):
        if self.frames is None:
            return
        try:
            self.frames.orchestrator.step(delta_time=0.0, rt_subframes=8)
        except Exception:
            self.frame_errors += 1
            if self.frame_errors > 3:
                self.frames = None


def find(stage, root, name=None, api=None):
    for p in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if (name is None or p.GetName() == name) and (api is None or p.HasAPI(api)):
            return str(p.GetPath())
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True)
    ap.add_argument('--mode', choices=['walk', 'pick'], required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--gyro', choices=['api', 'fd'], default='api')
    ap.add_argument('--frames', action='store_true')
    ap.add_argument('--cases', default='all', help='pick: all or comma-separated indices')
    a, _ = ap.parse_known_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    res = {'mode': a.mode, 'stage': a.stage, 'gyro': a.gyro, 'app_start_s': round(time.time() - T0, 2)}
    try:
        cfg = robot_model.load_robot_cfg()
        laptop = robot_model.laptop_dir(cfg)
        meta = json.loads(Path(a.stage + '.json').read_text())
        urdf = robot_model.Urdf(laptop / cfg['urdf_import']['urdf'])
        r = cfg['robot']
        policy, shaper, _ = walk.build_policy(laptop, cfg)
        dt = 1.0 / float(cfg['policy']['physics_hz'])
        t = time.time()
        ok, stage = stage_utils.open_stage(str(Path(a.stage).resolve()))
        if not ok:
            raise RuntimeError(f'open_stage failed: {a.stage}')
        while stage_utils.is_stage_loading():
            simulation_app.update()
        res['stage_open_s'] = round(time.time() - t, 2)
        robot_root = find(stage, meta['robot_prim'], api=UsdPhysics.ArticulationRootAPI)
        # the rigid-body prim: the importer also names a geometry child after its link (GPU D0-1)
        body = find(stage, meta['robot_prim'], name=r['body_link'], api=UsdPhysics.RigidBodyAPI)
        grip = find(stage, meta['robot_prim'], name=r['gripper_link'], api=UsdPhysics.RigidBodyAPI)
        res['paths'] = {'articulation_root': robot_root, 'body': body, 'gripper': grip}
        if not (robot_root and body and grip):
            raise RuntimeError(f'robot prims not found: {res["paths"]}')
        items = {n: f"{meta['pick_root']}/{n}" for n in meta.get('pick_items', [])}
        SimulationManager.set_physics_dt(dt)       # before play (6.2: raises while playing)
        app_utils.play()
        io = IsaacIO(stage, robot_root, body, {r['gripper_link']: grip}, items, dt,
                     out / 'frames' if a.frames else None,
                     '/World/Cameras/view_' + ('pick' if a.mode == 'pick' else 'walk'))
        io.step(1)
        res['dof_names'] = io.dof_names
        try:
            kp, kd = io.art.get_dof_gains()
            res['dof_gains'] = {'kp': np_(kp)[0].round(4).tolist(), 'kd': np_(kd)[0].round(4).tolist()}
        except Exception as e:
            res['dof_gains'] = f'{type(e).__name__}: {e}'
        res['physics_dt_reported'] = float(SimulationManager.get_physics_dt())
        sys.path.insert(0, str(HERE.parent / 'common'))
        import export_terrain
        import terrain as terrain_mod
        scene = export_terrain.load_scene(int(meta['seed']), Path(meta['scenes_yaml']))
        th = lambda x, y: float(terrain_mod.height_at(scene['terrain'], x, y))  # noqa: E731
        t = time.time()
        if a.mode == 'walk':
            w = d0_runner.run_walk(io, cfg, policy, shaper, th, a.gyro, cfg['capture']['every_s'])
            ok_, reasons, nums = d0_runner.judge_walk(w, meta['reference'], cfg['walk']['pass'])
            trace = w.pop('trace')
            np.save(out / 'trace.npy', np.asarray(trace))
            res.update(walk=w, ok=ok_, reasons=reasons, numbers=nums)
        else:
            import pick as pick_mod
            _, stance, cases = pick_mod.load_cases(laptop / cfg['pick']['plans'], cfg['pick']['stance'])
            if a.cases != 'all':
                cases = [cases[int(k)] for k in a.cases.split(',')]
            comp = robot_model.compartments(urdf, r['root_link'])
            bar_to_tcp = urdf.fixed_transform(r['gripper_link'], r['tcp_link'])
            body_to_base = urdf.fixed_transform(r['body_link'], r['root_link'])
            park = {n: io.item_pose(n) for n in items}
            results = d0_runner.run_picks(
                io, cfg, policy, shaper, cases,
                lambda b, p, m: robot_model.in_compartment(comp, b, p, m),
                bar_to_tcp, body_to_base, float(stance['ground_z_in_base_link']), park, a.gyro,
                cfg['capture']['every_s'], urdf.fixed_transform(r['gripper_link'], r['tips_link']))
            res.update(picks=results, ok=all(x['ok'] for x in results),
                       reasons=[f"case {x['case']} {x['item']}: lifted {x['lifted']}, in compartment "
                                f"{x['in_compartment']}, fell {x['fell']}" for x in results if not x['ok']])
        res['run_wall_s'] = round(time.time() - t, 2)
        res['sim_s'] = round(io.time(), 3)
        if a.frames:
            res['frames'] = 'ok' if io.frames is not None else getattr(io, 'frames_error', 'stopped after errors')
            try:
                io.frames.orchestrator.wait_until_complete()
            except Exception:
                pass
    except Exception:
        res['ok'] = False
        res['exception'] = traceback.format_exc()
    res['total_s'] = round(time.time() - T0, 2)
    (out / 'result.json').write_text(json.dumps(res, indent=2, default=lambda o: np.asarray(o).tolist()))
    print(json.dumps({k: res.get(k) for k in ('ok', 'reasons', 'numbers', 'exception', 'total_s')}, indent=2,
                     default=str), flush=True)
    simulation_app.close()
    os._exit(0)


if __name__ == '__main__':
    main()
