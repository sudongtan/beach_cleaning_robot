# REQUIRES_GPU
"""Import the robot URDF into USD with Isaac Sim 6.2's URDF importer (D0b).

    /isaac-sim/python.sh sim/isaac/import_robot.py --out sim/isaac/robots/generated

As the 6.2 example standalone_examples/api/isaacsim.asset.importer.urdf/urdf_import.py:
SimulationApp first, then `URDFImporterConfig` + `URDFImporter(config).import_urdf()` (returns the
USD path). Options from sim/isaac/robot.yaml `urdf_import`; mesh packages mapped through
`ros_package_paths`. Then check_robot_usd.py --apply-drives runs on the result (separate process,
pure pxr). Writes OUT/import.json (the returned path, config, timing); results judged from files.
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

T0 = time.time()
from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({'headless': True})

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import robot_model  # noqa: E402


def enable(ext):
    try:
        import omni.kit.app
        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(ext, True)
        return True
    except Exception as e:                       # logged, the import may still work
        return f'{type(e).__name__}: {e}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a, _ = ap.parse_known_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = robot_model.load_robot_cfg()
    laptop = robot_model.laptop_dir(cfg)
    ic = cfg['urdf_import']
    rep = {'app_start_s': round(time.time() - T0, 2), 'urdf': str(laptop / ic['urdf'])}
    try:
        # the 6.2 example enables these before importing
        rep['enable'] = {e: enable(e) for e in ('isaacsim.asset.importer.urdf', 'isaacsim.robot.schema',
                                                 'omni.scene.optimizer.core')}
        from isaacsim.asset.importer.urdf.impl import URDFImporter, URDFImporterConfig
        c = URDFImporterConfig()
        c.urdf_path = str(laptop / ic['urdf'])
        c.usd_path = str(out)
        c.merge_fixed_joints = bool(ic['merge_fixed_joints'])
        c.merge_mesh = bool(ic['merge_mesh'])
        c.collision_from_visuals = bool(ic['collision_from_visuals'])
        c.collision_type = ic['collision_type']
        c.allow_self_collision = bool(ic['allow_self_collision'])
        c.fix_base = bool(ic['fix_base'])
        c.joint_drive_type = ic['joint_drive_type']
        c.joint_target_type = ic['joint_target_type']
        c.ros_package_paths = [{'name': n, 'path': str(laptop / p)} for n, p in ic['ros_packages'].items()]
        rep['config'] = {k: getattr(c, k) for k in dir(c) if not k.startswith('_') and not callable(getattr(c, k))}
        t = time.time()
        rep['usd'] = str(URDFImporter(c).import_urdf())
        rep['import_s'] = round(time.time() - t, 2)
        rep['files'] = sorted(str(p.relative_to(out)) for p in out.rglob('*') if p.is_file())
        rep['ok'] = bool(rep['usd']) and Path(rep['usd']).is_file()
    except Exception:
        rep['ok'] = False
        rep['exception'] = traceback.format_exc()
    (out / 'import.json').write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps({k: rep.get(k) for k in ('ok', 'usd', 'import_s', 'exception')}, indent=2), flush=True)
    simulation_app.close()
    os._exit(0)              # close() exits 0 itself (D0 prep); the verdict is import.json


if __name__ == '__main__':
    main()
