# REQUIRES_GPU
"""Render and physics smoke test of one generated scene in Isaac Sim 6.2 (headless).

    /isaac-sim/python.sh sim/isaac/render_check.py --usd DIR/scene_0.usda --out OUT/scene_0

Opens the stage, saves one RGB image per scene camera (Replicator, as in the 6.2
example standalone_examples/api/isaacsim.replicator.examples/sdg_getting_started_01.py),
then plays 2 s of physics and checks the trash rests. Writes OUT/rgb/<camera>/rgb_*.png,
OUT/timing.json and OUT/checks.json; exit code 1 if a check fails. Restartable: it
rewrites OUT. All pure logic is in render_checks.py (tested on the Mac).

API sources (tag v6.2.0 of github.com/isaac-sim/IsaacSim): SimulationApp and Replicator
calls from sdg_getting_started_01.py; open_stage / is_stage_loading from
isaacsim.core.experimental.utils.stage; timeline play + app updates from
standalone_examples/api/isaacsim.simulation_app/load_stage.py.
"""

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

T0 = time.time()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp(launch_config={'headless': True})
T_APP = time.time()

# Omniverse imports only after the app has started.
import carb.settings  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import yaml  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import render_checks as rc  # noqa: E402

LOG_DIRS = ['/isaac-sim/kit/logs', '~/.nvidia-omniverse/logs', '/isaac-sim/.nvidia-omniverse/logs']   # 6.2: kit/logs/Kit/<app>/6.2/kit_*.log


def update(n=1):
    for _ in range(n):
        simulation_app.update()


def run(args, cfg, out, timing, checks):
    usd = str(Path(args.usd).resolve())
    t = time.time()
    ok, stage = stage_utils.open_stage(usd)
    if not ok or stage is None:
        raise RuntimeError(f'open_stage failed for {usd}')
    update(2)
    while stage_utils.is_stage_loading():
        update()
    timing['stage_open_s'] = round(time.time() - t, 3)

    cams = rc.scene_cameras(stage)
    checks['cameras'] = [c[0] for c in cams]
    before = rc.trash_positions(stage)

    # Replicator capture, as in sdg_getting_started_01.py: no capture on play, one render
    # product per camera, a DiskBackend + BasicWriter(rgb) each, step, wait.
    rep.orchestrator.set_capture_on_play(False)
    carb.settings.get_settings().set('rtx/post/dlss/execMode', 2)
    t = time.time()
    update(int(cfg['warmup_updates']))          # shader compile; the first frames are empty
    timing['warmup_s'] = round(time.time() - t, 3)
    rps, writers = [], []
    for name, path, res in cams:
        rp = rep.create.render_product(path, res, name=name)
        backend = rep.backends.get('DiskBackend')
        backend.initialize(output_dir=str(out / 'rgb' / name))
        writer = rep.writers.get('BasicWriter')
        writer.initialize(backend=backend, rgb=True)
        writer.attach(rp)
        rps.append(rp)
        writers.append(writer)
    timing['captures_s'] = []
    for i in range(int(cfg['captures'])):
        t = time.time()
        rep.orchestrator.step(rt_subframes=int(cfg['rt_subframes']))
        timing['captures_s'].append(round(time.time() - t, 3))
        if i == 0:
            timing['first_frame_s'] = round(time.time() - T0, 3)   # from script start
    t = time.time()
    rep.orchestrator.wait_until_complete()
    timing['write_wait_s'] = round(time.time() - t, 3)
    for w in writers:
        w.detach()
    for rp in rps:
        rp.destroy()

    images = {}
    for name, _, res in cams:
        files = rc.find_images(out / 'rgb', name)
        if not files:
            images[name] = None
            continue
        img = rc.load_image(files[-1])
        chk = rc.image_check(img, cfg['min_mean'], cfg['min_std'])
        chk['file'] = str(files[-1].relative_to(out))
        chk['expected_shape'] = [res[1], res[0], 3]
        images[name] = chk
    checks['images'] = images

    # Physics: play the timeline for physics_seconds of sim time, as load_stage.py plays it.
    # PhysX writes poses back to USD only with updateToUsd; the `updated` check catches it if not.
    carb.settings.get_settings().set('/physics/updateToUsd', True)
    timeline = omni.timeline.get_timeline_interface()
    t = time.time()
    timeline.play()
    update()
    sim0 = timeline.get_current_time()
    n = 0
    while timeline.get_current_time() - sim0 < float(cfg['physics_seconds']) and n < int(cfg['max_physics_updates']):
        update()
        n += 1
    timeline.pause()
    sim_s = timeline.get_current_time() - sim0
    after = rc.trash_positions(stage)
    timeline.stop()
    timing['physics_wall_s'] = round(time.time() - t, 3)
    timing['physics_sim_s'] = round(sim_s, 4)
    timing['physics_updates'] = n
    checks['physics'] = rc.trash_motion(before, after, float(cfg['max_trash_move_m']))
    checks['physics']['sim_seconds'] = round(sim_s, 4)
    if sim_s < float(cfg['physics_seconds']) - 1e-3:
        checks['physics']['rests'] = False
        checks['physics']['note'] = 'sim time did not reach physics_seconds'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--usd', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--config', default=str(HERE / 'render.yaml'))
    args, _ = ap.parse_known_args()
    cfg = yaml.safe_load(Path(args.config).read_text())['render_check']
    out = Path(args.out).resolve()
    if out.exists():
        shutil.rmtree(out)                       # restartable: a rerun starts clean
    out.mkdir(parents=True)
    timing = {'app_start_s': round(T_APP - T0, 3)}
    checks = {'usd': args.usd}
    try:
        run(args, cfg, out, timing, checks)
    except Exception:
        checks['exception'] = traceback.format_exc()
    log = rc.newest_log(LOG_DIRS, since=T0 - 5)
    checks['kit_log'] = str(log) if log else None
    matches = rc.scan_log(log.read_text(errors='replace'), cfg['log_patterns']) if log else {}
    checks['log_matches'] = matches
    verdict = rc.evaluate(checks.get('images', {}), checks.get('physics'), matches, cfg['fail_log_patterns'])
    if 'exception' in checks:
        verdict['ok'] = False
        verdict['failures'].insert(0, 'exception (see checks.json "exception")')
    checks.update(verdict)
    timing['total_s'] = round(time.time() - T0, 3)
    (out / 'timing.json').write_text(json.dumps(timing, indent=2))
    (out / 'checks.json').write_text(json.dumps(checks, indent=2))
    print(json.dumps({'ok': checks['ok'], 'failures': checks['failures'], 'timing': timing}, indent=2), flush=True)
    code = 0 if checks['ok'] else 1
    simulation_app.close()
    os._exit(code)          # close() ends the process with code 0 itself: gpu_session.sh reads checks.json


if __name__ == '__main__':
    main()
