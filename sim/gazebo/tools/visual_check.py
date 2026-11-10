#!/usr/bin/env python3
"""B5 visual check: the trash items on sand, seen by head-camera-like cameras at 1, 2 and 4 m and in
the CLASSIFY close-up (~0.3 m), rendered headless; frames saved and tiled into a contact sheet.

  python3 sim/gazebo/tools/visual_check.py [--out results/b5] [--seconds 20]

Builds results/b5/visual_check.sdf with load_scene.object_model() (the same visuals as the benchmark
worlds), runs `gz sim -s -r --headless-rendering` for --seconds of wall time with a subscriber on each
camera topic (gz only renders a camera somebody listens to), then tiles the newest frame of each camera
into contact_sheet.png. Camera: 640 x 480, hfov from config/sensors.yaml, 0.45 m up, pitched 20 deg
down (the Go2 head camera; beach_description). Run it with no other simulation on the machine.
"""
import argparse
import math
import os
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'sim' / 'gazebo'))
import load_scene as ls  # noqa: E402

ITEMS = ['pet_bottle_500', 'can_330', 'food_tin', 'glass_bottle_330', 'glass_jar', 'poly_cup', 'cigarette_pack',
         'food_container']
ROWS = {'1m': 1.0, '2m': 2.0, '4m': 4.0}
CAM_H, PITCH = 0.45, math.radians(20)


def build(out):
    objects = yaml.safe_load(open(REPO / 'sim' / 'common' / 'objects.yaml'))
    cam_cfg = yaml.safe_load(open(REPO / 'config' / 'sensors.yaml'))['camera']
    sdf = ET.Element('sdf', version='1.9')
    world = ls.sub(sdf, 'world', name='visual_check')
    ls.plugin(world, 'gz-sim-scene-broadcaster-system', 'gz::sim::systems::SceneBroadcaster')
    ls.plugin(world, 'gz-sim-sensors-system', 'gz::sim::systems::Sensors', [('render_engine', 'ogre2')])
    scn = ls.sub(world, 'scene')
    ls.sub(scn, 'ambient', '0.6 0.6 0.6 1')
    ls.sub(scn, 'background', '0.62 0.80 0.95 1')
    ls.sub(scn, 'shadows', 'true')
    sun = ls.sub(world, 'light', type='directional', name='sun')
    ls.sub(sun, 'cast_shadows', 'true')
    ls.sub(sun, 'pose', '0 0 10 0 0 0')
    ls.sub(sun, 'diffuse', '0.9 0.88 0.82 1')
    ls.sub(sun, 'specular', '0.2 0.2 0.2 1')
    ls.sub(sun, 'direction', '-0.4 0.3 -0.85')
    ground = ls.sub(world, 'model', name='sand')
    ls.sub(ground, 'static', 'true')
    link = ls.sub(ground, 'link', name='link')
    v = ls.sub(link, 'visual', name='visual')
    ls.sub(ls.sub(ls.sub(v, 'geometry'), 'plane'), 'size', '120 120')
    mat = ls.material(v, [1, 1, 1, 1])
    ls.sub(ls.sub(ls.sub(mat, 'pbr'), 'metal'), 'albedo_map', (ls.COMMON / 'textures' / 'sand_albedo.png').as_uri())
    topics = []
    for r, (row, d) in enumerate(list(ROWS.items()) + [('closeup', None)]):
        y0 = 10.0 * r - 15.0
        for k, name in enumerate(ITEMS):
            spec = objects['trash'][name]
            if d is None:                                   # close-up: one item at a time, held in the air
                x, y, z, yaw = 0.30 * math.cos(PITCH), y0 + 1.0 * k, CAM_H - 0.30 * math.sin(PITCH) - 0.03, math.pi / 2
            else:
                x, y, z, yaw = d, y0 + (k - 3.5) * 0.17 * d, 0.0, 0.0
            ls.object_model(world, f'{row}_{name}', spec, (x, y, yaw), z, 1, static=True)
        for k in (range(len(ITEMS)) if d is None else [None]):
            cname = f'cam_{row}' + (f'_{k}' if k is not None else '')
            cy = y0 + (1.0 * k if k is not None else 0.0)
            m = ls.sub(world, 'model', name=cname)
            ls.sub(m, 'static', 'true')
            ls.sub(m, 'pose', ls.fmt(0.0, cy, CAM_H, 0.0, PITCH, 0.0))
            s = ls.sub(ls.sub(m, 'link', name='link'), 'sensor', name='rgb', type='camera')
            topic = f'visual_check/{cname}'
            topics.append(topic)
            ls.sub(s, 'topic', topic)
            ls.sub(s, 'update_rate', '2')
            ls.sub(s, 'always_on', '1')
            cam = ls.sub(s, 'camera')
            ls.sub(cam, 'horizontal_fov', ls.fmt(cam_cfg['horizontal_fov_rad']))
            img = ls.sub(cam, 'image')
            ls.sub(img, 'width', str(cam_cfg['width']))
            ls.sub(img, 'height', str(cam_cfg['height']))
            clip = ls.sub(cam, 'clip')
            ls.sub(clip, 'near', '0.05')
            ls.sub(clip, 'far', '50')
            save = ls.sub(cam, 'save', enabled='true')
            ls.sub(save, 'path', str(out / 'frames' / cname))
    path = out / 'visual_check.sdf'
    ET.indent(sdf)
    path.write_text(ET.tostring(sdf, encoding='unicode'))
    return path, topics


def sheet(out, topics):
    tiles = []
    for t in topics:
        cname = t.split('/')[-1]
        frames = sorted((out / 'frames' / cname).glob('*.png'))
        if frames:
            im = Image.open(frames[-1]).convert('RGB').resize((320, 240))
            ImageDraw.Draw(im).text((5, 5), cname, fill=(255, 0, 0))
            tiles.append(im)
    if not tiles:
        return None
    cols = 4
    rows = math.ceil(len(tiles) / cols)
    s = Image.new('RGB', (cols * 320, rows * 240), (40, 40, 40))
    for i, im in enumerate(tiles):
        s.paste(im, ((i % cols) * 320, (i // cols) * 240))
    path = out / 'contact_sheet.png'
    s.save(path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(REPO / 'results' / 'b5'))
    ap.add_argument('--seconds', type=float, default=20.0)
    a = ap.parse_args()
    out = Path(a.out)
    (out / 'frames').mkdir(parents=True, exist_ok=True)
    world, topics = build(out)
    env = dict(os.environ, GZ_PARTITION='visual_check', GZ_IP='127.0.0.1')
    gz = subprocess.Popen(['gz', 'sim', '-s', '-r', '--headless-rendering', str(world)], env=env,
                          stdout=open(out / 'gz.log', 'w'), stderr=subprocess.STDOUT, start_new_session=True)
    time.sleep(3)
    subs = [subprocess.Popen(['gz', 'topic', '-e', '-t', '/' + t], env=env, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True) for t in topics]
    try:
        time.sleep(a.seconds)
    finally:
        for p in subs + [gz]:
            try:
                os.killpg(p.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        time.sleep(2)
        for p in subs + [gz]:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    print('contact sheet:', sheet(out, topics))


if __name__ == '__main__':
    main()
