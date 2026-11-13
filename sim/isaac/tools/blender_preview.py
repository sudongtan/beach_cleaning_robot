"""Blender preview of a generated scene (Mac, no GPU rental): renders every scene camera.

    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
        --python sim/isaac/tools/blender_preview.py -- --usd PATH --out DIR [--samples 32] [--scale 50]

Imports the .usda (bpy.ops.wm.usd_import; guide prims, i.e. the colliders, are skipped),
renders /World/Cameras/* with Cycles on the GPU (Metal) and writes DIR/<camera>.png.
Blender converts UsdPreviewSurface to its own material and its light units differ from
Isaac's, so this checks layout, missing textures, positions, camera directions and light
directions, NOT Isaac's look.

Lights: the sun is kept as imported. Blender honours USD's `normalize = false` (the default):
the intensity is spread over the sun's small disc, so 3000 in render.yaml is a plain daylight
exposure, not 3000 W/m^2. Overriding it with a Blender-style strength (e.g. 4) renders black.
Blender (5.3) makes no world from a DomeLight without a texture, so the preview adds a world in
the dome's colour at PREVIEW_WORLD_STRENGTH.
"""

import argparse
import math
import re
import sys
from pathlib import Path

import bpy

PREVIEW_WORLD_STRENGTH = 1.0    # the added sky world (no Blender equivalent of the dome's units)


def args_after_dashes():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument('--usd', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--samples', type=int, default=32)
    ap.add_argument('--scale', type=int, default=50, help='resolution percentage')
    ap.add_argument('--cpu', action='store_true', help='render on the CPU instead of Metal')
    return ap.parse_args(argv)


def usd_camera_resolutions(usd_text):
    """{camera name: (w, h)} from the builder's customData beach:resolution (text .usda)."""
    out = {}
    for m in re.finditer(r'def Camera "(\w+)"\s*\(\s*customData = \{.*?int2 resolution = \((\d+), (\d+)\)',
                         usd_text, re.S):
        out[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    return out


def usd_dome_color(usd_text):
    m = re.search(r'def DomeLight "\w+".*?color3f inputs:color = \(([^)]*)\)', usd_text, re.S)
    return tuple(float(v) for v in m.group(1).split(',')) if m else None


def setup_cycles(samples, cpu):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    device = 'CPU'
    if not cpu:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'METAL'
        prefs.refresh_devices()
        gpus = [d for d in prefs.devices if d.type == 'METAL']
        for d in prefs.devices:
            d.use = d.type == 'METAL'
        if gpus:
            device = 'GPU'
    scene.cycles.device = device
    print(f'cycles device: {device}', flush=True)


def main():
    a = args_after_dashes()
    usd = Path(a.usd).resolve()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    res = bpy.ops.wm.usd_import(filepath=str(usd), import_guide=False, import_proxy=False,
                                import_render=True, import_textures_mode='IMPORT_NONE',
                                create_world_material=True)
    print(f'usd_import: {res}', flush=True)
    usd_text = usd.read_text()
    resolutions = usd_camera_resolutions(usd_text)
    scene = bpy.context.scene

    # What came in (for the log): meshes, materials without an image where one was expected.
    meshes = [o for o in scene.objects if o.type == 'MESH']
    missing = [img.name for img in bpy.data.images if img.source == 'FILE' and not img.has_data
               and not Path(bpy.path.abspath(img.filepath)).exists()]
    print(f'objects {len(scene.objects)}, meshes {len(meshes)}, images {len(bpy.data.images)}, '
          f'missing images {missing}', flush=True)

    for o in scene.objects:
        if o.type == 'LIGHT':
            fwd = o.matrix_world.to_3x3() @ __import__('mathutils').Vector((0, 0, -1))
            print(f'light {o.name}: {o.data.type}, imported energy {o.data.energy:.3g}, '
                  f'direction {tuple(round(v, 3) for v in fwd)}'
                  + f', normalize {o.data.normalize}', flush=True)
    dome = usd_dome_color(usd_text)
    if scene.world is None and dome is not None:
        w = bpy.data.worlds.new('usd_dome')
        w.use_nodes = True
        bg = w.node_tree.nodes['Background']
        bg.inputs['Color'].default_value = (*dome, 1.0)
        bg.inputs['Strength'].default_value = PREVIEW_WORLD_STRENGTH
        scene.world = w
        print(f'world: none imported; added one in the dome colour {dome}', flush=True)

    setup_cycles(a.samples, a.cpu)
    scene.render.resolution_percentage = a.scale
    scene.render.image_settings.file_format = 'PNG'
    cams = sorted((o for o in scene.objects if o.type == 'CAMERA'), key=lambda o: o.name)
    if not cams:
        raise SystemExit('no cameras imported')
    for cam in cams:
        name = cam.name
        w, h = resolutions.get(name, (1280, 720))
        cam.data.sensor_fit = 'HORIZONTAL'      # keep the USD horizontal FOV
        hfov = 2 * math.atan(cam.data.sensor_width / (2 * cam.data.lens))
        scene.render.resolution_x, scene.render.resolution_y = w, h
        scene.camera = cam
        scene.render.filepath = str(out / f'{name}.png')
        print(f'render {name}: {w}x{h} at {a.scale} %, hfov {hfov:.4f} rad, clip '
              f'{cam.data.clip_start:.3g}-{cam.data.clip_end:.3g}, at '
              f'{tuple(round(v, 3) for v in cam.matrix_world.translation)}', flush=True)
        bpy.ops.render.render(write_still=True)
    print(f'wrote {len(cams)} images to {out}', flush=True)


main()
