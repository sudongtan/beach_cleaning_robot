#!/usr/bin/env python3
"""Build a Gazebo Harmonic SDF world for one scene of bench/scenes.yaml.

    python3 sim/gazebo/load_scene.py --seed 3

Writes sim/gazebo/worlds/generated/scene_<seed>.sdf and scene_<seed>.manifest.yaml
(spawn pose, attachable items, object classes, person timeline: read by
sim_bridge_gazebo at runtime), plus the terrain exports in
sim/common/generated/scene_<seed>/. All git-ignored; never edit them by hand.

The Gazebo world frame is the map frame: origin at the south-west corner of
the scene area, x east, y north, z = 0 at the water level.

The robot is not in the world file: bringup spawns it (A1b), and its Gazebo
xacro adds one DetachableJoint per trash item from the manifest.
"""

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
COMMON = REPO / 'sim' / 'common'
sys.path.insert(0, str(COMMON))
import export_terrain  # noqa: E402
import terrain as terrain_mod  # noqa: E402
from terrain import ground_z  # noqa: E402

OUT_DIR = HERE / 'worlds' / 'generated'
WORLD_NAME = 'beach'      # constant, so bridge topics (/world/beach/...) don't depend on the seed
SPAWN_GAP = 0.005         # objects start this far above the terrain and settle


def fmt(*vals):
    return ' '.join(f'{v:.4f}'.rstrip('0').rstrip('.') if isinstance(v, float) else str(v) for v in vals)


def sub(parent, tag, text=None, **attrib):
    e = ET.SubElement(parent, tag, attrib)
    if text is not None:
        e.text = text
    return e


def plugin(parent, filename, name, children=()):
    p = sub(parent, 'plugin', filename=filename, name=name)
    for tag, text in children:
        sub(p, tag, text)
    return p


def label(parent, class_id):
    plugin(parent, 'gz-sim-label-system', 'gz::sim::systems::Label', [('label', str(class_id))])


def material(parent, rgba):
    m = sub(parent, 'material')
    r, g, b, a = rgba
    sub(m, 'ambient', fmt(r * 0.6, g * 0.6, b * 0.6, a))
    sub(m, 'diffuse', fmt(r, g, b, a))
    sub(m, 'specular', '0.1 0.1 0.1 1')
    return m


def geometry(parent, prim):
    g = sub(parent, 'geometry')
    if prim['type'] == 'box':
        sub(sub(g, 'box'), 'size', fmt(*map(float, prim['size'])))
    elif prim['type'] == 'cylinder':
        c = sub(g, 'cylinder')
        sub(c, 'radius', fmt(float(prim['radius'])))
        sub(c, 'length', fmt(float(prim['length'])))
    else:
        raise ValueError(prim['type'])


def prim_extent(prim):
    if prim['type'] == 'box':
        return prim['size']
    d = 2 * prim['radius']
    return [d, d, prim['length']]


def inertial(link, mass, prims):
    """Solid-box inertia of the first collision primitive's bounding box."""
    sx, sy, sz = prim_extent(prims[0])
    pose = prims[0]['pose']
    i = sub(link, 'inertial')
    sub(i, 'pose', fmt(*map(float, pose[:3]), 0.0, 0.0, 0.0))
    sub(i, 'mass', fmt(float(mass)))
    t = sub(i, 'inertia')
    sub(t, 'ixx', f'{mass * (sy * sy + sz * sz) / 12.0:.3e}')
    sub(t, 'iyy', f'{mass * (sx * sx + sz * sz) / 12.0:.3e}')
    sub(t, 'izz', f'{mass * (sx * sx + sy * sy) / 12.0:.3e}')
    for k in ('ixy', 'ixz', 'iyz'):
        sub(t, k, '0')


def object_model(world, name, spec, pose, z, class_id, static, visual_dz=0.0):
    m = sub(world, 'model', name=name)
    sub(m, 'static', 'true' if static else 'false')
    x, y, yaw = pose
    sub(m, 'pose', fmt(float(x), float(y), z, 0.0, 0.0, float(yaw)))
    link = sub(m, 'link', name='link')
    if not static:
        inertial(link, spec['mass'], spec['collision'])
    for k, prim in enumerate(spec['collision']):
        c = sub(link, 'collision', name=f'collision_{k}')
        sub(c, 'pose', fmt(*map(float, prim['pose'])))
        geometry(c, prim)
    v = sub(link, 'visual', name='visual')
    sub(v, 'pose', fmt(0.0, 0.0, float(visual_dz), 0.0, 0.0, 0.0))
    mesh = sub(sub(v, 'geometry'), 'mesh')
    sub(mesh, 'uri', (COMMON / spec['mesh']).as_uri())
    material(v, spec['color'])
    if spec['color'][3] < 1.0:
        sub(v, 'transparency', fmt(1.0 - float(spec['color'][3])))
    label(m, class_id)
    return m


def surface(collision, mu):
    f = sub(sub(collision, 'surface'), 'friction')
    ode = sub(f, 'ode')
    sub(ode, 'mu', fmt(float(mu)))
    sub(ode, 'mu2', fmt(float(mu)))


def terrain_models(world, scene, hm_png):
    t = scene['terrain']
    hm = t['heightmap']
    size = hm['size']
    cx, cy = hm['center']
    z_lo, z_hi = terrain_mod.heightmap_range(t)   # the PNG spans exactly this range
    zr = z_hi - z_lo

    m = sub(world, 'model', name='terrain')
    sub(m, 'static', 'true')
    # Gazebo Harmonic places the two heightmaps differently (checked with probe
    # drops and top-down renders): DART collision ignores <pos> but follows the
    # pose; ogre2 rendering (Terra) ignores the model pose but follows <pos>.
    # So the model stays at the origin, the collision gets a <pose>, the
    # visual gets <pos>.
    sub(m, 'pose', '0 0 0 0 0 0')
    hm_pos = fmt(float(cx), float(cy), round(z_lo, 4))
    link = sub(m, 'link', name='link')
    c = sub(link, 'collision', name='collision')
    sub(c, 'pose', hm_pos + ' 0 0 0')
    h = sub(sub(c, 'geometry'), 'heightmap')
    sub(h, 'uri', hm_png.as_uri())
    sub(h, 'size', fmt(float(size), float(size), float(zr)))
    sub(h, 'pos', '0 0 0')
    surface(c, t['friction']['dry_sand'])
    v = sub(link, 'visual', name='visual')
    h = sub(sub(v, 'geometry'), 'heightmap')
    sub(h, 'uri', hm_png.as_uri())
    sub(h, 'size', fmt(float(size), float(size), float(zr)))
    sub(h, 'pos', hm_pos)
    tex = sub(h, 'texture')
    sub(tex, 'diffuse', (COMMON / 'textures' / 'sand_albedo.png').as_uri())
    sub(tex, 'normal', (COMMON / 'textures' / 'sand_normal.png').as_uri())
    sub(tex, 'size', '3')
    # (no <blend>: a single texture needs none)

    # Wet zone: separate flat box with its own friction, just above the heightmap.
    wl = t['waterline_y']
    wet_w = t['wet_zone_width']
    top = t['wet_zone_height'] + 0.005
    thick = 0.1
    m = sub(world, 'model', name='wet_zone')
    sub(m, 'static', 'true')
    sub(m, 'pose', fmt(float(cx), wl + wet_w / 2.0, top - thick / 2.0, 0.0, 0.0, 0.0))
    link = sub(m, 'link', name='link')
    c = sub(link, 'collision', name='collision')
    sub(sub(sub(c, 'geometry'), 'box'), 'size', fmt(float(size), float(wet_w), thick))
    surface(c, t['friction']['wet_sand'])
    v = sub(link, 'visual', name='visual')
    sub(sub(sub(v, 'geometry'), 'box'), 'size', fmt(float(size), float(wet_w), thick))
    mat = material(v, [1.0, 1.0, 1.0, 1.0])     # white: the albedo map gives the colour
    pbr = sub(sub(mat, 'pbr'), 'metal')
    sub(pbr, 'albedo_map', (COMMON / 'textures' / 'wet_sand_albedo.png').as_uri())
    sub(pbr, 'roughness', '0.6')

    # Water: visual only (no physics); lethal in /map and outside the geofence.
    sea_y0 = cy - size / 2.0
    m = sub(world, 'model', name='water')
    sub(m, 'static', 'true')
    sub(m, 'pose', fmt(float(cx), (sea_y0 + wl) / 2.0, -0.01, 0.0, 0.0, 0.0))
    link = sub(m, 'link', name='link')
    v = sub(link, 'visual', name='visual')
    sub(sub(sub(v, 'geometry'), 'box'), 'size', fmt(float(size), wl - sea_y0, 0.02))
    material(v, [0.10, 0.35, 0.55, 1.0])
    sub(v, 'transparency', '0.25')


PERSON_TURN_TIME = 0.5    # s, turning on the spot at each corner
PERSON_WAYPOINT_SPACING = 1.0   # m, so the actor follows the terrain height


def person_timeline(person, terrain):
    """The person's scripted path as [t, x, y, ground_z, yaw] samples.

    Used for the Gazebo actor's waypoints and, unchanged, for the /people
    ground truth (Gazebo doesn't publish the actor's pose). The person walks
    each leg at constant speed and turns on the spot at the corners; with
    `loop`, the timeline restarts at its last time stamp.
    """
    speed = person['speed']
    wps = person['waypoints'] + ([person['waypoints'][0]] if person['loop'] else [])
    out = []
    t = 0.0

    def add(time, x, y, yaw):
        out.append([round(time, 3), round(x, 3), round(y, 3),
                    round(float(terrain_mod.height_at(terrain, x, y)), 3), round(yaw, 4)])

    for k in range(len(wps) - 1):
        (x0, y0), (x1, y1) = wps[k], wps[k + 1]
        yaw = math.atan2(y1 - y0, x1 - x0)
        if k > 0:
            t += PERSON_TURN_TIME
        add(t, x0, y0, yaw)
        dist = math.hypot(x1 - x0, y1 - y0)
        steps = max(1, int(math.ceil(dist / PERSON_WAYPOINT_SPACING)))
        for s in range(1, steps + 1):
            f = s / steps
            add(t + f * dist / speed, x0 + f * (x1 - x0), y0 + f * (y1 - y0), yaw)
        t += dist / speed
    if person['loop']:                 # turn back to the first heading before the loop restarts
        (x0, y0), (x1, y1) = wps[0], wps[1]
        add(t + PERSON_TURN_TIME, x0, y0, math.atan2(y1 - y0, x1 - x0))
    return out


def actor(world, person, spec, terrain, class_id):
    a = sub(world, 'actor', name=person['id'])
    skin = sub(a, 'skin')
    sub(skin, 'filename', spec['gazebo_skin'])
    sub(skin, 'scale', '1.0')
    anim = sub(a, 'animation', name='walk')
    sub(anim, 'filename', spec['gazebo_skin'])
    sub(anim, 'interpolate_x', 'true')
    script = sub(a, 'script')
    sub(script, 'loop', 'true' if person['loop'] else 'false')
    sub(script, 'delay_start', '0')
    sub(script, 'auto_start', 'true')
    traj = sub(script, 'trajectory', id='0', type='walk', tension='0')
    for t, x, y, gz, yaw in person_timeline(person, terrain):
        w = sub(traj, 'waypoint')
        sub(w, 'time', fmt(t))
        sub(w, 'pose', fmt(x, y, round(gz + spec['hip_height'], 3), 0.0, 0.0, yaw))
    label(a, class_id)


def debug_cameras(world, scene):
    """Static cameras that save frames to disk: a top-down RGB + semantic
    segmentation pair over the area, and an oblique RGB view from the south-west.
    Used to check labels and headless rendering; not part of the real world.
    Note: gz-sim only renders a camera while its topic has a subscriber."""
    w, h = scene['area']['size']
    views = [
        ('top', (w / 2.0, h / 2.0, 42.0, 0.0, math.pi / 2, math.pi / 2), 1.2,
         (('camera', 'rgb'), ('segmentation', 'seg'))),
        ('oblique', (-6.0, -4.0, 5.0, 0.0, 0.18, 0.6), 1.3, (('camera', 'rgb'),)),
    ]
    for name, pose, hfov, sensors in views:
        m = sub(world, 'model', name=f'debug_{name}')
        sub(m, 'static', 'true')
        sub(m, 'pose', fmt(*map(float, pose)))
        link = sub(m, 'link', name='link')
        for kind, short in sensors:
            s = sub(link, 'sensor', name=f'{name}_{short}', type=kind)
            sub(s, 'topic', f'debug/{name}/{short}')
            sub(s, 'update_rate', '1')
            sub(s, 'always_on', '1')
            cam = sub(s, 'camera')
            if kind == 'segmentation':
                sub(cam, 'segmentation_type', 'semantic')
            sub(cam, 'horizontal_fov', fmt(hfov))
            img = sub(cam, 'image')
            sub(img, 'width', '1280')
            sub(img, 'height', '960' if name == 'top' else '720')
            clip = sub(cam, 'clip')
            sub(clip, 'near', '0.1')
            sub(clip, 'far', '150')
            save = sub(cam, 'save', enabled='true')
            sub(save, 'path', str(OUT_DIR / f"debug_scene_{scene['seed']}" / f'{name}_{short}'))


ROBOT_SPAWN_HEIGHT = 0.45   # m above the terrain, as in A0's flat-ground launch (z = 0.45)
PERSON_SIZE = [0.5, 0.5, 1.8]


def bbox(spec):
    """Axis-aligned size [x, y, z] of an object's collision primitives."""
    lo, hi = [1e9] * 3, [-1e9] * 3
    for p in spec['collision']:
        ext = prim_extent(p)
        for k in range(3):
            lo[k] = min(lo[k], p['pose'][k] - ext[k] / 2.0)
            hi[k] = max(hi[k], p['pose'][k] + ext[k] / 2.0)
    return [round(h - l, 4) for l, h in zip(lo, hi)]


ROBOT_VARIANTS = ('custom', 'stock')    # beach_description `variant` (notes/decisions.md)


def build_manifest(scene, objects, world_sdf, variant='custom'):
    """What the Gazebo bridge needs at runtime, derived from scenes.yaml +
    objects.yaml: spawn pose, attachable items, object classes and sizes, and
    the person's timeline (same samples as the actor's waypoints)."""
    t = scene['terrain']
    classes = objects['classes']
    x, y, yaw = scene['robot_start']['pose']
    objs = []
    for it in scene['trash']:
        spec = objects['trash'][it['model']]
        objs.append({'id': it['id'], 'kind': 'trash', 'class': it['material'],
                     'class_id': classes[it['material']], 'size': bbox(spec)})
    for d in scene['distractors']:
        spec = objects['distractors'][d['model']]
        objs.append({'id': d['id'], 'kind': 'distractor', 'class': d['type'],
                     'class_id': classes[d['type']], 'size': bbox(spec)})
    person = None
    if scene['person']:
        person = {'id': scene['person']['id'], 'class': 'person', 'class_id': classes['person'],
                  'size': PERSON_SIZE, 'loop': scene['person']['loop'],
                  'timeline': person_timeline(scene['person'], t)}
    return {
        'seed': scene['seed'],
        'variant': variant,             # robot variant: the launch uses it, A3 records it per run
        'world_name': WORLD_NAME,
        'world_sdf': str(world_sdf),
        'spawn': {'x': x, 'y': y, 'yaw': yaw,
                  'z': round(float(terrain_mod.height_at(t, x, y)) + ROBOT_SPAWN_HEIGHT, 3)},
        'attach_items': [it['id'] for it in scene['trash']],
        'objects': objs,
        'person': person,
    }


def build_world(scene, objects, debug_camera=False):
    terrain_dir = export_terrain.export(scene, mesh=False)
    classes = objects['classes']
    t = scene['terrain']

    sdf = ET.Element('sdf', version='1.9')
    world = sub(sdf, 'world', name=WORLD_NAME)
    world.append(ET.Comment(f" GENERATED by sim/gazebo/load_scene.py from bench/scenes.yaml, seed {scene['seed']}. Do not edit. "))

    phys = sub(world, 'physics', name='1ms', type='ignored')
    sub(phys, 'max_step_size', '0.001')       # revisit after A0: locomotion controller rate
    sub(phys, 'real_time_factor', '1.0')
    # ODE collision detection: probe drops on the beach heightmap rested within
    # ~1 cm of the terrain with ODE, but sank 2-5 cm with bullet (which gz-sim's
    # heightmap.sdf example suggests). FCL has no heightmap support.
    sub(sub(phys, 'dart'), 'collision_detector', 'ode')

    plugin(world, 'gz-sim-physics-system', 'gz::sim::systems::Physics')
    plugin(world, 'gz-sim-user-commands-system', 'gz::sim::systems::UserCommands')
    plugin(world, 'gz-sim-scene-broadcaster-system', 'gz::sim::systems::SceneBroadcaster')
    plugin(world, 'gz-sim-sensors-system', 'gz::sim::systems::Sensors', [('render_engine', 'ogre2')])
    # The Imu and NavSat systems come with the robot model (sim_bridge_gazebo xacro);
    # loading them here too would run them twice.

    geo = scene['geo']
    sc = sub(world, 'spherical_coordinates')
    sub(sc, 'surface_model', 'EARTH_WGS84')
    sub(sc, 'world_frame_orientation', 'ENU')
    sub(sc, 'latitude_deg', str(geo['latitude_deg']))
    sub(sc, 'longitude_deg', str(geo['longitude_deg']))
    sub(sc, 'elevation', str(geo['elevation_m']))
    sub(sc, 'heading_deg', '0')

    scn = sub(world, 'scene')
    sub(scn, 'ambient', '0.6 0.6 0.6 1')
    sub(scn, 'background', '0.62 0.80 0.95 1')
    sub(scn, 'shadows', 'true')

    sun = sub(world, 'light', type='directional', name='sun')
    sub(sun, 'cast_shadows', 'true')
    sub(sun, 'pose', '0 0 20 0 0 0')
    sub(sun, 'diffuse', '0.9 0.88 0.82 1')
    sub(sun, 'specular', '0.2 0.2 0.2 1')
    sub(sun, 'direction', '-0.4 0.3 -0.85')

    terrain_models(world, scene, terrain_dir / 'heightmap.png')

    for b in scene['bins']:
        spec = objects['bins'][b['model']]
        z = ground_z(t, b['pose'][0], b['pose'][1], 0.3, 'min')
        object_model(world, b['id'], spec, b['pose'], z, classes[spec['class']], static=True)
    for lm in scene['landmarks']:
        spec = objects['landmarks'][lm['model']]
        if lm['model'] == 'fence_post':
            z = ground_z(t, lm['pose'][0], lm['pose'][1], 0.05, 'min') - 0.2   # set into the sand
        else:
            z = ground_z(t, lm['pose'][0], lm['pose'][1], 1.0, 'min')
        object_model(world, lm['id'], spec, lm['pose'], z, classes[spec['class']], static=True)
    for it in scene['trash']:
        spec = objects['trash'][it['model']]
        z = ground_z(t, it['pose'][0], it['pose'][1], spec['footprint_radius'], 'max') + SPAWN_GAP
        # Half-buried: sink only the visual; the collision stays on the surface.
        height = max(p['pose'][2] + prim_extent(p)[2] / 2.0 for p in spec['collision'])
        dz = -it['bury_fraction'] * height if it['half_buried'] else 0.0
        object_model(world, it['id'], spec, it['pose'], z, classes[spec['material']], static=False,
                     visual_dz=dz)
    # Distractors are static (never picked up; dynamic ones resting on the heightmap cost
    # contact solving every step, RTF ~0.5 with the robot): set on the terrain at their centre.
    for d in scene['distractors']:
        spec = objects['distractors'][d['model']]
        z = ground_z(t, d['pose'][0], d['pose'][1], 0.0, 'min')
        object_model(world, d['id'], spec, d['pose'], z, classes[spec['class']], static=True)

    if scene['person']:
        actor(world, scene['person'], objects['person'], t, classes['person'])

    if debug_camera:
        debug_cameras(world, scene)

    ET.indent(sdf, space='  ')
    return '<?xml version="1.0" ?>\n' + ET.tostring(sdf, encoding='unicode') + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--scenes', default=str(REPO / 'bench' / 'scenes.yaml'))
    ap.add_argument('--debug-camera', action='store_true',
                    help='add overview RGB + segmentation cameras that save frames to disk')
    ap.add_argument('--out', help='output SDF path')
    ap.add_argument('--variant', choices=ROBOT_VARIANTS, default='custom',
                    help='robot variant written into the manifest (the launch spawns this robot)')
    args = ap.parse_args()

    scene = export_terrain.load_scene(args.seed, Path(args.scenes))
    with open(COMMON / 'objects.yaml') as f:
        objects = yaml.safe_load(f)
    out = Path(args.out) if args.out else OUT_DIR / f'scene_{args.seed}{"_debug" if args.debug_camera else ""}.sdf'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_world(scene, objects, args.debug_camera))
    manifest = out.with_suffix('.manifest.yaml')
    manifest.write_text('# GENERATED by sim/gazebo/load_scene.py. Do not edit.\n'
                        + yaml.safe_dump(build_manifest(scene, objects, out.resolve(), args.variant),
                                         sort_keys=False, default_flow_style=None, width=120))
    print(out)


if __name__ == '__main__':
    main()
