#!/usr/bin/env python3
"""Build an OpenUSD stage (Isaac Sim) for one scene of bench/scenes.yaml.

    python3 sim/isaac/load_scene.py --seed 3 [--out PATH]

Writes sim/isaac/scenes/generated/scene_<seed>.usda (git-ignored; never edit by
hand): the same scene as sim/gazebo/load_scene.py (terrain, wet zone, water,
bins, landmarks, trash, distractors, friction, lights), plus the cameras the
render check saves. No robot, no person (D0).

The stage's world frame is the map frame, as in Gazebo: origin at the
south-west corner of the scene area, x east, y north, z = 0 at the water level.
One self-contained text stage per seed: meshes are written into the stage,
textures are referenced relative to the .usda file, so the output folder plus
sim/common/textures can be copied to another machine as they are.

Plain OpenUSD only (UsdGeom, UsdShade UsdPreviewSurface, UsdLux, UsdPhysics):
it builds and is tested with `usd-core` on a machine without Isaac Sim.
Settings live in sim/isaac/render.yaml and config/sensors.yaml.
"""

import argparse
import math
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
COMMON = REPO / 'sim' / 'common'
sys.path.insert(0, str(COMMON))
import export_terrain  # noqa: E402
import terrain as terrain_mod  # noqa: E402
from terrain import ground_z  # noqa: E402

OUT_DIR = HERE / 'scenes' / 'generated'
RENDER_YAML = HERE / 'render.yaml'
SENSORS_YAML = REPO / 'config' / 'sensors.yaml'
ROBOT_XACRO = REPO / 'packages' / 'beach_description' / 'urdf' / 'beach_robot.urdf.xacro'

# The same placement rules as sim/gazebo/load_scene.py (the tests compare the poses).
SPAWN_GAP = 0.005           # m, trash starts this far above the terrain and settles
SAND_TILE_M = 3.0           # the sand texture repeats every 3 m (Gazebo heightmap <size>3</size>)
WET_ZONE_THICKNESS = 0.1
WET_ZONE_TOP_GAP = 0.005    # top of the wet-zone box above wet_zone_height
WET_SAND_ROUGHNESS = 0.6
WATER_COLOR = (0.10, 0.35, 0.55)
WATER_OPACITY = 0.75
WATER_Z = -0.01
WATER_THICKNESS = 0.02
DEFAULT_ROUGHNESS = 0.5
CAMERA_HORIZONTAL_APERTURE = 20.955   # USD default; only the ratio to focalLength matters

GROUPS = {'bins': 'Bins', 'landmarks': 'Landmarks', 'trash': 'Trash', 'distractors': 'Distractors'}


# ---------------------------------------------------------------- math

def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rpy_matrix(roll, pitch, yaw):
    """SDF / URDF fixed-axis roll-pitch-yaw: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def matrix_to_quat(r):
    """Rotation matrix (column-vector convention) -> Gf.Quatd (w, x, y, z)."""
    r = np.asarray(r, dtype=float)
    tr = np.trace(r)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        w, x, y, z = (r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        w, x, y, z = (r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        w, x, y, z = (r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s
    q = Gf.Quatd(w, Gf.Vec3d(x, y, z))
    return q.GetNormalized()


# A USD camera looks down its local -Z with +Y up. Columns: the USD camera's x, y, z axes
# written in the source camera frame.
_USD_CAM_IN = {
    # Gazebo camera / camera_link (REP 103 body): +x forward, +y left, +z up.
    'gazebo': np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=float),
    # ROS optical frame: +z forward, +x right, +y down.
    'optical': np.diag([1.0, -1.0, -1.0]),
}


def usd_camera_rotation(r_world_cam, convention='gazebo'):
    """World rotation of the USD camera prim, given the world rotation of a camera frame in
    the Gazebo (x forward, z up) or ROS optical (z forward, y down) convention."""
    return np.asarray(r_world_cam, dtype=float) @ _USD_CAM_IN[convention]


def light_rotation(direction):
    """Rotation that turns a UsdLux light's local -Z onto `direction` (the way light travels)."""
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    z = -d                                  # local +Z points back towards the light
    up = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def hfov_to_focal(hfov, aperture=CAMERA_HORIZONTAL_APERTURE):
    """focalLength for a horizontal FOV: hfov = 2 atan(aperture / (2 focal))."""
    return aperture / (2.0 * math.tan(hfov / 2.0))


# ---------------------------------------------------------------- helpers

def set_pose(xformable, pos=(0.0, 0.0, 0.0), quat=None, scale=None):
    """translate / orient / scale ops (the order Isaac Sim writes)."""
    xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, pos)))
    if quat is not None:
        xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(quat)
    if scale is not None:
        xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, scale)))


def yaw_quat(yaw):
    return Gf.Quatd(math.cos(yaw / 2.0), Gf.Vec3d(0.0, 0.0, math.sin(yaw / 2.0)))


def prim_extent(prim):
    if prim['type'] == 'box':
        return [float(v) for v in prim['size']]
    d = 2.0 * float(prim['radius'])
    return [d, d, float(prim['length'])]


def safe_name(s):
    s = re.sub(r'[^A-Za-z0-9_]', '_', str(s))
    return s if s and not s[0].isdigit() else '_' + s


def rel_asset(path, out_dir):
    """Asset path relative to the .usda's folder, with forward slashes."""
    rel = os.path.relpath(Path(path).resolve(), Path(out_dir).resolve()).replace(os.sep, '/')
    return rel if rel.startswith('.') else './' + rel


def set_class(prim, name, class_id):
    """Class label for the render check and D0 (Isaac's semantics schema is chosen in GPU session 1)."""
    prim.SetCustomDataByKey('beach:class', str(name))
    prim.SetCustomDataByKey('beach:class_id', int(class_id))


# ---------------------------------------------------------------- meshes

def read_obj(path):
    """Minimal OBJ reader: v, vt, vn and f (polygons triangulated as fans; v, v/vt, v//vn,
    v/vt/vn, negative indices). Returns points, face counts, point indices and, when the file
    has them, face-varying normals and UVs as (values, indices)."""
    v, vt, vn = [], [], []
    fv, ft, fn = [], [], []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == 'v':
            v.append([float(c) for c in parts[1:4]])
        elif parts[0] == 'vt':
            vt.append([float(c) for c in parts[1:3]])
        elif parts[0] == 'vn':
            vn.append([float(c) for c in parts[1:4]])
        elif parts[0] == 'f':
            corners = []
            for tok in parts[1:]:
                ids = (tok.split('/') + ['', ''])[:3]
                corners.append(tuple(_obj_index(ids[k], n) for k, n in
                                     ((0, len(v)), (1, len(vt)), (2, len(vn)))))
            for k in range(1, len(corners) - 1):
                for c in (corners[0], corners[k], corners[k + 1]):
                    fv.append(c[0])
                    ft.append(c[1])
                    fn.append(c[2])
    n_tris = len(fv) // 3
    out = {'points': np.array(v, dtype=float), 'counts': [3] * n_tris, 'indices': fv,
           'normals': None, 'st': None}
    if vn and all(i is not None for i in fn):
        out['normals'] = (np.array(vn, dtype=float), fn)
    if vt and all(i is not None for i in ft):
        out['st'] = (np.array(vt, dtype=float), ft)
    return out


def _obj_index(tok, n):
    if tok == '':
        return None
    i = int(tok)
    return i - 1 if i > 0 else n + i


def define_mesh(stage, path, points, counts, indices, normals=None, st=None, st_interp='faceVarying'):
    """A UsdGeom.Mesh with no subdivision. normals / st: (values, indices) face-varying, or
    for st, values per point with st_interp='vertex'."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    pts = Vt.Vec3fArray.FromNumpy(np.asarray(points, dtype=np.float32))
    mesh.CreatePointsAttr(pts)
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.asarray(counts, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(indices, dtype=np.int32)))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(pts))
    pv = UsdGeom.PrimvarsAPI(mesh)
    if normals is not None:
        vals, idx = normals
        p = pv.CreatePrimvar('normals', Sdf.ValueTypeNames.Normal3fArray, UsdGeom.Tokens.faceVarying)
        p.Set(Vt.Vec3fArray.FromNumpy(np.asarray(vals, dtype=np.float32)))
        p.SetIndices(Vt.IntArray.FromNumpy(np.asarray(idx, dtype=np.int32)))
    if st is not None:
        if st_interp == 'vertex':
            p = pv.CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
            p.Set(Vt.Vec2fArray.FromNumpy(np.asarray(st, dtype=np.float32)))
        else:
            vals, idx = st
            p = pv.CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
            p.Set(Vt.Vec2fArray.FromNumpy(np.asarray(vals, dtype=np.float32)))
            p.SetIndices(Vt.IntArray.FromNumpy(np.asarray(idx, dtype=np.int32)))
    return mesh


def box_mesh_data(size, tile):
    """A box centred at the origin, 4 points per face, UVs in metres / tile on each face's plane."""
    hx, hy, hz = (s / 2.0 for s in size)
    faces = [  # (normal axis, sign); corners counter-clockwise seen from outside
        ((0, 0, 1), [(-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)], (0, 1)),
        ((0, 0, -1), [(-hx, -hy, -hz), (-hx, hy, -hz), (hx, hy, -hz), (hx, -hy, -hz)], (0, 1)),
        ((1, 0, 0), [(hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz), (hx, -hy, hz)], (1, 2)),
        ((-1, 0, 0), [(-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz), (-hx, hy, -hz)], (1, 2)),
        ((0, 1, 0), [(-hx, hy, -hz), (-hx, hy, hz), (hx, hy, hz), (hx, hy, -hz)], (0, 2)),
        ((0, -1, 0), [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz)], (0, 2)),
    ]
    pts, st, nrm = [], [], []
    for n, corners, (a, b) in faces:
        for c in corners:
            pts.append(c)
            st.append((c[a] / tile, c[b] / tile))
            nrm.append(n)
    counts = [4] * 6
    idx = list(range(24))
    return pts, counts, idx, (nrm, idx), (st, idx)


# ---------------------------------------------------------------- materials

class Looks:
    """Materials under /World/Looks, one per texture or colour, shared between prims."""

    def __init__(self, stage, out_dir):
        self.stage = stage
        self.out_dir = out_dir
        self.root = UsdGeom.Scope.Define(stage, '/World/Looks').GetPath()
        self.cache = {}

    def get(self, color=(1.0, 1.0, 1.0), opacity=1.0, texture=None, normal_map=None,
            roughness=DEFAULT_ROUGHNESS, wrap='repeat'):
        key = (tuple(round(float(c), 4) for c in color), round(float(opacity), 4),
               str(texture), str(normal_map), round(float(roughness), 4))
        if key in self.cache:
            return self.cache[key]
        if texture is not None:
            name = 'tex_' + Path(texture).stem
            if normal_map is not None:
                name += '_n'
        else:
            name = 'col_' + '_'.join(f'{int(round(c * 255)):03d}' for c in key[0])
            if opacity < 1.0:
                name += f'_a{int(round(opacity * 100)):03d}'
        if abs(roughness - DEFAULT_ROUGHNESS) > 1e-9:
            name += f'_r{int(round(roughness * 100)):03d}'
        path = self.root.AppendChild(safe_name(name))
        mat = UsdShade.Material.Define(self.stage, path)
        sh = UsdShade.Shader.Define(self.stage, path.AppendChild('PreviewSurface'))
        sh.CreateIdAttr('UsdPreviewSurface')
        sh.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(float(roughness))
        sh.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.0)
        sh.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(float(opacity))
        if texture is None:
            sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, color)))
        else:
            reader = UsdShade.Shader.Define(self.stage, path.AppendChild('st_reader'))
            reader.CreateIdAttr('UsdPrimvarReader_float2')
            reader.CreateInput('varname', Sdf.ValueTypeNames.String).Set('st')
            st_out = reader.CreateOutput('result', Sdf.ValueTypeNames.Float2)
            tex = self._texture(path.AppendChild('albedo'), texture, st_out, 'sRGB', wrap)
            sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(
                tex.CreateOutput('rgb', Sdf.ValueTypeNames.Float3))
            if normal_map is not None:
                ntex = self._texture(path.AppendChild('normal'), normal_map, st_out, 'raw', wrap)
                ntex.CreateInput('scale', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
                ntex.CreateInput('bias', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
                sh.CreateInput('normal', Sdf.ValueTypeNames.Normal3f).ConnectToSource(
                    ntex.CreateOutput('rgb', Sdf.ValueTypeNames.Float3))
        mat.CreateSurfaceOutput().ConnectToSource(sh.CreateOutput('surface', Sdf.ValueTypeNames.Token))
        self.cache[key] = mat
        return mat

    def _texture(self, path, file, st_out, color_space, wrap):
        tex = UsdShade.Shader.Define(self.stage, path)
        tex.CreateIdAttr('UsdUVTexture')
        tex.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(rel_asset(file, self.out_dir)))
        tex.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(st_out)
        tex.CreateInput('wrapS', Sdf.ValueTypeNames.Token).Set(wrap)
        tex.CreateInput('wrapT', Sdf.ValueTypeNames.Token).Set(wrap)
        tex.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set(color_space)
        return tex


def bind(prim, material, purpose=None):
    api = UsdShade.MaterialBindingAPI.Apply(prim.GetPrim() if hasattr(prim, 'GetPrim') else prim)
    if purpose is None:
        api.Bind(material)
    else:
        api.Bind(material, UsdShade.Tokens.weakerThanDescendants, purpose)


def physics_material(stage, name, friction):
    mat = UsdShade.Material.Define(stage, f'/World/PhysicsMaterials/{name}')
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(friction))
    api.CreateDynamicFrictionAttr(float(friction))
    api.CreateRestitutionAttr(0.0)
    return mat


# ---------------------------------------------------------------- scene parts

def collider(stage, path, prim):
    """A box or cylinder collision primitive (purpose guide: not rendered)."""
    x, y, z, roll, pitch, yaw = map(float, prim['pose'])
    q = matrix_to_quat(rpy_matrix(roll, pitch, yaw))
    if prim['type'] == 'box':
        g = UsdGeom.Cube.Define(stage, path)
        g.CreateSizeAttr(1.0)
        set_pose(g, (x, y, z), q, prim['size'])
        g.CreateExtentAttr([(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)])
    elif prim['type'] == 'cylinder':
        g = UsdGeom.Cylinder.Define(stage, path)
        r, length = float(prim['radius']), float(prim['length'])
        g.CreateRadiusAttr(r)
        g.CreateHeightAttr(length)
        g.CreateAxisAttr(UsdGeom.Tokens.z)
        g.CreateExtentAttr([(-r, -r, -length / 2.0), (r, r, length / 2.0)])
        set_pose(g, (x, y, z), q)
    else:
        raise ValueError(prim['type'])
    g.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(g.GetPrim())
    return g


class MeshCache:
    def __init__(self):
        self.data = {}

    def get(self, rel):
        if rel not in self.data:
            self.data[rel] = read_obj(COMMON / rel)
        return self.data[rel]


def object_prim(stage, looks, meshes, parent, name, spec, pose, z, class_name, class_id,
                dynamic, visual_dz=0.0):
    """One object: an Xform at the Gazebo model pose, visual meshes, collision primitives."""
    x, y, yaw = map(float, pose)
    xf = UsdGeom.Xform.Define(stage, f'{parent}/{safe_name(name)}')
    set_pose(xf, (x, y, z), yaw_quat(yaw))
    prim = xf.GetPrim()
    prim.SetCustomDataByKey('beach:id', name)
    set_class(prim, class_name, class_id)
    for k, vis in enumerate(spec.get('visuals') or [{'mesh': spec['mesh'], 'color': spec['color']}]):
        d = meshes.get(vis['mesh'])
        m = define_mesh(stage, xf.GetPath().AppendChild(f'visual_{k}'), d['points'], d['counts'],
                        d['indices'], d['normals'], d['st'])
        set_pose(m, (0.0, 0.0, visual_dz))
        if 'texture' in vis:
            if d['st'] is None:
                raise ValueError(f"{vis['mesh']} has a texture but no UVs")
            mat = looks.get(texture=COMMON / vis['texture'], wrap='clamp')
        else:
            r, g, b, a = map(float, vis['color'])
            mat = looks.get(color=(r, g, b), opacity=a)
            m.CreateDisplayColorAttr([Gf.Vec3f(r, g, b)])
            if a < 1.0:
                m.CreateDisplayOpacityAttr([a])
        bind(m, mat)
    for k, p in enumerate(spec['collision']):
        collider(stage, xf.GetPath().AppendChild(f'collision_{k}'), p)
    if dynamic:
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        p0 = spec['collision'][0]
        sx, sy, sz = prim_extent(p0)
        mass = float(spec['mass'])
        mass_api.CreateMassAttr(mass)
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, p0['pose'][:3])))
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(mass * (sy * sy + sz * sz) / 12.0,
                                                    mass * (sx * sx + sz * sz) / 12.0,
                                                    mass * (sx * sx + sy * sy) / 12.0))
        mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    return xf


def terrain_prims(stage, looks, scene):
    t = scene['terrain']
    hm = t['heightmap']
    size = float(hm['size'])
    cx, cy = map(float, hm['center'])
    env = '/World/Environment'
    UsdGeom.Xform.Define(stage, env)
    fr = t['friction']
    dry = physics_material(stage, 'dry_sand', fr['dry_sand'])
    wet = physics_material(stage, 'wet_sand', fr['wet_sand'])

    # Terrain: the height grid as a triangle mesh (the triangulation of terrain.write_obj_mesh).
    xs, ys = terrain_mod.grid_axes(t)
    h = terrain_mod.height_grid(t)
    n = len(xs)
    X, Y = np.meshgrid(xs, ys)
    points = np.stack([X.ravel(), Y.ravel(), h.ravel()], axis=1)
    j, i = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing='ij')
    a = (j * n + i).ravel()
    b, c = a + 1, a + n
    d = c + 1
    tris = np.stack([a, b, d, a, d, c], axis=1).reshape(-1)
    xf = UsdGeom.Xform.Define(stage, f'{env}/terrain')
    set_pose(xf)
    set_class(xf.GetPrim(), 'background', 0)
    st = np.stack([X.ravel(), Y.ravel()], axis=1) / SAND_TILE_M
    mesh = define_mesh(stage, f'{env}/terrain/mesh', points, [3] * (len(tris) // 3), tris,
                       st=st, st_interp='vertex')
    bind(mesh, looks.get(texture=COMMON / 'textures' / 'sand_albedo.png',
                         normal_map=COMMON / 'textures' / 'sand_normal.png', roughness=0.9))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.none)
    bind(mesh, dry, 'physics')

    # Wet zone: a flat box just above the terrain, with its own friction.
    wl = float(t['waterline_y'])
    wet_w = float(t['wet_zone_width'])
    top = float(t['wet_zone_height']) + WET_ZONE_TOP_GAP
    box = (size, wet_w, WET_ZONE_THICKNESS)
    xf = UsdGeom.Xform.Define(stage, f'{env}/wet_zone')
    set_pose(xf, (cx, wl + wet_w / 2.0, top - WET_ZONE_THICKNESS / 2.0))
    set_class(xf.GetPrim(), 'background', 0)
    pts, counts, idx, nrm, uv = box_mesh_data(box, SAND_TILE_M)
    vis = define_mesh(stage, f'{env}/wet_zone/visual', pts, counts, idx, nrm, uv)
    bind(vis, looks.get(texture=COMMON / 'textures' / 'wet_sand_albedo.png', roughness=WET_SAND_ROUGHNESS))
    col = collider(stage, f'{env}/wet_zone/collision',
                   {'type': 'box', 'size': list(box), 'pose': [0, 0, 0, 0, 0, 0]})
    bind(col, wet, 'physics')

    # Water: visual only, no collider.
    sea_y0 = cy - size / 2.0
    xf = UsdGeom.Xform.Define(stage, f'{env}/water')
    set_pose(xf, (cx, (sea_y0 + wl) / 2.0, WATER_Z))
    set_class(xf.GetPrim(), 'background', 0)
    pts, counts, idx, nrm, _ = box_mesh_data((size, wl - sea_y0, WATER_THICKNESS), SAND_TILE_M)
    vis = define_mesh(stage, f'{env}/water/visual', pts, counts, idx, nrm)
    vis.CreateDisplayColorAttr([Gf.Vec3f(*WATER_COLOR)])
    vis.CreateDisplayOpacityAttr([WATER_OPACITY])
    bind(vis, looks.get(color=WATER_COLOR, opacity=WATER_OPACITY, roughness=0.1))


def lights(stage, cfg):
    UsdGeom.Scope.Define(stage, '/World/Lights')
    s = cfg['sun']
    sun = UsdLux.DistantLight.Define(stage, '/World/Lights/sun')
    set_pose(sun, quat=matrix_to_quat(light_rotation(s['direction'])))
    sun.CreateIntensityAttr(float(s['intensity']))
    sun.CreateColorAttr(Gf.Vec3f(*map(float, s['color'])))
    sun.CreateAngleAttr(float(s['angle_deg']))
    UsdLux.ShadowAPI.Apply(sun.GetPrim()).CreateShadowEnableAttr(True)
    d = cfg['dome']
    dome = UsdLux.DomeLight.Define(stage, '/World/Lights/sky')
    dome.CreateIntensityAttr(float(d['intensity']))
    dome.CreateColorAttr(Gf.Vec3f(*map(float, d['color'])))


def camera_joint_from_xacro(path=ROBOT_XACRO):
    """(xyz, rpy) of the fixed camera_joint (trunk -> camera_link) in the robot xacro."""
    root = ET.parse(path).getroot()
    for j in root.iter('joint'):
        if j.get('name') == 'camera_joint':
            o = j.find('origin')
            return ([_xacro_num(v) for v in o.get('xyz').split()],
                    [_xacro_num(v) for v in o.get('rpy').split()])
    raise ValueError(f'camera_joint not found in {path}')


def _xacro_num(tok):
    m = re.fullmatch(r'\$\{(.*)\}', tok)
    if not m:
        return float(tok)
    return float(eval(m.group(1), {'__builtins__': {}}, {'pi': math.pi, 'radians': math.radians}))


def robot_eye_pose(scene, standing_height, joint=None):
    """World position and rotation (Gazebo camera convention) of the robot's camera_link
    with the trunk standing at robot_start."""
    xyz, rpy = joint or camera_joint_from_xacro()
    x, y, yaw = map(float, scene['robot_start']['pose'])
    trunk_p = np.array([x, y, float(terrain_mod.height_at(scene['terrain'], x, y)) + standing_height])
    trunk_r = rot_z(yaw)
    pos = trunk_p + trunk_r @ np.asarray(xyz, dtype=float)
    return pos, trunk_r @ rpy_matrix(*rpy)


def camera(stage, name, pos, r_world_cam, hfov, resolution, clip):
    """A USD camera at a Gazebo-convention pose (x forward, z up)."""
    w, h = map(int, resolution)
    near, far = map(float, clip)
    if near <= 0.0:
        raise ValueError(f'camera {name}: near clip must be > 0 (a near clip of 0 renders black)')
    cam = UsdGeom.Camera.Define(stage, f'/World/Cameras/{name}')
    set_pose(cam, pos, matrix_to_quat(usd_camera_rotation(r_world_cam, 'gazebo')))
    cam.CreateProjectionAttr(UsdGeom.Tokens.perspective)
    cam.CreateHorizontalApertureAttr(CAMERA_HORIZONTAL_APERTURE)
    cam.CreateVerticalApertureAttr(CAMERA_HORIZONTAL_APERTURE * h / w)
    cam.CreateFocalLengthAttr(hfov_to_focal(float(hfov)))
    cam.CreateClippingRangeAttr(Gf.Vec2f(near, far))
    cam.GetPrim().SetCustomDataByKey('beach:resolution', Gf.Vec2i(w, h))
    return cam


def cameras(stage, scene, cfg, sensors):
    UsdGeom.Scope.Define(stage, '/World/Cameras')
    area_w, area_h = map(float, scene['area']['size'])
    for name in ('top', 'oblique'):
        c = cfg[name]
        x, y, z, roll, pitch, yaw = c['pose']
        x = area_w / 2.0 if x is None else x
        y = area_h / 2.0 if y is None else y
        camera(stage, name, (x, y, z), rpy_matrix(roll, pitch, yaw), c['horizontal_fov_rad'],
               c['resolution'], c['clip'])
    s = sensors['camera']
    pos, r = robot_eye_pose(scene, float(cfg['robot_eye']['standing_height_m']))
    camera(stage, 'robot_eye', pos, r, s['horizontal_fov_rad'], (s['width'], s['height']),
           (s['clip_near_m'], s['clip_far_m']))


# ---------------------------------------------------------------- stage

def load_configs():
    with open(COMMON / 'objects.yaml') as f:
        objects = yaml.safe_load(f)
    with open(RENDER_YAML) as f:
        render = yaml.safe_load(f)
    with open(SENSORS_YAML) as f:
        sensors = yaml.safe_load(f)
    return objects, render, sensors


def build_stage(scene, objects, render, sensors, out_path):
    """The scene as an in-memory stage; asset paths relative to out_path's folder."""
    out_dir = Path(out_path).resolve().parent
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, '/World')
    stage.SetDefaultPrim(world.GetPrim())
    stage.GetRootLayer().documentation = (
        f"GENERATED by sim/isaac/load_scene.py from bench/scenes.yaml, seed {scene['seed']}. Do not edit.")
    world.GetPrim().SetCustomDataByKey('beach:seed', int(scene['seed']))

    ps = UsdPhysics.Scene.Define(stage, '/World/PhysicsScene')
    ps.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    ps.CreateGravityMagnitudeAttr(9.81)

    looks = Looks(stage, out_dir)
    meshes = MeshCache()
    terrain_prims(stage, looks, scene)

    t = scene['terrain']
    classes = objects['classes']
    for g in GROUPS.values():
        UsdGeom.Xform.Define(stage, f'/World/{g}')
    for b in scene['bins']:
        spec = objects['bins'][b['model']]
        z = ground_z(t, b['pose'][0], b['pose'][1], 0.3, 'min')
        object_prim(stage, looks, meshes, '/World/Bins', b['id'], spec, b['pose'], z,
                    spec['class'], classes[spec['class']], dynamic=False)
    for lm in scene['landmarks']:
        spec = objects['landmarks'][lm['model']]
        if lm['model'] == 'fence_post':
            z = ground_z(t, lm['pose'][0], lm['pose'][1], 0.05, 'min') - 0.2   # set into the sand
        else:
            z = ground_z(t, lm['pose'][0], lm['pose'][1], 1.0, 'min')
        object_prim(stage, looks, meshes, '/World/Landmarks', lm['id'], spec, lm['pose'], z,
                    spec['class'], classes[spec['class']], dynamic=False)
    for it in scene['trash']:
        spec = objects['trash'][it['model']]
        z = ground_z(t, it['pose'][0], it['pose'][1], spec['footprint_radius'], 'max') + SPAWN_GAP
        # Half-buried: sink only the visual; the collision stays on the surface.
        height = max(p['pose'][2] + prim_extent(p)[2] / 2.0 for p in spec['collision'])
        dz = -it['bury_fraction'] * height if it['half_buried'] else 0.0
        object_prim(stage, looks, meshes, '/World/Trash', it['id'], spec, it['pose'], z,
                    spec['material'], classes[spec['material']], dynamic=True, visual_dz=dz)
    for d in scene['distractors']:
        spec = objects['distractors'][d['model']]
        z = ground_z(t, d['pose'][0], d['pose'][1], 0.0, 'min')
        object_prim(stage, looks, meshes, '/World/Distractors', d['id'], spec, d['pose'], z,
                    spec['class'], classes[spec['class']], dynamic=False)

    lights(stage, render['lights'])
    cameras(stage, scene, render['cameras'], sensors)
    return stage


def write_scene(seed, out=None, scenes_path=REPO / 'bench' / 'scenes.yaml'):
    """Build and write one scene. Returns (path, seconds)."""
    t0 = time.monotonic()
    scene = export_terrain.load_scene(seed, Path(scenes_path))
    objects, render, sensors = load_configs()
    out = Path(out) if out else OUT_DIR / f'scene_{seed}.usda'
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = build_stage(scene, objects, render, sensors, out)
    if not stage.GetRootLayer().Export(str(out)):
        raise RuntimeError(f'could not write {out}')
    return out, time.monotonic() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--scenes', default=str(REPO / 'bench' / 'scenes.yaml'))
    ap.add_argument('--out', help='output .usda path')
    args = ap.parse_args()
    out, secs = write_scene(args.seed, args.out, args.scenes)
    print(f'{out} ({secs:.1f} s, {out.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
