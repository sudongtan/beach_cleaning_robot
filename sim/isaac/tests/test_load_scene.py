"""Mac tests for sim/isaac/load_scene.py (notes/briefs/D0_prep.md §3, items 1-8).

Seed 0 first; the `all_seeds` tests repeat the checks for every seed in bench/scenes.yaml.
"""

import math
import re

import numpy as np
import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdSemantics, UsdShade, UsdUtils

from conftest import ALL_SEEDS, REPO, gazebo_scene, isaac_scene, open_stage

GROUP_PATHS = ['/World/Environment', '/World/Bins', '/World/Landmarks', '/World/Trash', '/World/Distractors']
T = Usd.TimeCode.Default()


def world_matrix(prim):
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(T)


def axes(m):
    """World x, y, z axes of a prim (USD matrices are row-vector: row i = image of axis i)."""
    return [np.array([m[i][0], m[i][1], m[i][2]]) for i in range(3)]


def scene_prims(stage):
    """Gazebo model name -> USD prim, for every scene object."""
    out = {}
    for g in GROUP_PATHS:
        for child in stage.GetPrimAtPath(g).GetChildren():
            out[child.GetName()] = child
    return out


def sdf_pose(model):
    return [float(v) for v in model.find('pose').text.split()]


def rpy(roll, pitch, yaw):
    return isaac_scene.rpy_matrix(roll, pitch, yaw)


# ------------------------------------------------------------------ 1. stage

@pytest.mark.parametrize('seed', ALL_SEEDS)
def test_stage_metadata(built, seed):
    path, _ = built(seed)
    stage = open_stage(path)
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert UsdPhysics.GetStageKilogramsPerUnit(stage) == 1.0
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert stage.GetDefaultPrim().GetPath() == Sdf.Path('/World')
    ps = UsdPhysics.Scene(stage.GetPrimAtPath('/World/PhysicsScene'))
    assert ps
    assert tuple(ps.GetGravityDirectionAttr().Get()) == (0.0, 0.0, -1.0)
    assert ps.GetGravityMagnitudeAttr().Get() == pytest.approx(9.81)
    assert path.read_text().startswith('#usda'), 'text .usda only'


# ------------------------------------------------------------------ 2. same scene as Gazebo

@pytest.mark.parametrize('seed', ALL_SEEDS)
def test_same_poses_as_gazebo(built, gazebo_sdf, seed):
    stage = open_stage(built(seed)[0])
    world = gazebo_sdf(seed).find('world')
    models = {m.get('name'): m for m in world.findall('model') if not m.get('name').startswith('debug_')}
    prims = scene_prims(stage)
    assert set(prims) == set(models), (set(prims) ^ set(models))
    for name, model in models.items():
        x, y, z, roll, pitch, yaw = sdf_pose(model)
        assert roll == 0 and pitch == 0
        m = world_matrix(prims[name])
        t = m.ExtractTranslation()
        assert np.allclose([t[0], t[1], t[2]], [x, y, z], atol=1e-3), name
        ax = axes(m)[0]
        assert abs(ax[2]) < 1e-9, name
        d = math.atan2(ax[1], ax[0]) - yaw
        assert abs(math.atan2(math.sin(d), math.cos(d))) < 1e-3, name


@pytest.mark.parametrize('seed', ALL_SEEDS)
def test_same_static_and_collisions_as_gazebo(built, gazebo_sdf, seed):
    """Dynamic iff Gazebo's model is not static; same number of collision primitives at the same offsets."""
    stage = open_stage(built(seed)[0])
    world = gazebo_sdf(seed).find('world')
    prims = scene_prims(stage)
    for model in world.findall('model'):
        name = model.get('name')
        if name.startswith('debug_') or name in ('terrain', 'wet_zone', 'water'):
            continue
        prim = prims[name]
        static = model.find('static').text == 'true'
        assert prim.HasAPI(UsdPhysics.RigidBodyAPI) == (not static), name
        sdf_cols = model.find('link').findall('collision')
        usd_cols = [c for c in prim.GetChildren() if c.HasAPI(UsdPhysics.CollisionAPI)]
        assert len(usd_cols) == len(sdf_cols), name
        for sc, uc in zip(sdf_cols, usd_cols):
            px, py, pz = (float(v) for v in sc.find('pose').text.split()[:3])
            t = UsdGeom.Xformable(uc).GetLocalTransformation().ExtractTranslation()
            assert np.allclose([t[0], t[1], t[2]], [px, py, pz], atol=1e-4), name
        # half-buried visuals: the same visual offset as Gazebo
        vz = float(model.find('link').find('visual').find('pose').text.split()[2])
        vis = prim.GetChild('visual_0')
        t = UsdGeom.Xformable(vis).GetLocalTransformation().ExtractTranslation()
        assert t[2] == pytest.approx(vz, abs=1e-4), name


@pytest.mark.parametrize('seed', [0, 101])
def test_time_code_range(built, configs, seed):
    """GPU session 1: without a time-code range Isaac's timeline loops over one frame and its
    clock stays at 0 (the physics check then never ends)."""
    _, render, _ = configs
    stage = open_stage(built(seed)[0])
    tcps = stage.GetTimeCodesPerSecond()
    assert tcps == render['stage']['time_codes_per_second']
    span_s = (stage.GetEndTimeCode() - stage.GetStartTimeCode()) / tcps
    assert span_s >= 10 * render['render_check']['physics_seconds']
    assert stage.GetFramesPerSecond() == tcps


@pytest.mark.parametrize('seed', [0, 101])
def test_semantic_labels(built, configs, scene_of, seed):
    """UsdSemantics LabelsAPI 'class' on every object (Isaac 6.2, session 1), none on the environment."""
    stage = open_stage(built(seed)[0])
    for name, p in scene_prims(stage).items():
        labels = UsdSemantics.LabelsAPI(p, 'class')
        if name in ('terrain', 'wet_zone', 'water'):
            assert not p.HasAPI(UsdSemantics.LabelsAPI, 'class'), name
            continue
        assert p.HasAPI(UsdSemantics.LabelsAPI, 'class'), name
        assert list(labels.GetLabelsAttr().Get()) == [p.GetCustomDataByKey('beach:class')], name


def test_class_labels(built, configs, scene_of):
    objects = configs[0]
    stage = open_stage(built(0)[0])
    scene = scene_of(0)
    prims = scene_prims(stage)
    for it in scene['trash']:
        p = prims[it['id']]
        assert p.GetCustomDataByKey('beach:class') == it['material']
        assert p.GetCustomDataByKey('beach:class_id') == objects['classes'][it['material']]
    for d in scene['distractors']:
        assert prims[d['id']].GetCustomDataByKey('beach:class_id') == objects['classes'][d['type']]


# ------------------------------------------------------------------ 3. terrain

def _terrain_points(stage):
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath('/World/Environment/terrain/mesh'))
    assert mesh
    return np.array(mesh.GetPointsAttr().Get(), dtype=float), mesh


def _mesh_height(pts, n, x, y):
    """Height of the terrain mesh at x, y: the cell's two triangles (a b d, a d c)."""
    xs, ys = pts[:n, 0], pts[::n, 1]
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    i = min(int((x - xs[0]) // dx), n - 2)
    j = min(int((y - ys[0]) // dy), n - 2)
    u, v = (x - xs[i]) / dx, (y - ys[j]) / dy
    za, zb = pts[j * n + i, 2], pts[j * n + i + 1, 2]
    zc, zd = pts[(j + 1) * n + i, 2], pts[(j + 1) * n + i + 1, 2]
    if u >= v:    # triangle a b d
        return za + u * (zb - za) + v * (zd - zb)
    return za + v * (zc - za) + u * (zd - zc)    # triangle a d c


@pytest.mark.parametrize('seed', [0, ALL_SEEDS[-1]])
def test_terrain_mesh(built, scene_of, seed):
    from terrain import grid_axes, height_at, height_grid
    t = scene_of(seed)['terrain']
    stage = open_stage(built(seed)[0])
    pts, mesh = _terrain_points(stage)
    h = height_grid(t)
    n = h.shape[0]
    assert pts.shape == (n * n, 3)
    assert np.allclose(pts[:, 2], h.ravel(), atol=1e-5)
    hm = t['heightmap']
    cx, cy = hm['center']
    half = hm['size'] / 2.0
    assert pts[:, 0].min() == pytest.approx(cx - half, abs=1e-4)
    assert pts[:, 0].max() == pytest.approx(cx + half, abs=1e-4)
    assert pts[:, 1].min() == pytest.approx(cy - half, abs=1e-4)
    assert pts[:, 1].max() == pytest.approx(cy + half, abs=1e-4)
    xs, ys = grid_axes(t)
    assert np.allclose(pts[:n, 0], xs, atol=1e-4)
    # triangles face up (counter-clockwise from above, USD rightHanded)
    idx = np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    a, b, c = pts[idx[:, 0]], pts[idx[:, 1]], pts[idx[:, 2]]
    assert (np.cross(b - a, c - a)[:, 2] > 0).all()
    assert mesh.GetSubdivisionSchemeAttr().Get() == UsdGeom.Tokens.none
    rng = np.random.default_rng(seed)
    for x, y in rng.uniform([cx - half + 0.1, cy - half + 0.1], [cx + half - 0.1, cy + half - 0.1], (20, 2)):
        assert _mesh_height(pts, n, x, y) == pytest.approx(float(height_at(t, x, y)), abs=0.01)


def test_terrain_same_triangulation_as_obj_export(built, scene_of, tmp_path):
    from terrain import write_obj_mesh
    write_obj_mesh(scene_of(0)['terrain'], tmp_path / 't.obj')
    faces = [list(map(int, ln.split()[1:])) for ln in (tmp_path / 't.obj').read_text().splitlines()
             if ln.startswith('f ')]
    stage = open_stage(built(0)[0])     # keep a reference: prims die with their stage
    _, mesh = _terrain_points(stage)
    usd = np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    assert np.array_equal(usd, np.array(faces) - 1)


def test_terrain_uv_tiles_every_3m(built):
    stage = open_stage(built(0)[0])
    pts, mesh = _terrain_points(stage)
    st = np.array(UsdGeom.PrimvarsAPI(mesh).GetPrimvar('st').Get())
    assert np.allclose(st, pts[:, :2] / 3.0, atol=1e-4)


# ------------------------------------------------------------------ 4. physics

def _friction(prim):
    mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
    assert mat, prim.GetPath()
    api = UsdPhysics.MaterialAPI(mat.GetPrim())
    return api.GetStaticFrictionAttr().Get(), api.GetDynamicFrictionAttr().Get(), api.GetRestitutionAttr().Get()


@pytest.mark.parametrize('seed', ALL_SEEDS)
def test_physics(built, configs, scene_of, seed):
    objects = configs[0]
    scene = scene_of(seed)
    stage = open_stage(built(seed)[0])
    trash = {it['id']: it for it in scene['trash']}
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    assert sorted(p.GetName() for p in bodies) == sorted(trash)
    for p in bodies:
        spec = objects['trash'][trash[p.GetName()]['model']]
        assert UsdPhysics.MassAPI(p).GetMassAttr().Get() == pytest.approx(spec['mass'])
        # no mesh collider on a moving body (PhysX would fall back to a convex hull)
        for d in Usd.PrimRange(p):
            if d.HasAPI(UsdPhysics.CollisionAPI):
                assert not d.IsA(UsdGeom.Mesh), d.GetPath()
                assert UsdGeom.Imageable(d).GetPurposeAttr().Get() == UsdGeom.Tokens.guide
    # the only mesh collider is the static terrain, exact triangles
    mesh_cols = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI) and p.IsA(UsdGeom.Mesh)]
    assert [str(p.GetPath()) for p in mesh_cols] == ['/World/Environment/terrain/mesh']
    assert UsdPhysics.MeshCollisionAPI(mesh_cols[0]).GetApproximationAttr().Get() == UsdPhysics.Tokens.none
    fr = scene['terrain']['friction']
    assert _friction(mesh_cols[0]) == pytest.approx((fr['dry_sand'], fr['dry_sand'], 0.0))
    wet = stage.GetPrimAtPath('/World/Environment/wet_zone/collision')
    assert wet.HasAPI(UsdPhysics.CollisionAPI)
    assert _friction(wet) == pytest.approx((fr['wet_sand'], fr['wet_sand'], 0.0))
    for d in Usd.PrimRange(stage.GetPrimAtPath('/World/Environment/water')):
        assert not d.HasAPI(UsdPhysics.CollisionAPI), d.GetPath()


def test_trash_inertia_matches_gazebo(built, gazebo_sdf):
    stage = open_stage(built(0)[0])
    world = gazebo_sdf(0).find('world')
    prims = scene_prims(stage)
    for model in world.findall('model'):
        inertial = model.find('link').find('inertial')
        if inertial is None:
            continue
        m = UsdPhysics.MassAPI(prims[model.get('name')])
        it = inertial.find('inertia')
        expect = [float(it.find(k).text) for k in ('ixx', 'iyy', 'izz')]
        assert np.allclose(list(m.GetDiagonalInertiaAttr().Get()), expect, rtol=2e-3)
        com = [float(v) for v in inertial.find('pose').text.split()[:3]]
        assert np.allclose(list(m.GetCenterOfMassAttr().Get()), com, atol=1e-6)


# ------------------------------------------------------------------ 5. materials

def _rendered_meshes(stage):
    for p in stage.Traverse():
        if p.IsA(UsdGeom.Gprim) and UsdGeom.Imageable(p).ComputePurpose() != UsdGeom.Tokens.guide:
            yield p


def _surface_input(mat, name):
    shader = mat.ComputeSurfaceSource()[0]
    return shader.GetInput(name).Get()


@pytest.mark.parametrize('seed', ALL_SEEDS)
def test_materials_and_assets(built, scene_of, seed):
    path, _ = built(seed)
    stage = open_stage(path)
    meshes = list(_rendered_meshes(stage))
    assert meshes
    for p in meshes:
        assert p.IsA(UsdGeom.Mesh), p.GetPath()
        mat, _ = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()
        assert mat, f'{p.GetPath()} has no material'
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(path))
    assert not unresolved, unresolved
    assert len(assets) >= 3                         # sand, normal, wet sand, + prints
    for a in re.findall(r'@([^@]+)@', path.read_text()):
        assert not a.startswith('/') and '://' not in a, f'absolute asset path {a}'
    # see-through bodies: PET and glass
    scene = scene_of(seed)
    for it in scene['trash']:
        if it['material'] in ('pet', 'glass'):
            body = stage.GetPrimAtPath(f"/World/Trash/{it['id']}/visual_0")
            mat, _ = UsdShade.MaterialBindingAPI(body).ComputeBoundMaterial()
            assert _surface_input(mat, 'opacity') < 1.0, it['id']


def test_water_is_see_through(built):
    stage = open_stage(built(0)[0])
    mat, _ = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath('/World/Environment/water/visual')).ComputeBoundMaterial()
    assert _surface_input(mat, 'opacity') == pytest.approx(0.75)
    assert tuple(_surface_input(mat, 'diffuseColor')) == pytest.approx((0.10, 0.35, 0.55))


def test_obj_reader_triangulates_and_reads_uvs(tmp_path):
    f = tmp_path / 'quad.obj'
    f.write_text('v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvn 0 0 1\n'
                 'f 1/1/1 2/2/1 3/3/1 4/4/1\nf -4//-1 -2//-1 -1//-1\n')
    d = isaac_scene.read_obj(f)
    assert d['counts'] == [3, 3, 3]
    assert d['indices'] == [0, 1, 2, 0, 2, 3, 0, 2, 3]
    assert d['normals'] is not None and d['st'] is None       # the second face has no UVs
    f.write_text('v 0 0 0\nv 1 0 0\nv 1 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nf 1/1 2/2 3/3\n')
    d = isaac_scene.read_obj(f)
    assert d['st'][1] == [0, 1, 2] and d['normals'] is None


# ------------------------------------------------------------------ 6. cameras

def _hfov(cam):
    return 2.0 * math.atan(cam.GetHorizontalApertureAttr().Get() / (2.0 * cam.GetFocalLengthAttr().Get()))


def _cam_vectors(prim):
    """World position, forward (-Z) and up (+Y) of a USD camera."""
    m = world_matrix(prim)
    x, y, z = axes(m)
    t = m.ExtractTranslation()
    return np.array([t[0], t[1], t[2]]), -z, y


def test_robot_eye_camera(built, configs, scene_of):
    from terrain import height_at
    _, render, sensors = configs
    s = sensors['camera']
    stage = open_stage(built(0)[0])
    cam = UsdGeom.Camera(stage.GetPrimAtPath('/World/Cameras/robot_eye'))
    assert _hfov(cam) == pytest.approx(s['horizontal_fov_rad'], abs=1e-4)
    assert tuple(cam.GetClippingRangeAttr().Get()) == pytest.approx((s['clip_near_m'], s['clip_far_m']))
    assert cam.GetVerticalApertureAttr().Get() == pytest.approx(
        cam.GetHorizontalApertureAttr().Get() * s['height'] / s['width'])
    assert tuple(cam.GetPrim().GetCustomDataByKey('beach:resolution')) == (s['width'], s['height'])
    # independent pose: the xacro's camera_joint read with a regex, plain trigonometry
    xacro = (REPO / 'packages/beach_description/urdf/beach_robot.urdf.xacro').read_text()
    m = re.search(r'<joint name="camera_joint".*?<origin xyz="([^"]+)" rpy="0 \$\{radians\((\d+(?:\.\d+)?)\)\} 0"',
                  xacro, re.S)
    jx, jy, jz = map(float, m.group(1).split())
    pitch = math.radians(float(m.group(2)))
    scene = scene_of(0)
    x, y, yaw = scene['robot_start']['pose']
    trunk_z = float(height_at(scene['terrain'], x, y)) + render['cameras']['robot_eye']['standing_height_m']
    expect_pos = [x + math.cos(yaw) * jx - math.sin(yaw) * jy, y + math.sin(yaw) * jx + math.cos(yaw) * jy,
                  trunk_z + jz]
    expect_fwd = [math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), -math.sin(pitch)]
    pos, fwd, up = _cam_vectors(cam.GetPrim())
    assert np.allclose(pos, expect_pos, atol=1e-6)
    assert np.allclose(fwd, expect_fwd, atol=1e-6)
    assert up[2] > 0.9                             # image up is roughly world up


@pytest.mark.parametrize('seed', [0, 5])
def test_debug_cameras_match_gazebo(built, gazebo_sdf, seed):
    stage = open_stage(built(seed)[0])
    world = gazebo_sdf(seed).find('world')
    for name in ('top', 'oblique'):
        model = world.find(f"model[@name='debug_{name}']")
        x, y, z, roll, pitch, yaw = sdf_pose(model)
        sensor = model.find('link').find(f"sensor[@name='{name}_rgb']")
        c = sensor.find('camera')
        r = rpy(roll, pitch, yaw)
        cam = UsdGeom.Camera(stage.GetPrimAtPath(f'/World/Cameras/{name}'))
        pos, fwd, up = _cam_vectors(cam.GetPrim())
        assert np.allclose(pos, [x, y, z], atol=1e-3), name
        assert np.allclose(fwd, r[:, 0], atol=1e-4), name     # Gazebo camera looks along +x
        assert np.allclose(up, r[:, 2], atol=1e-4), name      # and has +z up
        assert _hfov(cam) == pytest.approx(float(c.find('horizontal_fov').text), abs=1e-4)
        w, h = int(c.find('image/width').text), int(c.find('image/height').text)
        assert tuple(cam.GetPrim().GetCustomDataByKey('beach:resolution')) == (w, h)
        assert cam.GetVerticalApertureAttr().Get() == pytest.approx(cam.GetHorizontalApertureAttr().Get() * h / w)
        near, far = float(c.find('clip/near').text), float(c.find('clip/far').text)
        assert tuple(cam.GetClippingRangeAttr().Get()) == pytest.approx((near, far))


@pytest.mark.parametrize('seed', ALL_SEEDS)
def test_no_zero_near_clip(built, seed):
    stage = open_stage(built(seed)[0])
    cams = [p for p in stage.Traverse() if p.IsA(UsdGeom.Camera)]
    assert {c.GetName() for c in cams} == {'top', 'oblique', 'robot_eye'}
    for c in cams:
        assert UsdGeom.Camera(c).GetClippingRangeAttr().Get()[0] > 0.0


def test_camera_builder_refuses_zero_near_clip():
    stage = Usd.Stage.CreateInMemory()
    with pytest.raises(ValueError):
        isaac_scene.camera(stage, 'c', (0, 0, 0), np.eye(3), 1.2, (640, 480), (0.0, 30.0))


def _ndc(cam_prim, point):
    frustum = UsdGeom.Camera(cam_prim).GetCamera(T).frustum
    m = frustum.ComputeViewMatrix() * frustum.ComputeProjectionMatrix()
    p = m.Transform(Gf.Vec3d(*point))
    return np.array([p[0], p[1], p[2]])


@pytest.mark.parametrize('convention,r', [
    ('gazebo', np.eye(3)),
    # the same camera in the ROS optical frame: z = world x, x = world -y, y = world -z
    ('optical', np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)),
])
def test_camera_projection_convention(convention, r):
    """A camera at the origin looking along +x: a point straight ahead lands in the image centre,
    a point to the right (-y) right of centre, a point above (+z) above centre."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    cam = UsdGeom.Camera.Define(stage, '/cam')
    isaac_scene.set_pose(cam, (0, 0, 0), isaac_scene.matrix_to_quat(isaac_scene.usd_camera_rotation(r, convention)))
    cam.CreateHorizontalApertureAttr(20.955)
    cam.CreateVerticalApertureAttr(20.955 * 0.75)
    cam.CreateFocalLengthAttr(isaac_scene.hfov_to_focal(1.2))
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.1, 30.0))
    centre = _ndc(cam.GetPrim(), (5, 0, 0))
    assert np.allclose(centre[:2], 0.0, atol=1e-9)
    assert -1.0 < centre[2] < 1.0                   # in front, inside the clip range
    assert _ndc(cam.GetPrim(), (5, -1, 0))[0] > 0.1
    assert _ndc(cam.GetPrim(), (5, 0, 1))[1] > 0.1
    # behind the camera: outside the depth range
    behind = _ndc(cam.GetPrim(), (-5, 0, 0))
    assert not (-1.0 < behind[2] < 1.0)


def test_hfov_formula():
    for hfov in (0.5, 1.2, 1.3, 2.0):
        assert 2 * math.atan(20.955 / (2 * isaac_scene.hfov_to_focal(hfov))) == pytest.approx(hfov)


def test_sun_direction(built, configs):
    _, render, _ = configs
    stage = open_stage(built(0)[0])
    m = world_matrix(stage.GetPrimAtPath('/World/Lights/sun'))
    travel = -axes(m)[2]                           # UsdLux lights shine along local -Z
    d = np.array(render['lights']['sun']['direction'], dtype=float)
    assert np.allclose(travel, d / np.linalg.norm(d), atol=1e-6)


# ------------------------------------------------------------------ 7. validation

@pytest.mark.parametrize('seed', [0])
def test_usd_validation(built, seed):
    UsdValidation = pytest.importorskip('pxr.UsdValidation', reason='usd-core has no UsdValidation')
    reg = UsdValidation.ValidationRegistry()
    validators = [reg.GetOrLoadValidatorByName(m.name) for m in reg.GetAllValidatorMetadata()]
    assert validators
    stage = open_stage(built(seed)[0])
    from pxr import Tf
    bad, skipped = [], []
    for v in validators:      # one at a time: a broken validator mustn't hide the others
        try:
            errors = UsdValidation.ValidationContext([v]).Validate(stage)
        except Tf.ErrorException as e:
            # usd-core 25.11's wheel (pinned to Isaac 6.2's USD) lacks shaderDefs.usda, so the
            # shader validators can't load; that's the package, not the stage.
            if 'Could not find shader resource' in str(e):
                skipped.append(v.GetMetadata().name)
                continue
            raise
        bad += [f'{e.GetName()}: {e.GetMessage()}' for e in errors
                if e.GetType() == UsdValidation.ValidationErrorType.Error]
    print(f'validators: {len(validators) - len(skipped)} run, skipped (missing shaderDefs.usda): {skipped}')
    assert len(skipped) <= 3, skipped
    assert not bad, '\n'.join(bad[:20])


# ------------------------------------------------------------------ 8. size and time

@pytest.mark.parametrize('seed', ALL_SEEDS)
def test_size_and_time(built, configs, seed):
    _, render, _ = configs
    path, secs = built(seed)
    mb = path.stat().st_size / 1e6
    print(f'seed {seed}: {secs:.2f} s, {mb:.1f} MB')
    assert secs < render['build']['max_seconds']
    assert mb < render['build']['max_megabytes']
