#!/usr/bin/env python3
"""Check an imported robot USD against its URDF, and write the joint drives (D0b).

    python sim/isaac/check_robot_usd.py --usd ROBOT.usd [--apply-drives] [--report OUT.json]

Pure pxr: runs on the Mac (usd-core) and on the GPU host right after the import (Isaac's pxr),
so a bad import shows up within the same session. Checks: one articulation root; every URDF
movable joint present with its type, axis and limits (USD revolute limits are in degrees); every
link with mass is a rigid body with that mass; no exact triangle-mesh collider on a link (PhysX
would fall back to a convex hull on a moving body); the frames D0 needs exist; no extra physics
scene. `--apply-drives` writes force drives with position targets: legs at the policy's kp / kd,
the arm at arm_gains.yaml, angular gains per degree (USD), max force = the URDF effort (legs: the
policy's torque limit), targets = the stand pose.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kit_pxr  # noqa: E402

kit_pxr.ensure_pxr()
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

import robot_model  # noqa: E402

FRAMES = ['base_link', 'trunk', 'imu_link', 'camera_link', 'arm_base_link', 'arm_gripper_bar_link',
          'arm_ee_gripper_link', 'arm_jaw_tips_link', 'basket_link', 'basket_recyclable_link',
          'basket_residual_link']


def by_name(stage, kind=None):
    """{prim name: prim} for every prim (or every prim of a UsdPhysics joint kind)."""
    out = {}
    for p in stage.Traverse():
        if kind is None or p.IsA(kind):
            out.setdefault(p.GetName(), p)
    return out


def joint_prims(stage):
    return by_name(stage, UsdPhysics.Joint)


def check(stage, urdf):
    errors, notes = [], []
    roots = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if len(roots) != 1:
        errors.append(f'articulation roots: {roots} (want exactly one)')
    joints = joint_prims(stage)
    for name in urdf.movable():
        j = urdf.joints[name]
        p = joints.get(name)
        if p is None:
            errors.append(f'joint {name} missing')
            continue
        want = {'revolute': UsdPhysics.RevoluteJoint, 'continuous': UsdPhysics.RevoluteJoint,
                'prismatic': UsdPhysics.PrismaticJoint}[j['type']]
        if not p.IsA(want):
            errors.append(f'joint {name} is {p.GetTypeName()}, URDF {j["type"]}')
            continue
        jj = want(p)
        lo, hi = jj.GetLowerLimitAttr().Get(), jj.GetUpperLimitAttr().Get()
        if j['type'] == 'revolute' and j['lower'] is not None:
            scale = 180.0 / math.pi
            if lo is None or hi is None or abs(lo - j['lower'] * scale) > 0.05 or abs(hi - j['upper'] * scale) > 0.05:
                errors.append(f'joint {name} limits {lo}..{hi} deg, URDF {j["lower"]}..{j["upper"]} rad')
        if j['type'] == 'prismatic' and j['lower'] is not None:
            if lo is None or abs(lo - j['lower']) > 1e-4 or abs(hi - j['upper']) > 1e-4:
                errors.append(f'joint {name} limits {lo}..{hi} m, URDF {j["lower"]}..{j["upper"]}')
        axis = jj.GetAxisAttr().Get()
        notes.append(f'{name}: axis {axis}, URDF {j["axis"]}')
    links = by_name(stage)
    for name, link in urdf.links.items():
        if link['mass'] is None or link['mass'] < 0.01:
            continue                                     # massless / 1 g frames may be merged or kept
        p = links.get(name)
        if p is None:
            errors.append(f'link {name} ({link["mass"]} kg) missing')
            continue
        if not p.HasAPI(UsdPhysics.RigidBodyAPI):
            errors.append(f'link {name} has no RigidBodyAPI')
        m = UsdPhysics.MassAPI(p).GetMassAttr().Get() if p.HasAPI(UsdPhysics.MassAPI) else None
        if m is None or abs(m - link['mass']) > 1e-4 * max(1.0, link['mass']):
            errors.append(f'link {name} mass {m}, URDF {link["mass"]}')
        for d in Usd.PrimRange(p):
            if d.IsA(UsdGeom.Mesh) and d.HasAPI(UsdPhysics.CollisionAPI):
                appr = UsdPhysics.MeshCollisionAPI(d).GetApproximationAttr().Get() if d.HasAPI(UsdPhysics.MeshCollisionAPI) else None
                if appr in (None, UsdPhysics.Tokens.none):
                    errors.append(f'{d.GetPath()}: triangle-mesh collider on a moving link (approximation {appr})')
    missing = [f for f in FRAMES if f not in links]
    if missing:
        errors.append(f'frames missing: {missing}')
    scenes = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if scenes:
        notes.append(f'physics scenes in the robot asset (compose.py deactivates them): {scenes}')
    return {'ok': not errors, 'errors': errors, 'notes': notes, 'articulation_roots': roots,
            'joints_found': sorted(set(joints) & set(urdf.movable())),
            'meters_per_unit': UsdGeom.GetStageMetersPerUnit(stage), 'up_axis': UsdGeom.GetStageUpAxis(stage),
            'default_prim': str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim() else None}


def apply_drives(stage, urdf, gains, targets, leg_effort, leg_joints):
    """Force drives with position targets on every movable joint. gains / targets in SI (rad, m)."""
    joints = joint_prims(stage)
    done = []
    for name in urdf.movable():
        p = joints.get(name)
        if p is None:
            continue
        angular = urdf.joints[name]['type'] in ('revolute', 'continuous')
        drive = UsdPhysics.DriveAPI.Apply(p, 'angular' if angular else 'linear')
        kp, kd = gains[name]
        conv = robot_model.usd_angular_gain if angular else (lambda g: g)
        drive.CreateTypeAttr('force')
        drive.CreateStiffnessAttr(float(conv(kp)))
        drive.CreateDampingAttr(float(conv(kd)))
        effort = leg_effort if name in leg_joints else urdf.joints[name]['effort']
        if effort:
            drive.CreateMaxForceAttr(float(effort))
        tgt = float(targets.get(name, 0.0))
        drive.CreateTargetPositionAttr(math.degrees(tgt) if angular else tgt)
        done.append(name)
    return done


def stand_targets(cfg, stand_pos):
    r = cfg['robot']
    t = dict(zip(r['leg_joints'], stand_pos))
    t.update(zip(r['arm_joints'], r['arm_rest']))
    t[r['finger_joints'][0]] = r['finger_rest']
    t[r['finger_joints'][1]] = -r['finger_rest']
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--usd', required=True)
    ap.add_argument('--apply-drives', action='store_true')
    ap.add_argument('--report')
    a = ap.parse_args()
    cfg = robot_model.load_robot_cfg()
    laptop = robot_model.laptop_dir(cfg)
    urdf = robot_model.Urdf(laptop / cfg['urdf_import']['urdf'])
    stage = Usd.Stage.Open(a.usd)
    rep = {}
    if a.apply_drives:
        import walk
        with open(laptop / cfg['policy']['config']) as f:
            pcfg = yaml.safe_load(f)
        gains = robot_model.drive_gains(cfg, pcfg, cfg['variant'])
        tg = stand_targets(cfg, walk.load_controller()['stand_pos'])
        rep['drives'] = apply_drives(stage, urdf, gains, tg, cfg['leg_effort_limit'], cfg['robot']['leg_joints'])
        stage.GetRootLayer().Save()
    rep.update(check(stage, urdf))
    text = json.dumps(rep, indent=2)
    if a.report:
        Path(a.report).write_text(text)
    print(text)
    kit_pxr.close(0 if rep['ok'] else 1)


if __name__ == '__main__':
    main()
