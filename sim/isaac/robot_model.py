"""The robot's URDF facts that D0b/D0c need, in plain numpy (no ROS, no Isaac).

- joints (type, axis, limits) and fixed transforms between links (`fixed_transform`), e.g.
  base_link -> trunk, gripper bar -> TCP, base_link -> basket compartments;
- the basket compartments' inner boxes, as `grasping.arm.compartment` / `in_basket` compute them
  (same formulas, from the URDF's collision boxes), in base_link;
- the drive gains for every movable joint (legs from the policy config, arm from
  beach_description/config/arm_gains.yaml), and the USD unit conversion for angular drives.

The URDF comes from the laptop (d0_from_laptop/urdf, notes/briefs/D0_laptop.md).
"""

import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROBOT_YAML = HERE / 'robot.yaml'
ARM_GAINS = REPO / 'packages' / 'beach_description' / 'config' / 'arm_gains.yaml'


def load_robot_cfg(path=ROBOT_YAML):
    with open(path) as f:
        return yaml.safe_load(f)


def laptop_dir(cfg=None):
    """The laptop files folder: $BEACH_D0_LAPTOP_DIR, else robot.yaml's laptop_dir."""
    cfg = cfg or load_robot_cfg()
    return Path(os.environ.get('BEACH_D0_LAPTOP_DIR') or cfg['laptop_dir']).expanduser()


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def transform(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    t = np.eye(4)
    t[:3, :3] = rpy_matrix(*rpy)
    t[:3, 3] = xyz
    return t


def _floats(s, n=3, default=0.0):
    return [float(v) for v in s.split()] if s else [default] * n


class Urdf:
    def __init__(self, path):
        self.path = Path(path)
        root = ET.parse(self.path).getroot()
        self.links = {}
        for link in root.findall('link'):
            inertial = link.find('inertial')
            mass = float(inertial.find('mass').get('value')) if inertial is not None else None
            boxes = []
            for c in link.findall('collision'):
                box = c.find('geometry/box')
                if box is not None:
                    o = c.find('origin')
                    boxes.append({'size': _floats(box.get('size')),
                                  'xyz': _floats(o.get('xyz') if o is not None else None)})
            self.links[link.get('name')] = {'mass': mass, 'boxes': boxes}
        self.joints = {}
        for j in root.findall('joint'):
            o = j.find('origin')
            lim = j.find('limit')
            ax = j.find('axis')
            self.joints[j.get('name')] = {
                'type': j.get('type'), 'parent': j.find('parent').get('link'), 'child': j.find('child').get('link'),
                'xyz': _floats(o.get('xyz') if o is not None else None),
                'rpy': _floats(o.get('rpy') if o is not None else None),
                'axis': _floats(ax.get('xyz') if ax is not None else None) if ax is not None else [1.0, 0.0, 0.0],
                'lower': float(lim.get('lower')) if lim is not None and lim.get('lower') else None,
                'upper': float(lim.get('upper')) if lim is not None and lim.get('upper') else None,
                'effort': float(lim.get('effort')) if lim is not None and lim.get('effort') else None,
            }
        self.parent_joint = {j['child']: name for name, j in self.joints.items()}
        roots = [n for n in self.links if n not in self.parent_joint]
        if len(roots) != 1:
            raise ValueError(f'expected one root link, got {roots}')
        self.root = roots[0]

    def movable(self):
        return [n for n, j in self.joints.items() if j['type'] != 'fixed']

    def chain(self, link):
        """Joint names from the root down to `link`."""
        out = []
        while link != self.root:
            j = self.parent_joint[link]
            out.append(j)
            link = self.joints[j]['parent']
        return out[::-1]

    def fixed_transform(self, frm, to):
        """4x4 transform of link `to` in link `frm`; every joint between them must be fixed."""
        def from_root(link):
            t = np.eye(4)
            for j in self.chain(link):
                jj = self.joints[j]
                if jj['type'] != 'fixed':
                    raise ValueError(f'{j} on the way to {link} is {jj["type"]}, not fixed')
                t = t @ transform(jj['xyz'], jj['rpy'])
            return t
        # walk only the part below the common ancestor, so movable joints above it are fine
        ca, cb = self.chain(frm), self.chain(to)
        k = 0
        while k < min(len(ca), len(cb)) and ca[k] == cb[k]:
            k += 1

        def below(chain_):
            t = np.eye(4)
            for j in chain_[k:]:
                jj = self.joints[j]
                if jj['type'] != 'fixed':
                    raise ValueError(f'{j} is {jj["type"]}, not fixed')
                t = t @ transform(jj['xyz'], jj['rpy'])
            return t
        return np.linalg.inv(below(ca)) @ below(cb)


def basket_size(urdf, link='basket_link'):
    """(length, width, height, wall) from basket_link's collision boxes (grasping.arm.basket_size)."""
    sizes = np.array([b['size'] for b in urdf.links[link]['boxes']])
    return float(sizes[:, 0].max()), float(sizes[:, 1].max()), float(sizes[:, 2].max()), float(sizes.min())


def compartments(urdf, frame='base_link'):
    """{bin_id: {x: (lo, hi), y: (lo, hi), floor, rim}} in `frame`, as grasping.arm.compartment."""
    length, width, height, wall = basket_size(urdf)
    out = {}
    for bin_id in ('recyclable', 'residual'):
        rim = urdf.fixed_transform(frame, f'basket_{bin_id}_link')[:3, 3]
        half_y = width / 4 - wall
        out[bin_id] = {'x': (rim[0] - length / 2 + wall, rim[0] + length / 2 - wall),
                       'y': (rim[1] - half_y, rim[1] + half_y),
                       'floor': rim[2] - height + wall, 'rim': rim[2], 'height': height}
    return out


def in_basket(comp, p, margin=0.03):
    """grasping.arm.in_basket on precomputed compartments: point p (base_link) in the basket."""
    rec, res = comp['recyclable'], comp['residual']
    y_lo, y_hi = min(rec['y'][0], res['y'][0]), max(rec['y'][1], res['y'][1])
    return bool(rec['x'][0] - margin <= p[0] <= rec['x'][1] + margin and y_lo - margin <= p[1] <= y_hi + margin
                and rec['floor'] - margin <= p[2] <= rec['rim'] + rec['height'])


def in_compartment(comp, bin_id, p, margin=0.03):
    """The right half: inside `bin_id`'s compartment in y (+ margin), and in the basket."""
    c = comp[bin_id]
    return in_basket(comp, p, margin) and c['y'][0] - margin <= p[1] <= c['y'][1] + margin


def drive_gains(cfg, policy_cfg, variant='custom', arm_gains_path=ARM_GAINS):
    """{joint: (kp, kd)} in SI (N m/rad, N m s/rad; N/m, N s/m for the fingers)."""
    r = cfg['robot']
    gains = {}
    for k, j in enumerate(r['leg_joints']):
        gains[j] = (float(policy_cfg['rl_kp'][k]), float(policy_cfg['rl_kd'][k]))
    with open(arm_gains_path) as f:
        ag = yaml.safe_load(f)
    table = dict(ag['gains'])
    table.update((ag.get('variants') or {}).get(variant) or {})
    for j in r['arm_joints'] + r['finger_joints']:
        gains[j] = (float(table[j]['kp']), float(table[j]['kd']))
    return gains


def usd_angular_gain(si_gain):
    """USD angular drives store stiffness / damping per degree (Isaac 6.2's Articulation converts
    with `1.0 / np.rad2deg(1.0 / stiffness)`): N m/rad -> N m/deg."""
    return si_gain * math.pi / 180.0
