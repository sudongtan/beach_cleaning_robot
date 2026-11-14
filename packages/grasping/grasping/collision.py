"""Collision check of planned arm moves (B5), plain Python + Pinocchio / coal (no ROS).

Every arm move is a single-point trajectory goal from rest to rest, so the controller runs it as the
straight line qa + s (qb - qa) in joint space. A move is sampled so that no joint changes more than
step_rad between samples; at every sample each active pair must be at least clearance_m apart
(coal's CollisionRequest.security_margin: closer counts as a collision).

What is modelled, all in base_link (the legs in Arm's stand pose):
  - the robot from the URDF collision geometry: every arm link from the shoulder outward against the
    body (trunk, legs, plate, IMU, basket walls), and arm self pairs at least 2 links apart (as
    beach_description/scripts/arm_pose_check.py);
  - items lying in the basket (ground-truth boxes) and bin walls (objects.yaml collision boxes) as
    obstacles fixed in base_link, against every arm link from the shoulder outward;
  - the held item, a box fixed to the jaw-tips frame, against the body, the arm from the shoulder to
    the wrist (not the hand that holds it), the items and the bins.
An item to be picked (from the ground or the basket) is no obstacle until its attach step; after its
detach it stays where it was let go. Not modelled: the ground, items on the ground, the other bin.
"""

import math

import coal
import numpy as np
import pinocchio as pin

from grasping.arm import TIPS

CHAIN = ['arm_base_link', 'arm_shoulder_link', 'arm_upper_arm_link', 'arm_upper_forearm_link',
         'arm_lower_forearm_link', 'arm_wrist_link', 'arm_gripper_link']
SEPARATING_TOL = 1e-4                      # m: a pair in contact at a move's start may not close by more
HAND = {'arm_gripper_link', 'arm_gripper_prop_link', 'arm_gripper_bar_link',
        'arm_left_finger_link', 'arm_right_finger_link'}


def link_of(name):
    return name.rsplit('_', 1)[0]           # pinocchio names geometries <link>_<index>


def rank(link):
    return len(CHAIN) - 1 if link in HAND else CHAIN.index(link)


def aabb_item(o):
    """An item in plan_store's form ('centre', 'half' (x, y), 'top') as an obstacle box."""
    c = np.asarray(o['centre'], dtype=float)
    return {'centre': c, 'R': np.eye(3), 'size': (2 * o['half'][0], 2 * o['half'][1], 2 * (o['top'] - c[2]))}


def bin_boxes(objects_bin, opening, yaw):
    """Collision boxes of a bin (objects.yaml entry) as obstacles: [(name, size, pin.SE3)] in base_link.
    opening: centre of the bin opening at the rim (base_link), yaw: the bin's yaw relative to base_link."""
    rim = max(b['pose'][2] + b['size'][2] / 2 for b in objects_bin['collision'])
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    origin = np.asarray(opening, dtype=float) - [0.0, 0.0, rim]
    return [(f'bin_{k}', tuple(b['size']), pin.SE3(R, origin + R @ np.asarray(b['pose'][:3], dtype=float)))
            for k, b in enumerate(objects_bin['collision'])]


class State:
    """What the arm moves among: obstacle items {id: {'centre', 'R', 'size'}}, `excluded` ids (items to
    be picked: no obstacle until attached), bin boxes, the held item (id, centre and rotation in the
    jaw-tips frame, size) and the finger joint value."""

    def __init__(self, items=None, excluded=(), bins=(), held=None, fingers=None):
        self.items = dict(items or {})
        self.excluded = set(excluded)
        self.bins = list(bins)
        self.held = held
        self.fingers = fingers

    def copy(self):
        return State(self.items, self.excluded, self.bins, self.held, self.fingers)


class Collision:

    def __init__(self, arm, clearance, step, refine=None, refine_step=None):
        self.arm, self.clearance, self.step = arm, float(clearance), float(step)
        self.refine = float(refine or 0.0)                       # 0: no refinement
        self.refine_step = float(refine_step or step)
        m = arm.model
        self.data = m.createData()
        base = pin.buildGeomFromUrdfString(m, arm.urdf.replace('file://', ''), pin.GeometryType.COLLISION)
        names = [g.name for g in base.geometryObjects]
        arm_links = {f.name for f in m.frames if f.name.startswith('arm_') and f.name != 'arm_plate_link'}
        self.moving = [i for i, n in enumerate(names) if link_of(n) in arm_links and link_of(n) != 'arm_base_link']
        self.body = [i for i in range(len(names)) if i not in self.moving]
        self.held_vs_arm = [i for i in self.moving if link_of(names[i]) not in HAND]
        for i, a in enumerate(names):
            for j in range(i + 1, len(names)):
                la, lb = link_of(a), link_of(names[j])
                in_a, in_b = la in arm_links, lb in arm_links
                if (in_a and in_b and abs(rank(la) - rank(lb)) >= 2) or \
                        (in_a != in_b and (la if in_a else lb) != 'arm_base_link'):
                    base.addCollisionPair(pin.CollisionPair(i, j))
        self.base = base
        self.robot_pairs = len(base.collisionPairs)
        fid = m.getFrameId(TIPS)
        self.tips_joint, self.tips_in_joint = m.frames[fid].parentJoint, m.frames[fid].placement
        self.tips_frame = fid
        self.fl = m.joints[m.getJointId('arm_left_finger')].idx_q
        self.fr = m.joints[m.getJointId('arm_right_finger')].idx_q
        self._cache = {}

    @classmethod
    def from_config(cls, arm, col):
        """From the grasping config's `collision` entry (None: no check)."""
        if not col:
            return None
        return cls(arm, col['clearance_m'], col['step_rad'], col.get('refine_m'), col.get('refine_step_rad'))

    # --- geometry of a state -------------------------------------------------------------------

    def _geometry(self, state):
        """(GeometryModel, GeometryData) with the state's obstacles and held item; cached per state."""
        return self._entry(state)[:2]

    def _entry(self, state):
        """(GeometryModel, GeometryData at clearance_m, GeometryData at clearance_m + refine_m)."""
        key = (id(state.items), frozenset(state.items) - state.excluded, id(state.bins),
               None if state.held is None else (state.held['id'], id(state.held)))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        if len(self._cache) > 64:
            self._cache.clear()
        geom = self.base.copy()
        obstacles = []
        for oid, it in state.items.items():
            if oid in state.excluded:
                continue
            obstacles.append(geom.addGeometryObject(pin.GeometryObject(
                f'item:{oid}', 0, pin.SE3(np.asarray(it['R'], dtype=float), np.asarray(it['centre'], dtype=float)),
                coal.Box(*map(float, it['size'])))))
        for name, size, M in state.bins:
            obstacles.append(geom.addGeometryObject(pin.GeometryObject(name, 0, M, coal.Box(*map(float, size)))))
        for o in obstacles:
            for i in self.moving:
                geom.addCollisionPair(pin.CollisionPair(i, o))
        if state.held is not None:
            h = state.held
            M = self.tips_in_joint * pin.SE3(np.asarray(h['R'], dtype=float), np.asarray(h['p'], dtype=float))
            k = geom.addGeometryObject(pin.GeometryObject(f"held:{h['id']}", self.tips_joint, M,
                                                          coal.Box(*map(float, h['size']))))
            for o in self.body + self.held_vs_arm + obstacles:
                geom.addCollisionPair(pin.CollisionPair(min(o, k), max(o, k)))
        gdata, near = pin.GeometryData(geom), pin.GeometryData(geom)
        for req in gdata.collisionRequests:
            req.security_margin = self.clearance
        for req in near.collisionRequests:
            req.security_margin = self.clearance + self.refine
        self._cache[key] = (geom, gdata, near)
        return self._cache[key]

    def full(self, arm_q, fingers):
        q = self.arm.full(arm_q)
        f = self.arm.finger_closed if fingers is None else fingers
        q[self.fl], q[self.fr] = f, -f
        return q

    def _pair_names(self, geom, k):
        cp = geom.collisionPairs[k]
        return (link_of(geom.geometryObjects[cp.first].name) if ':' not in geom.geometryObjects[cp.first].name
                else geom.geometryObjects[cp.first].name,
                link_of(geom.geometryObjects[cp.second].name) if ':' not in geom.geometryObjects[cp.second].name
                else geom.geometryObjects[cp.second].name)

    def distance(self, arm_q, state):
        """(minimum distance over the active pairs, (a, b)) at one pose."""
        geom, gdata = self._geometry(state)
        pin.computeDistances(self.arm.model, self.data, geom, gdata, self.full(arm_q, state.fingers))
        k = int(np.argmin([r.min_distance for r in gdata.distanceResults]))
        return gdata.distanceResults[k].min_distance, self._pair_names(geom, k)

    def _hit(self, q, geom, gdata):
        """None if every pair is clear at q (full configuration), else (distance, (a, b))."""
        if not pin.computeCollisions(self.arm.model, self.data, geom, gdata, q, True):
            return None
        for k, res in enumerate(gdata.collisionResults):
            if res.isCollision():
                return pin.computeDistance(geom, gdata, k).min_distance, self._pair_names(geom, k)
        return None

    # --- moves and step lists --------------------------------------------------------------------

    def samples(self, qa, qb, fa, fb, step=None):
        """Fractions s in (0, 1] along a move, no joint (fingers included) changing more than step."""
        step = self.step if step is None else step
        span = max(float(np.max(np.abs(np.subtract(qb, qa)))) if len(qa) else 0.0, abs(fb - fa))
        n = max(1, int(math.ceil(span / step)))
        return [k / n for k in range(1, n + 1)]

    def _contacts(self, q, geom, gdata):
        """{pair index: distance} of every pair closer than clearance_m at q (full configuration)."""
        if not pin.computeCollisions(self.arm.model, self.data, geom, gdata, q, False):
            return {}
        return {k: pin.computeDistance(geom, gdata, k).min_distance
                for k, res in enumerate(gdata.collisionResults) if res.isCollision()}

    def _refined(self, qa, qb, fa, fb, state):
        """The samples of a move: every step_rad, and every refine_step_rad on both sides of a sample (or the
        start) where a pair is within clearance_m + refine_m (a dip below clearance_m between two samples is
        at most ~0.2 mm at step_rad 0.02, B5.3 sweep 2; refine_m covers it with margin)."""
        coarse = self.samples(qa, qb, fa, fb)
        if not self.refine:
            return coarse
        geom, _, near = self._entry(state)
        fine = self.samples(qa, qb, fa, fb, self.refine_step)
        out = set(coarse)
        for k, s in enumerate([0.0] + coarse):
            q = self.full(qa + s * (qb - qa), fa + s * (fb - fa))
            if pin.computeCollisions(self.arm.model, self.data, geom, near, q, True):
                a, b = (coarse[k - 2] if k >= 2 else 0.0), (coarse[k] if k < len(coarse) else 1.0)
                out.update(f for f in fine if a < f <= b)
        return sorted(out)

    def check_move(self, qa, qb, state, fb=None):
        """None if the move qa -> qb (fingers state.fingers -> fb) keeps clearance_m everywhere, else
        {'s', 'pair', 'distance'} at the first failing sample. The start pose is not checked (the arm is
        there already); a pair already closer than clearance_m there (an item lifted off the one it lies
        on) may stay that close but not get closer."""
        geom, gdata = self._geometry(state)
        qa, qb = np.asarray(qa, dtype=float), np.asarray(qb, dtype=float)
        fa = self.arm.finger_closed if state.fingers is None else state.fingers
        fb = fa if fb is None else fb
        start = self._contacts(self.full(qa, fa), geom, gdata)
        for s in self._refined(qa, qb, fa, fb, state):
            q = self.full(qa + s * (qb - qa), fa + s * (fb - fa))
            if not start:
                hit = self._hit(q, geom, gdata)
                if hit is not None:
                    return {'s': s, 'pair': hit[1], 'distance': hit[0]}
                continue
            for k, d in self._contacts(q, geom, gdata).items():
                if k not in start or d < start[k] - SEPARATING_TOL:
                    return {'s': s, 'pair': self._pair_names(geom, k), 'distance': d}
        return None

    def min_move_distance(self, qa, qb, state, fb=None, step=None):
        """(minimum distance, pair, s) along a move (a measuring tool: every pair, every sample)."""
        geom, gdata = self._geometry(state)
        qa, qb = np.asarray(qa, dtype=float), np.asarray(qb, dtype=float)
        fa = self.arm.finger_closed if state.fingers is None else state.fingers
        fb = fa if fb is None else fb
        best = (math.inf, None, None)
        for s in self.samples(qa, qb, fa, fb, step):
            pin.computeDistances(self.arm.model, self.data, geom, gdata, self.full(qa + s * (qb - qa), fa + s * (fb - fa)))
            d = [r.min_distance for r in gdata.distanceResults]
            k = int(np.argmin(d))
            if d[k] < best[0]:
                best = (d[k], self._pair_names(geom, k), s)
        return best

    def scan_move(self, qa, qb, state, fb=None, step=None):
        """Every sample along a move (finer step allowed) where a pair is closer than clearance_m:
        [(s, distance, pair, separating)], separating: the pair was at least that close at the start
        (a measuring tool: no early stop, every pair)."""
        geom, gdata = self._geometry(state)
        qa, qb = np.asarray(qa, dtype=float), np.asarray(qb, dtype=float)
        fa = self.arm.finger_closed if state.fingers is None else state.fingers
        fb = fa if fb is None else fb
        start = self._contacts(self.full(qa, fa), geom, gdata)
        return [(s, d, self._pair_names(geom, k), k in start and d >= start[k] - SEPARATING_TOL)
                for s in self.samples(qa, qb, fa, fb, step)
                for k, d in self._contacts(self.full(qa + s * (qb - qa), fa + s * (fb - fa)), geom, gdata).items()]

    def attach(self, arm_q, state, object_id):
        """State after attaching object_id at arm pose arm_q: the item moves from the obstacles to the jaws
        (its pose relative to the jaw tips, as attach mode fixes it). Unknown items are not modelled."""
        st = state.copy()
        it = st.items.pop(object_id, None)
        if it is None:
            st.held = None
            return st
        st.items = dict(st.items)
        t, R = self.arm.pose(arm_q, TIPS)
        st.held = {'id': object_id, 'p': R.T @ (np.asarray(it['centre'], dtype=float) - t),
                   'R': R.T @ np.asarray(it['R'], dtype=float), 'size': tuple(it['size'])}
        st.excluded = st.excluded - {object_id}
        return st

    def detach(self, arm_q, state):
        """State after letting go: the held item stays where it was, as an obstacle."""
        st = state.copy()
        if st.held is None:
            return st
        h = st.held
        t, R = self.arm.pose(arm_q, TIPS)
        st.items = dict(st.items)
        st.items[h['id']] = {'centre': t + R @ np.asarray(h['p']), 'R': R @ np.asarray(h['R']), 'size': h['size']}
        st.held = None
        return st

    def check_steps(self, steps, start_q, state, measure=None):
        """Walk a step list (grasping.sequence) from arm pose start_q. Returns (failure, end pose, end
        state): failure None, or {'phase', 'index', 's', 'pair', 'distance'} of the first failing move.
        measure(phase, qa, qb, state, fb): called for every move instead of stopping (sweeps)."""
        q = np.asarray(start_q, dtype=float)
        for i, step in enumerate(steps):
            if step.kind in ('arm', 'fingers'):
                qb = np.asarray(step.value, dtype=float) if step.kind == 'arm' else q
                fb = float(step.value) if step.kind == 'fingers' else None
                if measure is not None:
                    measure(step.phase, q, qb, state, fb)
                else:
                    hit = self.check_move(q, qb, state, fb)
                    if hit is not None:
                        return dict(hit, phase=step.phase, index=i), q, state
                if fb is not None:
                    state = state.copy()
                    state.fingers = fb
                q = qb
            elif step.kind == 'attach':
                state = self.attach(q, state, step.value)
            elif step.kind == 'detach':
                state = self.detach(q, state)
        return None, q, state


def reason(failure):
    """Refusal text of a failure from check_steps: collision:<phase>:<a>/<b>."""
    return f"collision:{failure['phase']}:{failure['pair'][0]}/{failure['pair'][1]}"


def describe(failure):
    return (f"{reason(failure)} {failure['distance'] * 1000:.1f} mm at s {failure['s']:.2f} "
            f"(step {failure['index']})")
