"""World model as plain Python (no ROS): items, basket compartments, camera coverage.

Items come from detections already converted to the map frame. A detection of the
same class within dedup_radius of an item still on the ground (SEEN or SKIPPED) is
that item (pose averaged); otherwise it is a new item. Items in the basket or a bin
(COLLECTED, PLACED) are no longer on the ground and never match.
"""

import math
from dataclasses import dataclass, field

import numpy as np

SEEN, COLLECTED, PLACED, SKIPPED = 0, 1, 2, 3       # beach_interfaces/TrashItem constants
COMPARTMENTS = ('recyclable', 'residual')
TRANSITIONS = {SEEN: {COLLECTED, SKIPPED}, COLLECTED: {PLACED, SEEN}, SKIPPED: set(), PLACED: set()}


@dataclass
class Item:
    id: str
    object_class: str
    x: float
    y: float
    z: float
    confidence: float
    scene_id: str = ''
    yaw: float = 0.0
    size: tuple = (0.0, 0.0, 0.0)                    # bbox (x, y, z = height); zeros = unknown
    mass_kg: float = 0.0
    material: str = ''
    bin_id: str = ''
    status: int = SEEN
    skip_reason: str = ''
    times_seen: int = 1
    last_seen: float = 0.0


@dataclass
class WorldModel:
    dedup_radius_m: float
    max_items_per_compartment: int
    max_load_kg: float
    default_item_mass_kg: float
    masses: dict = field(default_factory=dict)       # scene id -> mass (ground truth, if known)
    items: dict = field(default_factory=dict)
    compartments: dict = field(default_factory=lambda: {c: [] for c in COMPARTMENTS})
    next_id: int = 0

    # --- detections -----------------------------------------------------------------
    def add_detection(self, t, object_class, x, y, z, confidence, scene_id='', yaw=0.0, size=(0.0, 0.0, 0.0)):
        """Merge one map-frame detection; returns the item id. Position is averaged over the
        detections; yaw and size are taken from the latest one (a yaw average is ill-defined for
        items that look the same turned by 180 degrees)."""
        best, best_d = None, self.dedup_radius_m
        for it in self.items.values():
            if it.status in (SEEN, SKIPPED) and it.object_class == object_class:
                d = math.hypot(it.x - x, it.y - y)
                if d <= best_d:
                    best, best_d = it, d
        if best is None:
            iid = f'wm_{self.next_id:03d}'
            self.next_id += 1
            self.items[iid] = Item(iid, object_class, x, y, z, confidence, scene_id, yaw, tuple(size),
                                   self.masses.get(scene_id, self.default_item_mass_kg), last_seen=t)
            return iid
        n = best.times_seen
        best.x, best.y, best.z = ((best.x * n + x) / (n + 1), (best.y * n + y) / (n + 1), (best.z * n + z) / (n + 1))
        best.yaw, best.size = yaw, tuple(size)
        best.confidence = max(best.confidence, confidence)
        best.times_seen, best.last_seen = n + 1, t
        best.scene_id = best.scene_id or scene_id
        return best.id

    # --- basket ---------------------------------------------------------------------
    def load_kg(self):
        return sum(self.items[i].mass_kg for c in COMPARTMENTS for i in self.compartments[c])

    def can_store(self, bin_id, mass_kg=None):
        mass = self.default_item_mass_kg if mass_kg is None else mass_kg
        return (bin_id in COMPARTMENTS and len(self.compartments[bin_id]) < self.max_items_per_compartment
                and self.load_kg() + mass <= self.max_load_kg + 1e-9)

    def basket_full(self):
        """No compartment can take another item of unknown mass (the task then goes to the bins)."""
        return not any(self.can_store(c) for c in COMPARTMENTS)

    # --- task updates -----------------------------------------------------------------
    def update(self, iid, status, material='', bin_id='', skip_reason=''):
        """Apply a status change from the task. Returns (ok, message)."""
        it = self.items.get(iid)
        if it is None:
            return False, f'unknown item {iid}'
        if status != it.status and status not in TRANSITIONS[it.status]:
            return False, f'{iid}: status {it.status} -> {status} not allowed'
        if status == COLLECTED and it.status != COLLECTED:
            if not self.can_store(bin_id, it.mass_kg):
                return False, f'{iid}: compartment {bin_id!r} full or unknown'
            self.compartments[bin_id].append(iid)
        if it.status == COLLECTED and status != COLLECTED:
            self.compartments[it.bin_id].remove(iid)
        it.material = material or it.material
        it.status = status
        it.bin_id = bin_id if status in (COLLECTED, PLACED) else ''
        it.skip_reason = skip_reason if status == SKIPPED else ''
        return True, 'ok'

    def remaining(self):
        """Trash items still to pick up."""
        return [it for it in self.items.values() if it.status == SEEN and it.object_class == 'trash']


class Coverage:
    """Cells of the scene area swept by the camera: 100 swept, 0 not yet, -1 outside the geofence."""

    def __init__(self, width, height, resolution, inside_mask, ground_z):
        """inside_mask, ground_z: arrays (ny, nx) over the cell centres, row 0 = south."""
        self.res = resolution
        ny, nx = int(round(height / resolution)), int(round(width / resolution))
        xs = (np.arange(nx) + 0.5) * resolution
        ys = (np.arange(ny) + 0.5) * resolution
        self.X, self.Y = np.meshgrid(xs, ys)
        self.Z = np.asarray(ground_z, dtype=float)
        self.inside = np.asarray(inside_mask, dtype=bool)
        self.grid = np.where(self.inside, 0, -1).astype(np.int8)

    def mark(self, R, t, fov):
        """Mark cells whose ground point is in view of a camera at rotation R / position t (map
        frame, camera_link axes: x forward). Only cells within the depth range are tested."""
        near = np.hypot(self.X - t[0], self.Y - t[1]) <= fov.far
        pts = np.stack([self.X[near], self.Y[near], self.Z[near]], axis=1) - np.asarray(t)
        cam = pts @ np.asarray(R)                         # R^T p for each row
        seen = fov.sees_many(cam[:, 0], cam[:, 1], cam[:, 2])
        idx = np.flatnonzero(near)[seen]
        flat = self.grid.reshape(-1)
        flat[idx[self.inside.reshape(-1)[idx]]] = 100

    def fraction(self):
        inside = self.grid >= 0
        return float((self.grid[inside] == 100).mean()) if inside.any() else 0.0
