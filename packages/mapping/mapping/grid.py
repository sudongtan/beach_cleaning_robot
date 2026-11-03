"""Static traversability grid of a scene, as plain numpy (no ROS).

Terrain part (water, slopes over the limit) from sim/common/terrain.py, the same
function the scene generator and the A1 exports use. On top: lethal footprints of
the static obstacles listed in config/mapping.yaml, and everything outside the
geofence. Row 0 = south (y = 0), column 0 = west (x = 0), as nav_msgs/OccupancyGrid.
"""

import math

import numpy as np

FREE, LETHAL = 0, 100


def cell_centres(occ_shape, resolution):
    ny, nx = occ_shape
    xs = (np.arange(nx) + 0.5) * resolution
    ys = (np.arange(ny) + 0.5) * resolution
    return np.meshgrid(xs, ys)


def primitive_mask(X, Y, obj_pose, prim, margin):
    """Cells whose centre lies in a collision primitive's footprint (grown by margin)."""
    ox, oy, yaw = obj_pose
    px, py = prim['pose'][0], prim['pose'][1]
    c, s = math.cos(yaw), math.sin(yaw)
    cx, cy = ox + c * px - s * py, oy + s * px + c * py
    if prim['type'] == 'cylinder':
        return np.hypot(X - cx, Y - cy) <= prim['radius'] + margin
    a = yaw + prim['pose'][5]
    ca, sa = math.cos(a), math.sin(a)
    u = ca * (X - cx) + sa * (Y - cy)            # into the box frame
    v = -sa * (X - cx) + ca * (Y - cy)
    return (np.abs(u) <= prim['size'][0] / 2 + margin) & (np.abs(v) <= prim['size'][1] / 2 + margin)


def inside_polygon(X, Y, poly):
    """Even-odd rule, vectorised over the cell centres."""
    inside = np.zeros(X.shape, dtype=bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        crosses = (y1 > Y) != (y2 > Y)
        with np.errstate(divide='ignore', invalid='ignore'):
            xint = x1 + (Y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (X < xint)
    return inside


def static_objects(scene, objects):
    """(model spec, pose) for every bin, landmark and distractor in the scene."""
    for b in scene['bins']:
        yield b['model'], objects['bins'][b['model']], b['pose']
    for lm in scene['landmarks']:
        yield lm['model'], objects['landmarks'][lm['model']], lm['pose']
    for d in scene['distractors']:
        yield d['model'], objects['distractors'][d['model']], d['pose']


def build(scene, objects, terrain_occupancy, resolution, obstacle_models, margin, lethal_outside_geofence):
    """Occupancy (int8, row 0 south) for the scene."""
    occ = np.array(terrain_occupancy, dtype=np.int8)
    X, Y = cell_centres(occ.shape, resolution)
    # grown by half a cell diagonal: every cell the footprint touches, so a thin rotated
    # object (driftwood, 8 cm) can't fall between cell centres
    grow = margin + resolution / math.sqrt(2)
    for model, spec, pose in static_objects(scene, objects):
        if model not in obstacle_models:
            continue
        for prim in spec['collision']:
            occ[primitive_mask(X, Y, pose, prim, grow)] = LETHAL
    if lethal_outside_geofence:
        occ[~inside_polygon(X, Y, scene['geofence']['polygon'])] = LETHAL
    return occ


def lethal_at(occ, resolution, x, y):
    return occ[int(y // resolution), int(x // resolution)] == LETHAL
