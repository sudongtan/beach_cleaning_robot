"""Beach terrain: one height function, several exports.

The height field is computed from a scene's `terrain` parameters (see
bench/SCENES_SCHEMA.md). Everything that needs terrain heights uses
`height_at`, so the Gazebo heightmap, the Isaac mesh, the traversability grid
and the generator's placement checks all agree.

Frame: map (ENU), origin at the south-west corner of the scene area. z = 0 is
the water level. Pure numpy, no simulator imports.
"""

import math
from pathlib import Path

import numpy as np
from PIL import Image

# Smooth step over the wet/dry boundary, so roughness fades in without a step.
DRY_RAMP_M = 1.0
# Seabed slope below the waterline.
SEABED_SLOPE = 0.05


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def height_at(terrain, x, y):
    """Terrain height (m) at map coordinates x, y (scalars or arrays)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    wl = terrain['waterline_y']
    wet_top = wl + terrain['wet_zone_width']
    h_wet = terrain['wet_zone_height']

    # Cross-shore profile: seabed (from the water level down), flat wet zone,
    # rising dry beach. The small step at the waterline is under the wet-zone patch.
    seabed = -(wl - y) * SEABED_SLOPE
    dry = h_wet + (y - wet_top) * terrain['beach_slope']
    h = np.where(y < wl, np.maximum(seabed, terrain['z_min']),
                 np.where(y < wet_top, h_wet, dry))

    # Roughness and dunes only on dry sand, faded in over DRY_RAMP_M.
    w = _smoothstep((y - wet_top) / DRY_RAMP_M)
    rough = np.zeros_like(h)
    for c in terrain['roughness']:
        k = 2.0 * math.pi / c['wavelength']
        a = math.radians(c['direction_deg'])
        rough += c['amplitude'] * np.sin(k * (x * math.cos(a) + y * math.sin(a)) + c['phase'])
    dunes = np.zeros_like(h)
    for d in terrain['dunes']:
        dunes += d['height'] * np.exp(-0.5 * (((x - d['x']) / d['sigma_x']) ** 2
                                              + ((y - d['y']) / d['sigma_y']) ** 2))
    h = h + w * (rough + dunes)
    return np.clip(h, terrain['z_min'], terrain['z_max'])


def slope_deg_at(terrain, x, y, eps=0.05):
    """Terrain slope (deg) by central differences of `height_at`."""
    dzdx = (height_at(terrain, np.add(x, eps), y) - height_at(terrain, np.subtract(x, eps), y)) / (2 * eps)
    dzdy = (height_at(terrain, x, np.add(y, eps)) - height_at(terrain, x, np.subtract(y, eps))) / (2 * eps)
    return np.degrees(np.arctan(np.hypot(dzdx, dzdy)))


def grid_axes(terrain):
    """x and y coordinates of the heightmap vertices (south to north)."""
    hm = terrain['heightmap']
    n = hm['pixels']
    cx, cy = hm['center']
    half = hm['size'] / 2.0
    xs = np.linspace(cx - half, cx + half, n)
    ys = np.linspace(cy - half, cy + half, n)
    return xs, ys


def height_grid(terrain):
    """Height array, shape (n, n), row 0 = south (y min), column 0 = west."""
    xs, ys = grid_axes(terrain)
    X, Y = np.meshgrid(xs, ys)
    return height_at(terrain, X, Y)


def heightmap_range(terrain):
    """(lowest, highest) height of the heightmap grid, in metres."""
    h = height_grid(terrain)
    return float(h.min()), float(h.max())


def write_heightmap_png(terrain, path):
    """16-bit grayscale PNG for Gazebo, stretched to the full pixel range:
    pixel 0 = lowest grid height, 65535 = highest (see `heightmap_range`).

    Gazebo's heightmap collision (gz-physics/DART, checked in Harmonic with
    probe drops) scales by the image's own maximum pixel value, not by 65535.
    Using the full range makes both conventions agree. The world builder places
    the heightmap at z = lowest with size z = highest - lowest.
    Image row 0 is north (images are stored top-down).
    """
    h = height_grid(terrain)
    z0, z1 = float(h.min()), float(h.max())
    px = np.round((h - z0) / (z1 - z0) * 65535.0).astype(np.uint16)
    img = Image.fromarray(np.flipud(px), mode='I;16')
    img.save(path)


def write_obj_mesh(terrain, path):
    """Terrain as a triangle mesh (OBJ, map frame, metres) for Isaac."""
    xs, ys = grid_axes(terrain)
    h = height_grid(terrain)
    n = len(xs)
    lines = ['# beach terrain, map frame (ENU), metres']
    for j in range(n):
        for i in range(n):
            lines.append(f'v {xs[i]:.4f} {ys[j]:.4f} {h[j, i]:.4f}')
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i + 1
            b = a + 1
            c = a + n
            d = c + 1
            lines.append(f'f {a} {b} {d}')
            lines.append(f'f {a} {d} {c}')
    Path(path).write_text('\n'.join(lines) + '\n')


def traversability(scene, max_slope_deg, resolution):
    """Slope, water mask and occupancy over the scene area.

    Returns a dict of arrays with row 0 = south. Cells are lethal (100) when
    they are water or steeper than max_slope_deg, free (0) otherwise.
    """
    t = scene['terrain']
    w, hgt = scene['area']['size']
    nx, ny = int(round(w / resolution)), int(round(hgt / resolution))
    xs = (np.arange(nx) + 0.5) * resolution
    ys = (np.arange(ny) + 0.5) * resolution
    X, Y = np.meshgrid(xs, ys)
    slope = slope_deg_at(t, X, Y)
    water = Y < t['waterline_y']
    occ = np.where(water | (slope > max_slope_deg), 100, 0).astype(np.int8)
    return {'slope_deg': slope, 'water': water, 'height': height_at(t, X, Y),
            'occupancy': occ, 'resolution': resolution}


def write_traversability(scene, max_slope_deg, resolution, out_dir):
    """Writes traversability.npz plus a map_server style PGM + YAML."""
    out_dir = Path(out_dir)
    g = traversability(scene, max_slope_deg, resolution)
    np.savez_compressed(out_dir / 'traversability.npz', slope_deg=g['slope_deg'],
                        water=g['water'], height=g['height'], occupancy=g['occupancy'],
                        resolution=resolution)
    # map_server convention: image row 0 = top (north); black = occupied.
    pgm = np.where(g['occupancy'] > 0, 0, 254).astype(np.uint8)
    Image.fromarray(np.flipud(pgm), mode='L').save(out_dir / 'traversability.pgm')
    (out_dir / 'traversability.yaml').write_text(
        'image: traversability.pgm\n'
        'mode: trinary\n'
        f'resolution: {resolution}\n'
        'origin: [0.0, 0.0, 0.0]\n'
        'negate: 0\n'
        'occupied_thresh: 0.65\n'
        'free_thresh: 0.25\n')
