"""Lawnmower search path over the geofence polygon, as plain Python (no ROS).

Lanes run parallel to x (the shore), spaced by the camera footprint width with overlap,
and are walked back and forth starting with the lane and end nearest the robot.
Waypoints sit every waypoint_spacing_m along each lane, inset from the geofence by
edge_margin_m; waypoints on lethal map cells are dropped (the planner routes around
obstacles between the remaining ones).
"""

import math


def lane_spacing(horizontal_fov, view_distance, overlap):
    return 2.0 * view_distance * math.tan(horizontal_fov / 2.0) * (1.0 - overlap)


def line_segments(poly, y):
    """x intervals where the horizontal line at y is inside the polygon (even-odd rule)."""
    xs = []
    n = len(poly)
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xs.append(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
    xs.sort()
    return list(zip(xs[0::2], xs[1::2]))


def lanes(poly, spacing, margin):
    """Lane y values from the bottom to the top of the polygon, inset by margin."""
    y0 = min(p[1] for p in poly) + margin
    y1 = max(p[1] for p in poly) - margin
    if y1 < y0:
        return []
    n = max(1, math.ceil((y1 - y0) / spacing))
    step = (y1 - y0) / n                           # spread evenly, never wider than spacing
    return [y0 + k * step for k in range(n + 1)]


def plan(poly, start, spacing, margin, waypoint_spacing, lethal=lambda x, y: False):
    """Waypoints [(x, y)] in the map frame."""
    ys = lanes(poly, spacing, margin)
    if not ys:
        return []
    if abs(ys[-1] - start[1]) < abs(ys[0] - start[1]):
        ys = ys[::-1]
    path = []
    forward = None
    for y in ys:
        lane = []
        for a, b in line_segments(poly, y):
            a, b = a + margin, b - margin
            if b < a:
                continue
            n = max(1, math.ceil((b - a) / waypoint_spacing))
            lane += [(a + (b - a) * k / n, y) for k in range(n + 1)]
        if not lane:
            continue
        if forward is None:                         # first lane: begin at the end nearer the start
            forward = abs(lane[0][0] - start[0]) <= abs(lane[-1][0] - start[0])
        path += [p for p in (lane if forward else lane[::-1]) if not lethal(*p)]
        forward = not forward
    return path
