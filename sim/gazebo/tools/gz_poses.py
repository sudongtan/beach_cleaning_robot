"""Parse the text output of `gz topic -e -t /world/<w>/pose/info -n 1`."""

import re


def parse(text):
    """Returns (sim_time_s, {name: (x, y, z)})."""
    m = re.search(r'stamp \{\s*(?:sec: (\d+))?\s*(?:nsec: (\d+))?', text)
    t = (int(m.group(1) or 0) + int(m.group(2) or 0) * 1e-9) if m else 0.0
    poses = {}
    for blk in re.split(r'\npose \{', text):
        n = re.search(r'name: "([^"]+)"', blk)
        if not n:
            continue
        pos = re.search(r'position \{([^}]*)\}', blk)
        vals = dict(re.findall(r'(\w+): ([-\d.e]+)', pos.group(1))) if pos else {}
        poses.setdefault(n.group(1), tuple(float(vals.get(k, 0.0)) for k in 'xyz'))
    return t, poses
