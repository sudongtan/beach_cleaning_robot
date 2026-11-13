"""Pure helpers for the GPU render check (sim/isaac/render_check.py), tested on the Mac.

No Omniverse imports: numpy, Pillow and pxr (usd-core on the Mac, Isaac's own pxr on the GPU).
Thresholds come from sim/isaac/render.yaml `render_check`.
"""

import math
import re
from pathlib import Path

import numpy as np

CAMERA_ROOT = '/World/Cameras'
TRASH_ROOT = '/World/Trash'


# ---------------------------------------------------------------- stage

def scene_cameras(stage):
    """[(name, prim path, (width, height))] for the cameras the builder wrote."""
    from pxr import UsdGeom
    out = []
    root = stage.GetPrimAtPath(CAMERA_ROOT)
    if not root:
        return out
    for p in root.GetChildren():
        if p.IsA(UsdGeom.Camera):
            res = p.GetCustomDataByKey('beach:resolution')
            if res is None:
                raise ValueError(f'{p.GetPath()} has no beach:resolution')
            out.append((p.GetName(), str(p.GetPath()), (int(res[0]), int(res[1]))))
    return out


def trash_positions(stage):
    """{trash id: (x, y, z)} world positions of the trash prims, read through USD."""
    from pxr import Usd, UsdGeom
    out = {}
    root = stage.GetPrimAtPath(TRASH_ROOT)
    if not root:
        return out
    for p in root.GetChildren():
        t = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
        out[p.GetName()] = (float(t[0]), float(t[1]), float(t[2]))
    return out


# ---------------------------------------------------------------- images

def load_image(path):
    """RGB(A) PNG -> float array (h, w, 3), values 0-255."""
    from PIL import Image
    a = np.asarray(Image.open(path).convert('RGB'), dtype=np.float64)
    return a


def image_check(img, min_mean, min_std):
    """Not black (mean > min_mean) and not uniform (std > min_std), on 0-255 RGB values."""
    a = np.asarray(img, dtype=np.float64)
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[..., :3]
    mean, std = float(a.mean()), float(a.std())
    return {'shape': list(a.shape), 'mean': round(mean, 3), 'std': round(std, 3),
            'not_black': mean > min_mean, 'not_uniform': std > min_std,
            'ok': mean > min_mean and std > min_std}


def find_images(out_dir, camera):
    """RGB PNGs written for one camera, oldest first: any rgb*.png under <out_dir>/<camera>/."""
    d = Path(out_dir) / camera
    return sorted(d.rglob('rgb*.png')) if d.is_dir() else []


# ---------------------------------------------------------------- physics

def trash_motion(before, after, max_move):
    """Per-item displacement over the physics check. `rests`: every item moved < max_move.
    `updated`: at least one item moved at all (they start SPAWN_GAP above the sand and
    settle), so all-zero motion means the positions were never written back to USD."""
    moves = {}
    missing = sorted(set(before) ^ set(after))
    for k in sorted(set(before) & set(after)):
        moves[k] = round(math.dist(before[k], after[k]), 5)
    finite = all(math.isfinite(v) for v in moves.values())
    worst = max(moves.values()) if moves else None
    return {'moves_m': moves, 'max_move_m': worst, 'missing': missing,
            'rests': bool(moves) and finite and not missing and worst < max_move,
            'updated': bool(moves) and worst is not None and worst > 1e-5}


# ---------------------------------------------------------------- log

def scan_log(text, patterns, max_lines=20):
    """Case-insensitive pattern matches in a Kit log: {pattern: {'count': n, 'lines': [...]}}."""
    out = {}
    lines = text.splitlines()
    for pat in patterns:
        rx = re.compile(re.escape(pat), re.IGNORECASE)
        hits = [ln.strip()[:300] for ln in lines if rx.search(ln)]
        out[pat] = {'count': len(hits), 'lines': hits[:max_lines]}
    return out


def newest_log(dirs, since=None, pattern='kit_*.log'):
    """Newest Kit app log (kit_<date>.log) under the given folders, modified after `since`
    (epoch s), or None. Only kit_*.log: other logs there (omni.replicator.extinfo.log, ...)
    are newer but aren't the app's log (GPU session 1)."""
    best = None
    for d in dirs:
        d = Path(d).expanduser()
        if not d.is_dir():
            continue
        for f in d.rglob(pattern):
            m = f.stat().st_mtime
            if since is not None and m < since:
                continue
            if best is None or m > best[0]:
                best = (m, f)
    return best[1] if best else None


# ---------------------------------------------------------------- verdict

def verdict_from_file(path):
    """0 if checks.json says ok, else 1. gpu_session.sh uses this, not the process exit code:
    SimulationApp.close() ends the process with code 0 itself (GPU session 1)."""
    import json
    try:
        return 0 if json.loads(Path(path).read_text()).get('ok') is True else 1
    except (OSError, ValueError):
        return 1


def evaluate(images, motion, log_matches, fail_patterns):
    """Overall verdict. images: {camera: image_check dict or None (no image)}."""
    failures = []
    for cam, chk in images.items():
        if chk is None:
            failures.append(f'{cam}: no image')
        elif not chk['ok']:
            what = [w for w, k in (('black', 'not_black'), ('uniform', 'not_uniform')) if not chk[k]]
            failures.append(f"{cam}: {' and '.join(what)} (mean {chk['mean']}, std {chk['std']})")
    if not images:
        failures.append('no cameras')
    if motion is None:
        failures.append('physics check did not run')
    else:
        if not motion['updated']:
            failures.append('trash positions never changed (not written back to USD?)')
        if not motion['rests']:
            failures.append(f"trash moved up to {motion['max_move_m']} m (missing: {motion['missing']})")
    for pat in fail_patterns:
        n = (log_matches.get(pat) or {}).get('count', 0)
        if n:
            failures.append(f"log: {n} line(s) with '{pat}'")
    return {'ok': not failures, 'failures': failures}


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == '--verdict':
        sys.exit(verdict_from_file(sys.argv[2]))
    sys.exit('usage: render_checks.py --verdict CHECKS_JSON')
