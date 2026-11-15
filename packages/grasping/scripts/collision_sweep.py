#!/usr/bin/env python3
"""B5.3 offline component check of the collision check (grasping.collision), no ROS, no sim.

  (a) check off, the module as a measuring tool on today's plans:
      a1 store `lower` with wrist_branch any, 8 items x 12 yaws x 2 compartments (lower_sweep3, 192 plans);
      a2 second store into recyclable, every trash pair x 4 x 4 yaws (pair_store, 1024 plans): any store move
         below clearance_m with the first item lying where it landed.
  (b) check on: every executed move (the plan the planner returned, walked step by step with attach / detach)
      measured on its own, every pair at step_rad / 9 (a grid apart from the check's refine_step_rad):
      count moves below clearance_m over second stores (every pair x 4 x 4 yaws x both compartments), third
      stores (200 random triples), unloads from those layouts, bin releases over a grid of robot poses at the
      bin, picks over the envelope.
  (c) second-store success rate (and third).
  (d) plan times with the check (p95).

Usage (sourced ~/wt/install, PYTHONPATH with packages/grasping):
  python3 packages/grasping/scripts/collision_sweep.py --out ~/wt/logs/b5_sweep.json [--jobs 7]
      [--set store.slots_x=4 --set store.slots_y=3]
"""

import argparse
import copy
import itertools
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'packages' / 'grasping' / 'test'))

from test_arm import CFG, LIMITS, OBJECTS, box_size, landed, picked, rz  # noqa: E402

from grasping.arm import (Arm, bin_release_target, compartment, held_item, near, plan_pick,  # noqa: E402
                          plan_store, plan_unload, release_clearances)
from grasping.collision import Collision, State, bin_boxes, reason  # noqa: E402
from grasping.sequence import bin_steps, pick_steps, store_steps  # noqa: E402

ITEMS = list(OBJECTS['trash'])
YAWS4 = np.linspace(-np.pi, np.pi, 5)[:-1]
COMPS = ('recyclable', 'residual')
W = {}                                               # per worker: arm, col, cfg


def init(cfg):
    W['cfg'] = cfg
    W['arm'] = Arm(LIMITS['variant'], cfg['ik_restarts'], cfg['ik_tolerance_m'])
    W['col'] = Collision.from_config(W['arm'], cfg['collision'])
    W['show'] = np.asarray(cfg['show_pose'], dtype=float)
    W['rest'] = np.asarray(W['arm'].poses['rest']['arm'], dtype=float)


# --- helpers ---------------------------------------------------------------------------------------

def obstacle(o, Rb, e, size):
    """A landed item (test_arm.landed) as an oriented obstacle box: its own size, rotation Rb."""
    return {'centre': np.asarray(o['centre'], dtype=float), 'R': Rb, 'size': tuple(size)}


def checker(steps_of, start, state):
    col = W['col']

    def check(plan):
        failure, _, _ = col.check_steps(steps_of(plan), start, state)
        return None if failure is None else reason(failure)
    return check


def measure(steps, start, state):
    """Executed moves of a step list, measured on their own (every pair, step_rad / 9, no early stop):
    [(phase, distance, pair)], per move its closest sample below clearance_m, or (phase, inf, ()) if clear."""
    col, out = W['col'], []

    def m(phase, qa, qb, st, fb):
        hits = col.scan_move(qa, qb, st, fb, col.step / 9)
        out.append(min(((phase, d, pair) for _, d, pair, sep in hits if not sep), key=lambda x: x[1],
                       default=(phase, math.inf, ())))
        if any(sep for *_, sep in hits):
            out[-1] = out[-1] + ('separating',)
    col.check_steps(steps, start, state, measure=m)
    return out


def worst(moves):
    return min(moves, key=lambda x: x[1]) if moves else None


def timed(f, *a):
    t = time.monotonic()
    r = f(*a)
    return r, time.monotonic() - t


def store(comp, held, hid, layout, check_on=True):
    """Store `held` into comp with layout [(id, landed, Rb, e, size)] in the basket. Returns a record."""
    arm, cfg = W['arm'], W['cfg']
    items = {i: obstacle(o, Rb, e, s) for i, o, Rb, e, s in layout}
    state = State(items, held=dict(held, id=hid))
    steps_of = lambda plan: store_steps(arm.poses, comp, arm.finger_open, hid, plan)  # noqa: E731
    check = checker(steps_of, W['show'], state) if check_on else None
    (ok, why, plan, info), dt = timed(plan_store, arm, comp, held, [o for _, o, *_ in layout], cfg, check)
    rec = {'ok': ok, 'why': why, 'dt': dt, 'rejected': len(info.get('rejected', []))}
    if ok:
        rec['moves'] = measure(steps_of(plan), W['show'], state)
        o, Rb, e = landed(arm, plan['release'], held, info['drop'])
        rec['landed'] = (hid, o, Rb, e, held['size'])
    return rec


def unload(comp, target, layout, bins=(), release=None):
    """Unload `target` (one of layout) from comp: the prefix to `up` (or the whole bin sequence with
    `release`(plan, check_state) -> (q, dt) | None). Returns a record."""
    arm, cfg = W['arm'], W['cfg']
    tid, o, Rb, e, size = target
    items = {i: obstacle(oo, R, ee, s) for i, oo, R, ee, s in layout}
    state = State(items, excluded={tid}, bins=bins)
    centre, height = np.asarray(o['centre'], dtype=float), float(e[2])
    after = {}

    def check(plan):
        steps = bin_steps(arm.poses, comp, plan, plan['above'], arm.finger_open, arm.finger_closed, tid)
        k = next(i for i, st in enumerate(steps) if st.phase == 'over_bin')
        failure, q, st = W['col'].check_steps(steps[:k], W['rest'], state)
        after.update(q=q, state=st)
        return None if failure is None else reason(failure)
    (ok, why, plan), dt = timed(plan_unload, arm, comp, centre, height, cfg, check)
    rec = {'ok': ok, 'why': why, 'dt': dt}
    if not ok:
        return rec
    if release is None:
        steps = bin_steps(arm.poses, comp, plan, plan['above'], arm.finger_open, arm.finger_closed, tid)
        steps = steps[:next(i for i, st in enumerate(steps) if st.phase == 'over_bin')]
    else:
        def release_check(q):
            steps = bin_steps(arm.poses, comp, plan, q, arm.finger_open, arm.finger_closed, tid)
            k = next(i for i, st in enumerate(steps) if st.phase == 'over_bin')
            failure, _, _ = W['col'].check_steps(steps[k:], after['q'], after['state'])
            return None if failure is None else reason(failure)
        (q, rdt) = release(height, release_check)
        rec['release_dt'] = rdt
        if q is None:
            rec.update(ok=False, why='bin_unreachable')
            return rec
        steps = bin_steps(arm.poses, comp, plan, q, arm.finger_open, arm.finger_closed, tid)
    rec['moves'] = measure(steps, W['rest'], state)
    return rec


def slim(rec):
    """A record without the landed pose; moves reduced to the worst one and the count below clearance."""
    out = {k: v for k, v in rec.items() if k not in ('moves', 'landed')}
    if 'moves' in rec:
        clr = W['col'].clearance
        w = worst(rec['moves'])
        out['n_moves'] = len(rec['moves'])
        out['below'] = [(p, round(d, 5), list(pair)) for p, d, pair, *_ in rec['moves'] if d < clr]
        out['separating'] = sum(len(m) > 3 for m in rec['moves'])
        out['min'] = None if w is None else (w[0], w[1] if w[1] == math.inf else round(w[1], 5), list(w[2]))
    return out


# --- tasks -----------------------------------------------------------------------------------------

def task_a1(args):
    """(a1) one item x 12 yaws x 2 compartments, wrist_branch any, check off: is `lower` flagged?"""
    name, = args
    arm, col = W['arm'], W['col']
    cfg = {**W['cfg'], 'wrist_branch': 'any'}
    out = []
    for comp in COMPS:
        for yaw in np.linspace(-np.pi, np.pi, 13)[:-1]:
            held = picked(arm, OBJECTS['trash'][name], yaw)
            ok, why, plan, _ = plan_store(arm, comp, held, [], cfg)
            if not ok:
                out.append({'item': name, 'comp': comp, 'yaw': yaw, 'ok': False, 'why': why})
                continue
            hit = col.check_move(plan['above'], plan['release'], State(held=dict(held, id='h')))
            out.append({'item': name, 'comp': comp, 'yaw': yaw, 'ok': True, 'flag': hit is not None,
                        'turn': abs(plan['release'][-1] - plan['above'][-1])})
    return out


def task_a2(args):
    """(a2) second store into recyclable, check off: any store move below clearance?"""
    a, b = args
    arm = W['arm']
    out = []
    for y1 in YAWS4:
        h1 = picked(arm, OBJECTS['trash'][a], y1)
        ok, _, p1, i1 = plan_store(arm, 'recyclable', h1, [], W['cfg'])
        if not ok:
            continue
        o, Rb, e = landed(arm, p1['release'], h1, i1['drop'])
        layout = [('first', o, Rb, e, h1['size'])]
        for y2 in YAWS4:
            h2 = picked(arm, OBJECTS['trash'][b], y2)
            ok, why, p2, _ = plan_store(arm, 'recyclable', h2, [o], W['cfg'])
            if not ok:
                out.append({'pair': (a, b), 'ok': False, 'why': why})
                continue
            state = State({'first': obstacle(o, Rb, e, h1['size'])}, held=dict(h2, id='second'))
            failure, _, _ = W['col'].check_steps(store_steps(arm.poses, 'recyclable', arm.finger_open, 'second', p2),
                                                 W['show'], state)
            out.append({'pair': (a, b), 'ok': True, 'flag': None if failure is None else reason(failure)})
    return out


def task_b_second(args):
    """(b) second stores (check on) and the unloads from their layouts, one pair x 4 x 4 yaws x comp."""
    a, b, comp = args
    arm = W['arm']
    out = {'first': [], 'second': [], 'unload': []}
    for y1 in YAWS4:
        h1 = picked(arm, OBJECTS['trash'][a], y1)
        r1 = store(comp, h1, 'first', [])
        out['first'].append(dict(slim(r1), item=a, comp=comp, yaw=y1))
        if not r1['ok']:
            continue
        for y2 in YAWS4:
            h2 = picked(arm, OBJECTS['trash'][b], y2)
            r2 = store(comp, h2, 'second', [r1['landed']])
            out['second'].append(dict(slim(r2), pair=(a, b), comp=comp, yaws=(y1, y2)))
            layout = [r1['landed']] + ([r2['landed']] if r2['ok'] else [])
            for t in layout:
                out['unload'].append(dict(slim(unload(comp, t, layout)), comp=comp, n=len(layout)))
    return out


def task_b_third(args):
    """(b) a random triple stored into one compartment (check on), then every item unloaded."""
    k, names, yaws, comp = args
    arm = W['arm']
    layout, recs, unl = [], [], []
    for i, (name, yaw) in enumerate(zip(names, yaws)):
        r = store(comp, picked(arm, OBJECTS['trash'][name], yaw), f'i{i}', layout)
        recs.append(slim(r))
        if not r['ok']:
            break
        layout.append(r['landed'])
    for t in layout:
        unl.append(dict(slim(unload(comp, t, layout)), comp=comp, n=len(layout)))
    return {'k': k, 'names': names, 'comp': comp, 'stores': recs, 'unload': unl}


def bin_pose(along, lateral, heading, body_height):
    """Opening centre at the rim (base_link) and the bin's yaw relative to base_link, with base_link `along`
    out on the bin's axis, `lateral` beside it, `heading` off facing it (as test_cross_phase.arm_hits_off_axis)."""
    walls = OBJECTS['bins']['bin_recyclable']['collision']
    rim = max(p['pose'][2] + p['size'][2] / 2 for p in walls) - body_height
    yaw = math.pi + heading
    c, s = math.cos(yaw), math.sin(yaw)
    bx, by = -(c * along + s * lateral), s * along - c * lateral
    return (bx, by, rim), -yaw


def task_b_bin(args):
    """(b) bin releases: one robot pose at the bin x every item lying alone in each compartment."""
    along, lateral, heading, body_height, yaw_item = args
    arm, cfg = W['arm'], W['cfg']
    opening, byaw = bin_pose(along, lateral, heading, body_height)
    bins = bin_boxes(OBJECTS['bins']['bin_recyclable'], opening, byaw)
    bd = LIMITS['bin_drop']
    half = (max(abs(b['pose'][0]) + b['size'][0] / 2 for b in OBJECTS['bins']['bin_recyclable']['collision']),
            max(abs(b['pose'][1]) + b['size'][1] / 2 for b in OBJECTS['bins']['bin_recyclable']['collision']))
    depth = abs(math.cos(byaw)) * half[0] + abs(math.sin(byaw)) * half[1]
    up = arm.poses['up']['arm']

    def release(height, check):                      # node.bin_release, offline
        t = time.monotonic()
        hang = height + cfg['hover_above_item_m']
        for clearance in release_clearances(bd):
            target = bin_release_target(opening, opening[0] - depth, bd, hang, clearance)
            ok, q, err, tilt = arm.ik(target, bd['max_tilt_deg'], seed_q=up,
                                      bounds=near(up, cfg['release_joint_spread_rad']))
            if ok and check(np.asarray(q)):
                continue
            if ok:
                return np.asarray(q), time.monotonic() - t
        return None, time.monotonic() - t
    out = []
    for comp in COMPS:
        for name in ITEMS:
            r1 = store(comp, picked(arm, OBJECTS['trash'][name], yaw_item), 'item', [])
            if not r1['ok']:
                continue
            rec = unload(comp, r1['landed'], [r1['landed']], bins, release)
            out.append(dict(slim(rec), comp=comp, item=name, pose=(along, lateral, heading, body_height)))
    return out


def task_b_pick(args):
    """(b) picks over the envelope: one item at one yaw, every grid point and body height."""
    name, yaw = args
    arm, cfg = W['arm'], W['cfg']
    size = box_size(OBJECTS['trash'][name])
    out = []
    for body_height in LIMITS['bin_drop']['body_heights_m']:
        for x in np.linspace(*cfg['reach_x_m'], 4):
            for y in np.linspace(-cfg['reach_y_max_m'], cfg['reach_y_max_m'], 3):
                centre = np.array([x, y, -body_height + size[2] / 2])
                state = State({'target': {'centre': centre, 'R': rz(yaw), 'size': tuple(size)}}, excluded={'target'})
                steps_of = lambda plan: pick_steps(plan, arm.poses, cfg['show_pose'], arm.finger_open,  # noqa: E731
                                                   arm.finger_closed, 'target')
                check = checker(steps_of, W['rest'], state)
                (ok, why, plan), dt = timed(plan_pick, arm, centre, size[2], cfg, check)
                rec = {'ok': ok, 'why': why, 'dt': dt}
                if ok:
                    rec['moves'] = measure(steps_of(plan), W['rest'], state)
                out.append(dict(slim(rec), item=name, yaw=yaw, at=(x, y, body_height)))
    return out


# --- main ------------------------------------------------------------------------------------------

def setting(cfg, kv):
    key, val = kv.split('=', 1)
    d = cfg
    *path, last = key.split('.')
    for p in path:
        d = d[p]
    d[last] = yaml.safe_load(val)


def flat(results):
    return [r for rs in results for r in rs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=7)
    ap.add_argument('--set', action='append', default=[], help='config override key.sub=value (grasping.yaml)')
    ap.add_argument('--parts', default='a1,a2,second,third,bin,pick')
    ap.add_argument('--triples', type=int, default=200)
    args = ap.parse_args()
    cfg = copy.deepcopy(CFG)
    for kv in args.set:
        setting(cfg, kv)
    assert cfg.get('collision'), 'config collision is off'
    parts = args.parts.split(',')
    clr = cfg['collision']['clearance_m']
    res = {'config': {'collision': cfg['collision'], 'store': cfg['store'], 'set': args.set},
           'start_utc': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}
    times = {}
    with mp.get_context('fork').Pool(args.jobs, initializer=init, initargs=(cfg,)) as pool:
        def run(name, fn, tasks):
            t = time.monotonic()
            r, step = [], max(1, len(tasks) // 10)
            for i, x in enumerate(pool.imap(fn, tasks, chunksize=1)):
                r.append(x)
                if (i + 1) % step == 0:
                    print(f'  {name}: {i + 1}/{len(tasks)} at {time.monotonic() - t:.0f} s', flush=True)
            print(f'{name}: {len(tasks)} tasks in {time.monotonic() - t:.0f} s', flush=True)
            return r
        if 'a1' in parts:
            a1 = flat(run('a1', task_a1, [(n,) for n in ITEMS]))
            res['a1'] = {'plans': sum(r['ok'] for r in a1), 'flagged': sum(r.get('flag', False) for r in a1),
                         'half_turns': sum(r.get('turn', 0) > 2.5 for r in a1),
                         'half_turns_flagged': sum(r.get('turn', 0) > 2.5 and r['flag'] for r in a1),
                         'refused': sum(not r['ok'] for r in a1)}
            print('a1', res['a1'], flush=True)
        if 'a2' in parts:
            a2 = flat(run('a2', task_a2, list(itertools.product(ITEMS, ITEMS))))
            ok = [r for r in a2 if r['ok']]
            phases = {}
            for r in ok:
                if r['flag']:
                    ph = r['flag'].split(':')[1]
                    phases[ph] = phases.get(ph, 0) + 1
            res['a2'] = {'stores': len(a2), 'planned': len(ok), 'flagged': sum(bool(r['flag']) for r in ok),
                         'by_phase': phases, 'refused': len(a2) - len(ok)}
            print('a2', res['a2'], flush=True)
        if 'second' in parts:
            sec = run('second', task_b_second, [(a, b, c) for a in ITEMS for b in ITEMS for c in COMPS])
            first = flat(r['first'] for r in sec)
            second = flat(r['second'] for r in sec)
            unl = flat(r['unload'] for r in sec)
            res['second'] = summary(second, clr)
            res['second']['first_stores'] = summary(first, clr)
            res['second']['refused_by_item'] = {n: sum(not r['ok'] for r in second if r['pair'][1] == n) for n in ITEMS}
            res['second']['first_refused_by_item'] = {n: sum(not r['ok'] for r in first if r['item'] == n) for n in ITEMS}
            res['unload_second'] = summary(unl, clr)
            times['store'] = [r['dt'] for r in first + second]
            times['unload'] = [r['dt'] for r in unl]
            print('second', res['second'], '\nunload', res['unload_second'], flush=True)
        if 'third' in parts:
            rng = np.random.default_rng(5)
            tasks = [(k, [ITEMS[i] for i in rng.integers(0, len(ITEMS), 3)], [float(y) for y in rng.choice(YAWS4, 3)],
                      COMPS[int(rng.integers(0, 2))]) for k in range(args.triples)]
            th = run('third', task_b_third, tasks)
            thirds = [r['stores'][2] for r in th if len(r['stores']) == 3]
            res['third'] = summary(thirds, clr)
            res['third']['reached'] = len(thirds)
            res['third']['all_stores'] = summary(flat(r['stores'] for r in th), clr)
            res['unload_third'] = summary(flat(r['unload'] for r in th), clr)
            times['store'] = times.get('store', []) + [r['dt'] for r in flat(r['stores'] for r in th)]
            times['unload'] = times.get('unload', []) + [r['dt'] for r in flat(r['unload'] for r in th)]
            print('third', res['third'], '\nunload', res['unload_third'], flush=True)
        if 'bin' in parts:
            grid = [(al, la, he, bh, float(yi)) for al in np.linspace(0.84, 0.94, 3) for la in (-0.10, 0.0, 0.10)
                    for he in (-0.15, 0.0, 0.15) for bh in LIMITS['bin_drop']['body_heights_m'] for yi in (0.0, 1.2)]
            bins = flat(run('bin', task_b_bin, grid))
            res['bin'] = summary(bins, clr)
            times['unload'] = times.get('unload', []) + [r['dt'] for r in bins]
            times['release'] = [r['release_dt'] for r in bins if 'release_dt' in r]
            print('bin', res['bin'], flush=True)
        if 'pick' in parts:
            picks = flat(run('pick', task_b_pick, [(n, float(y)) for n in ITEMS
                                                   for y in np.linspace(-np.pi, np.pi, 13)[:-1]]))
            res['pick'] = summary(picks, clr)
            times['pick'] = [r['dt'] for r in picks]
            print('pick', res['pick'], flush=True)
    every = [t for ts in times.values() for t in ts]
    res['time'] = {k: {'n': len(v), 'p50': float(np.percentile(v, 50)), 'p95': float(np.percentile(v, 95)),
                       'max': float(np.max(v))} for k, v in list(times.items()) + [('all', every)] if v}
    res['end_utc'] = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())
    print('time', res['time'], flush=True)
    json.dump(res, open(args.out, 'w'), indent=1, default=str)


def summary(recs, clr):
    ok = [r for r in recs if r['ok']]
    below = [r for r in ok if r.get('below')]
    why = {}
    for r in recs:
        if not r['ok']:
            why[r['why']] = why.get(r['why'], 0) + 1
    mins = [r['min'][1] for r in ok if r.get('min') and r['min'][1] < math.inf]
    worst = min((r for r in ok if r.get('min')), key=lambda r: r['min'][1], default=None)
    return {'n': len(recs), 'ok': len(ok), 'rate': len(ok) / len(recs) if recs else None, 'refused': why,
            'moves': sum(r.get('n_moves', 0) for r in ok), 'moves_below_clearance': sum(len(r['below']) for r in below),
            'plans_below_clearance': len(below), 'examples_below': [r['below'][:2] for r in below[:5]],
            'min_below_clearance': min(mins) if mins else None,
            'worst': None if worst is None or worst['min'][1] == math.inf else worst['min'],
            'rejected_slots': sum(r.get('rejected', 0) for r in recs),
            'separating_moves': sum(r.get('separating', 0) for r in ok)}


if __name__ == '__main__':
    main()
