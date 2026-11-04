"""Benchmark metrics for one run (notes/benchmark.md "Metrics"): pure functions of the task's event log
(results/runs/<run>.jsonl), the recorder snapshot (bench/recording.py), the scene (bench/scenes.yaml),
config/sorting_rules.yaml and bench/bench.yaml `metrics`. Items collected, sorting and false pickups come
from the final ground truth, never from the task's own claims."""
import json
import math
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MATERIALS = ['pet', 'metal', 'glass', 'residual']


def load_events(path):
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:            # a line cut by a killed run
                pass
    return events


def bin_base_z(terrain, x, y, radius):
    """Lowest terrain point within `radius` (9 points), where the static bin stands
    (sim/common/terrain.py ground_z(..., 'min'), as sim/gazebo/load_scene.py spawns it)."""
    sys.path.insert(0, str(REPO / 'sim' / 'common'))
    import terrain as terrain_mod
    return terrain_mod.ground_z(terrain, x, y, radius, 'min')


def scene_bins(scene, cfg):
    """{bin_id: (x, y, yaw, base_z)} from the scene."""
    return {b['bin_id']: (b['pose'][0], b['pose'][1], b['pose'][2],
                          bin_base_z(scene['terrain'], b['pose'][0], b['pose'][1], cfg['bin_ground_radius_m']))
            for b in scene['bins']}


def in_bin(p, b, half, rim):
    """Is point p (map x, y, z) inside the inner box of bin b = (x, y, yaw, base_z)? Same rule as
    grasping/node.py in_bin (inner half sizes in the bin frame, below the rim), with the rim at the
    bin's real height on the terrain."""
    x, y, yaw, base_z = b
    dx, dy = p[0] - x, p[1] - y
    u, v = math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy
    return abs(u) <= half[0] and abs(v) <= half[1] and p[2] < base_z + rim


def in_basket(p, robot, half, below):
    """Is p within the box around base_link (robot = (x, y, z, yaw)) that holds the basket?"""
    x, y, z, yaw = robot
    dx, dy = p[0] - x, p[1] - y
    u, v = math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy
    return abs(u) <= half[0] and abs(v) <= half[1] and p[2] >= z - below


def locate(objects, robot, bins, cfg):
    """{object id: 'bin:<bin_id>' | 'basket' | 'ground' | 'missing'} from the final ground truth."""
    out = {}
    for oid, p in objects.items():
        where = 'ground'
        for bid, b in bins.items():
            if in_bin(p, b, cfg['bin_inner_half'], cfg['bin_rim_m']):
                where = f'bin:{bid}'
                break
        else:
            if robot is not None and in_basket(p, robot, cfg['basket_half'], cfg['basket_below_base_m']):
                where = 'basket'
        out[oid] = where
    return out


def sorting(scene, where, rules, weight):
    """Items collected, sorting accuracy, contamination, false pickups (distractors)."""
    trash = {t['id']: t['material'] for t in scene['trash']}
    binned = {i: w[4:] for i, w in where.items() if i in trash and w.startswith('bin:')}
    correct = sum(1 for i, b in binned.items() if rules[trash[i]] == b)
    in_recyc = [i for i, b in binned.items() if b == 'recyclable']
    contamination = sum(1 for i in in_recyc if rules[trash[i]] == 'residual')
    reverse = sum(1 for i, b in binned.items() if b == 'residual' and rules[trash[i]] == 'recyclable')
    missing = [i for i in trash if i not in where]
    dis = {d['id'] for d in scene['distractors']}
    return {
        'items_total': len(trash), 'items_collected': len(binned),
        'items_in_basket': sum(1 for i in trash if where.get(i) == 'basket'),
        'items_missing_from_ground_truth': missing,
        'sorting_correct': correct,
        'sorting_accuracy': correct / len(binned) if binned else None,
        'contamination_count': contamination,
        'contamination_rate': contamination / len(in_recyc) if in_recyc else None,
        'recyclable_in_residual': reverse,
        'weighted_sorting_errors': weight * contamination + reverse,
        'distractors_binned_or_basket': sorted(i for i in dis if where.get(i, 'ground') != 'ground'),
        'placement': {i: where.get(i, 'missing') for i in sorted(trash)},
    }


def event_metrics(events, wm_ids, scene):
    """From the task's event log: grasps, false pickups by the picks' scene ids, the confusion matrix, skips."""
    truth = {t['id']: t['material'] for t in scene['trash']}
    dis = {d['id'] for d in scene['distractors']}
    picks = [e for e in events if e.get('event') == 'pick']
    ok = [e for e in picks if e.get('success')]
    picked_distractors = sorted({wm_ids[e['item']] for e in ok if wm_ids.get(e['item']) in dis})
    picked_unknown = sorted({e['item'] for e in ok if e['item'] not in wm_ids})
    confusion = {m: Counter() for m in MATERIALS}
    unmatched = 0
    for e in events:
        if e.get('event') != 'classify':
            continue
        sid = wm_ids.get(e['item'])
        if sid not in truth:
            unmatched += 1
            continue
        confusion[truth[sid]][e.get('material') or 'unknown'] += 1
    n_cls = sum(sum(c.values()) for c in confusion.values())
    diag = sum(confusion[m][m] for m in MATERIALS)
    skips = Counter(e.get('reason', '') for e in events if e.get('event') == 'skipped')
    return {
        'picks': len(picks), 'picks_ok': len(ok),
        'grasp_success_rate': len(ok) / len(picks) if picks else None,
        'picked_distractors': picked_distractors, 'picked_unknown_ids': picked_unknown,
        'confusion': {m: dict(confusion[m]) for m in MATERIALS},
        'classified': n_cls, 'classified_unmatched': unmatched,
        'classification_accuracy': diag / n_cls if n_cls else None,
        'skipped': dict(skips),
        'task_done': any(e.get('event') == 'done' for e in events),
        'task_fell': any(e.get('event') == 'fell' for e in events),
        'stores_ok': sum(1 for e in events if e.get('event') == 'store' and e.get('success')),
        'places_ok': sum(1 for e in events if e.get('event') == 'place' and e.get('success')),
    }


def safety(rec, lead_s):
    """Violations (notes/benchmark.md "Safety") before the fall cut-off (first tilt > fall limit minus
    `lead_s`); events after it are the fall's effects and are listed apart."""
    cut = rec['fall_t'] - lead_s if rec.get('fall_t') is not None else math.inf

    def split(eps):
        return [e for e in eps if e['t_start'] < cut], [e for e in eps if e['t_start'] >= cut]

    person, person_after = split(rec['person_violations'])
    arm, arm_after = split(rec['arm_violations'])
    out, out_after = split(rec['outside_geofence'])
    outside_s = sum(min(e['t_end'], cut) - e['t_start'] for e in out)
    return {
        'violations': len(person) + len(arm) + len(out),
        'person_violations': person, 'arm_violations': arm, 'geofence_episodes': out,
        'outside_geofence_s': outside_s,
        'min_person_distance': rec['min_person'],
        'arm_max_change_not_ok_rad': rec['arm_max_change_not_ok_rad'],
        'safety_state_changes': rec['safety_counts'],
        'after_fall': {'person': person_after, 'arm': arm_after, 'geofence': out_after},
        'fall_cutoff_t': None if cut == math.inf else cut,
    }


def run_metrics(events, rec, scene, rules, cfg, end_reason):
    """All metrics of one run (the 'metrics' part of results/runs/<run>.json)."""
    bins = scene_bins(scene, cfg)
    objects = rec.get('objects') or {}
    where = locate(objects, rec.get('objects_robot'), bins, cfg)
    s = sorting(scene, where, rules, cfg['contamination_weight'])
    ev = event_metrics(events, rec.get('wm_ids') or {}, scene)
    dis_pickups = sorted(set(s['distractors_binned_or_basket']) | set(ev['picked_distractors']))
    sim_s = (rec['sim_t'] - rec['sim_t0']) if rec.get('sim_t') is not None else None
    wm = rec.get('world_model') or {}
    return {
        **{k: v for k, v in s.items() if k != 'distractors_binned_or_basket'},
        'false_pickups': len(dis_pickups), 'false_pickup_ids': dis_pickups,
        'grasp_success_rate': ev['grasp_success_rate'], 'grasp_mode': 'simulated attach',
        'picks': ev['picks'], 'picks_ok': ev['picks_ok'],
        'confusion': ev['confusion'], 'classified': ev['classified'],
        'classified_unmatched': ev['classified_unmatched'],
        'items_per_min': s['items_collected'] / (sim_s / 60) if sim_s else None,
        'sim_s': sim_s,
        'falls': int(ev['task_fell'] or rec.get('fall_t') is not None or end_reason == 'fell'),
        'fall_t': rec.get('fall_t'), 'fall_pose': rec.get('fall_pose'), 'max_tilt_deg': rec.get('max_tilt_deg'),
        'stuck_events': int(end_reason == 'stall') + ev['skipped'].get('no_progress', 0),
        'timeout': end_reason in ('sim_cap', 'wall_cap'),
        'skipped': ev['skipped'],
        'distance_m': rec.get('distance_m'),
        'duplicates': wm.get('duplicates'), 'duplicate_ids': wm.get('duplicate_ids'),
        'coverage': wm.get('coverage'),
        'safety': safety(rec, cfg['fall_lead_s']),
        'components': {
            'detection': 'n/a (ground truth detector: reads /sim/ground_truth/objects)',
            'classification_accuracy': ev['classification_accuracy'],
            'localization_drift_m': rec.get('drift'),
        },
    }
