#!/usr/bin/env python3
"""Aggregates the run JSONs of one row (results/runs/<row>_s<seed>.json) into results/results.md:
one line in the summary table plus a per-seed section. Numbers only from the run JSONs.

  report.py <row> [--print]      (--print: show the row, don't write)"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / 'results' / 'results.md'
SUMMARY_MARK = '<!-- summary rows above this line -->'

HEADER = f"""# Benchmark results

Rows are appended by `make bench` (`bench/runner.py`, `bench/report.py`), only from the run JSONs in
`results/runs/<row>_s<seed>.json`; never edited by hand. Spec: 10 scenes (seeds 0–9, `bench/scenes.yaml`),
one run each, headless Gazebo. Compare every new row with the previous one.

Columns (per-run values; "mean (min–max)" over the 10 runs unless it says sum):
- **Row / date / commit / sim:** row name, date the row was written, git commit of the runs (`*` = uncommitted changes), simulator.
- **Components:** locomotion / localization / perception / classifier / grasp mode. `gt` = ground-truth baseline.
- **Caps:** sim-time cap, wall-time cap, stall limit per run (`bench/bench.yaml`).
- **Ends:** how the runs ended: done (task finished), fell, stall (no progress), sim_cap / wall_cap (timeout), dead_stack / startup_failed (infrastructure).
- **Collected:** trash items inside a bin at the end of the run (final ground truth: bin inner box, below the rim), sum over all runs / items in all scenes.
- **Sorting acc.:** items in the correct bin (`config/sorting_rules.yaml`) / items in any bin, over all runs.
- **Contam.:** residual items in the recyclable bin, sum (weighted ×2 in "weighted errors" in the run JSONs).
- **False pickups:** distractors in a bin or the basket, or picked (`pick` event on a world-model item whose ground-truth id is a distractor), sum.
- **Grasp:** successful `pick` events / all, over all runs; "simulated attach" while `grasp_mode:=attach`.
- **Items/min:** collected items per sim minute (from the recorder's start after the stand-up to the end).
- **Falls / stuck / timeouts:** runs with a fall (tilt > 30°); stall ends plus `no_progress` skips; runs ended by a cap. Sums.
- **Distance:** metres walked (ground truth, steps > 1 cm).
- **Dup.:** extra world-model entries for the same ground-truth item (last `/world_model`), sum.
- **Coverage:** share of geofenced coverage cells swept (last `/world_model`).
- **Safety viol.:** person rule (robot moves while a person is within the stop distance), arm rule (an arm joint changes > 0.03 rad within 0.5 s while `/safety/state` ≠ ok; the per-seed table also gives the largest change since the state left ok, slow creep included), geofence rule (`base_link` outside). Sum over all runs; must be 0. Events from 2 s before a fall on are listed apart in the run JSON (`safety.after_fall`).
- **Min person d.:** closest robot–person distance over all runs with a person (m).
- **Class. acc.:** classifier accuracy on the held items (`classify` events vs the true material).
- **Drift:** localization error `/odom/global` vs ground truth, mean of the runs' means (max) in m.
- **Detection:** P / R / mAP; `n/a` while the detector reads ground truth.

| Row | Date | Commit | Sim | Components | Caps | Ends | Collected | Sorting acc. | Contam. | False pickups | Grasp | Items/min | Falls / stuck / timeouts | Distance (m) | Dup. | Coverage | Safety viol. | Min person d. (m) | Class. acc. | Drift (m) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
{SUMMARY_MARK}
"""


def fmt(x, nd=2):
    return '–' if x is None else (f'{x:.{nd}f}' if isinstance(x, float) else str(x))


def spread(vals, nd=2):
    v = [x for x in vals if x is not None]
    if not v:
        return '–'
    return f'{sum(v) / len(v):.{nd}f} ({min(v):.{nd}f}–{max(v):.{nd}f})'


def ratio(num, den, nd=2):
    return f'{num / den:.{nd}f} ({num}/{den})' if den else '–'


def load_runs(row, seeds):
    runs = []
    for s in seeds:
        p = REPO / 'results' / 'runs' / f'{row}_s{s}.json'
        if not p.exists():
            sys.exit(f'{p} missing: run make bench first')
        runs.append(json.loads(p.read_text()))
    return runs


def short(comp):
    g = lambda v: 'gt' if v == 'ground_truth' else v   # noqa: E731
    return (f"loco {comp['locomotion']} / loc {g(comp['localization'])} / perc {g(comp['perception'])} / "
            f"cls {g(comp['classifier'])} / grasp {comp['grasp_mode']}")


def build(row, runs):
    m = [r['metrics'] for r in runs]
    lim = runs[0]['limits']
    ends = {}
    for r in runs:
        ends[r['end_reason']] = ends.get(r['end_reason'], 0) + 1
    collected, total = sum(x['items_collected'] for x in m), sum(x['items_total'] for x in m)
    binned = collected
    correct = sum(x['sorting_correct'] for x in m)
    picks, picks_ok = sum(x['picks'] for x in m), sum(x['picks_ok'] for x in m)
    cls_n = sum(x['classified'] for x in m)
    cls_ok = sum(x['confusion'][k].get(k, 0) for x in m for k in x['confusion'])
    viol = sum(x['safety']['violations'] for x in m)
    persons = [x['safety']['min_person_distance']['d'] for x in m if x['safety']['min_person_distance']]
    drift = [x['components']['localization_drift_m'] or {} for x in m]
    drift_mean = [d.get('mean_m') for d in drift if d.get('mean_m') is not None]
    drift_max = [d.get('max_m') for d in drift if d.get('max_m') is not None]
    commit = runs[0]['commit'] + ('*' if any(r['dirty'] for r in runs) else '')
    if len({r['commit'] for r in runs}) > 1:
        commit = '/'.join(sorted({r['commit'] for r in runs})) + ('*' if any(r['dirty'] for r in runs) else '')
    line = '| ' + ' | '.join([
        f'`{row}`', datetime.now(timezone.utc).strftime('%Y-%m-%d'), commit, runs[0]['sim'], short(runs[0]['components']),
        f"{lim['sim_s']:.0f} s sim / {lim['wall_min']} min wall / stall {lim['stall_s']} s",
        ', '.join(f'{k} {v}' for k, v in sorted(ends.items())),
        f'{collected}/{total}', ratio(correct, binned), str(sum(x['contamination_count'] for x in m)),
        str(sum(x['false_pickups'] for x in m)), ratio(picks_ok, picks) + ' simulated attach',
        spread([x['items_per_min'] for x in m]),
        f"{sum(x['falls'] for x in m)} / {sum(x['stuck_events'] for x in m)} / {sum(int(x['timeout']) for x in m)}",
        spread([x['distance_m'] for x in m], 0), str(sum(x['duplicates'] or 0 for x in m)),
        spread([x['coverage'] for x in m]), str(viol),
        fmt(min(persons)) if persons else '–', ratio(cls_ok, cls_n),
        f'{sum(drift_mean) / len(drift_mean):.3f} ({max(drift_max):.3f})' if drift_mean else '–',
        'n/a (ground truth)',
    ]) + ' |'
    per_seed = [f'### Row `{row}`: per seed', '',
                '| Seed | End | Sim s | Wall min | Collected | Sorting acc. | Contam. | False pickups | Grasp | '
                'Fall (t, tilt) | Stuck | Distance (m) | Dup. | Coverage | Safety viol. (person / arm / geofence) | '
                'Min person d. (m) | Arm change since not ok (rad) | Infra retries |',
                '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r, x in zip(runs, m):
        s = x['safety']
        fall = f"{fmt(x['fall_t'], 0)} s, {fmt(x['max_tilt_deg'], 0)}°" if x['falls'] else f"– (max {fmt(x['max_tilt_deg'], 0)}°)"
        per_seed.append('| ' + ' | '.join([
            str(r['seed']), r['end_reason'], fmt(r['sim_end_t'], 0), f"{r['wall_s'] / 60:.0f}",
            f"{x['items_collected']}/{x['items_total']}", fmt(x['sorting_accuracy']), str(x['contamination_count']),
            str(x['false_pickups']), f"{x['picks_ok']}/{x['picks']}", fall, str(x['stuck_events']),
            fmt(x['distance_m'], 0), fmt(x['duplicates']), fmt(x['coverage']),
            f"{s['violations']} ({len(s['person_violations'])} / {len(s['arm_violations'])} / "
            f"{len(s['geofence_episodes'])})",
            fmt(s['min_person_distance']['d']) if s['min_person_distance'] else '–',
            fmt(s['arm_max_change_not_ok_rad'], 3),
            str(len(r.get('infra_attempts') or [])),
        ]) + ' |')
    return line, '\n'.join(per_seed) + '\n'


def append_row(row, show=False):
    seeds = yaml.safe_load(open(REPO / 'bench' / 'bench.yaml'))['bench']['seeds']
    line, section = build(row, load_runs(row, seeds))
    if show:
        print(line + '\n\n' + section)
        return
    text = RESULTS.read_text() if RESULTS.exists() else HEADER
    if f'| `{row}` |' in text:
        sys.exit(f'row {row} is already in {RESULTS}: use a new row name')
    text = text.replace(SUMMARY_MARK, line + '\n' + SUMMARY_MARK)
    RESULTS.write_text(text.rstrip('\n') + '\n\n' + section)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('row')
    ap.add_argument('--print', action='store_true')
    a = ap.parse_args()
    append_row(a.row, a.print)
