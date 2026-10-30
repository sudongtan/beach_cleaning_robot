"""Checks a scenes.yaml document against bench/SCENES_SCHEMA.md.

    python3 sim/common/scene_schema.py bench/scenes.yaml

Returns a list of error strings; empty means valid. Kept dependency-free
(no jsonschema) so every simulator's loader can call it.
"""

import sys

import yaml

NUM = (int, float)


def _pose(errs, where, v, n=3):
    if not (isinstance(v, list) and len(v) == n and all(isinstance(a, NUM) for a in v)):
        errs.append(f'{where}: expected a list of {n} numbers, got {v!r}')


def _keys(errs, where, d, required):
    if not isinstance(d, dict):
        errs.append(f'{where}: expected a mapping')
        return False
    missing = [k for k in required if k not in d]
    if missing:
        errs.append(f'{where}: missing {missing}')
    return not missing


def validate(doc, objects):
    errs = []
    if not _keys(errs, 'root', doc, ['schema_version', 'scenes']):
        return errs
    if doc['schema_version'] != 1:
        errs.append(f"unsupported schema_version {doc['schema_version']}")
    seeds = [s.get('seed') for s in doc['scenes']]
    if len(set(seeds)) != len(seeds):
        errs.append('duplicate seeds')
    for s in doc['scenes']:
        w = f"scene {s.get('seed')}"
        if not _keys(errs, w, s, ['seed', 'area', 'geo', 'terrain', 'geofence', 'robot_start',
                                  'bins', 'landmarks', 'trash', 'distractors', 'person']):
            continue
        _keys(errs, f'{w}.geo', s['geo'], ['latitude_deg', 'longitude_deg', 'elevation_m'])
        _keys(errs, f'{w}.terrain', s['terrain'],
              ['z_min', 'z_max', 'heightmap', 'waterline_y', 'wet_zone_width', 'wet_zone_height',
               'beach_slope', 'roughness', 'dunes', 'friction'])
        _keys(errs, f'{w}.terrain.friction', s['terrain'].get('friction'), ['dry_sand', 'wet_sand'])
        poly = s['geofence'].get('polygon', [])
        if len(poly) < 3:
            errs.append(f'{w}.geofence: polygon needs >= 3 points')
        for p in poly:
            _pose(errs, f'{w}.geofence.polygon', p, 2)
        _pose(errs, f'{w}.robot_start.pose', s['robot_start'].get('pose'))
        ids = []
        for b in s['bins']:
            if _keys(errs, f'{w}.bins', b, ['id', 'model', 'bin_id', 'pose']):
                _pose(errs, f"{w}.bins.{b['id']}", b['pose'])
                if b['model'] not in objects['bins']:
                    errs.append(f"{w}: unknown bin model {b['model']}")
                ids.append(b['id'])
        if sorted(b['bin_id'] for b in s['bins']) != ['recyclable', 'residual']:
            errs.append(f'{w}: need exactly one recyclable and one residual bin')
        for lm in s['landmarks']:
            if _keys(errs, f'{w}.landmarks', lm, ['id', 'model', 'pose']):
                _pose(errs, f"{w}.landmarks.{lm['id']}", lm['pose'])
                if lm['model'] not in objects['landmarks']:
                    errs.append(f"{w}: unknown landmark model {lm['model']}")
                ids.append(lm['id'])
        for it in s['trash']:
            if _keys(errs, f'{w}.trash', it, ['id', 'model', 'material', 'pose', 'half_buried']):
                _pose(errs, f"{w}.trash.{it['id']}", it['pose'])
                spec = objects['trash'].get(it['model'])
                if spec is None:
                    errs.append(f"{w}: unknown trash model {it['model']}")
                elif spec['material'] != it['material']:
                    errs.append(f"{w}: {it['id']} material does not match objects.yaml")
                if it['half_buried'] and not 0 < it.get('bury_fraction', 0) < 1:
                    errs.append(f"{w}: {it['id']} half_buried needs bury_fraction in (0, 1)")
                ids.append(it['id'])
        for d in s['distractors']:
            if _keys(errs, f'{w}.distractors', d, ['id', 'model', 'type', 'pose']):
                _pose(errs, f"{w}.distractors.{d['id']}", d['pose'])
                if d['model'] not in objects['distractors']:
                    errs.append(f"{w}: unknown distractor model {d['model']}")
                ids.append(d['id'])
        if s['person'] is not None:
            p = s['person']
            if _keys(errs, f'{w}.person', p, ['id', 'speed', 'loop', 'waypoints']):
                if not p['speed'] > 0:
                    errs.append(f'{w}.person: speed must be > 0')
                if len(p['waypoints']) < 2:
                    errs.append(f'{w}.person: needs >= 2 waypoints')
                for q in p['waypoints']:
                    _pose(errs, f'{w}.person.waypoints', q, 2)
                ids.append(p['id'])
        if len(set(ids)) != len(ids):
            errs.append(f'{w}: duplicate ids')
    return errs


def main():
    from pathlib import Path
    here = Path(__file__).resolve().parent
    with open(sys.argv[1]) as f:
        doc = yaml.safe_load(f)
    with open(here / 'objects.yaml') as f:
        objects = yaml.safe_load(f)
    errs = validate(doc, objects)
    print('\n'.join(errs) if errs else 'valid')
    sys.exit(1 if errs else 0)


if __name__ == '__main__':
    main()
