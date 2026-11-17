#!/usr/bin/env python3
"""Offline localization evaluation on a recorded bag (B4).

  python3 bench/loc_eval.py <bag> --seed 5 [--own] [--json out.json]

Without --own: compares the bag's /odom/global (e.g. recorded while robot_localization replayed the
sensors) with /sim/ground_truth/pose. With --own: runs our own EKF (localization/ekf_core.py, the
settings of config/localization.yaml) over the bag's /leg_odom, /imu, /gnss/fix first.

Reports (sim time, first --skip-s seconds left out: start-up): horizontal error RMS / p50 / p95 /
p99.9 / max, yaw error RMS, and the change of map -> odom over --window-s windows (p95): how far
a target seen in the map moves in that time, the number behind "close range in odom".
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'packages' / 'localization'))
from localization.ekf_core import LocalEnu, PlanarEkf  # noqa: E402


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def read_bag(path, topics):
    """[(topic, msg)] in recording order."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id=''), rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    reader.set_filter(rosbag2_py.StorageFilter(topics=[t for t in topics if t in types]))
    out = []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        out.append((topic, deserialize_message(data, get_message(types[topic]))))
    return out


def stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def run_own(msgs, scene, cfg, imu_std):
    """Own EKF over the recorded sensors -> {'map': [(t, x, y, yaw)], 'odom': [...]}."""
    geo = scene['geo']
    enu = LocalEnu(geo['latitude_deg'], geo['longitude_deg'], geo.get('elevation_m', 0.0))
    start = scene['robot_start']['pose']
    sc = cfg['odom_scale']
    local = PlanarEkf(cfg['process_noise'], x0=(0.0, 0.0, start[2]), p0=cfg['initial_stddev_local'],
                      scale0=sc['initial'])
    glob = PlanarEkf(cfg['process_noise'], x0=tuple(start[:3]), p0=cfg['initial_stddev_global'],
                     gate_chi2=cfg['gnss_gate_chi2'], estimate_scale=sc['estimate'], scale0=sc['initial'],
                     scale_std=sc['initial_stddev'], scale_noise=sc['noise'])
    lever = None
    from tf2_ros import Buffer
    from rclpy.time import Time
    buf = Buffer()
    for topic, m in msgs:
        if topic == '/tf_static':
            for tf in m.transforms:
                buf.set_transform_static(tf, 'bag')
    try:
        tr = buf.lookup_transform('base_link', 'gnss_link', Time()).transform.translation
        lever = (tr.x, tr.y)
    except Exception as e:                        # noqa: BLE001
        print(f'no base_link <- gnss_link in the bag ({e}): lever 0')
    lever = lever or (0.0, 0.0)
    out = {'map': [], 'odom': []}
    for topic, m in msgs:
        t = stamp(m) if hasattr(m, 'header') else None
        if topic == '/leg_odom':
            v, c = m.twist.twist, m.twist.covariance
            var = [max(c[0], 1e-4), max(c[7], 1e-4), max(c[35], 1e-4)]
            for f in (local, glob):
                f.update_body_velocity(t, (v.linear.x, v.linear.y, v.angular.z), var)
        elif topic == '/imu':
            wz_var = max(m.angular_velocity_covariance[8], 1e-8)
            for f in (local, glob):
                f.update_imu(t, yaw_of(m.orientation), m.angular_velocity.z, imu_std[2] ** 2, wz_var)
        elif topic == '/gnss/fix':
            e, n, _ = enu(m.latitude, m.longitude, m.altitude)
            c = m.position_covariance
            glob.update_gnss(t, e, n, max(c[0], 1e-4), max(c[4], 1e-4), lever)
            local.s[6] = glob.scale()                  # the odom filter uses the calibrated scale
            out['map'].append((t, *glob.pose()))
            out['odom'].append((t, *local.pose()))
    print(f'own EKF: GNSS used {glob.gnss_used}, gated out {glob.gnss_rejected}, lever {lever}, '
          f'odom scale {glob.scale():.3f}')
    return out


def recorded(msgs, topic):
    return [(stamp(m), m.pose.pose.position.x, m.pose.pose.position.y, yaw_of(m.pose.pose.orientation))
            for tp, m in msgs if tp == topic]


def nearest(series, t):
    ts = series[:, 0]
    k = np.searchsorted(ts, t)
    k = np.clip(k, 1, len(ts) - 1)
    k = np.where(np.abs(ts[k - 1] - t) < np.abs(ts[k] - t), k - 1, k)
    return series[k]


def map_odom(est_map, est_odom):
    """map -> odom (x, y, yaw) per sample (map_T_base * inverse(odom_T_base))."""
    yaw = np.arctan2(np.sin(est_map[:, 3] - est_odom[:, 3]), np.cos(est_map[:, 3] - est_odom[:, 3]))
    c, s = np.cos(yaw), np.sin(yaw)
    x = est_map[:, 1] - (c * est_odom[:, 1] - s * est_odom[:, 2])
    y = est_map[:, 2] - (s * est_odom[:, 1] + c * est_odom[:, 2])
    return np.stack([est_map[:, 0], x, y, yaw], axis=1)


def evaluate(est, gt, skip_s, window_s, est_odom=None):
    est = np.asarray(est)
    gt = np.asarray(gt)
    est = est[est[:, 0] >= gt[0, 0] + skip_s]
    g = nearest(gt, est[:, 0])
    err = np.hypot(est[:, 1] - g[:, 1], est[:, 2] - g[:, 2])
    dyaw = np.arctan2(np.sin(est[:, 3] - g[:, 3]), np.cos(est[:, 3] - g[:, 3]))
    res = {'samples': int(len(est)), 'duration_s': round(float(est[-1, 0] - est[0, 0]), 1),
           'err_rms_m': round(float(np.sqrt(np.mean(err ** 2))), 3),
           'err_p50_m': round(float(np.percentile(err, 50)), 3),
           'err_p95_m': round(float(np.percentile(err, 95)), 3),
           'err_p999_m': round(float(np.percentile(err, 99.9)), 3),
           'err_max_m': round(float(err.max()), 3),
           'yaw_rms_deg': round(float(np.degrees(np.sqrt(np.mean(dyaw ** 2)))), 2)}
    if est_odom is not None and len(est_odom):
        o = nearest(np.asarray(est_odom), est[:, 0])
        mo = map_odom(est, o)
        # change of map -> odom over window_s: how far a point fixed in odom moves in the map
        later = nearest(mo, mo[:, 0] + window_s)
        ok = later[:, 0] - mo[:, 0] > 0.8 * window_s
        shift = np.hypot(later[ok, 1] - mo[ok, 1], later[ok, 2] - mo[ok, 2])
        res[f'map_odom_change_{window_s:g}s_p95_m'] = round(float(np.percentile(shift, 95)), 3) if ok.any() else None
        res[f'map_odom_change_{window_s:g}s_max_m'] = round(float(shift.max()), 3) if ok.any() else None
    if est_odom is not None and len(est_odom):
        res.update(odom_drift(np.asarray(est_odom), gt, window_s))
    return res


def odom_drift(odom, gt, window_s):
    """Relative error of the odom track over window_s: |odom displacement - true displacement| (both
    frames share the IMU's absolute heading), and that per metre walked (moving windows only)."""
    gt = np.asarray(gt)
    later = nearest(odom, odom[:, 0] + window_s)
    ok = later[:, 0] - odom[:, 0] > 0.8 * window_s
    g0, g1 = nearest(gt, odom[ok, 0]), nearest(gt, later[ok, 0])
    d_est = later[ok, 1:3] - odom[ok, 1:3]
    d_gt = g1[:, 1:3] - g0[:, 1:3]
    err = np.linalg.norm(d_est - d_gt, axis=1)
    walked = np.linalg.norm(d_gt, axis=1)
    moving = walked > 0.2
    out = {f'odom_drift_{window_s:g}s_p50_m': round(float(np.percentile(err, 50)), 3),
           f'odom_drift_{window_s:g}s_p95_m': round(float(np.percentile(err, 95)), 3)}
    if moving.any():
        out['odom_drift_per_m_p50'] = round(float(np.percentile(err[moving] / walked[moving], 50)), 3)
        out['odom_drift_per_m_p95'] = round(float(np.percentile(err[moving] / walked[moving], 95)), 3)
        along = np.sum(d_est[moving] * d_gt[moving], axis=1) / walked[moving] ** 2
        out['odom_scale_p50'] = round(float(np.percentile(along, 50)), 3)    # 1 = right speed
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--own', action='store_true')
    ap.add_argument('--gt-bag', default='', help='bag with /sim/ground_truth/pose (default: the same bag)')
    ap.add_argument('--skip-s', type=float, default=10.0)
    ap.add_argument('--window-s', type=float, default=10.0)
    ap.add_argument('--json', default='')
    a = ap.parse_args()
    scene = next(s for s in yaml.safe_load(open(REPO / 'bench' / 'scenes.yaml'))['scenes'] if s['seed'] == a.seed)
    loc = yaml.safe_load(open(REPO / 'config' / 'localization.yaml'))['localization']
    topics = ['/leg_odom', '/imu', '/gnss/fix', '/tf_static', '/sim/ground_truth/pose', '/odom/global', '/odom',
              '/localization/gnss_odom']
    msgs = read_bag(a.bag, topics)
    gt_msgs = read_bag(a.gt_bag, ['/sim/ground_truth/pose']) if a.gt_bag else msgs
    gt = recorded(gt_msgs, '/sim/ground_truth/pose')
    if a.own:
        series = run_own(msgs, scene, loc['own_ekf'], loc['imu_relay']['orientation_stddev_rad'])
        est, est_odom = series['map'], series['odom']
    else:
        est, est_odom = recorded(msgs, '/odom/global'), recorded(msgs, '/odom')
    res = {'bag': a.bag, 'seed': a.seed, 'estimator': 'own_ekf' if a.own else 'recorded /odom/global',
           **evaluate(est, gt, a.skip_s, a.window_s, est_odom)}
    gnss = recorded(msgs, '/localization/gnss_odom')
    if gnss:                                       # navsat_transform's GNSS in the map vs truth (datum check)
        gn = np.asarray(gnss)
        g = nearest(np.asarray(gt), gn[:, 0])
        res['gnss_odom_mean_offset_m'] = [round(float(np.mean(gn[:, 1] - g[:, 1])), 3),
                                          round(float(np.mean(gn[:, 2] - g[:, 2])), 3)]
    print(json.dumps(res, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
