"""Run recording without ROS: the recorder node (bench/recorder.py) feeds this accumulator from the
topics, and the unit tests feed it by hand. Everything in sim time (message stamps) except the
progress / heartbeat times, which are wall time (time.time()) for the runner's stall and dead-stack checks.

Records what can't be rebuilt from the task's event log afterwards (notes/benchmark.md, brief A3):
distance, tilt / fall time, person distance and the person rule, the arm rule, the geofence rule,
localization drift, the last /world_model (coverage, duplicates, id -> scene id) and the final
ground truth of every object."""
import math
from collections import Counter, deque


def tilt_deg(q):
    """Angle between the body z axis and the world z axis; q = (x, y, z, w)."""
    x, y = q[0], q[1]
    return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))


def yaw_of(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def inside_polygon(x, y, poly):
    """Even-odd rule; poly = [[x, y], ...]."""
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def coverage_share(data):
    """Share of geofenced cells (!= -1) that are swept (== 100); None without any."""
    inside = [v for v in data if v != -1]
    return sum(1 for v in inside if v == 100) / len(inside) if inside else None


def duplicates(items):
    """Extra world-model entries per non-empty scene id: {scene_id: extra}."""
    c = Counter(i['scene_id'] for i in items if i['scene_id'])
    return {k: n - 1 for k, n in c.items() if n > 1}


class Episodes:
    """Groups violation samples (sim t) into episodes: a gap > `gap` starts a new one."""

    def __init__(self, gap):
        self.gap, self.items = gap, []

    def add(self, t, **info):
        e = self.items[-1] if self.items else None
        if e is None or t - e['t_end'] > self.gap:
            e = {'t_start': t, 't_end': t, 'samples': 0}
            self.items.append(e)
        e['t_end'] = t
        e['samples'] += 1
        for k, v in info.items():                    # keep the worst value per key
            if isinstance(v, (int, float)) and k in e and isinstance(e[k], (int, float)):
                e[k] = min(e[k], v) if k.startswith('min_') else max(e[k], v)
            else:
                e.setdefault(k, v)
        return e


class Recording:
    def __init__(self, cfg, stop_distance_m, geofence=None):
        self.cfg = cfg
        self.stop_d = stop_distance_m
        self.geofence = geofence                      # [[x, y], ...], replaced by /geofence when it arrives
        self.geofence_source = 'scenes.yaml' if geofence else None
        self.sim_t0 = self.sim_t = None
        self.pose = None                              # (x, y, z, yaw)
        self.distance = 0.0
        self.dist_anchor = None
        self.max_tilt, self.fall_t, self.fall_pose = 0.0, None, None
        self.track = deque()                          # (t, x, y) for the speed window
        self.speed = 0.0
        self.cmd = (0.0, 0.0, 0.0)
        self.people, self.people_t = [], None         # [(x, y)], stamp
        self.min_person = None                        # {'d', 't', 'speed', 'cmd', 'robot'}
        self.person_viol = Episodes(cfg['episode_gap_s'])
        self.safety = None
        self.safety_counts = Counter()
        self.safety_since = None                      # sim t of the last /safety/state change
        self.transitions = []                         # [t, state, person d, robot speed] per change
        self.inside_since = None                      # sim t the person came within the stop distance
        self.arm = None
        self.arm_ref = None                           # arm q when the state left ok
        self.arm_max_dev = 0.0                        # max change while not ok
        self.arm_max_at = None                        # {'t', 'joint', 'state'} of that maximum
        self.arm_viol = Episodes(cfg['episode_gap_s'])
        self.arm_window = deque()                     # (t, q) of the last arm_window_s while not ok
        self.outside = Episodes(cfg['episode_gap_s'])
        self.outside_s = 0.0
        self.drift = {'n': 0, 'sum': 0.0, 'max': 0.0, 'last': None}
        self.drift_next = None
        self.odom_global = None                       # (t, x, y)
        self.task_state, self.task_changes = '', 0
        self.wm = None                                # last world model summary
        self.wm_ids = {}                              # world-model id -> scene id, every one ever seen
        self.objects, self.objects_t, self.objects_robot = None, None, None
        self.last_pose_wall = None
        self.progress_wall = None                     # last task state change or 0.5 m move (wall)
        self.progress_anchor = None

    # --- inputs -------------------------------------------------------------------------
    def on_pose(self, t, pos, quat, wall):
        x, y, z = pos
        if self.sim_t0 is None:
            self.sim_t0 = t
            self.progress_wall = wall
        self.sim_t, self.last_pose_wall = t, wall
        prev_t = self.track[-1][0] if self.track else None
        self.pose = (x, y, z, yaw_of(quat))
        # distance walked: steps > distance_step_m
        if self.dist_anchor is None:
            self.dist_anchor = (x, y)
        else:
            d = math.hypot(x - self.dist_anchor[0], y - self.dist_anchor[1])
            if d > self.cfg['distance_step_m']:
                self.distance += d
                self.dist_anchor = (x, y)
        # stall progress: 0.5 m from the anchor
        if self.progress_anchor is None or math.hypot(x - self.progress_anchor[0],
                                                      y - self.progress_anchor[1]) > self.cfg['stall_move_m']:
            self.progress_anchor = (x, y)
            self.progress_wall = wall
        # tilt and the fall time
        tilt = tilt_deg(quat)
        self.max_tilt = max(self.max_tilt, tilt)
        if self.fall_t is None and tilt > self.cfg['fall_tilt_deg']:
            self.fall_t, self.fall_pose = t, [round(x, 3), round(y, 3)]
        # planar speed over the window
        self.track.append((t, x, y))
        while len(self.track) > 2 and t - self.track[1][0] >= self.cfg['speed_window_s']:
            self.track.popleft()
        t0, x0, y0 = self.track[0]
        if t - t0 >= 0.8 * self.cfg['speed_window_s']:
            self.speed = math.hypot(x - x0, y - y0) / (t - t0)
        # geofence: base_link outside the polygon
        if self.geofence and not inside_polygon(x, y, self.geofence):
            if prev_t is not None:
                self.outside_s += t - prev_t
            self.outside.add(t, robot=[round(x, 3), round(y, 3)])
        self.check_person(t)
        self.sample_drift(t)

    def on_cmd(self, vx, vy, wz):
        self.cmd = (vx, vy, wz)

    def on_people(self, t, people):
        self.people, self.people_t = people, t

    def on_safety(self, state):
        if state != self.safety:
            self.safety_counts[state] += 1
            self.safety_since = self.sim_t
            self.transitions.append([self.sim_t, state, self.person_d(), round(self.speed, 3)])
            if state == 'ok':
                self.arm_ref = None
            elif self.safety in ('ok', None):
                self.arm_ref = self.arm
                self.arm_window.clear()
        self.safety = state

    def on_joints(self, t, positions):
        """positions: {joint name: rad}. Arm rule (b2 reading, decision 2026-11-04): a joint changes by
        more than arm_change_rad within arm_window_s while the state isn't ok. The change since the
        state left ok (slow creep included) is reported as a number."""
        names = self.cfg['arm_joints']
        if not all(n in positions for n in names):
            return
        self.arm = [positions[n] for n in names]
        if self.safety == 'ok' or self.safety is None:
            self.arm_window.clear()
            return
        if self.arm_ref is None:
            self.arm_ref = self.arm
        devs = [abs(a - b) for a, b in zip(self.arm, self.arm_ref)]
        dev = max(devs)
        if dev > self.arm_max_dev:
            self.arm_max_dev = dev
            self.arm_max_at = {'t': round(t, 2), 'joint': names[devs.index(dev)], 'state': self.safety}
        self.arm_window.append((t, self.arm))
        while self.arm_window and t - self.arm_window[0][0] > self.cfg['arm_window_s']:
            self.arm_window.popleft()
        moves = [max(abs(a - b) for a, b in zip(self.arm, q)) for _, q in self.arm_window]
        move = max(moves)
        if move > self.cfg['arm_change_rad']:
            k = moves.index(move)
            jd = [abs(a - b) for a, b in zip(self.arm, self.arm_window[k][1])]
            since = None if self.safety_since is None else round(t - self.safety_since, 3)
            e = self.arm_viol.add(t, state=self.safety, max_change_rad=round(move, 4),
                                  joint=names[jd.index(max(jd))], since_not_ok_s=since)
            if e['samples'] <= self.cfg['episode_samples']:
                e.setdefault('trace', []).append([round(t, 3), round(move, 4), round(dev, 4)])

    def on_odom_global(self, t, x, y):
        self.odom_global = (t, x, y)

    def on_task(self, state, wall):
        if state != self.task_state:
            self.task_state = state
            self.task_changes += 1
            self.progress_wall = wall

    def on_geofence(self, poly):
        if len(poly) >= 3:
            self.geofence, self.geofence_source = poly, '/geofence'

    def on_world_model(self, items, coverage_data):
        """items: [{'id', 'scene_id', 'object_class', 'material', 'bin_id', 'status', 'skip_reason'}]."""
        for i in items:
            if i['scene_id']:
                self.wm_ids[i['id']] = i['scene_id']
        dup = duplicates(items)
        self.wm = {'items': items, 'coverage': coverage_share(coverage_data),
                   'duplicates': sum(dup.values()), 'duplicate_ids': dup}

    def on_objects(self, t, objects):
        """objects: {id: [x, y, z]} (bbox centre, map frame), every trash item and distractor."""
        self.objects, self.objects_t = objects, t
        self.objects_robot = list(self.pose) if self.pose else None

    # --- derived ------------------------------------------------------------------------
    def person_d(self):
        """Horizontal distance base_link -> nearest person (None without a fresh /people)."""
        if (not self.people or self.people_t is None or self.pose is None
                or self.sim_t - self.people_t > self.cfg['people_max_age_s']):
            return None
        return round(min(math.hypot(px - self.pose[0], py - self.pose[1]) for px, py in self.people), 3)

    def check_person(self, t):
        d = self.person_d()
        if d is None:
            return
        x, y = self.pose[0], self.pose[1]
        if d >= self.stop_d:
            self.inside_since = None
        elif self.inside_since is None:
            self.inside_since = t
        cmd = max(abs(v) for v in self.cmd)
        if self.min_person is None or d < self.min_person['d']:
            self.min_person = {'d': round(d, 3), 't': round(t, 2), 'speed': round(self.speed, 3),
                               'cmd': [round(v, 3) for v in self.cmd], 'robot': [round(x, 3), round(y, 3)]}
        if d < self.stop_d:
            by_cmd = cmd > self.cfg['cmd_eps']
            by_speed = self.speed > self.cfg['moving_speed_mps']
            if by_cmd or by_speed:
                since_stop = (round(t - self.safety_since, 3)
                              if self.safety == 'stop' and self.safety_since is not None else None)
                e = self.person_viol.add(t, min_d=round(d, 3), max_speed=round(self.speed, 3),
                                         max_cmd=round(cmd, 3), by_cmd=by_cmd, by_speed=by_speed,
                                         robot=[round(x, 3), round(y, 3)], state=self.safety,
                                         since_inside_s=round(t - self.inside_since, 3))
                if e['samples'] <= self.cfg['episode_samples']:
                    # [t, d, speed, max |cmd|, safety state, s since the state became stop]
                    e.setdefault('trace', []).append([round(t, 3), round(d, 3), round(self.speed, 3),
                                                      round(cmd, 3), self.safety, since_stop])

    def sample_drift(self, t):
        if self.odom_global is None or abs(self.odom_global[0] - t) > 0.1:
            return
        if self.drift_next is not None and t < self.drift_next:
            return
        self.drift_next = t + self.cfg['drift_every_s']
        e = math.hypot(self.odom_global[1] - self.pose[0], self.odom_global[2] - self.pose[1])
        d = self.drift
        d['n'] += 1
        d['sum'] += e
        d['max'] = max(d['max'], e)
        d['last'] = e

    def snapshot(self, wall, final=False):
        d = self.drift
        return {
            'final': final, 'wall': wall,
            'sim_t0': self.sim_t0, 'sim_t': self.sim_t,
            'last_pose_wall': self.last_pose_wall, 'progress_wall': self.progress_wall,
            'task_state': self.task_state, 'task_changes': self.task_changes,
            'pose': self.pose, 'distance_m': self.distance,
            'max_tilt_deg': self.max_tilt, 'fall_t': self.fall_t, 'fall_pose': self.fall_pose,
            'stop_distance_m': self.stop_d, 'min_person': self.min_person,
            'person_violations': self.person_viol.items,
            'safety_state': self.safety, 'safety_counts': dict(self.safety_counts),
            'safety_transitions': self.transitions,
            'arm_max_change_not_ok_rad': self.arm_max_dev, 'arm_max_change_at': self.arm_max_at,
            'arm_violations': self.arm_viol.items,
            'geofence_source': self.geofence_source, 'outside_geofence_s': self.outside_s,
            'outside_geofence': self.outside.items,
            'drift': {'samples': d['n'], 'mean_m': d['sum'] / d['n'] if d['n'] else None,
                      'max_m': d['max'] if d['n'] else None, 'last_m': d['last']},
            'world_model': self.wm, 'wm_ids': self.wm_ids,
            'objects_t': self.objects_t, 'objects': self.objects, 'objects_robot': self.objects_robot,
        }
