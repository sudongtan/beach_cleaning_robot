"""The attach-mode pick and store (D0c) as a timed schedule of joint targets, plus its checks.

The steps come from our own planner (grasping.arm / grasping.sequence on `main`), saved by the
laptop as d0_from_laptop/grasp/pick_plans.json (notes/briefs/D0_laptop.md). As in the Gazebo
grasping node: an arm step moves all 6 arm joints to the step's values in `move_time_s`, a finger
step moves both fingers to [v, -v] in `finger_time_s`; attach / detach happen at their step.
Moves are smooth (cubic, zero speed at both ends), like a two-point joint trajectory.
No Isaac here: run_d0.py plays the schedule.
"""

import json
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Segment:
    t0: float
    t1: float
    kind: str                       # arm, fingers, attach, detach
    phase: str
    arm_from: np.ndarray
    arm_to: np.ndarray
    finger_from: float
    finger_to: float
    item: str = ''


@dataclass
class Schedule:
    segments: list = field(default_factory=list)
    duration: float = 0.0

    def events(self):
        """[(t, kind, item, phase)] for attach / detach."""
        return [(s.t0, s.kind, s.item, s.phase) for s in self.segments if s.kind in ('attach', 'detach')]

    def phase_end(self, phase, nth=0):
        hits = [s for s in self.segments if s.phase == phase]
        return hits[nth].t1 if len(hits) > nth else None

    def targets(self, t):
        """(arm[6], finger) at time t."""
        seg = self.segments[-1]
        for s in self.segments:
            if t < s.t1:
                seg = s
                break
        if t >= self.duration:
            return seg.arm_to.copy(), seg.finger_to
        f = smooth((t - seg.t0) / (seg.t1 - seg.t0)) if seg.t1 > seg.t0 else 1.0
        return seg.arm_from + f * (seg.arm_to - seg.arm_from), seg.finger_from + f * (seg.finger_to - seg.finger_from)


def smooth(f):
    f = min(max(f, 0.0), 1.0)
    return f * f * (3.0 - 2.0 * f)


def build_schedule(steps, arm_start, finger_start, move_time, finger_time, settle):
    arm = np.asarray(arm_start, dtype=float)
    finger = float(finger_start)
    t = 0.0
    sch = Schedule()
    for st in steps:
        kind, phase, value = st['kind'], st['phase'], st['value']
        if kind == 'arm':
            to = np.asarray(value, dtype=float)
            sch.segments.append(Segment(t, t + move_time + settle, kind, phase, arm, to, finger, finger))
            arm, t = to, t + move_time + settle
        elif kind == 'fingers':
            sch.segments.append(Segment(t, t + finger_time + settle, kind, phase, arm, arm, finger, float(value)))
            finger, t = float(value), t + finger_time + settle
        elif kind in ('attach', 'detach'):
            sch.segments.append(Segment(t, t, kind, phase, arm, arm, finger, finger, item=str(value)))
        else:
            raise ValueError(f'unknown step kind {kind}')
    sch.duration = t
    return sch


def load_cases(plans_path, stance):
    with open(plans_path) as f:
        plans = json.load(f)
    st = plans['stances'][stance]
    return plans, st, [c for c in st['cases'] if c.get('pick_ok') and c.get('store_ok')]


def case_schedule(case, cfg_pick, arm_rest, finger_rest):
    """Pick then store: one schedule (the store starts where the pick ended)."""
    steps = list(case['pick_steps']) + list(case['store_steps'])
    return build_schedule(steps, arm_rest, finger_rest, float(cfg_pick['move_time_s']),
                          float(cfg_pick['finger_time_s']), float(cfg_pick['settle_s']))


def item_pose_base_link(case):
    """Item model origin (footprint centre on the ground, objects.yaml frame) and yaw in base_link.
    The plans give the bbox centre and assume the item's long axis along base_link x."""
    c = np.asarray(case['centre_base_link'], dtype=float)
    return np.array([c[0], c[1], c[2] - case['size'][2] / 2.0]), 0.0


def evaluate_case(case, trace, comp_fn, cfg_pass, ground_z_base):
    """trace: dict with t, item_base_link (n x 3, item centre in base_link), attached (bool per row),
    tilt_deg, tips_item_top_at_attach. comp_fn(bin, p_base_link, margin) -> in the right compartment."""
    t = np.asarray(trace['t'])
    p = np.asarray(trace['item_base_link'])
    lift_t = trace.get('lift_end_t')
    out = {'item': case['item'], 'bin': case['bin'], 'spot': case['centre_base_link']}
    if lift_t is not None and len(t):
        k = int(np.searchsorted(t, lift_t))
        k = min(k, len(t) - 1)
        out['height_after_lift_m'] = round(float(p[k, 2] - ground_z_base - case['size'][2] / 2.0), 4)
        out['lifted'] = out['height_after_lift_m'] >= cfg_pass['lift_min_m']
    else:
        out['lifted'] = False
    final = p[-1] if len(p) else np.full(3, np.nan)
    out['final_base_link'] = [round(float(v), 4) for v in final]
    out['in_compartment'] = bool(len(p)) and comp_fn(case['bin'], final, cfg_pass['basket_margin_m'])
    out['max_tilt_deg'] = round(float(np.max(trace['tilt_deg'])), 2) if len(trace['tilt_deg']) else None
    out['fell'] = out['max_tilt_deg'] is None or out['max_tilt_deg'] > trace.get('fall_tilt_deg', 30.0)
    out['tips_item_top_at_attach_m'] = trace.get('tips_item_top_at_attach')
    out['ok'] = bool(out['lifted'] and out['in_compartment'] and not out['fell'])
    return out
