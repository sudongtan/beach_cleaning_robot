#!/usr/bin/env python3
"""Benchmark runner (notes/benchmark.md, brief A3): runs the benchmark seeds one after another, headless,
unattended; one JSON per run in results/runs/<row>_s<seed>.json, then the row in results/results.md.

  runner.py [--sim gazebo] [--seeds all|0-9|0,3] [--row baseline] [--no-report] [--sim-cap S]

Per seed: stop_sim.sh, `make run` (own partition / domain, loopback), wait for the stand-up, start the
recorder (bench/recorder.py), watch for the end (done, fell, stall, sim / wall cap, dead stack, startup
failure; bench/bench.yaml), stop the recorder (it flushes), stop_sim.sh, write the run JSON.
Restartable: a seed whose run JSON exists is skipped. Only a startup failure or a dead stack is retried
(once); falls, stalls and timeouts are results. Doesn't start a seed on battery or with too little disk.
Progress: results/bench/<row>/status.json; per-run logs in results/bench/<row>/<run>_<utc>/."""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics  # noqa: E402

STOP_SIM = REPO / 'sim' / 'gazebo' / 'tools' / 'stop_sim.sh'
ROS_SETUP = '/opt/ros/jazzy/setup.bash'
# the Makefile exports WS_SETUP (an overlay install, e.g. a worktree's)
WS_SETUP = os.environ.get('WS_SETUP') or str(Path.home() / 'ros2_ws' / 'install' / 'setup.bash')
INFRA = ('startup_failed', 'dead_stack')


def utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def parse_seeds(text, default):
    if text in ('', 'all'):
        return list(default)
    out = []
    for part in text.split(','):
        a, _, b = part.partition('-')
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def decide_end(now, t_start, rec_start, snap, flags, lim, launch_alive):
    """End reason of a running run, or None. flags: done / fell / fall_line / dead_line (bools) from the
    event log and the launch log; snap: last recorder snapshot or None."""
    state = (snap or {}).get('task_state', '')
    if flags.get('done') or state.startswith('DONE'):
        return 'done'
    if flags.get('fell') or flags.get('fall_line') or state.startswith('FELL'):
        return 'fell'
    if not launch_alive or flags.get('dead_line'):
        return 'dead_stack'
    dead = lim['dead_after_s']
    if snap is None or snap.get('last_pose_wall') is None:
        return 'dead_stack' if now - rec_start > dead else None
    if now - snap['wall'] > dead or now - snap['last_pose_wall'] > dead:
        return 'dead_stack'
    if snap['sim_t'] >= lim['sim_s']:
        return 'sim_cap'
    if now - t_start >= lim['wall_min'] * 60:
        return 'wall_cap'
    if now - snap['progress_wall'] > lim['stall_s']:
        return 'stall'
    return None


class Tail:
    """Reads the lines a growing file has gained since the last call."""

    def __init__(self, path):
        self.path, self.pos, self.rest = Path(path), 0, ''

    def lines(self):
        if not self.path.exists():
            return []
        with open(self.path, errors='replace') as f:
            f.seek(self.pos)
            data = f.read()
            self.pos = f.tell()
        data = self.rest + data
        parts = data.split('\n')
        self.rest = parts.pop()
        return parts


class Runner:
    def __init__(self, args):
        self.bench_cfg = yaml.safe_load(open(REPO / 'bench' / 'bench.yaml'))
        self.cfg = self.bench_cfg['bench']
        self.lim = dict(self.cfg['limits'])
        if args.sim_cap:
            self.lim['sim_s'] = args.sim_cap
        if args.ac_path:
            self.cfg['ac_path'] = args.ac_path
        self.row = args.row or self.cfg['row']
        self.seeds = parse_seeds(args.seeds, self.cfg['seeds'])
        self.dir = REPO / 'results' / 'bench' / self.row
        self.dir.mkdir(parents=True, exist_ok=True)
        self.env = dict(os.environ, **{k: str(v) for k, v in self.cfg['env'].items()})
        self.status = {'row': self.row, 'pid': os.getpid(), 'started': utc(), 'seeds': self.seeds,
                       'finished': {}, 'current': None, 'paused': None}
        self.scenes = {s['seed']: s for s in yaml.safe_load(open(REPO / 'bench' / 'scenes.yaml'))['scenes']}
        self.rules = yaml.safe_load(open(REPO / 'config' / 'sorting_rules.yaml'))['sorting']['rules']
        self.commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO, capture_output=True,
                                     text=True).stdout.strip()
        self.dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=REPO,
                                         capture_output=True, text=True).stdout.strip())

    def log(self, *a):
        print(f'[{utc()}]', *a, flush=True)

    def write_status(self, **kw):
        self.status.update(kw, updated=utc())
        tmp = self.dir / 'status.tmp'
        tmp.write_text(json.dumps(self.status, indent=1))
        os.replace(tmp, self.dir / 'status.json')

    def stop_sim(self, tries=3):
        """stop_sim.sh until it reports `remaining: 0` (a slow process can survive one call)."""
        for k in range(tries):
            out = subprocess.run([str(STOP_SIM)], capture_output=True, text=True).stdout.strip()
            if out.endswith('remaining: 0'):
                return True, out
            time.sleep(5)
        return False, out

    def wait_for_power(self):
        path = Path(self.cfg['ac_path'])
        paused = False
        while path.exists() and path.read_text().strip() == '0':
            if not paused:
                self.log('paused: on battery (waiting for AC)')
                self.write_status(paused='on battery')
                paused = True
            time.sleep(60)
        if paused:
            self.log('AC back: continuing')
            self.write_status(paused=None)

    def disk_ok(self):
        free = shutil.disk_usage(REPO).free / 1e9
        if free < self.cfg['min_free_gb']:
            self.log(f'stopping: only {free:.1f} GB free')
            self.write_status(paused=f'stopped: {free:.1f} GB free disk')
            return False
        return True

    def run(self):
        self.log(f'bench row {self.row}, seeds {self.seeds}, commit {self.commit}{" (dirty)" if self.dirty else ""}, '
                 f'limits {self.lim}')
        self.write_status()
        for seed in self.seeds:
            run_id = f'{self.row}_s{seed}'
            out = REPO / 'results' / 'runs' / f'{run_id}.json'
            if out.exists():
                self.log(f'{run_id}: run JSON exists, skipped')
                self.status['finished'][str(seed)] = json.loads(out.read_text())['end_reason']
                continue
            attempts = []
            while True:
                self.wait_for_power()
                if not self.disk_ok():
                    return 1
                res = self.run_once(seed, run_id, len(attempts) + 1)
                if res['end_reason'] in INFRA and len(attempts) < self.cfg['infra_retries']:
                    self.log(f'{run_id}: {res["end_reason"]} ({res["end_detail"]}): infrastructure, retrying once')
                    attempts.append({k: res[k] for k in ('end_reason', 'end_detail', 'log_dir', 'wall_s')})
                    continue
                break
            res['infra_attempts'] = attempts
            tmp = out.with_suffix('.tmp')
            tmp.write_text(json.dumps(res, indent=1))
            os.replace(tmp, out)
            self.status['finished'][str(seed)] = res['end_reason']
            self.log(f'{run_id}: {res["end_reason"]}, {res["metrics"]["items_collected"]}/'
                     f'{res["metrics"]["items_total"]} collected, sim {res["sim_end_t"]} s, wall {res["wall_s"] / 60:.0f} min')
            self.write_status(current=None)
        return 0

    def kill_stale_recorders(self):
        """Recorders left by a killed runner (stop_sim.sh doesn't know them); by PID, never pkill -f."""
        ps = subprocess.run(['ps', '-eo', 'pid=,args='], capture_output=True, text=True).stdout
        for line in ps.splitlines():
            pid, _, args = line.strip().partition(' ')
            if args.startswith('python3 ') and 'bench/recorder.py' in args:
                self.log(f'killing a stale recorder (pid {pid})')
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def run_once(self, seed, run_id, attempt):
        self.kill_stale_recorders()
        ok, out = self.stop_sim()
        work = self.dir / f'{run_id}_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}'
        work.mkdir(parents=True)
        launch_log, rec_out, rec_log = work / 'launch.log', work / 'recorder.json', work / 'recorder.log'
        events_path = REPO / 'results' / 'runs' / f'{run_id}.jsonl'
        if events_path.exists():                    # the task appends: keep an earlier attempt's log apart
            shutil.move(str(events_path), work / f'previous_{run_id}.jsonl')
        t_start = time.time()
        base = {'run_id': run_id, 'row': self.row, 'seed': seed, 'sim': 'gazebo', 'attempt': attempt,
                'commit': self.commit, 'dirty': self.dirty, 'components': self.cfg['components'],
                'limits': self.lim, 'started': utc(), 'log_dir': str(work.relative_to(REPO))}
        self.write_status(current={'seed': seed, 'run_id': run_id, 'attempt': attempt, 'started': base['started'],
                                   'phase': 'startup', 'log_dir': base['log_dir']})
        if not ok:
            self.log(f'{run_id}: stop_sim.sh left processes: {out}')
            return self.finish(base, t_start, 'startup_failed', f'stop_sim.sh: {out}', None, events_path)
        self.log(f'{run_id}: launching (attempt {attempt}), logs in {work}')
        launch = subprocess.Popen(
            ['make', 'run', f'SEED={seed}', 'HEADLESS=true', f'GZ_PARTITION={self.env["GZ_PARTITION"]}',
             f'RUN_ID={run_id}', f'LOCO={self.cfg["components"]["locomotion"]}'],
            cwd=REPO, env=self.env, stdout=open(launch_log, 'w'), stderr=subprocess.STDOUT, start_new_session=True)
        tail = Tail(launch_log)
        flags = {}
        # startup
        up = False
        while time.time() - t_start < self.lim['startup_s']:
            time.sleep(2)
            for line in tail.lines():
                if self.cfg['startup_ok'] in line:
                    up = True
                elif any(p in line for p in self.cfg['startup_fail']):
                    flags['startup_line'] = line.strip()[:300]
            if flags.get('startup_line'):
                return self.end(base, t_start, launch, None, 'startup_failed', flags['startup_line'], None, events_path)
            if launch.poll() is not None:
                return self.end(base, t_start, launch, None, 'startup_failed', f'make run exited {launch.returncode}',
                                None, events_path)
            if up:
                break
        if not up:
            return self.end(base, t_start, launch, None, 'startup_failed',
                            f'no "{self.cfg["startup_ok"]}" within {self.lim["startup_s"]} s', None, events_path)
        self.log(f'{run_id}: stood up after {time.time() - t_start:.0f} s, starting the recorder')
        recorder = subprocess.Popen(
            ['bash', '-c', f'source {ROS_SETUP} && source {WS_SETUP} && exec python3 {REPO}/bench/recorder.py '
                           f'{rec_out} {seed} --ros-args -p use_sim_time:=true -r /clock:=/sim/clock_throttled'],
            cwd=REPO, env=self.env, stdout=open(rec_log, 'w'), stderr=subprocess.STDOUT, start_new_session=True)
        rec_start = time.time()
        ev_tail = Tail(events_path)
        last_event, tracebacks = None, 0
        while True:
            time.sleep(self.cfg['poll_s'])
            for line in tail.lines():
                if self.cfg['fall_pattern'] in line:
                    flags['fall_line'] = True
                if any(p in line for p in self.cfg['dead_patterns']):
                    flags['dead_line'] = line.strip()[:300]
                if 'Traceback' in line:
                    tracebacks += 1
            for line in ev_tail.lines():
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get('event') in ('done', 'fell'):
                    flags[e['event']] = True
                if e.get('event') != 'state':
                    last_event = {k: e[k] for k in ('t', 'event', 'item', 'success', 'reason') if k in e}
            snap = None
            if rec_out.exists():
                try:
                    snap = json.loads(rec_out.read_text())
                except json.JSONDecodeError:
                    pass
            if recorder.poll() is not None and not flags.get('dead_line'):
                flags['dead_line'] = f'recorder exited {recorder.returncode}'
            now = time.time()
            reason = decide_end(now, t_start, rec_start, snap, flags, self.lim, launch.poll() is None)
            cur = dict(self.status['current'], phase='running', wall_min=round((now - t_start) / 60, 1),
                       sim_t=snap and snap.get('sim_t'), task_state=snap and snap.get('task_state'),
                       last_event=last_event)
            self.write_status(current=cur)
            if reason:
                detail = {'done': 'task done', 'fell': 'task fell / FALL: in the launch log',
                          'stall': f'no task state change and < {self.lim["stall_move_m"]} m movement for '
                                   f'{self.lim["stall_s"]} s wall',
                          'sim_cap': f'sim time {self.lim["sim_s"]} s', 'wall_cap': f'{self.lim["wall_min"]} min wall',
                          'dead_stack': str(flags.get('dead_line') or ('make run exited' if launch.poll() is not None
                                                                       else 'no ground-truth pose / recorder heartbeat'))
                          }[reason]
                base['tracebacks_in_launch_log'] = tracebacks
                return self.end(base, t_start, launch, recorder, reason, detail, rec_out, events_path)

    def end(self, base, t_start, launch, recorder, reason, detail, rec_out, events_path):
        self.log(f'{base["run_id"]}: end {reason} ({detail})')
        if recorder is not None and recorder.poll() is None:
            os.killpg(recorder.pid, signal.SIGINT)
            try:
                recorder.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(recorder.pid, signal.SIGKILL)
        ok, out = self.stop_sim()
        self.log(f'{base["run_id"]}: stop_sim.sh: {out.splitlines()[-1] if out else "?"}')
        try:
            launch.wait(30)
        except subprocess.TimeoutExpired:
            os.killpg(launch.pid, signal.SIGKILL)
        return self.finish(base, t_start, reason, detail, rec_out, events_path)

    def finish(self, base, t_start, reason, detail, rec_out, events_path):
        rec = None
        if rec_out is not None and rec_out.exists():
            try:
                rec = json.loads(rec_out.read_text())
            except json.JSONDecodeError:
                pass
        events = metrics.load_events(events_path) if events_path.exists() else []
        if events_path.exists():
            shutil.copy(events_path, Path(REPO / base['log_dir']) / events_path.name)
        empty = {'sim_t0': None, 'sim_t': None, 'objects': None, 'objects_robot': None, 'wm_ids': {},
                 'world_model': None, 'fall_t': None, 'person_violations': [], 'arm_violations': [],
                 'outside_geofence': [], 'min_person': None, 'arm_max_change_not_ok_rad': None,
                 'safety_counts': {}, 'drift': None, 'distance_m': None, 'max_tilt_deg': None}
        m = metrics.run_metrics(events, rec or empty, self.scenes[base['seed']], self.rules,
                                self.bench_cfg['metrics'], reason)
        return dict(base, end_reason=reason, end_detail=detail, ended=utc(), wall_s=round(time.time() - t_start, 1),
                    sim_end_t=rec and rec.get('sim_t'), recorder_final=bool(rec and rec.get('final')),
                    metrics=m, recorder=rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim', default='gazebo')
    ap.add_argument('--seeds', default='all')
    ap.add_argument('--row', default='')
    ap.add_argument('--no-report', action='store_true')
    ap.add_argument('--sim-cap', type=float, default=0, help='override the sim-time cap (failure-path tests only)')
    ap.add_argument('--ac-path', default='', help='override the AC "online" file (failure-path tests only)')
    args = ap.parse_args()
    if args.sim != 'gazebo':
        sys.exit(f'--sim {args.sim}: only gazebo exists so far (isaac comes in B0)')
    r = Runner(args)
    code = r.run()
    if code == 0 and not args.no_report:
        if sorted(r.seeds) == sorted(r.cfg['seeds']) and not args.sim_cap and not args.ac_path:
            import report
            report.append_row(r.row)
            r.log(f'row {r.row} appended to results/results.md')
        else:
            r.log('not the benchmark seed set (or a test cap): no row written')
    r.write_status(done=code == 0, current=None)
    sys.exit(code)


if __name__ == '__main__':
    main()
