#!/bin/bash
# Start `make bench` detached from this shell / agent session, with the laptop kept awake
# (sleep, idle and the lid switch blocked for exactly as long as the bench runs; brief A3).
#   bench/start_detached.sh [make args, e.g. ROW=baseline SEEDS=0-9]
# Log: results/bench_<row>.log; progress: results/bench/<row>/status.json
REPO=$(cd "$(dirname "$0")/.." && pwd)
ROW=baseline
for a in "$@"; do case $a in ROW=*) ROW=${a#ROW=};; esac; done
cd "$REPO"
setsid nohup systemd-inhibit --what=sleep:idle:handle-lid-switch --mode=block --who=beach-bench \
  --why="make bench (overnight)" make bench "$@" > "results/bench_$ROW.log" 2>&1 < /dev/null &
sleep 3
echo "started (log results/bench_$ROW.log)"
systemd-inhibit --list --no-pager | grep beach-bench || { echo "WARNING: no beach-bench inhibitor lock"; exit 1; }
