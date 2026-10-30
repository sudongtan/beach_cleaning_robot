#!/usr/bin/env bash
# Run the simulation (bringup sim.launch.py sim:=gazebo) headless, isolated from other
# Gazebo/ROS instances on this machine,
# wait until all controllers are active, run a command against it, then shut everything down.
#   sim/gazebo/run_headless.sh [launch args ...] -- <command ...>
#   e.g. sim/gazebo/run_headless.sh basket_load_kg:=2.0 -- python3 drive_test.py --stack legubiao ...
# Launch output goes to $LOG (default: /tmp/run_headless_<pid>.log).
LAUNCH_ARGS=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do LAUNCH_ARGS+=("$1"); shift; done
shift
source ~/ros2_ws/install/setup.bash
export GZ_PARTITION=headless_$$
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID_HEADLESS:-44}
LOG=${LOG:-/tmp/run_headless_$$.log}

ros2 launch bringup sim.launch.py sim:=gazebo headless:=true "${LAUNCH_ARGS[@]}" > "$LOG" 2>&1 &
LPID=$!
descendants() { for c in $(ps -o pid= --ppid "$1"); do echo "$c"; descendants "$c"; done; }
cleanup() {
  local tree
  tree=$(descendants $LPID)   # only this run's processes, never other sessions' sims
  kill -INT $LPID 2>/dev/null
  for _ in $(seq 1 20); do kill -0 $LPID 2>/dev/null || break; sleep 0.5; done
  kill -KILL $LPID $tree 2>/dev/null
}
trap cleanup EXIT

for i in $(seq 1 60); do
  [ "$(ros2 control list_controllers 2>/dev/null | grep -c ' active')" -ge 5 ] && break
  kill -0 $LPID 2>/dev/null || { echo "launch died, see $LOG"; exit 1; }
  sleep 2
done
echo "[run_headless] controllers active after ~$((i * 2)) s (log: $LOG)"
sleep 3
"$@"
