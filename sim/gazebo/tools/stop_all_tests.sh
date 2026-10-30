#!/bin/bash
# Stop every test campaign started from sim/gazebo/tools (loops, drivers, scenes),
# on both test partitions. Kills by PID, never its own process tree.
TOOLS=$(cd "$(dirname "$0")" && pwd)
self=$$
for pat in "a1c_walk_campaign"  "slope_test.sh" "physics_grasp_run.sh" "physics_grasp_test.py" "tools/goto.py" "turn_test.py" "startup_all.sh"; do
  for p in $(pgrep -f "$pat"); do
    [ "$p" = "$self" ] && continue
    grep -q "stop_all_tests" /proc/$p/cmdline 2>/dev/null && continue
    kill "$p" 2>/dev/null
  done
done
sleep 2
"$TOOLS/scene_down.sh"
OUT=$TOOLS/../worlds/generated/tools/pgrasp SCENE_GZ_PARTITION=pgrasp SCENE_ROS_DOMAIN_ID=44 "$TOOLS/scene_down.sh"
