#!/bin/bash
# Turn-then-walk cycles in a fresh scene, through the safety supervisor (A2b problem E).
#   turn_walk_test.sh <seed> [turn_walk_test.py args...]
#   e.g. turn_walk_test.sh 3 --mode combined --cycles 10
# Cycles start at the spawn point (seed 3: dry flat sand) and stay within ~1 m of it.
# Writes $OUT/turnwalk_<seed>_<mode>.{launch,test}.log.
set -e
TOOLS=$(cd "$(dirname "$0")" && pwd)
source "$TOOLS/scene_env.sh"
SEED=$1; shift
MODE=$(echo "$@" | grep -o 'combined\|separate' || echo combined)
V=$OUT/turnwalk_${SEED}_${MODE}
MANIFEST=$REPO/sim/gazebo/worlds/generated/scene_${SEED}.manifest.yaml
[ -f "$MANIFEST" ] || (cd "$REPO" && make -s sim/gazebo/worlds/generated/scene_${SEED}.sdf SEED=$SEED > /dev/null)
"$TOOLS/stop_sim.sh" > /dev/null 2>&1 || true
SAFETY=$(ros2 pkg prefix safety)/share/safety/config/safety.yaml
setsid bash -c "
  ros2 launch bringup sim.launch.py scene:=$MANIFEST headless:=true publish_odom_tf:=false &
  ros2 run localization ground_truth --ros-args -p use_sim_time:=true &
  ros2 run mapping ground_truth --ros-args -p repo:=$REPO -p seed:=$SEED -p use_sim_time:=true &
  ros2 run safety supervisor --ros-args --params-file $SAFETY -p use_sim_time:=true &
  wait" > "$V.launch.log" 2>&1 &
for _ in $(seq 120); do sleep 2; grep -q "stand-up done" "$V.launch.log" && break; done
sleep 5
python3 "$TOOLS/turn_walk_test.py" "$@" > "$V.test.log" 2>&1 || true
grep -h "FALL:" "$V.launch.log" | head -1 | sed 's/^/   adapter: /' || true
"$TOOLS/stop_sim.sh" > /dev/null 2>&1 || true
grep RESULT "$V.test.log" | tail -1
