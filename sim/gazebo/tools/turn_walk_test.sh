#!/bin/bash
# Turn-then-walk cycles in a fresh scene, through the safety supervisor (A2b problem E).
#   turn_walk_test.sh <seed> [turn_walk_test.py args...]
#   e.g. turn_walk_test.sh 3 --mode combined --cycles 10
# Cycles start at the spawn point (seed 3: dry flat sand), or where env GOTO="x,y [x,y ...]" walks
# the robot first (goto.py through the supervisor, smoothed), and stay within ~1 m of it.
# env LABEL: log name suffix (default: the mode); TOOL: test script instead of turn_walk_test.py
# (e.g. cmd_replay.py, then its arguments); LOCO: locomotion:= (default legubiao).
# Writes $OUT/turnwalk_<seed>_<mode>.{launch,test}.log.
set -e
TOOLS=$(cd "$(dirname "$0")" && pwd)
source "$TOOLS/scene_env.sh"
SEED=$1; shift
MODE=$(echo "$@" | grep -o 'combined\|separate' || echo combined)
V=$OUT/turnwalk_${SEED}_${LABEL:-$MODE}
MANIFEST=$REPO/sim/gazebo/worlds/generated/scene_${SEED}.manifest.yaml
[ -f "$MANIFEST" ] || (cd "$REPO" && make -s sim/gazebo/worlds/generated/scene_${SEED}.sdf SEED=$SEED > /dev/null)
"$TOOLS/stop_sim.sh" > /dev/null 2>&1 || true
SAFETY=$(ros2 pkg prefix safety)/share/safety/config/safety.yaml
setsid bash -c "
  ros2 launch bringup sim.launch.py scene:=$MANIFEST headless:=true publish_odom_tf:=false locomotion:=${LOCO:-legubiao} &
  ros2 run localization ground_truth --ros-args -p use_sim_time:=true &
  ros2 run mapping ground_truth --ros-args -p repo:=$REPO -p seed:=$SEED -p use_sim_time:=true &
  ros2 run safety supervisor --ros-args --params-file $SAFETY -p use_sim_time:=true &
  wait" > "$V.launch.log" 2>&1 &
for _ in $(seq 120); do sleep 2; grep -q "stand-up done" "$V.launch.log" && break; done
sleep 5
[ -n "$GOTO" ] && python3 "$TOOLS/goto.py" $GOTO --seed "$SEED" --accel 0.3 --ang-accel 0.5 > "$V.goto.log" 2>&1
python3 "$TOOLS/${TOOL:-turn_walk_test.py}" "$@" > "$V.test.log" 2>&1 || true
grep -h "FALL:" "$V.launch.log" | head -1 | sed 's/^/   adapter: /' || true
"$TOOLS/stop_sim.sh" > /dev/null 2>&1 || true
grep RESULT "$V.test.log" | tail -1
