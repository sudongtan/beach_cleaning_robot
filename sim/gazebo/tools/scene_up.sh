#!/bin/bash
# Start `make scene` headless in the background on its own Gazebo partition and
# ROS domain (isolated from anything else running), and wait until the robot has
# finished standing up.
#   scene_up.sh <seed> [extra make args]     -> writes $OUT/scene.pid, log $OUT/scene_<seed>.log
#   then:  source scene_env.sh   (sets GZ_PARTITION / ROS_DOMAIN_ID for ros2/gz commands)
#   stop:  scene_down.sh
set -e
TOOLS=$(cd "$(dirname "$0")" && pwd)
source "$TOOLS/scene_env.sh"
SEED=${1:?seed}; shift || true
LOG=$OUT/scene_$SEED.log
"$TOOLS/scene_down.sh" >/dev/null 2>&1 || true
cd "$REPO"
setsid make scene SEED="$SEED" HEADLESS=${HEADLESS:-true} GZ_PARTITION="$GZ_PARTITION" "$@" > "$LOG" 2>&1 &
echo $! > "$OUT/scene.pid"
for _ in $(seq 120); do
  sleep 2
  grep -q "stand-up done" "$LOG" && break
  if grep -q -E "Error 1|Traceback|still attached" "$LOG"; then echo "startup failed, see $LOG"; exit 1; fi
done
sleep 3
echo "scene $SEED up (log: $LOG)"
