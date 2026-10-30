#!/bin/bash
# Physics-grasp feasibility run (A1c item 7): custom robot on flat ground (world:=empty),
# headless, on its own Gazebo partition and ROS domain. Extra args go to physics_grasp_test.py.
#   physics_grasp_run.sh --items all --trials 10 --out results/physics_grasp/baseline.json
# env: LAUNCH_ARGS="..." extra launch arguments (e.g. a gripper friction override)
set -e
TOOLS=$(cd "$(dirname "$0")" && pwd)
export SCENE_GZ_PARTITION=${SCENE_GZ_PARTITION:-pgrasp} SCENE_ROS_DOMAIN_ID=${SCENE_ROS_DOMAIN_ID:-44}
# own OUT dir: scene_down.sh stops the process group in $OUT/scene.pid, keep other runs alive
export OUT=${OUT:-$TOOLS/../worlds/generated/tools/pgrasp}
source "$TOOLS/scene_env.sh"
LOG=$OUT/physics_grasp_launch.log
"$TOOLS/scene_down.sh"
setsid ros2 launch bringup sim.launch.py world:=empty headless:=true variant:=custom $LAUNCH_ARGS > "$LOG" 2>&1 &
echo $! > "$OUT/scene.pid"
for _ in $(seq 120); do sleep 2; grep -q "stand-up done" "$LOG" && break; done
sleep 3
cd "$REPO"
python3 "$TOOLS/physics_grasp_test.py" "$@" || true
"$TOOLS/scene_down.sh"
