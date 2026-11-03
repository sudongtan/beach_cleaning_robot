#!/bin/bash
# Long Nav2 walks for the fall rate (A2b problem E): the full skeleton (make run), with the task,
# perception, world model, classifier and grasping stopped (by PID), then nav_walk_test.py.
#   nav_walk_test.sh <seed> <label> [nav_walk_test.py args...]
#   e.g. nav_walk_test.sh 3 base --minutes 60
# Own GZ_PARTITION / ROS_DOMAIN_ID (navwalk / 71). Writes $OUT/navwalk_<seed>_<label>.{launch,test}.log;
# env RECORD=1 also records the command chain to .cmd.jsonl (cmd_record.py); TOOL: another test
# script on the same stack (e.g. store_drop_test.py).
TOOLS=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$TOOLS/../../.." && pwd)
OUT=${OUT:-$REPO/sim/gazebo/worlds/generated/tools}
mkdir -p "$OUT"
SEED=$1; LABEL=$2; shift 2
V=$OUT/navwalk_${SEED}_${LABEL}
source /opt/ros/jazzy/setup.bash && source "$HOME/ros2_ws/install/setup.bash"
export GZ_PARTITION=navwalk ROS_DOMAIN_ID=71 GZ_IP=127.0.0.1 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST BEACH_REPO=$REPO
"$TOOLS/stop_sim.sh" > /dev/null
cd "$REPO"
(setsid bash -c "exec make run SEED=$SEED HEADLESS=true GZ_PARTITION=navwalk RUN_ID=navwalk_$LABEL" > "$V.launch.log" 2>&1 &)
for _ in $(seq 120); do sleep 2; grep -q "stand-up done" "$V.launch.log" && break; done
sleep 5
# the task would send its own goals; perception and the arm only cost CPU here
for exe in task/lib/task/task perception/lib/perception/ground_truth_detector \
           perception/lib/perception/ground_truth_classifier world_model/lib/world_model/world_model \
           grasping/lib/grasping/grasping; do
  ps -eo pid=,args= | awk -v e="install/$exe" '$3 ~ e"$" {print $1}' | xargs -r kill
done
sleep 3
ros2 service call /nav/goal/_action/cancel_goal action_msgs/srv/CancelGoal "{}" > /dev/null
[ -n "$RECORD" ] && (python3 "$TOOLS/cmd_record.py" 20000 "$V.cmd.jsonl" > /dev/null 2>&1 &)
python3 "$TOOLS/${TOOL:-nav_walk_test.py}" --seed "$SEED" "$@" > "$V.test.log" 2>&1
ps -eo pid=,args= | awk -v f="$V.cmd.jsonl" '$3 ~ /cmd_record.py$/ && $5 == f {print $1}' | xargs -r kill
grep -h "FALL:" "$V.launch.log" | head -1 | sed 's/^/   adapter: /'
grep -h "STUCK" "$V.test.log" | head -1
"$TOOLS/stop_sim.sh" > /dev/null
grep RESULT "$V.test.log" | tail -1
