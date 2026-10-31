#!/bin/bash
# Walk a lane pattern in one terrain zone of a scene, through the safety supervisor (A2b:
# why does the robot fall in the wet zone?). One fresh simulation per walk: after a fall the
# locomotion adapter keeps the legs passive.
#   zone_walk_test.sh <seed> <label> <x0> <x1> <y_a> <y_b> [lanes]
#   e.g. zone_walk_test.sh 3 wet 6 14 5.5 7.5 4     -> lanes x 6 <-> 14 alternating at y 5.5 / 7.5
# env: SMOOTH (default "--accel 0.3 --ang-accel 0.5", like Nav2's smoother), SPEED (default 0.3),
#      ARM_POSE (named arm pose set before walking, e.g. up or rest; default: as spawned)
# Writes $OUT/zone_<seed>_<label>.{launch,walk}.log; prints the RESULT line.
set -e
TOOLS=$(cd "$(dirname "$0")" && pwd)
source "$TOOLS/scene_env.sh"
SEED=$1; LABEL=$2; X0=$3; X1=$4; YA=$5; YB=$6; LANES=${7:-4}
V=$OUT/zone_${SEED}_${LABEL}
MANIFEST=$REPO/sim/gazebo/worlds/generated/scene_${SEED}.manifest.yaml
[ -f "$MANIFEST" ] || (cd "$REPO" && make -s sim/gazebo/worlds/generated/scene_${SEED}.sdf SEED=$SEED > /dev/null)
"$TOOLS/scene_down.sh" > /dev/null 2>&1 || true
SAFETY=$(ros2 pkg prefix safety)/share/safety/config/safety.yaml
setsid bash -c "
  ros2 launch bringup sim.launch.py scene:=$MANIFEST headless:=true publish_odom_tf:=false &
  ros2 run localization ground_truth --ros-args -p use_sim_time:=true &
  ros2 run mapping ground_truth --ros-args -p repo:=$REPO -p seed:=$SEED -p use_sim_time:=true &
  ros2 run safety supervisor --ros-args --params-file $SAFETY -p use_sim_time:=true &
  wait" > "$V.launch.log" 2>&1 &
echo $! > "$OUT/scene.pid"
for _ in $(seq 120); do sleep 2; grep -q "stand-up done" "$V.launch.log" && break; done
sleep 3
[ -n "$ARM_POSE" ] && python3 "$TOOLS/arm_pose.py" "$ARM_POSE" > "$V.arm.log" 2>&1
WPS=""
for k in $(seq 0 $((LANES - 1))); do
  y=$([ $((k % 2)) -eq 0 ] && echo "$YA" || echo "$YB")
  if [ $((k % 2)) -eq 0 ]; then WPS="$WPS $X0,$y $X1,$y"; else WPS="$WPS $X1,$y $X0,$y"; fi
done
python3 "$TOOLS/goto.py" $WPS --seed "$SEED" --speed "${SPEED:-0.3}" --timeout ${WALK_TIMEOUT:-900} \
  ${SMOOTH:---accel 0.3 --ang-accel 0.5} > "$V.walk.log" 2>&1 || true
grep -h "FALL:" "$V.launch.log" | head -1 | sed 's/^/   adapter: /' || true
"$TOOLS/scene_down.sh" > /dev/null 2>&1 || true
echo "zone $LABEL (seed $SEED, x $X0..$X1, y $YA/$YB, arm ${ARM_POSE:-as spawned}): $(grep RESULT "$V.walk.log" | tail -1)"
