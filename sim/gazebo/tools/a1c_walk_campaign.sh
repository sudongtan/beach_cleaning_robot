#!/bin/bash
# A1c item 3: walking checks with the custom robot. Abrupt walk->turn on flat ground with the
# full load, then the steepest line of seed 9 with the dunes scaled, with velocity smoothing,
# at three loads. Prints one result line per run.
TOOLS=$(cd "$(dirname "$0")" && pwd)
source "$TOOLS/scene_env.sh"
"$TOOLS/scene_down.sh"
setsid ros2 launch bringup sim.launch.py world:=empty headless:=true basket_load_kg:=2.0 > "$OUT/turn_flat.log" 2>&1 &
echo $! > "$OUT/scene.pid"
for _ in $(seq 120); do sleep 2; grep -q "stand-up done" "$OUT/turn_flat.log" && break; done
sleep 3
echo "turn test (custom, 2 kg, flat): $(python3 "$TOOLS/turn_test.py" --cycles 3 | tail -1)"
"$TOOLS/scene_down.sh"
export SMOOTH="--accel 0.3 --ang-accel 0.5"
for r in 1 2; do for d in 5.0 6.0 7.0; do LOAD=2.0 "$TOOLS/slope_test.sh" 9 $d | head -1; done; done
for d in 5.0 6.0; do LOAD=1.25 "$TOOLS/slope_test.sh" 9 $d | head -1; done
for d in 5.0 6.0 7.0; do LOAD=0.0 "$TOOLS/slope_test.sh" 9 $d | head -1; done
