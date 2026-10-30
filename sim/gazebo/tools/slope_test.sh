#!/bin/bash
# Climb the steepest line of a seed with its dunes scaled to TARGET degrees.
#   slope_test.sh <seed> <target_deg>      (0 = dunes as generated)
# env: LOAD=2.0 (basket load, kg; default the full 2 kg), VARIANT=custom|stock (default custom),
#      SMOOTH="--accel 0.3 --ang-accel 0.5" (velocity smoothing for goto.py; default none)
set -e
TOOLS=$(cd "$(dirname "$0")" && pwd)
source "$TOOLS/scene_env.sh"
SEED=$1; TGT=$2; V=$OUT/slope_${SEED}_${TGT}
read -r _ LINE < <(python3 "$TOOLS/slope_variant.py" "$SEED" "$TGT" "$V.yaml" | grep LINE)
python3 "$REPO/sim/gazebo/load_scene.py" --seed "$SEED" --scenes "$V.yaml" --out "$V.sdf" \
  --variant "${VARIANT:-custom}" > /dev/null
"$TOOLS/scene_down.sh"
setsid ros2 launch bringup sim.launch.py scene:="$V.manifest.yaml" headless:=true \
  basket_load_kg:="${LOAD:-2.0}" > "$V.launch.log" 2>&1 &
echo $! > "$OUT/scene.pid"
for _ in $(seq 120); do sleep 2; grep -q "stand-up done" "$V.launch.log" && break; done
sleep 3
python3 "$TOOLS/goto.py" $LINE $(echo $LINE | awk "{print \$2, \$1}") --seed "$SEED" --scene "$V.yaml" --timeout ${WALK_TIMEOUT:-900} $SMOOTH > "$V.walk.log" 2>&1 || true
"$TOOLS/scene_down.sh"
echo "slope ${TGT} deg (${VARIANT:-custom}, load ${LOAD:-2.0} kg${SMOOTH:+, smoothed}): $(tail -1 "$V.walk.log")"
grep -E "slope= ?[6-9]\.|slope=1[0-9]" "$V.walk.log" | awk '{print "   ", $0}' | cut -c1-125
