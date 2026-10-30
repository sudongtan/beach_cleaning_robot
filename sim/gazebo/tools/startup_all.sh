#!/bin/bash
# Start every benchmark seed once (headless): did the items detach, did the robot
# stand up, real-time factor and camera rate (wall clock) while standing.
#   startup_all.sh [seeds...]      default 0..9
TOOLS=$(cd "$(dirname "$0")" && pwd)
source "$TOOLS/scene_env.sh"
SEEDS=${*:-0 1 2 3 4 5 6 7 8 9}
for s in $SEEDS; do
  if ! "$TOOLS/scene_up.sh" "$s" > /dev/null 2>&1; then
    echo "seed $s: STARTUP FAILED (log $OUT/scene_$s.log)"; "$TOOLS/scene_down.sh"; continue
  fi
  det=$(grep -o "detached all [0-9]* items after startup ([0-9.]* s)" "$OUT/scene_$s.log")
  rtf=$(for i in 1 2 3 4 5; do timeout 5 gz topic -e -t /world/beach/stats -n 1 | grep "^real_time_factor" | cut -d' ' -f2 | cut -c1-4; sleep 1; done | tr '\n' ' ')
  cam=$(timeout 15 ros2 topic hz /camera/rgb --window 20 2>/dev/null | grep average | tail -1 | awk '{print $3}')
  echo "seed $s: $det; stand-up done; RTF $rtf; /camera/rgb ${cam} Hz (wall); load $(cut -d' ' -f1 /proc/loadavg)"
  "$TOOLS/scene_down.sh"
done
