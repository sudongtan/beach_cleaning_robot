#!/bin/bash
# Run a scene headless for ~70 s wall time: print the real-time factor every
# 8 s, then check that every item settled where it spawned.
#   settle_check.sh <seed>        (builds the world first)
set -e
source "$(dirname "$0")/common.sh"
SEED=${1:?seed}
cd "$REPO"
python3 sim/gazebo/load_scene.py --seed "$SEED" > /dev/null
SDF=sim/gazebo/worlds/generated/scene_$SEED.sdf
gz sim -s -r --headless-rendering "$SDF" > "$OUT/settle_$SEED.log" 2>&1 &
PID=$!
sleep 15
for _ in 1 2 3 4 5 6; do
  sleep 8
  timeout 5 gz topic -e -t /world/beach/stats -n 1 | python3 -c "import sys,re; t=sys.stdin.read(); print('rtf', re.search(r'real_time_factor: ([\d.]+)',t).group(1)[:5], 'sim_s', re.search(r'sim_time \{\s*sec: (\d+)',t).group(1))"
done
echo "load average: $(cut -d' ' -f1-3 /proc/loadavg)"
timeout 10 gz topic -e -t /world/beach/pose/info -n 1 > "$OUT/settle_poses_$SEED.txt"
stop_gz $PID
python3 "$TOOLS/compare_poses.py" "$SDF" "$OUT/settle_poses_$SEED.txt" | grep -E "large|MISSING|worst|sim time"
