#!/bin/bash
# Drop probe boxes on a scene's heightmap and compare their resting height
# with terrain.height_at. "rest" should match "terrain" within ~1 cm inside
# the area (y 5-28); probes on the steep back dunes (y >= 32) slide down.
#   run_heightmap_probe.sh [detector=ode] [seed=3]
set -e
source "$(dirname "$0")/common.sh"
DET=${1:-ode}; SEED=${2:-3}
python3 "$TOOLS/heightmap_probe.py" --detector "$DET" --seed "$SEED" \
  --out "$OUT/probe_$DET.sdf" --expected "$OUT/probe_expected.txt"
gz sim -s -r "$OUT/probe_$DET.sdf" > "$OUT/probe_$DET.log" 2>&1 &
PID=$!
for _ in $(seq 60); do sleep 3; [ "$(sim_seconds)" -ge 5 ] && break; done
timeout 8 gz topic -e -t /world/beach/pose/info -n 1 > "$OUT/probe_$DET.txt"
stop_gz $PID
cd "$TOOLS" && python3 - "$OUT/probe_$DET.txt" "$OUT/probe_expected.txt" <<'PY'
import sys, gz_poses
t, got = gz_poses.parse(open(sys.argv[1]).read())
print(f'sim time {t:.1f} s')
for line in open(sys.argv[2]):
    name, x, y, h = line.split()
    z = got.get(name, (0, 0, float('nan')))[2] - 0.05      # box centre -> bottom
    print(f'x={x:>5} y={y:>5}  rest={z:+8.3f}  terrain={float(h):+.3f}  diff={z - float(h):+.3f}')
PY
