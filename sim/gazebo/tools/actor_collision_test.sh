#!/bin/bash
# Checks that the walking person (Gazebo actor) has no collision: places a
# 2 kg box on the person's first leg in scene 3 and shows that it doesn't
# move after the person has walked through it (~11 s).
#   actor_collision_test.sh [seed=3]   (seed must have a person: odd seeds)
set -e
source "$(dirname "$0")/common.sh"
SEED=${1:-3}
cd "$REPO"
python3 sim/gazebo/load_scene.py --seed "$SEED" > /dev/null
python3 - "$SEED" "$OUT/actor_test.sdf" <<'PY'
import sys, math, yaml
sys.path.insert(0, 'sim/common'); import terrain as T
seed, out = int(sys.argv[1]), sys.argv[2]
s = [s for s in yaml.safe_load(open('bench/scenes.yaml'))['scenes'] if s['seed'] == seed][0]
p = s['person']; (x0, y0), (x1, y1) = p['waypoints'][:2]
bx = 5.0; by = y0 + (bx - x0) / (x1 - x0) * (y1 - y0)
h = float(T.height_at(s['terrain'], bx, by))
print(f'box at ({bx:.2f}, {by:.2f}); person reaches it at t = {math.hypot(bx - x0, by - y0) / p["speed"]:.1f} s')
box = (f'<model name="actor_probe"><pose>{bx} {by} {h + 0.16} 0 0 0</pose><link name="link">'
       '<inertial><mass>2</mass><inertia><ixx>0.03</ixx><iyy>0.03</iyy><izz>0.03</izz></inertia></inertial>'
       '<collision name="c"><geometry><box><size>0.3 0.3 0.3</size></box></geometry></collision>'
       '<visual name="v"><geometry><box><size>0.3 0.3 0.3</size></box></geometry></visual></link></model>\n  </world>')
open(out, 'w').write(open(f'sim/gazebo/worlds/generated/scene_{seed}.sdf').read().replace('</world>', box))
PY
gz sim -s -r "$OUT/actor_test.sdf" > "$OUT/actor_test.log" 2>&1 &
PID=$!
sleep 5
for _ in $(seq 8); do
  timeout 8 gz topic -e -t /world/beach/pose/info -n 1 | (cd "$TOOLS" && python3 -c "
import sys, gz_poses
t, p = gz_poses.parse(sys.stdin.read())
print('t=%5.1f s  box at (%.3f, %.3f, %.3f)' % ((t,) + p['actor_probe']))")
  sleep 3
done
stop_gz $PID
