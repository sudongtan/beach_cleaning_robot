#!/bin/bash
# Stop what scene_up.sh started (the whole process group), and any Gazebo
# server left on the tools partition.
TOOLS=$(cd "$(dirname "$0")" && pwd)
source "$TOOLS/scene_env.sh"
if [ -f "$OUT/scene.pid" ]; then
  PGID=$(cat "$OUT/scene.pid")
  kill -INT -- -"$PGID" 2>/dev/null
  for _ in $(seq 15); do kill -0 -- -"$PGID" 2>/dev/null || break; sleep 1; done
  kill -9 -- -"$PGID" 2>/dev/null
  rm -f "$OUT/scene.pid"
fi
# gz sim sometimes ignores SIGINT on shutdown: kill servers on our partition
for p in $(pgrep -f "gz sim"); do
  tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "GZ_PARTITION=$GZ_PARTITION" && kill -9 "$p"
done
true
