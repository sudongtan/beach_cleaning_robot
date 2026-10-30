# Shared helpers for the tool scripts. Source it, don't run it.
source /opt/ros/jazzy/setup.bash
export GZ_PARTITION=${GZ_PARTITION:-a1b}
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$TOOLS/../../.." && pwd)
OUT=${OUT:-$REPO/sim/gazebo/worlds/generated/tools}   # git-ignored
mkdir -p "$OUT"

# sim_seconds: current sim time (whole seconds) from /world/beach/stats
sim_seconds() {
  timeout 5 gz topic -e -t /world/beach/stats -n 1 | python3 -c \
    "import sys,re; m=re.search(r'sim_time \{\s*sec: (\d+)', sys.stdin.read()); print(m.group(1) if m else 0)"
}

# stop_gz PID: gz sim -s sometimes ignores SIGINT/SIGTERM on shutdown
stop_gz() {
  kill -INT "$1" 2>/dev/null
  for _ in $(seq 20); do kill -0 "$1" 2>/dev/null || return 0; sleep 1; done
  kill -9 "$1" 2>/dev/null
}
