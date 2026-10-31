#!/bin/bash
# Real-time factor per seed: sim-time advance over 30 s of wall time, robot standing
# (headless, isolated partition/domain).   rtf_measure.sh <seeds...>
TOOLS=$(cd "$(dirname "$0")" && pwd)
source "$TOOLS/scene_env.sh"
simt() { timeout 5 gz topic -e -t /world/beach/stats -n 1 | python3 -c "import sys,re;t=sys.stdin.read();s=re.search(r'sim_time \{\s*sec: (\d+)(?:\s*nsec: (\d+))?',t);print(int(s.group(1))+int(s.group(2) or 0)*1e-9)"; }
for s in "$@"; do
  "$TOOLS/scene_up.sh" "$s" >/dev/null 2>&1 || { echo "seed $s: startup failed"; "$TOOLS/scene_down.sh"; continue; }
  sleep 5
  t0=$(date +%s.%N); s0=$(simt); sleep 30; t1=$(date +%s.%N); s1=$(simt)
  echo "seed $s: RTF $(python3 -c "print(round(($s1-$s0)/($t1-$t0),3))") load $(cut -d' ' -f1 /proc/loadavg) freq $(($(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq)/1000)) MHz"
  "$TOOLS/scene_down.sh" >/dev/null 2>&1
done
