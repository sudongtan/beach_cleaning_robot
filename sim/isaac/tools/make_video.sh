#!/usr/bin/env bash
# Frames from run_d0.py --frames (BasicWriter rgb_*.png under DIR) -> an H.264 MP4.
#   bash sim/isaac/tools/make_video.sh DIR OUT.mp4 [FPS]
# FPS = 1 / the run's --capture-every (10 for 0.1 s), so the video plays at sim speed.
set -euo pipefail
dir=$(cd "$1" && pwd); out=$2; fps=${3:-10}   # absolute: concat resolves paths from the list file
list=$(mktemp)
find "$dir" -name 'rgb_*.png' | sort | sed "s/^/file '/; s/\$/'/" > "$list"
n=$(wc -l < "$list"); [ "$n" -gt 0 ] || { echo "no rgb_*.png under $dir" >&2; exit 1; }
ffmpeg -y -loglevel error -r "$fps" -f concat -safe 0 -i "$list" -c:v libx264 -pix_fmt yuv420p -crf 23 \
  -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" "$out"
rm -f "$list"
echo "$out: $n frames at $fps fps"
