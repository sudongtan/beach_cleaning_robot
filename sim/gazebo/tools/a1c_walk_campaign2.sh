#!/bin/bash
# A1c item 3, second part (after the Wi-Fi drop invalidated runs): smoothed climbs at
# 7 deg with 2 kg (x2), 1.25 kg and 0 kg loads, and one unsmoothed 7 deg / 2 kg run.
TOOLS=$(cd "$(dirname "$0")" && pwd)
export SMOOTH="--accel 0.3 --ang-accel 0.5"
for r in 1 2; do LOAD=2.0 "$TOOLS/slope_test.sh" 9 7.0 | head -1; done
for d in 5.0 6.0 7.0; do LOAD=1.25 "$TOOLS/slope_test.sh" 9 $d | head -1; done
for d in 5.0 6.0 7.0; do LOAD=0.0 "$TOOLS/slope_test.sh" 9 $d | head -1; done
SMOOTH= LOAD=2.0 "$TOOLS/slope_test.sh" 9 7.0 | head -1
