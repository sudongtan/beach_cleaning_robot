#!/usr/bin/env bash
# REQUIRES_GPU
# D0b + D0c on ONE GPU instance (notes/briefs/D0.md): import the robot, stand + walk (flat, seed 3),
# the attach-mode picks, then frame sequences. Inside the Isaac Sim 6.2.0 container, from the
# uploaded repo root:
#
#   BEACH_D0_LAPTOP_DIR=/workspace/d0_from_laptop bash sim/isaac/tools/gpu_session_d0.sh /workspace/d0
#
# Restartable: each step writes $OUT/.done/<step>; delete a marker to rerun a step. Results are
# judged from the JSON files each step writes (SimulationApp.close() exits 0 itself: D0 prep).
# Walk / pick failures don't stop the session: the next steps still run, so one session collects
# everything. Only a failed import stops it (nothing after it can work).

set -uo pipefail

# vast.ai ssh shells don't inherit the container's --env (D0 prep): take the Isaac flags from PID 1
INIT_ENVIRON=${INIT_ENVIRON:-/proc/1/environ}
for v in ACCEPT_EULA PRIVACY_CONSENT OMNI_KIT_ALLOW_ROOT NVIDIA_DRIVER_CAPABILITIES; do
    if [ -z "${!v:-}" ] && [ -r "$INIT_ENVIRON" ]; then
        val=$(tr '\0' '\n' < "$INIT_ENVIRON" | sed -n "s/^$v=//p" | head -1)
        [ -n "$val" ] && export "$v=$val"
    fi
done

if [ $# -lt 1 ]; then echo "usage: $0 OUT" >&2; exit 2; fi
OUT=$(realpath -m "$1")
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
ISAAC=${ISAAC_ROOT:-/isaac-sim}
PY="$ISAAC/python.sh"
export BEACH_D0_LAPTOP_DIR=${BEACH_D0_LAPTOP_DIR:-/workspace/d0_from_laptop}
L="$BEACH_D0_LAPTOP_DIR"
ROBOT_DIR="$REPO/sim/isaac/robots/generated"
STAGES="$REPO/sim/isaac/stages/generated"
mkdir -p "$OUT/.done" "$STAGES"
exec > >(tee -a "$OUT/session.log") 2>&1

stamp() { echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
done_() { [ -f "$OUT/.done/$1" ]; }
mark() { date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/.done/$1"; }
fail() { stamp "FAILED: $*"; exit 1; }
# ok FILE -> 0 if FILE's JSON has "ok": true
ok() { "$PY" -c "import json,sys; sys.exit(0 if json.load(open('$1')).get('ok') is True else 1)" 2>/dev/null; }
field() { "$PY" -c "import json; d=json.load(open('$1')); print(d$2)" 2>/dev/null; }

stamp "D0 session start; repo $REPO; laptop files $L"

if ! done_ host; then
    stamp "step 1: host facts, Vulkan, compatibility check"
    { echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
      free -g; nproc; cat "$ISAAC/VERSION"; } > "$OUT/host.txt" 2>&1
    cat "$OUT/host.txt"
    command -v vulkaninfo >/dev/null || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq vulkan-tools; }
    vulkaninfo --summary > "$OUT/vulkaninfo.txt" 2>&1
    grep -qi nvidia "$OUT/vulkaninfo.txt" || fail "vulkaninfo lists no NVIDIA device"
    (cd "$ISAAC" && timeout 900 ./isaac-sim.compatibility_check.sh --/app/quitAfter=10 --no-window) > "$OUT/compat.txt" 2>&1
    grep -q 'System checking result: PASSED' "$OUT/compat.txt" || fail "compatibility check did not pass"
    mark host
fi

if ! done_ laptop; then
    stamp "step 2: laptop files present and unchanged (MANIFEST md5s)"
    [ -f "$L/MANIFEST.txt" ] || fail "no $L/MANIFEST.txt: upload d0_from_laptop first"
    (cd "$L" && grep -E '^[0-9a-f]{32}  \./' MANIFEST.txt | md5sum -c -) || fail "laptop files differ from MANIFEST.txt"
    mark laptop
fi

if ! done_ import; then
    stamp "step 3: import the URDF (isaacsim.asset.importer.urdf)"
    rm -rf "$ROBOT_DIR"
    timeout 1200 "$PY" "$REPO/sim/isaac/import_robot.py" --out "$ROBOT_DIR"
    mkdir -p "$OUT/import"
    cp "$ROBOT_DIR/import.json" "$OUT/import/" 2>/dev/null
    ok "$ROBOT_DIR/import.json" || fail "import failed: see $OUT/import/import.json"
    mark import
fi
ROBOT_USD=$(field "$ROBOT_DIR/import.json" "['usd']")
[ -f "$ROBOT_USD" ] || fail "imported USD not found: $ROBOT_USD"

if ! done_ check; then
    stamp "step 4: check the robot USD against the URDF, write the drives"
    timeout 600 "$PY" "$REPO/sim/isaac/check_robot_usd.py" --usd "$ROBOT_USD" --apply-drives \
        --report "$OUT/import/robot_check.json"
    ok "$OUT/import/robot_check.json" || stamp "robot check has errors (continuing: see robot_check.json)"
    (cd "$ROBOT_DIR" && tar czf "$OUT/import/robot_usd.tgz" .)          # the USD for the Mac tests
    mark check
fi

if ! done_ compose; then
    stamp "step 5: compose the test stages"
    for spec in "flat walk" "3 walk" "flat pick"; do
        set -- $spec
        extra=""; [ "$2" = pick ] && extra="--pick"
        timeout 600 "$PY" "$REPO/sim/isaac/compose.py" --scene "$1" --robot "$ROBOT_USD" \
            --out "$STAGES/stage_$1_$2.usda" $extra || fail "compose $spec"
        [ -f "$STAGES/stage_$1_$2.usda.json" ] || fail "compose $spec wrote no metadata"
    done
    mark compose
fi

run_d0() {  # run_d0 NAME STAGE MODE [args] -> $OUT/NAME/result.json
    local name=$1 stage=$2 mode=$3; shift 3
    if done_ "$name"; then return 0; fi
    stamp "run_d0 $name ($mode $*)"
    timeout 1800 "$PY" "$REPO/sim/isaac/run_d0.py" --stage "$STAGES/$stage" --mode "$mode" --out "$OUT/$name" "$@"
    if ok "$OUT/$name/result.json"; then stamp "$name: ok"; else stamp "$name: NOT ok (see $OUT/$name/result.json)"; fi
    mark "$name"
}

# step 6: walk on flat ground; if the API's angular velocity is the wrong frame the policy falls:
# then the same run with the finite-difference gyro (d0_runner.gyro_from_quats)
run_d0 walk_flat_api stage_flat_walk.usda walk --gyro api
GYRO=api
if ! ok "$OUT/walk_flat_api/result.json"; then
    run_d0 walk_flat_fd stage_flat_walk.usda walk --gyro fd
    ok "$OUT/walk_flat_fd/result.json" && GYRO=fd
fi
echo "$GYRO" > "$OUT/gyro_used.txt"
stamp "gyro source for the next runs: $GYRO"

# step 7: walk on seed 3 (dunes, slope, roughness)
run_d0 walk_seed3 stage_3_walk.usda walk --gyro "$GYRO"

# step 8: all pick cases (3 items x 3 spots), flat ground
run_d0 pick_flat stage_flat_pick.usda pick --gyro "$GYRO"

# step 9: frame sequences for the log / README (not judged: Replicator steps may add render ticks)
run_d0 frames_walk stage_flat_walk.usda walk --gyro "$GYRO" --frames
run_d0 frames_pick stage_flat_pick.usda pick --gyro "$GYRO" --frames --cases 0,3,6

stamp "step 10: summary, logs, tarball"
"$PY" - "$OUT" > "$OUT/summary.txt" 2>&1 <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
for name in ('walk_flat_api', 'walk_flat_fd', 'walk_seed3', 'pick_flat', 'frames_walk', 'frames_pick'):
    f = out / name / 'result.json'
    if not f.is_file():
        continue
    r = json.loads(f.read_text())
    print(f"{name}: ok={r.get('ok')} reasons={r.get('reasons')} numbers={r.get('numbers')} "
          f"exception={'yes' if r.get('exception') else 'no'} total_s={r.get('total_s')}")
EOF
cat "$OUT/summary.txt"
mkdir -p "$OUT/kit_logs"
cp -r "$ISAAC/kit/logs" "$OUT/kit_logs/" 2>/dev/null
tar -czf "$OUT.tgz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
ls -la "$OUT.tgz"
stamp "done"
