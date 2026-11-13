#!/usr/bin/env bash
# REQUIRES_GPU
# D0 prep GPU session (notes/briefs/D0_prep.md §5-§7), inside the Isaac Sim 6.2.0 container.
#
#   bash sim/isaac/tools/gpu_session.sh OUT SEED [SEED ...]
#   e.g. bash sim/isaac/tools/gpu_session.sh /workspace/d0_s1 0
#
# Run from the repo root (the uploaded copy: sim/isaac with scenes/generated/scene_<seed>.usda,
# sim/common/textures). Restartable: each step writes $OUT/.done/<step> and is skipped when
# the marker exists (delete a marker to rerun a step). Everything goes to $OUT/session.log.
# Ends with $OUT.tar.gz for download.

set -uo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 OUT SEED [SEED ...]" >&2
    exit 2
fi
OUT=$(realpath -m "$1"); shift
SEEDS=("$@")
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
ISAAC=${ISAAC_ROOT:-/isaac-sim}
PY="$ISAAC/python.sh"
mkdir -p "$OUT/.done"
exec > >(tee -a "$OUT/session.log") 2>&1

stamp() { echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
done_() { [ -f "$OUT/.done/$1" ]; }
mark() { date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/.done/$1"; }
fail() { stamp "FAILED: $*"; exit 1; }

stamp "session start; repo $REPO; seeds ${SEEDS[*]}"

# 1. Host facts
if ! done_ host; then
    stamp "step 1: host facts"
    {
        echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "hostname: $(hostname)"
        nvidia-smi
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
        free -g
        df -h / /workspace 2>/dev/null
        echo "nproc: $(nproc)"
        echo "isaac VERSION: $(cat "$ISAAC/VERSION" 2>/dev/null)"
        env | grep -E '^NVIDIA_(DRIVER_CAPABILITIES|VISIBLE_DEVICES)='
    } > "$OUT/host.txt" 2>&1
    cat "$OUT/host.txt"
    mark host
fi

# 2. Vulkan sees the NVIDIA GPU (needs NVIDIA_DRIVER_CAPABILITIES with graphics, e.g. all)
if ! done_ vulkan; then
    stamp "step 2: vulkaninfo"
    if ! command -v vulkaninfo >/dev/null; then
        apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq vulkan-tools \
            || fail "could not install vulkan-tools"
    fi
    vulkaninfo --summary > "$OUT/vulkaninfo.txt" 2>&1
    cat "$OUT/vulkaninfo.txt"
    grep -qi 'nvidia' "$OUT/vulkaninfo.txt" || fail "vulkaninfo lists no NVIDIA device (brief §8: capabilities / next host)"
    mark vulkan
fi

# 3. NVIDIA's compatibility check
if ! done_ compat; then
    stamp "step 3: compatibility check"
    (cd "$ISAAC" && timeout 900 ./isaac-sim.compatibility_check.sh --/app/quitAfter=10 --no-window) \
        > "$OUT/compat.txt" 2>&1
    tail -40 "$OUT/compat.txt"
    grep -q 'System checking result: PASSED' "$OUT/compat.txt" || fail "compatibility check did not pass"
    mark compat
fi

# Image check for step 4, with our tested helper.
check_png() {   # check_png DIR -> 0 if the newest rgb*.png under DIR is neither black nor uniform
    "$PY" - "$1" "$REPO/sim/isaac" <<'EOF'
import sys, yaml
from pathlib import Path
sys.path.insert(0, sys.argv[2])
import render_checks as rc
cfg = yaml.safe_load(open(Path(sys.argv[2]) / 'render.yaml'))['render_check']
files = sorted(Path(sys.argv[1]).rglob('rgb*.png'), key=lambda p: p.stat().st_mtime)
if not files:
    print('no rgb image'); sys.exit(1)
c = rc.image_check(rc.load_image(files[-1]), cfg['min_mean'], cfg['min_std'])
print(files[-1], c); sys.exit(0 if c['ok'] else 1)
EOF
}

# 4. NVIDIA's own example renders (proves the host renders, separately from our code)
if ! done_ nvidia_example; then
    stamp "step 4: NVIDIA example sdg_getting_started_01.py"
    EX="$ISAAC/standalone_examples/api/isaacsim.replicator.examples/sdg_getting_started_01.py"
    [ -f "$EX" ] || fail "example not found: $EX"
    D="$OUT/nvidia_example"
    mkdir -p "$D"
    # Unchanged first. It asks for a window ("headless": False), which a display-less
    # container may refuse; then a copy with only that flag flipped.
    (cd "$D" && timeout 1200 "$PY" "$EX") > "$D/run_unchanged.log" 2>&1
    echo "unchanged exit code: $?"
    if check_png "$D"; then
        echo "variant: unchanged" > "$D/variant.txt"
    else
        sed 's/"headless": False/"headless": True/' "$EX" > "$D/sdg_getting_started_01_headless.py"
        diff "$EX" "$D/sdg_getting_started_01_headless.py"
        (cd "$D" && timeout 1200 "$PY" "$D/sdg_getting_started_01_headless.py") > "$D/run_headless.log" 2>&1
        echo "headless copy exit code: $?"
        check_png "$D" || fail "NVIDIA's example gives no usable image: the host can't render (brief §8)"
        echo "variant: headless copy" > "$D/variant.txt"
    fi
    cat "$D/variant.txt"
    mark nvidia_example
fi

# 5. Isaac's USD version and the semantics schemas it has
if ! done_ versions; then
    stamp "step 5: USD version, semantics schemas"
    "$PY" - > "$OUT/versions.txt" 2>&1 <<'EOF'
from isaacsim import SimulationApp
app = SimulationApp(launch_config={'headless': True})
from pxr import Usd
print('usd_version:', Usd.GetVersion())
for mod in ('UsdSemantics', 'Semantics'):
    try:
        m = __import__('pxr', fromlist=[mod])
        print(f'pxr.{mod}:', hasattr(m, mod) and [n for n in dir(getattr(m, mod)) if 'API' in n])
    except Exception as e:
        print(f'pxr.{mod}: no ({e})')
try:
    import Semantics
    print('Semantics (module):', [n for n in dir(Semantics) if 'API' in n])
except Exception as e:
    print('Semantics (module): no', e)
try:
    import isaacsim.core.experimental.utils.semantics as sem
    print('isaacsim.core.experimental.utils.semantics:', [n for n in dir(sem) if not n.startswith('_')])
except Exception as e:
    print('isaacsim.core.experimental.utils.semantics: no', e)
import sys
print('python:', sys.version)
app.close()
EOF
    cat "$OUT/versions.txt"
    mark versions
fi

# 6. Our render check per seed
FAILED=()
for s in "${SEEDS[@]}"; do
    if done_ "scene_$s"; then continue; fi
    stamp "step 6: render_check seed $s"
    USD="$REPO/sim/isaac/scenes/generated/scene_$s.usda"
    [ -f "$USD" ] || { echo "missing $USD"; FAILED+=("$s"); continue; }
    timeout 1800 "$PY" "$REPO/sim/isaac/render_check.py" --usd "$USD" --out "$OUT/scene_$s"
    rc=$?
    stamp "seed $s exit code $rc"
    if [ $rc -eq 0 ]; then mark "scene_$s"; else FAILED+=("$s"); fi
done

# 7. Logs and tarball for download (always, also after failures)
stamp "step 7: collect logs, tar"
mkdir -p "$OUT/kit_logs"
for d in "$HOME/.nvidia-omniverse/logs" "$ISAAC/.nvidia-omniverse/logs"; do
    [ -d "$d" ] && cp -r "$d" "$OUT/kit_logs/$(echo "$d" | tr '/' '_')" 2>/dev/null
done
tar -czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
ls -la "$OUT.tar.gz"
if [ ${#FAILED[@]} -gt 0 ]; then
    stamp "done, FAILED seeds: ${FAILED[*]}"
    exit 1
fi
stamp "done, all seeds passed"
