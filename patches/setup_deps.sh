#!/usr/bin/env bash
# Clone the pinned third-party repos into ~/ros2_ws/src, apply our patches,
# and COLCON_IGNORE the packages we don't build (so a plain `colcon build` works).
set -e
REPO=$(cd "$(dirname "$0")/.." && pwd)
SRC=$(cd "$REPO/.." && pwd)

vcs import "$SRC" < "$REPO/deps.repos"

# patches/<repo>/*.patch are applied in order (skipped if already applied)
for dir in "$REPO"/patches/*/; do
  name=$(basename "$dir")
  for p in "$dir"*.patch; do
    [ -e "$p" ] || continue
    if git -C "$SRC/$name" apply --check "$p" 2>/dev/null; then
      git -C "$SRC/$name" apply "$p" && echo "applied $name/$(basename "$p")"
    elif git -C "$SRC/$name" apply --reverse --check "$p" 2>/dev/null; then
      echo "already applied: $name/$(basename "$p")"
    else
      echo "ERROR: $name/$(basename "$p") does not apply" >&2; exit 1
    fi
  done
done

# not built: need OCS2 / unitree_sdk2 (real robot). rl_quadruped_controller is built
# (needs libtorch, see VERSIONS.md); drop a COLCON_IGNORE left by older versions of this script
rm -f "$SRC/quadruped_ros2_control/controllers/rl_quadruped_controller/COLCON_IGNORE"
for pkg in controllers/ocs2_quadruped_controller \
           hardwares/hardware_unitree_sdk2 commands/unitree_joystick_input; do
  touch "$SRC/quadruped_ros2_control/$pkg/COLCON_IGNORE"
done
# interbotix: only the arm descriptions (meshes); everything else needs the Interbotix SDK stack
IB="$SRC/interbotix_ros_manipulators"
find "$IB" -name package.xml -not -path "*/interbotix_xsarm_descriptions/*" -printf '%h\n' |
  while read -r d; do touch "$d/COLCON_IGNORE"; done

echo "done. Next: cd ~/ros2_ws && rosdep install --from-paths src --ignore-src -r -y && colcon build --symlink-install"
