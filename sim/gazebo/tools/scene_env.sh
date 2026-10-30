# Source this: environment of the isolated test scene (scene_up.sh).
source /opt/ros/jazzy/setup.bash
source "$HOME/ros2_ws/install/setup.bash"
export GZ_PARTITION=${SCENE_GZ_PARTITION:-a1b}
export ROS_DOMAIN_ID=${SCENE_ROS_DOMAIN_ID:-43}
# Loopback only: gz-transport and DDS discover peers by multicast on a network interface;
# when the Wi-Fi drops ("Network is unreachable"), running sims lose their topics.
export GZ_IP=127.0.0.1
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
TOOLS=${TOOLS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
REPO=$(cd "$TOOLS/../../.." && pwd)
OUT=${OUT:-$REPO/sim/gazebo/worlds/generated/tools}
mkdir -p "$OUT"
