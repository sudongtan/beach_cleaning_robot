#!/usr/bin/env bash
# A0 locomotion evaluation: run one stack headless, isolated from other Gazebo/ROS instances,
# drive it with drive_test.py, save results/<stack>_<payload>.json and the launch log.
#   ./run_eval.sh legubiao true|false
#   ./run_eval.sh champ    true|false

STACK=$1
PAYLOAD=$2
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HERE/results
mkdir -p "$OUT"
TAG=${STACK}_payload-${PAYLOAD}

source ~/ros2_ws/install/setup.bash
export GZ_PARTITION=a0eval_$$
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID_EVAL:-42}

case $STACK in
  legubiao)
    WORLD=empty
    LAUNCH=(ros2 launch "$HERE/legubiao_payload.launch.py" headless:=true payload:=$PAYLOAD) ;;
  champ)
    WORLD=default
    WORLD_SDF=$(ros2 pkg prefix --share unitree_go2_description)/worlds/default.sdf
    # their launch doesn't forward extra xacro args, so pick the file instead
    DESC=$(ros2 pkg prefix --share unitree_go2_description)/urdf/unitree_go2_robot.xacro
    [ "$PAYLOAD" = true ] && DESC=$HERE/champ_payload.xacro
    LAUNCH=(ros2 launch unitree_go2_sim unitree_go2_launch.py rviz:=false "world:=$WORLD_SDF -s"
            "unitree_go2_description_path:=$DESC"
            disable_camera:=true disable_d455:=true disable_lidar_l1:=true disable_velodyne_lidar:=true) ;;
  *) echo "unknown stack $STACK"; exit 1 ;;
esac

echo "[$TAG] launching (GZ_PARTITION=$GZ_PARTITION ROS_DOMAIN_ID=$ROS_DOMAIN_ID)"
setsid "${LAUNCH[@]}" > "$OUT/$TAG.launch.log" 2>&1 &
LPID=$!
cleanup() { kill -INT -- -$LPID 2>/dev/null; sleep 5; kill -KILL -- -$LPID 2>/dev/null; }
trap cleanup EXIT

# wait until the leg controller is active (champ spawns it on a timer, ~30 s after launch)
case $STACK in legubiao) CTRL=basic_quadruped_controller ;; champ) CTRL=joint_group_effort_controller ;; esac
for i in $(seq 1 60); do
  ros2 control list_controllers 2>/dev/null | grep -q "^$CTRL .* active" && break
  sleep 2
done
echo "[$TAG] $CTRL active after ~$((i * 2)) s"
sleep 5    # README: wait for a stable frame rate before commanding
python3 "$HERE/drive_test.py" --stack "$STACK" --world "$WORLD" --out "$OUT/$TAG.json"
echo "[$TAG] done"
