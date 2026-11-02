#!/bin/bash
# Stop every Gazebo / A2 stack process by PID (never pkill -f: it matches the caller's own shell).
pids=$(ps -eo pid=,args= | awk -v me=$$ -v pp=$PPID '$1 != me && $1 != pp' | grep -E \
  ' (gz sim|ruby .*gz sim|/usr/bin/python3 /opt/ros/jazzy/bin/ros2 launch bringup|make (run|scene) SEED)| /opt/ros/jazzy/lib/(nav2_|robot_state_publisher|ros_gz_bridge|controller_manager)| /home/xd/ros2_ws/install/[a-z_]+/lib/' \
  | grep -v -E 'grep|stop_sim' | awk '{print $1}')
[ -n "$pids" ] && kill $pids 2>/dev/null
sleep 4
left=$(ps -eo pid=,args= | grep -E ' (gz sim|/usr/bin/python3 /home/xd/ros2_ws/install/[a-z_]+/lib/)' | grep -v grep | awk '{print $1}')
[ -n "$left" ] && kill -9 $left 2>/dev/null
sleep 1
echo "remaining: $(ps -eo args= | grep -E '^(gz sim|/usr/bin/python3 /home/xd/ros2_ws/install/[a-z_]+/lib/|/opt/ros/jazzy/lib/nav2_)' | grep -v grep | wc -l)"
