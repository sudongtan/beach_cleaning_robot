cmake_minimum_required(VERSION 3.8)
project(sim_bridge_gazebo)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosgraph_msgs REQUIRED)

# /clock at 100 Hz sim time for the ROS nodes (src/clock_throttle.cpp)
add_executable(clock_throttle src/clock_throttle.cpp)
target_link_libraries(clock_throttle PUBLIC rclcpp::rclcpp ${rosgraph_msgs_TARGETS})
install(TARGETS clock_throttle DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY urdf config launch rviz DESTINATION share/${PROJECT_NAME})
# robot_lab policy (locomotion:=legubiao_robot_lab): our config/robot_lab/config.yaml + the
# policy file from the pinned go2_description (not copied into this repo)
find_package(go2_description REQUIRED)
install(FILES ${go2_description_DIR}/../config/robot_lab/policy.pt DESTINATION share/${PROJECT_NAME}/config/robot_lab)
# Gazebo worlds live in the repo's sim/ tree (simulator-specific, not a ROS package)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../sim/gazebo/worlds/ DESTINATION share/${PROJECT_NAME}/worlds
        FILES_MATCHING PATTERN "*.sdf")

# Shared sensor parameters (repo config/, simulator-independent)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../../config/sensors.yaml DESTINATION share/${PROJECT_NAME}/config)
install(PROGRAMS scripts/attach_server.py scripts/ground_truth.py scripts/gnss_relay.py
        DESTINATION lib/${PROJECT_NAME})

ament_package()
