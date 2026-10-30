cmake_minimum_required(VERSION 3.8)
project(sim_bridge_gazebo)

find_package(ament_cmake REQUIRED)

install(DIRECTORY urdf config launch rviz DESTINATION share/${PROJECT_NAME})
# Gazebo worlds live in the repo's sim/ tree (simulator-specific, not a ROS package)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../sim/gazebo/worlds/ DESTINATION share/${PROJECT_NAME}/worlds
        FILES_MATCHING PATTERN "*.sdf")

# Shared sensor parameters (repo config/, simulator-independent)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../../config/sensors.yaml DESTINATION share/${PROJECT_NAME}/config)
install(PROGRAMS scripts/attach_server.py scripts/ground_truth.py scripts/gnss_relay.py
        DESTINATION lib/${PROJECT_NAME})

ament_package()
