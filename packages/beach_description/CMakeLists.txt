cmake_minimum_required(VERSION 3.8)
project(beach_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY urdf config launch rviz DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/reach_check.py scripts/arm_pose_check.py DESTINATION lib/${PROJECT_NAME})

ament_package()
