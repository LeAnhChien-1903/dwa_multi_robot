cmake_minimum_required(VERSION 3.0.2)
project(dwa_multi_robot)

find_package(catkin REQUIRED COMPONENTS
    dynamic_reconfigure
    roscpp
    rospy
)

include_directories(
# include
    ${catkin_INCLUDE_DIRS}
    include/${PROJECT_NAME}
)

catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS 
    src/main.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)