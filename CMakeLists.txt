cmake_minimum_required(VERSION 3.0.2)
project(data_preprocessing)


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -Wall -Wuninitialized -fno-omit-frame-pointer")


find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
  pcl_ros  
  cv_bridge  
  livox_ros_driver2

)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(PCL REQUIRED)


generate_messages(
  DEPENDENCIES
  std_msgs
)


catkin_package(
 CATKIN_DEPENDS roscpp rospy std_msgs
DEPENDS
CATKIN_DEPENDS roscpp rospy std_msgs 
)


include_directories(
include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}  
  ${yaml-cpp_INCLUDE_DIRS}
  )


add_executable(data_preprocessing src/main.cpp)
target_link_libraries(data_preprocessing ${catkin_LIBRARIES} ${OpenCV_LIBS})

