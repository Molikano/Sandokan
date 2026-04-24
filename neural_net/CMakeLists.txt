cmake_minimum_required(VERSION 3.15)
project(neural_net)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)

add_executable(neural_net main.cpp)
target_link_libraries(neural_net Eigen3::Eigen)
