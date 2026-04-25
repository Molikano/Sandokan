cmake_minimum_required(VERSION 3.15)
project(TestSandokanProject)

add_executable(my_app main.cpp)

# This will look for SandokanConfig.cmake which we generated earlier
find_package(Sandokan REQUIRED)

# Link it exactly like an external user would
target_link_libraries(my_app PRIVATE Sandokan::sandokan)
