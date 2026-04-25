cmake_minimum_required(VERSION 3.15)
project(neural_net)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)
find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)

set(PMAD_DIR ${CMAKE_SOURCE_DIR}/../pmad)

set(PMAD_SOURCES
    ${PMAD_DIR}/src/PMAD.c
    ${PMAD_DIR}/src/BlockHeader.c
    ${PMAD_DIR}/src/MemoryPool.c
    ${PMAD_DIR}/src/incPmad.c
)

# EIGEN_USE_BLAS: routes all matrix-matrix products through the external BLAS
# interface (cblas_dgemm / cblas_sgemm). Accelerate provides these and routes
# them through AMX on Apple Silicon.
add_compile_options(-O3 -march=native)
add_compile_definitions(EIGEN_USE_BLAS)

add_executable(neural_net main.cpp ${PMAD_SOURCES})
target_include_directories(neural_net PRIVATE ${PMAD_DIR}/include)
target_link_libraries(neural_net Eigen3::Eigen ${ACCELERATE_FRAMEWORK})

add_executable(bench bench.cpp ${PMAD_SOURCES})
target_include_directories(bench PRIVATE ${PMAD_DIR}/include)
target_link_libraries(bench Eigen3::Eigen ${ACCELERATE_FRAMEWORK})
