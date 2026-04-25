cmake_minimum_required(VERSION 3.15)
project(Sandokan VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD   11)

find_package(Eigen3 REQUIRED)
find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)

# EIGEN_USE_BLAS: routes all matrix-matrix products through the external BLAS
# interface (cblas_sgemm). Accelerate provides these and routes them through
# AMX on Apple Silicon.
add_compile_options(-O3 -march=native)
add_compile_definitions(EIGEN_USE_BLAS)

# ---------- pmad — C allocator, compiled once into a static lib ----------

set(PMAD_DIR ${CMAKE_SOURCE_DIR}/pmad)

add_library(pmad STATIC
    ${PMAD_DIR}/src/PMAD.c
    ${PMAD_DIR}/src/BlockHeader.c
    ${PMAD_DIR}/src/MemoryPool.c
    ${PMAD_DIR}/src/incPmad.c
)
target_include_directories(pmad PUBLIC ${PMAD_DIR}/include)

# ---------- sandokan — header-only INTERFACE target ----------
#
# Consumers link against sandokan and get:
#   • include/sandokan/ on their include path   → #include <sandokan/network.h>
#   • pmad/include/ on their include path       → PMAD C headers resolved
#   • Eigen3, pmad static lib, Accelerate       → all transitive deps

add_library(sandokan INTERFACE)
target_include_directories(sandokan INTERFACE
    ${CMAKE_SOURCE_DIR}/include
    ${PMAD_DIR}/include
)
target_link_libraries(sandokan INTERFACE
    Eigen3::Eigen
    pmad
    ${ACCELERATE_FRAMEWORK}
)

add_subdirectory(examples)
