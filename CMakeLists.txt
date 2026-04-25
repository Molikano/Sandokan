cmake_minimum_required(VERSION 3.15)
project(Sandokan VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD   11)

include(GNUInstallDirs)

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
target_include_directories(pmad PUBLIC
    $<BUILD_INTERFACE:${PMAD_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/pmad>
)

# ---------- sandokan — header-only INTERFACE target ----------
#
# Consumers link against sandokan and get:
#   • include/sandokan/ on their include path   → #include <sandokan/network.h>
#   • pmad/include/ on their include path       → PMAD C headers resolved
#   • Eigen3, pmad static lib, Accelerate       → all transitive deps

add_library(sandokan INTERFACE)
target_include_directories(sandokan INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PMAD_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/pmad>
)
target_link_libraries(sandokan INTERFACE
    Eigen3::Eigen
    pmad
    ${ACCELERATE_FRAMEWORK}
)

option(SANDOKAN_BUILD_EXAMPLES "Build Sandokan examples" ON)
if(SANDOKAN_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ---------- Installation (Make it universal) ----------

# 1. Install the targets (compiled pmad static library and sandokan interface)
install(TARGETS sandokan pmad
    EXPORT SandokanTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# 2. Install your master header and the sandokan directory
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# 3. Install the pmad headers
install(DIRECTORY ${PMAD_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pmad
)

# 4. Export CMake configs so other projects can find it using find_package(Sandokan)
install(EXPORT SandokanTargets
    FILE SandokanTargets.cmake
    NAMESPACE Sandokan::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Sandokan
)

# Generate a config file
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/SandokanConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Sandokan
    NO_SET_AND_CHECK_MACRO
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/SandokanConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY AnyNewerVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/SandokanConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/SandokanConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Sandokan
)
