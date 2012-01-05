project(OP2-Laplace1D)

# Require CMake 2.8
cmake_minimum_required(VERSION 2.8)

# Load OP2 configuration (Set OP2_DIR to the directory containing OP2Config.cmake)
find_package(OP2 REQUIRED)

# Default build type (can be overridden by user)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
    "Choose the type of build, options are: Debug MinSizeRel Release RelWithDebInfo." FORCE)
endif()

include_directories(${OP2_INCLUDE_DIRS})

# simple sequential version
if (TARGET op2_seq)
  add_executable(laplace1d_seq laplace.cpp)
  target_link_libraries(laplace1d_seq op2_seq)
  install(TARGETS laplace1d_seq RUNTIME DESTINATION ${OP2_APPS_DIR} COMPONENT RuntimeExecutables)
endif()
