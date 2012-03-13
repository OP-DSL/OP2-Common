project(OP2-Laplace1d)

# Require CMake 2.8
cmake_minimum_required(VERSION 2.8)

include(../common.cmake)

set(KERNEL_HEADERS laplace.h)
# simple sequential version
if (TARGET op2_seq)
  add_executable(laplace1d_seq laplace.cpp)
  target_link_libraries(laplace1d_seq op2_seq)
  install(TARGETS laplace1d_seq RUNTIME DESTINATION ${OP2_APPS_DIR} COMPONENT RuntimeExecutables)
endif()
