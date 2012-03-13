project(OP2-Mass2d)

# Require CMake 2.8
cmake_minimum_required(VERSION 2.8)

include(../common.cmake)

set(KERNEL_HEADERS mass.h)
# simple sequential version
if (TARGET op2_seq)
  add_executable(mass2d_seq mass.cpp)
  target_link_libraries(mass2d_seq op2_seq)
  install(TARGETS mass2d_seq RUNTIME DESTINATION ${OP2_APPS_DIR} COMPONENT RuntimeExecutables)
endif()
