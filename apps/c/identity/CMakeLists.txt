project(OP2-Identity)

# Require CMake 2.8
cmake_minimum_required(VERSION 2.8)

include(../common.cmake)

# simple sequential version
if (TARGET op2_seq)
  add_executable(identity_seq identity.cpp)
  target_link_libraries(identity_seq op2_seq)
  install(TARGETS identity_seq RUNTIME DESTINATION ${OP2_APPS_DIR} COMPONENT RuntimeExecutables)
endif()
# hand-written CUDA version
if (TARGET op2_cuda)
    find_package(CUDA)
    if (CUDA_FOUND)
      cuda_add_executable(identity_cuda rose_identity_cuda.cpp identity_kernel.cu)
      target_link_libraries(identity_cuda op2_cuda)
      install(TARGETS identity_cuda RUNTIME DESTINATION ${OP2_APPS_DIR} COMPONENT RuntimeExecutables)
    endif()
endif()
