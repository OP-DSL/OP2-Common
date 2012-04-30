project(OP2-Airfoil-MPI-CUDA-DP)

# Require CMake 2.8
cmake_minimum_required(VERSION 2.8)

include(../airfoil-common.cmake)

# MPI with sequential-nodes version
if (TARGET op2_mpi_cuda AND OP2_BUILD_DP)
  find_package(CUDA)
  add_definitions(${OP2_MPI_DEFINITIONS})
  include_directories(${OP2_MPI_INCLUDE_DIRS})
  cuda_add_executable(airfoil_mpi_cuda_dp dp/airfoil_mpi_cuda.cpp dp/airfoil_mpi_kernels.cu
    dp/save_soln.h dp/adt_calc.h dp/res_calc.h dp/bres_calc.h dp/update.h)
  add_dependencies(airfoil_mpi_cuda_dp grid)
  target_link_libraries(airfoil_mpi_cuda_dp op2_mpi_cuda)
  install(TARGETS airfoil_mpi_cuda_dp RUNTIME DESTINATION ${OP2_APPS_DIR} COMPONENT RuntimeExecutables)
endif()
