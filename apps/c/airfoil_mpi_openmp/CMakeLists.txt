project(OP2-Airfoil-MPI-DP)

# Require CMake 2.8
cmake_minimum_required(VERSION 2.8)

include(../common.cmake)

# MPI with sequential-nodes version
if (TARGET op2_mpi AND OP2_BUILD_DP)
  add_definitions(${OP2_MPI_DEFINITIONS})
  include_directories(${OP2_MPI_INCLUDE_DIRS})
  add_executable(airfoil_mpi_openmp_dp dp/airfoil_mpi_openmp.cpp dp/airfoil_mpi_kernels.cpp
    dp/save_soln.h dp/adt_calc.h dp/res_calc.h dp/bres_calc.h dp/update.h)
  target_link_libraries(airfoil_mpi_openmp_dp op2_mpi)
  set_target_properties(airfoil_mpi_openmp_dp PROPERTIES COMPILE_FLAGS
    "${OpenMP_CXX_FLAGS}" LINK_FLAGS "${OpenMP_CXX_FLAGS}")
  install(TARGETS airfoil_mpi_openmp_dp RUNTIME DESTINATION ${OP2_APPS_DIR} COMPONENT RuntimeExecutables)
endif()
