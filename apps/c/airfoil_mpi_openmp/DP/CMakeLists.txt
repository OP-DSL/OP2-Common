project(OP2-Airfoil-MPI-DP)

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

# MPI with sequential-nodes version
if (TARGET op2_mpi)
  add_definitions(${OP2_MPI_DEFINITIONS})
  include_directories(${OP2_MPI_INCLUDE_DIRS})
  add_executable(airfoil_mpi_openmp_dp airfoil_mpi_openmp.cpp airfoil_mpi_kernels.cpp
    save_soln.h adt_calc.h res_calc.h bres_calc.h update.h)
  target_link_libraries(airfoil_mpi_openmp_dp op2_mpi)
  install(TARGETS airfoil_mpi_openmp_dp RUNTIME DESTINATION ${OP2_APPS_DIR} COMPONENT RuntimeExecutables)
endif()
