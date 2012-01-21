project(OP2-APPS)

# Require CMake 2.8
cmake_minimum_required(VERSION 2.8)

include(common.cmake)

if(OP2_FOUND)
  set(OP2_APPS
    airfoil/DP
    airfoil/SP
    airfoil_hdf5/DP
    airfoil_mpi_cuda/DP
    airfoil_mpi_openmp/DP
    jac1/DP
    jac1/SP
    jac2
    )

  # Add each OP2 app directory
  foreach (APP_DIR ${OP2_APPS})
    option(OP2_WITH_${APP_DIR} "Build the ${APP_DIR} apps" ON)
    if(OP2_WITH_${APP_DIR})
      add_subdirectory(${APP_DIR})
    endif()
  endforeach()
endif()
