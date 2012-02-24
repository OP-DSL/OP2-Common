project(OP2-APPS)

# Require CMake 2.8
cmake_minimum_required(VERSION 2.8)

set(OP2_ALL_APPS TRUE)
include(common.cmake)
unset(OP2_ALL_APPS)

if(OP2_FOUND)
  set(OP2_APPS
    airfoil
    airfoil_hdf5
    airfoil_mpi_cuda
    airfoil_mpi_openmp
    jac1
    jac2
    )

  # Add each OP2 app directory
  foreach (APP_DIR ${OP2_APPS})
    option(OP2_BUILD_${APP_DIR} "Build the ${APP_DIR} apps" ON)
    if(OP2_BUILD_${APP_DIR})
      add_subdirectory(${APP_DIR})
    endif()
  endforeach()
endif()
