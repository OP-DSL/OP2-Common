project(OP2-APPS)

# Require CMake 2.8
cmake_minimum_required(VERSION 2.8)

# Find OP2 config file (not used here, but check that the app will be able
# to find it)
find_package(OP2 PATHS ${CMAKE_BINARY_DIR} REQUIRED)

if(OP2_FOUND)
  set(OP2_APPS
    airfoil/DP
    airfoil/SP
    airfoil_hdf5/DP
    airfoil_mpi_openmp/DP
    jac1/DP
    jac1/SP
    jac2
    )

  # Add each OP2 app directory
  foreach (app_dir ${OP2_APPS})
    add_subdirectory(${app_dir})
  endforeach()
endif()
