project(OP2-APPS)

# Require CMake 2.8
cmake_minimum_required(VERSION 2.8)

set(OP2_ALL_APPS TRUE)
include(common.cmake)
unset(OP2_ALL_APPS)

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

# Create the op2.conf configuration file

# FIXME: Can CMake provide the library path name variable?
if (APPLE)
  set(OS_LIBRARY_PATH_NAME "DYLD_LIBRARY_PATH")
else()
  set(OS_LIBRARY_PATH_NAME "LD_LIBRARY_PATH")
endif()

if(CUDA_FOUND)
  get_filename_component(CUDA_LIB_DIR ${CUDA_CUDART_LIBRARY} PATH CACHE)
else()
  set(CUDA_LIB_DIR '')
endif()

set(OP2_DATA_DIR ${CMAKE_SOURCE_DIR}/../data)
configure_file(${OP2_CMAKE_DIR}/op2.conf.in "${OP2_APPS_DIR}/op2.conf" @ONLY)
