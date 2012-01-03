project(OP2-APPS)

# Require CMake 2.8
cmake_minimum_required(VERSION 2.8)

# Find OP2 config file (not used here, but check that the app will be able
# to find it)
find_package(OP2 PATHS ${CMAKE_BINARY_DIR} REQUIRED)

# Default build type (can be overridden by user)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
    "Choose the type of build, options are: Debug MinSizeRel Release RelWithDebInfo." FORCE)
endif()
string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE)

# FIXME: at the moment we can't build with -Werror flags due to some warnings
# not easily fixable
if(CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE})
  STRING(REPLACE "-Werror" "" CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}})
endif()
if(CMAKE_C_FLAGS_${CMAKE_BUILD_TYPE})
  STRING(REPLACE "-Werror" "" CMAKE_C_FLAGS_${CMAKE_BUILD_TYPE} ${CMAKE_C_FLAGS_${CMAKE_BUILD_TYPE}})
endif()

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
  foreach (app_dir ${OP2_APPS})
    add_subdirectory(${app_dir})
  endforeach()
endif()
