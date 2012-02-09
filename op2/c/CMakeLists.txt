# Top level CMakeLists.txt file for OP2

# Require CMake 2.8
cmake_minimum_required(VERSION 2.8)

#------------------------------------------------------------------------------
# Set project name and version number

project(OP2)

set(OP2_VERSION_MAJOR 0)
set(OP2_VERSION_MINOR 1)
set(OP2_VERSION_PATCH 0)
set(OP2_VERSION
  ${OP2_VERSION_MAJOR}.${OP2_VERSION_MINOR}.${OP2_VERSION_PATCH})

#------------------------------------------------------------------------------
# General configuration

# Set location of our FindFoo.cmake modules
set(OP2_CMAKE_DIR "${OP2_SOURCE_DIR}/../../cmake" CACHE INTERNAL
  "OP2 CMake utilities directory")
set(CMAKE_MODULE_PATH "${OP2_CMAKE_DIR}/modules")

# Set CMake options, see `cmake --help-policy CMP000x`
if (COMMAND cmake_policy)
  # Libraries linked via full path no longer produce linker search paths.
  cmake_policy(SET CMP0003 NEW)
  # Silently strip whitespace from target_link_libraries
  cmake_policy(SET CMP0004 OLD)
endif()

#------------------------------------------------------------------------------
# Configurable options for what/how we want to build

option(OP2_WITH_SEQ      "Build a sequential version of the OP2 library."  ON)
option(OP2_WITH_OPENMP   "Build an OpenMP version of the OP2 library."     ON)
option(OP2_WITH_HDF5     "Build an HDF5 version of the OP2 library."       ON)
option(OP2_WITH_MPI      "Build an MPI version of the OP2 library."        ON)
option(OP2_WITH_CUDA     "Build a NVIDIA CUDA version of the OP2 library." ON)
option(BUILD_SHARED_LIBS "Build OP2 with shared libraries."                ON)
option(CMAKE_VERBOSE_CONFIGURE "Enable verbose configuration output."     OFF)

#------------------------------------------------------------------------------
# Compiler flags

# Default build type (can be overridden by user)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
    "Choose the type of build, options are:
    Debug Developer MinSizeRel Release RelWithDebInfo." FORCE)
endif()

# Check for some C compiler flags
include(CheckCCompilerFlag)

# C99 is required
CHECK_C_COMPILER_FLAG(-std=c99 HAVE_C99)
if (HAVE_C99)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
else()
  message(FATAL_ERROR "C99 support for the C compiler is required.")
endif()

# Flags specific to intel compilers
# ignore remark #981: operands are evaluated in unspecified order
set(OP2_INTEL_FLAGS "-wd981" CACHE STRING
  "Flags specifically for Intel compilers")
if(CMAKE_C_COMPILER_ID STREQUAL "Intel")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OP2_INTEL_FLAGS}")
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OP2_INTEL_FLAGS}")
endif()

CHECK_C_COMPILER_FLAG(-pipe HAVE_PIPE)
if (HAVE_PIPE)
  set(OP2_C_DEVELOPER_FLAGS "-pipe ${OP2_C_DEVELOPER_FLAGS}")
endif()

CHECK_C_COMPILER_FLAG("-Wall -Werror" HAVE_PEDANTIC)
if (HAVE_PEDANTIC)
  set(OP2_C_DEVELOPER_FLAGS "-Wall -Werror ${OP2_C_DEVELOPER_FLAGS}")
endif()

CHECK_C_COMPILER_FLAG("-pedantic" HAVE_WERR)
if (HAVE_WERR)
  set(OP2_C_DEVELOPER_FLAGS "-pedantic ${OP2_C_DEVELOPER_FLAGS}")
endif()

CHECK_C_COMPILER_FLAG(-g HAVE_DEBUG)
if (HAVE_DEBUG)
  set(OP2_C_DEVELOPER_FLAGS "-g ${OP2_C_DEVELOPER_FLAGS}")
endif()

CHECK_C_COMPILER_FLAG(-O2 HAVE_O2_OPTIMISATION)
if (HAVE_O2_OPTIMISATION)
  set(OP2_C_DEVELOPER_FLAGS "-O2 ${OP2_C_DEVELOPER_FLAGS}")
endif()

# Set 'Developer' build type flags
set(CMAKE_C_FLAGS_DEVELOPER ${OP2_C_DEVELOPER_FLAGS} CACHE STRING
  "Flags used by the compiler during development.")

# Check for some CXX compiler flags
include(CheckCXXCompilerFlag)

CHECK_CXX_COMPILER_FLAG(-pipe HAVE_PIPE)
if (HAVE_PIPE)
  set(OP2_CXX_DEVELOPER_FLAGS "-pipe ${OP2_CXX_DEVELOPER_FLAGS}")
endif()

CHECK_CXX_COMPILER_FLAG("-Wall -Werror" HAVE_WERR)
if (HAVE_WERR)
  set(OP2_CXX_DEVELOPER_FLAGS "-Wall -Werror ${OP2_CXX_DEVELOPER_FLAGS}")
endif()

CHECK_CXX_COMPILER_FLAG(-std=c++98 HAVE_STD)
if (HAVE_STD)
  set(OP2_CXX_DEVELOPER_FLAGS "-std=c++98 ${OP2_CXX_DEVELOPER_FLAGS}")
endif()

CHECK_CXX_COMPILER_FLAG(-g HAVE_DEBUG)
if (HAVE_DEBUG)
  set(OP2_CXX_DEVELOPER_FLAGS "-g ${OP2_CXX_DEVELOPER_FLAGS}")
endif()

CHECK_CXX_COMPILER_FLAG(-O2 HAVE_O2_OPTIMISATION)
if (HAVE_O2_OPTIMISATION)
  set(OP2_CXX_DEVELOPER_FLAGS "-O2 ${OP2_CXX_DEVELOPER_FLAGS}")
endif()

# Set 'Developer' build type flags
set(CMAKE_CXX_FLAGS_DEVELOPER ${OP2_CXX_DEVELOPER_FLAGS} CACHE STRING
  "Flags used by the compiler during development.")

#------------------------------------------------------------------------------
# Installation directories

set(INSTALLATION_BIN_DIR bin CACHE PATH "Binary installation directory.")
set(INSTALLATION_LIB_DIR lib CACHE PATH "Library installation directory.")
set(INSTALLATION_INCLUDE_DIR include CACHE PATH
  "C/C++ header installation directory.")
set(INSTALLATION_APPS_DIR apps CACHE PATH "Apps installation directory.")
set(INSTALLATION_CMAKE_DIR lib/op2 CACHE PATH
    "Installation directory for CMake files")

# Make relative paths absolute (needed later on)
foreach(p LIB BIN INCLUDE APPS CMAKE)
  set(var INSTALLATION_${p}_DIR)
  if(${var} AND NOT IS_ABSOLUTE "${${var}}")
    set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  endif()
endforeach()

#------------------------------------------------------------------------------
# Install header files

file(GLOB HEADERS include/*.h)
install(FILES ${HEADERS} DESTINATION ${INSTALLATION_INCLUDE_DIR} COMPONENT
  Development)

#------------------------------------------------------------------------------
# Set include directories

include_directories(${OP2_SOURCE_DIR}/include)

#------------------------------------------------------------------------------
# OP2 source directories

add_subdirectory(src)

# Store the current build directory in the CMake user package registry.
# This helps dependent projects find and use a package from the current
# project's build tree without help from the user
export(PACKAGE OP2)
