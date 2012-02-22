# Top level CMakeLists.txt file for OP2

# Require CMake 2.8
cmake_minimum_required(VERSION 2.8)

#------------------------------------------------------------------------------
# General configuration

# Set location of our FindFoo.cmake modules
set(OP2_CMAKE_DIR "${CMAKE_SOURCE_DIR}/../../cmake" CACHE INTERNAL
  "OP2 CMake utilities directory")
set(CMAKE_MODULE_PATH "${OP2_CMAKE_DIR}/modules")

# Set CMake options, see `cmake --help-policy CMP000x`
# Libraries linked via full path no longer produce linker search paths.
cmake_policy(SET CMP0003 NEW)
# Silently strip whitespace from target_link_libraries
cmake_policy(SET CMP0004 OLD)
if(POLICY CMP0017)
  # Always prefer files from CMAKE_MODULE_PATH over files from the CMake
  # modules directory.
  # This is necessary to make overriding the platfrom-specific compiler
  # flags work.
  cmake_policy(SET CMP0017 OLD)
endif()

#------------------------------------------------------------------------------
# Set project name and version number

project(OP2 C CXX)

set(OP2_VERSION_MAJOR 0)
set(OP2_VERSION_MINOR 1)
set(OP2_VERSION_PATCH 0)
set(OP2_VERSION
  ${OP2_VERSION_MAJOR}.${OP2_VERSION_MINOR}.${OP2_VERSION_PATCH})

#------------------------------------------------------------------------------
# Configurable options for what/how we want to build

option(OP2_WITH_SEQ      "Build a sequential version of the OP2 library."  ON)
option(OP2_WITH_OPENMP   "Build an OpenMP version of the OP2 library."     ON)
option(OP2_WITH_HDF5     "Build an HDF5 version of the OP2 library."       ON)
option(OP2_WITH_MPI      "Build an MPI version of the OP2 library."        ON)
option(OP2_WITH_PTSCOTCH "Use PTScotch for partitioning in parallel."      ON)
option(OP2_WITH_PARMETIS "Use ParMETIS for partitioning in parallel."      ON)
option(OP2_WITH_CUDA     "Build a NVIDIA CUDA version of the OP2 library." ON)
option(BUILD_SHARED_LIBS "Build OP2 with shared libraries."                ON)
option(CMAKE_VERBOSE_CONFIGURE "Enable verbose configuration output."     OFF)

#------------------------------------------------------------------------------
# Default build type (can be overridden by user)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
    "Choose the type of build, options are:
    Debug Developer MinSizeRel Release RelWithDebInfo." FORCE)
endif()

#------------------------------------------------------------------------------
# Custom user definitions

set(OP2_USER_DEFINITIONS "${OP2_USER_DEFINITIONS}" CACHE STRING
  "Space separated list of custom preprocessor definitions -DVAR=FOO.")
add_definitions(${OP2_USER_DEFINITIONS})

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
