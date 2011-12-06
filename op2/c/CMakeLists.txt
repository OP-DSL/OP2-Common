# Open source copyright declaration based on BSD open source template:
# http://www.opensource.org/licenses/bsd-license.php
#
# This file is part of the OP2 distribution.
#
# Copyright (c) 2011, Florian Rathgeber and others. Please see the AUTHORS
# file in the main source directory for a full list of copyright holders.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Florian Rathgeber may not be used to endorse or promote
#       products derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY Florian Rathgeber ''AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL Florian Rathgeber BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Top level CMakeLists.txt file for OP2

# Require CMake 2.8
cmake_minimum_required(VERSION 2.8)

#------------------------------------------------------------------------------
# General configuration

# Set location of our FindFoo.cmake modules
find_path(OP2_CMAKE_DIR modules PATHS "${CMAKE_SOURCE_DIR}/../../cmake"
  NO_DEFAULT_PATH)
set(OP2_CMAKE_DIR ${OP2_CMAKE_DIR} CACHE INTERNAL
  "OP2 CMake utilities directory")
set(OP2_CMAKE_MODULE_DIR ${OP2_CMAKE_DIR}/modules CACHE INTERNAL
  "OP2 CMake modules directory")

# Add packaged CMake scripts for backwards compatibility of features not in
# CMake 2.8.0 to module path
set(CMAKE_MODULE_PATH_INIT ${OP2_CMAKE_MODULE_DIR})
file(GLOB BACKCOMP_VERSIONS RELATIVE ${OP2_CMAKE_MODULE_DIR} ${OP2_CMAKE_MODULE_DIR}/2.8.*)
foreach(BACKCOMP_VERSION ${BACKCOMP_VERSIONS})
  if (${CMAKE_VERSION} VERSION_LESS ${BACKCOMP_VERSION})
    set(CMAKE_MODULE_PATH_INIT ${CMAKE_MODULE_PATH_INIT} ${OP2_CMAKE_MODULE_DIR}/${BACKCOMP_VERSION})
  endif()
endforeach()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH_INIT} CACHE FILEPATH
  "Search path for CMake scripts and modules")

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

# Used languages
set(OP2_ENABLED_LANGUAGES C CXX)

#------------------------------------------------------------------------------
# Enable Fortran if the Fortran interface is enabled
option(OP2_ENABLE_FORTRAN_INTERFACE "Enable the Fortran interface to the OP2 library." OFF)
if(OP2_ENABLE_FORTRAN_INTERFACE)
  set(OP2_ENABLED_LANGUAGES ${OP2_ENABLED_LANGUAGES} Fortran)
endif()

#------------------------------------------------------------------------------
# Set project name and version number

project(OP2 ${OP2_ENABLED_LANGUAGES})

set(OP2_VERSION_MAJOR 0)
set(OP2_VERSION_MINOR 1)
set(OP2_VERSION_PATCH 0)
set(OP2_VERSION
  ${OP2_VERSION_MAJOR}.${OP2_VERSION_MINOR}.${OP2_VERSION_PATCH})

#------------------------------------------------------------------------------
# Configurable options for what/how we want to build

option(OP2_WITH_CUDA     "Build a NVIDIA CUDA version of the OP2 library." ON)
option(OP2_WITH_HDF5     "Build an HDF5 version of the OP2 library."       ON)
option(OP2_WITH_MPI      "Build an MPI version of the OP2 library."        ON)
option(OP2_WITH_PTSCOTCH "Use PTScotch for partitioning in parallel."      ON)
option(OP2_WITH_PARMETIS "Use ParMETIS for partitioning in parallel."      ON)
option(OP2_WITH_OPENMP   "Build an OpenMP version of the OP2 library."     ON)
option(OP2_WITH_MATRICES "Enable support for matrix kernel arguments."     ON)
option(OP2_WITH_SEQ      "Build a sequential version of the OP2 library."  ON)
option(BUILD_SHARED_LIBS "Build OP2 with shared libraries."                ON)
if(BUILD_SHARED_LIBS)
  option(USE_INSTALL_RPATH "Set rpath for installed shared libraries."     ON)
endif()
option(CMAKE_VERBOSE_CONFIGURE "Enable verbose configuration output."     OFF)

#------------------------------------------------------------------------------
# Print a summary of enabled/disabled features
if(CMAKE_VERBOSE_CONFIGURE)
  include(FeatureSummary)
  add_feature_info("   Fortran interface" OP2_ENABLE_FORTRAN_INTERFACE "Enables the OP2 Fortran interface to the OP2 C libraries")
  add_feature_info("   Sequential library" OP2_WITH_SEQ "For reference testing")
  add_feature_info("   OpenMP library" OP2_WITH_OPENMP "Requires OpenMP support by the compiler")
  add_feature_info("   HDF5 library" OP2_WITH_HDF5 "Requires the HDF5 library")
  add_feature_info("   MPI library" OP2_WITH_MPI "Requires an MPI library")
  add_feature_info("   Matrix support" OP2_WITH_MATRICES "Requires PETSc")
  add_feature_info("   Parallel graph partitioning with PT-Scotch"
    OP2_WITH_PTSCOTCH "Requires the PT-Scotch library")
  add_feature_info("   Parallel graph partitioning with ParMETIS"
    OP2_WITH_PARMETIS "Requires the ParMETIS library")
  add_feature_info("   CUDA library" OP2_WITH_CUDA "Requires the NVIDIA CUDA toolkit")
  feature_summary(WHAT ENABLED_FEATURES DESCRIPTION
    "Configure with the following features enabled:")
  feature_summary(WHAT DISABLED_FEATURES DESCRIPTION
    "Configure with the following features disabled:")
endif()

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
