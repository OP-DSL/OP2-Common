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

#------------------------------------------------------------------------------
# Check for packages

message(STATUS "Checking for optional packages...")

if(OP2_WITH_OPENMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    message(STATUS "OpenMP found")
    set(OP2_OPENMP_DEFINITIONS "-DHAVE_OPENMP")
  else()
    message(STATUS "Optional package OpenMP not found - disabling")
    set(OP2_WITH_OPENMP OFF)
  endif()
endif()

if(OP2_WITH_CUDA)
  ## Override CUDA default cache variables
  # These need to be separated by semicolon
  set(CUDA_NVCC_FLAGS -arch=sm_20;-Xptxas;-dlcm=ca;-Xptxas=-v
    CACHE STRING "Flags used by NVCC during release builds
      (semi-colon delimit multiple arguments).")
  set(CUDA_NVCC_FLAGS_RELEASE -O3;-use_fast_math
    CACHE STRING "Flags used by NVCC during all build types
      (semi-colon delimit multiple arguments).")
  # FIXME: This assumes a GNU host compiler for nvcc
  set(CUDA_NVCC_FLAGS_DEVELOPER -g;-G;-Xcompiler=-O2;-Xcompiler=-g;-Xcompiler=-Wall;-Xcompiler=-pipe
    CACHE STRING "Flags used by NVCC during developer builds
      (semi-colon delimit multiple arguments).")
  # Do not propagate C/CXX flags to nvcc by default
  option(CUDA_PROPAGATE_HOST_FLAGS
    "Propage C/CXX_FLAGS and friends to the host compiler via -Xcompile" OFF)

  find_package(CUDA)
  if(CUDA_FOUND)
    message(STATUS "CUDA found")
  else()
    message(STATUS "Optional package CUDA not found - disabling")
    set(OP2_WITH_CUDA OFF)
  endif()
endif()

if(OP2_WITH_MPI)
  find_package(MPI)
  if(MPI_FOUND)
    message(STATUS "MPI found")
    set(OP2_MPI_DEFINITIONS "${OP2_MPI_DEFINITIONS} -DHAVE_MPI")
    set(OP2_MPI_INCLUDE_DIRS ${MPI_INCLUDE_PATH} ${OP2_MPI_INCLUDE_DIRS})
    set(OP2_MPI_LIBRARIES ${MPI_LIBRARIES} ${OP2_MPI_LIBRARIES})
    if(CMAKE_VERBOSE_CONFIGURE)
      set(SCOTCH_DEBUG TRUE)
    endif()
    # Parallel graph partitioner: Try PTScotch first, fall back to ParMETIS
    if(OP2_WITH_PTSCOTCH)
      find_package(ScotchPT)
      if(SCOTCH_FOUND)
        message(STATUS "PTScotch found")
        set(OP2_MPI_DEFINITIONS "${OP2_MPI_DEFINITIONS} -DHAVE_PTSCOTCH")
        set(OP2_MPI_INCLUDE_DIRS ${SCOTCH_INCLUDE_DIRS} ${OP2_MPI_INCLUDE_DIRS})
        set(OP2_MPI_LIBRARIES ${OP2_MPI_LIBRARIES} ${SCOTCH_LIBRARIES})
      else()
        message(STATUS "Optional package PTScotch not found - disabling")
        set(OP2_WITH_PTSCOTCH OFF)
      endif()
    endif()
    if(OP2_WITH_PARMETIS AND NOT OP2_WITH_PTSCOTCH)
      find_package(ParMETIS)
      if(PARMETIS_FOUND)
        message(STATUS "ParMETIS found")
        set(OP2_MPI_DEFINITIONS "${OP2_MPI_DEFINITIONS} -DHAVE_PARMETIS")
        set(OP2_MPI_INCLUDE_DIRS ${PARMETIS_INCLUDE_DIRS} ${OP2_MPI_INCLUDE_DIRS})
        set(OP2_MPI_LIBRARIES ${OP2_MPI_LIBRARIES} ${PARMETIS_LIBRARIES})
      else()
        message(STATUS "Optional package ParMETIS not found - disabling")
        set(OP2_WITH_PARMETIS OFF)
      endif()
    endif()
  else()
    message(STATUS "Optional package MPI not found - disabling")
    set(OP2_WITH_MPI OFF)
  endif()
endif()

if(OP2_WITH_HDF5)
  if(NOT BUILD_SHARED_LIBS)
    set(HDF5_USE_STATIC_LIBRARIES)
  endif()
  find_package(HDF5)
  if(HDF5_FOUND)
    message(STATUS "HDF5 found")
    set(OP2_HDF5_DEFINITIONS ${HDF5_DEFINITIONS})
    set(OP2_HDF5_INCLUDE_DIRS ${HDF5_INCLUDE_DIRS})
    set(OP2_HDF5_LIBRARIES ${HDF5_LIBRARIES})
    # If HDF5 is build with parallel support it requires MPI, which needs
    # to be enabled
    if (HDF5_IS_PARALLEL)
      if (OP2_WITH_MPI)
        set(OP2_HDF5_INCLUDE_DIRS ${OP2_HDF5_INCLUDE_DIRS} ${MPI_INCLUDE_PATH})
        set(OP2_HDF5_LIBRARIES ${OP2_HDF5_LIBRARIES} ${MPI_LIBRARIES})
        # MPI is always built with HDF5 if available
        set(OP2_MPI_INCLUDE_DIRS ${HDF5_INCLUDE_DIRS} ${OP2_MPI_INCLUDE_DIRS})
        set(OP2_MPI_LIBRARIES ${HDF5_LIBRARIES} ${OP2_MPI_LIBRARIES})
      else()
        message(STATUS "HDF5 is built with parallel support requiring MPI, but MPI is disabled")
        message(STATUS "Disabling HDF5 support")
        set(OP2_WITH_HDF5 OFF)
      endif()
    endif()
  else()
    message(STATUS "Optional package HDF5 not found - disabling")
    set(OP2_WITH_HDF5 OFF)
  endif()
endif()

if(OP2_WITH_MATRICES)
  find_package(PETSc)
  if(PETSC_FOUND)
    message(STATUS "PETSc found")
  else()
    message(STATUS "Optional package PETSc not found - disabling matrix support")
    set(OP2_WITH_MATRICES OFF)
  endif()
endif()

#------------------------------------------------------------------------------
# Common source files

if(OP2_ENABLE_FORTRAN_INTERFACE)
  # Fortran interface sources
  set(FORTRAN_SRC ${OP2_SOURCE_DIR}/../fortran/src/backend/op2_for_declarations.F90
    ${OP2_SOURCE_DIR}/../fortran/src/backend/op2_for_C_wrappers.c)
endif()
# Core library sources
set(COMMON_SRC ${OP2_SOURCE_DIR}/src/core/op_lib_core.c ${FORTRAN_SRC})
# Utility sources
set(UTIL_SRC ${OP2_SOURCE_DIR}/src/externlib/op_util.c)
# Runtime support sources
set(RT_SRC ${OP2_SOURCE_DIR}/src/core/op_rt_support.c)

#------------------------------------------------------------------------------
# Configuration

# Set rpath to use on installation if desired
if(USE_INSTALL_RPATH)
  # Append directories in the linker search path of the built libraries to
  # the rpath. This makes the installed libraries also find libraries in non-
  # standard locations without having to set LD_LIBRARY_PATH
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

set(OP2_TARGETS_EXPORT_SET OP2LibraryDepends)
# Reset build-tree export set
export(TARGETS FILE "${PROJECT_BINARY_DIR}/${OP2_TARGETS_EXPORT_SET}.cmake")

#------------------------------------------------------------------------------
# Configure targets

message(STATUS "Configuring targets...")

if(OP2_WITH_SEQ)
  message(STATUS "Configuring OP2 sequential library")
  add_subdirectory(sequential)
endif()

if(OP2_WITH_OPENMP)
  message(STATUS "Configuring OP2 OpenMP library")
  add_subdirectory(openmp)
endif()

if(OP2_WITH_HDF5)
  message(STATUS "Configuring OP2 HDF5 library")
  add_subdirectory(externlib)
endif()

if(OP2_WITH_MPI)
  message(STATUS "Configuring OP2 MPI library")
  add_subdirectory(mpi)
endif()

if(OP2_WITH_CUDA)
  message(STATUS "Configuring OP2 CUDA library")
  add_subdirectory(cuda)
endif()

#------------------------------------------------------------------------------
# Export configuration to build tree

# Create the build tree OP2Config.cmake files
set(OP2_CONF_DIR ${PROJECT_BINARY_DIR})
configure_file(${OP2_CMAKE_DIR}/OP2Config.cmake.in
  "${PROJECT_BINARY_DIR}/OP2Config.cmake" @ONLY)

# Store the current build directory in the CMake user package registry.
# This helps dependent projects find and use a package from the current
# project's build tree without help from the user
export(PACKAGE OP2)

#------------------------------------------------------------------------------
# Export configuration to install tree

# Create the install tree OP2Config.cmake files

# Set directory from where to import target export set
set(OP2_CONF_DIR ${INSTALLATION_CMAKE_DIR})
configure_file(${OP2_CMAKE_DIR}/OP2Config.cmake.in
  "${PROJECT_BINARY_DIR}/install/OP2Config.cmake" @ONLY)

# Install the configuration and export set for use with the install-tree
install(FILES "${PROJECT_BINARY_DIR}/install/OP2Config.cmake"
  DESTINATION "${INSTALLATION_CMAKE_DIR}")
install(EXPORT ${OP2_TARGETS_EXPORT_SET} DESTINATION "${INSTALLATION_CMAKE_DIR}"
  COMPONENT dev)
