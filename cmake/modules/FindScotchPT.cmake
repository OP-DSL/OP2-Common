# - Try to find SCOTCH
# Once done this will define
#
#  SCOTCH_FOUND        - system has found SCOTCH
#  SCOTCH_INCLUDE_DIRS - include directories for SCOTCH
#  SCOTCH_LIBRARIES    - libraries for SCOTCH
#  SCOTCH_VERSION      - version for SCOTCH
#
# Variables used by this module. They can change the default behaviour and
# need to be set before calling find_package:
#
#  SCOTCH_DIR          - Prefix directory of the Scotch installation
#  SCOTCH_INCLUDE_DIR  - Include directory of the Scotch installation
#                        (set only if different from ${SCOTCH_DIR}/include)
#  SCOTCH_LIB_DIR      - Library directory of the Scotch installation
#                        (set only if different from ${SCOTCH_DIR}/lib)
#  SCOTCH_DEBUG        - Set this to TRUE to enable debugging output
#                        of FindScotchPT.cmake if you are having problems.
#                        Please enable this before filing any bug reports.
#  SCOTCH_TEST_RUNS    - Skip tests building and running a test
#                        executable linked against PTScotch libraries
#  SCOTCH_LIB_SUFFIX   - Also search for non-standard library names with the
#                        given suffix appended

#=============================================================================
# Copyright (C) 2010-2012 Garth N. Wells, Johannes Ring, Anders Logg
# and Florian Rathgeber. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#=============================================================================

if (SCOTCH_DEBUG)
  message(STATUS "Checking for package 'SCOTCH-PT'")
endif()

# Check for header file
find_path(SCOTCH_INCLUDE_DIR scotch.h ptscotch.h
  HINTS ${SCOTCH_DIR}/include $ENV{SCOTCH_DIR}/include
    ${SCOTCH_INCLUDE_DIR} $ENV{SCOTCH_INCLUDE_DIR}
  PATH_SUFFIXES scotch
  DOC "Directory where the SCOTCH-PT header is located"
)

# Check for scotch
find_library(SCOTCH_LIBRARY
  NAMES scotch scotch${SCOTCH_LIB_SUFFIX}
  HINTS ${SCOTCH_DIR}/lib $ENV{SCOTCH_DIR}/lib
    ${SCOTCH_LIB_DIR} $ENV{SCOTCH_LIB_DIR}
  DOC "The SCOTCH library"
)

find_library(SCOTCHERR_LIBRARY
  NAMES scotcherr scotcherr${SCOTCH_LIB_SUFFIX}
  HINTS ${SCOTCH_DIR}/lib $ENV{SCOTCH_DIR}/lib
    ${SCOTCH_LIB_DIR} $ENV{SCOTCH_LIB_DIR}
  DOC "The SCOTCH-ERROR library"
)

# Check for ptscotch
find_library(PTSCOTCH_LIBRARY
  NAMES ptscotch ptscotch${SCOTCH_LIB_SUFFIX}
  HINTS ${SCOTCH_DIR}/lib $ENV{SCOTCH_DIR}/lib
    ${SCOTCH_LIB_DIR} $ENV{SCOTCH_LIB_DIR}
  DOC "The PTSCOTCH library"
)

# Check for ptscotcherr
find_library(PTSCOTCHERR_LIBRARY
  NAMES ptscotcherr ptscotcherr${SCOTCH_LIB_SUFFIX}
  HINTS ${SCOTCH_DIR}/lib $ENV{SCOTCH_DIR}/lib
    ${SCOTCH_LIB_DIR} $ENV{SCOTCH_LIB_DIR}
  DOC "The PTSCOTCH-ERROR library"
)

# Get Scotch version
if(NOT SCOTCH_VERSION_STRING AND SCOTCH_INCLUDE_DIR AND EXISTS "${SCOTCH_INCLUDE_DIR}/scotch.h")
  set(version_pattern "^#define[\t ]+SCOTCH_(VERSION|RELEASE|PATCHLEVEL)[\t ]+([0-9\\.]+)$")
  file(STRINGS "${SCOTCH_INCLUDE_DIR}/scotch.h" scotch_version REGEX ${version_pattern})

  foreach(match ${scotch_version})
    if(SCOTCH_VERSION_STRING)
      set(SCOTCH_VERSION_STRING "${SCOTCH_VERSION_STRING}.")
    endif()
    string(REGEX REPLACE ${version_pattern} "${SCOTCH_VERSION_STRING}\\2" SCOTCH_VERSION_STRING ${match})
    set(SCOTCH_${CMAKE_MATCH_1} ${CMAKE_MATCH_2})
  endforeach()
  unset(scotch_version)
  unset(version_pattern)
endif()

# Try compiling and running test program
if (SCOTCH_INCLUDE_DIR AND PTSCOTCH_LIBRARY AND PTSCOTCHERR_LIBRARY)

  if (SCOTCH_DEBUG)
    message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                   "location of ptscotch.h: ${SCOTCH_INCLUDE_DIR}/ptscotch.h")
    message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                   "location of libptscotch: ${PTSCOTCH_LIBRARY}")
    message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                   "location of libptscotcherr: ${PTSCOTCHERR_LIBRARY}")
  endif()

  # Test requires MPI
  find_package(MPI QUIET REQUIRED)

  # Set flags for building test program
  set(CMAKE_REQUIRED_INCLUDES ${SCOTCH_INCLUDE_DIR} ${MPI_INCLUDE_PATH})
  set(CMAKE_REQUIRED_LIBRARIES ${PTSCOTCH_LIBRARY} ${PTSCOTCHERR_LIBRARY} ${MPI_LIBRARIES})
  set(CMAKE_REQUIRED_FLAGS ${MPI_COMPILE_FLAGS})

  # PT-SCOTCH was first introduced in SCOTCH version 5.0
  # FIXME: parallel graph partitioning features in PT-SCOTCH was first
  #        introduced in 5.1. Do we require version 5.1?
  if (NOT ${SCOTCH_VERSION} VERSION_LESS "5.0")
    include(CheckCXXSourceRuns)
    set(SCOTCH_CXX_TEST_SOURCE "
#include <sys/types.h>
#include <stdio.h>
#include <mpi.h>
#include <ptscotch.h>
#include <iostream>
#include <cstdlib>

int main() {
  int provided;
  SCOTCH_Dgraph dgrafdat;

  MPI_Init_thread(0, 0, MPI_THREAD_MULTIPLE, &provided);

  if (SCOTCH_dgraphInit(&dgrafdat, MPI_COMM_WORLD) != 0) {
    if (MPI_THREAD_MULTIPLE > provided) {
      std::cout << \"MPI implementation is not thread-safe:\" << std::endl;
      std::cout << \"SCOTCH should be compiled without SCOTCH_PTHREAD\" << std::endl;
      exit(1);
    }
    else {
      std::cout << \"libptscotch linked to libscotch or other unknown error\" << std::endl;
      exit(2);
    }
  }
  else {
    SCOTCH_dgraphExit(&dgrafdat);
  }

  MPI_Finalize();

  return 0;
}
")
    check_cxx_source_runs("${SCOTCH_CXX_TEST_SOURCE}" SCOTCH_TEST_RUNS)

    # If program does not run, try adding zlib library and test again
    if(NOT SCOTCH_TEST_RUNS)
      find_package(ZLIB QUIET)

      if (ZLIB_INCLUDE_DIRS AND ZLIB_LIBRARIES)
        set(CMAKE_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES} ${ZLIB_INCLUDE_DIRS})
        set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} ${ZLIB_LIBRARIES})
        check_cxx_source_runs("${SCOTCH_CXX_TEST_SOURCE}" SCOTCH_ZLIB_TEST_RUNS)

        # Add zlib flags if required and set test run to 'true'
        if (SCOTCH_ZLIB_TEST_RUNS)
          set(SCOTCH_INCLUDE_DIR ${SCOTCH_INCLUDE_DIR} ${ZLIB_INCLUDE_DIRS})
          set(SCOTCH_EXTRA_LIBRARY ${ZLIB_LIBRARIES})
          # Flag test as successful and do not re-run
          set(SCOTCH_TEST_RUNS 1 CACHE INTERNAL "Test SCOTCH_TEST_RUNS")
        endif()

      endif()
    endif()
  endif()
endif()

# Standard package handling
include(FindPackageHandleStandardArgs)
if(CMAKE_VERSION VERSION_GREATER 2.8.2)
  find_package_handle_standard_args(SCOTCH
    REQUIRED_VARS PTSCOTCH_LIBRARY PTSCOTCHERR_LIBRARY SCOTCH_INCLUDE_DIR SCOTCH_TEST_RUNS
    VERSION_VAR SCOTCH_VERSION_STRING)
else()
  find_package_handle_standard_args(SCOTCH
    REQUIRED_VARS PTSCOTCH_LIBRARY PTSCOTCHERR_LIBRARY SCOTCH_INCLUDE_DIR SCOTCH_TEST_RUNS)
endif()

if(SCOTCH_FOUND)
  set(SCOTCH_LIBRARIES ${PTSCOTCH_LIBRARY} ${PTSCOTCHERR_LIBRARY} ${SCOTCH_EXTRA_LIBRARY})
  set(SCOTCH_INCLUDE_DIRS ${SCOTCH_INCLUDE_DIR})
endif()

mark_as_advanced(PTSCOTCH_LIBRARY PTSCOTCHERR_LIBRARY SCOTCH_LIBRARY SCOTCHERR_LIBRARY SCOTCH_EXTRA_LIBRARY SCOTCH_INCLUDE_DIR)
