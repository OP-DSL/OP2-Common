# - Try to find cusp and thrust headers
# Once done this will define
#
#  CUSP_FOUND  - system has cusp
#  CUSP_INCLUDE_DIRS - include directories for cusp
#  THRUST_FOUND  - system has thrust
#  THRUST_INCLUDE_DIRS - include directories for thrust

#=============================================================================
# Copyright (C) 2010-2012 Florian Rathgeber
# All rights reserved.
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

message(STATUS "Checking for package 'cusp'")

# Find dependcy package CUDA
find_package(CUDA REQUIRED)

# Check for thrust header files
find_path(THRUST_INCLUDE_DIRS NAMES thrust
  HINTS ${CUDA_INCLUDE_DIRS}
  PATHS ${THRUST_DIR} ENV THRUST_DIR
  PATH_SUFFIXES include
  DOC "Directory where the thrust headers are located"
)
mark_as_advanced(THRUST_INCLUDE_DIRS)

# Check for cusp header files
find_path(CUSP_INCLUDE_DIRS NAMES cusp
  HINTS ${CUDA_INCLUDE_DIRS}
  PATHS ${CUSP_DIR} ENV CUSP_DIR
  PATH_SUFFIXES include
  DOC "Directory where the cusp headers are located"
)
mark_as_advanced(CUSP_INCLUDE_DIRS)

# Standard package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Cusp
  "Cusp C header could not be found. Be sure to set CUSP_DIR and THRUST_DIR."
  THRUST_INCLUDE_DIRS CUSP_INCLUDE_DIRS
)
