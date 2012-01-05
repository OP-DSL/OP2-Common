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

project(OP2-APPS)

# Require CMake 2.8
cmake_minimum_required(VERSION 2.8)

option(OP2_BUILD_ALL_AIRFOIL "Build all airfoil apps" ON)
option(OP2_BUILD_ALL_JACOBI "Build all jacobi apps" ON)
option(OP2_BUILD_ALL_AERO "Build all aero apps" ON)
option(OP2_BUILD_ALL_MATRIX "Build all matrix apps" ON)
set(OP2_AIRFOIL_APPS
  airfoil/airfoil_plain
  airfoil/airfoil_hdf5
  )
set(OP2_JACOBI_APPS
  jac1
  jac2
  )
set(OP2_AERO_APPS
  aero/aero_plain
  aero/aero_hdf5
  )
set(OP2_MATRIX_APPS
  laplace1d
  )

include(common.cmake)
# Add each OP2 app directory
if(OP2_BUILD_ALL_AIRFOIL)
  foreach (APP_DIR ${OP2_AIRFOIL_APPS})
    option(OP2_BUILD_${APP_DIR} "Build the ${APP_DIR} apps" ON)
    if(OP2_BUILD_${APP_DIR})
      add_subdirectory(${APP_DIR})
    endif()
  endforeach()
endif()
if(OP2_BUILD_ALL_JACOBI)
  foreach (APP_DIR ${OP2_JACOBI_APPS})
    option(OP2_BUILD_${APP_DIR} "Build the ${APP_DIR} apps" ON)
    if(OP2_BUILD_${APP_DIR})
      add_subdirectory(${APP_DIR})
    endif()
  endforeach()
endif()
if(OP2_BUILD_ALL_AERO)
  foreach (APP_DIR ${OP2_AERO_APPS})
    option(OP2_BUILD_${APP_DIR} "Build the ${APP_DIR} apps" ON)
    if(OP2_BUILD_${APP_DIR})
      add_subdirectory(${APP_DIR})
    endif()
  endforeach()
endif()
if(OP2_BUILD_ALL_MATRIX)
  foreach (APP_DIR ${OP2_MATRIX_APPS})
    option(OP2_BUILD_${APP_DIR} "Build the ${APP_DIR} apps" ON)
    if(OP2_BUILD_${APP_DIR})
      add_subdirectory(${APP_DIR})
    endif()
  endforeach()
endif()
