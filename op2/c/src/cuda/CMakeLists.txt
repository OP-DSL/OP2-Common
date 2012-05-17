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

# CUDA library
include_directories(${CUDA_INCLUDE_DIRS} ${THRUST_INCLUDE_DIRS} ${CUSP_INCLUDE_DIRS})
cuda_add_library(op2_cuda ${COMMON_SRC} ${RT_SRC}
  op_cuda_decl.c op_cuda_rt_support.c op_cuda_utils.cu ../core/op_dummy_singlenode.c)

# Add target to the build-tree export set
export(TARGETS op2_cuda APPEND
  FILE "${PROJECT_BINARY_DIR}/${OP2_TARGETS_EXPORT_SET}.cmake")

# Install
install(TARGETS op2_cuda
  EXPORT ${OP2_TARGETS_EXPORT_SET}
  LIBRARY DESTINATION ${INSTALLATION_LIB_DIR} COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION ${INSTALLATION_LIB_DIR} COMPONENT Development
)
