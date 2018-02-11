/*
 * Open source copyright declaration based on BSD open source template:
 * http://www.opensource.org/licenses/bsd-license.php
 *
 * This file is part of the OP2 distribution.
 *
 * Copyright (c) 2011, Mike Giles and others. Please see the AUTHORS file in
 * the main source directory for a full list of copyright holders.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * The name of Mike Giles may not be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY Mike Giles ''AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL Mike Giles BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#include <op_lib_core.h>
 #include <op_lib_c.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <op_cuda_rt_support.h>
/* Functions with a different implementation in CUDA than other backends */
void op_upload_dat(op_dat dat);
void op_download_dat(op_dat dat);

void op_put_dat(op_dat dat) {
  op_upload_dat(dat);
}

void op_get_dat(op_dat dat) {
  op_download_dat(dat);
}

void
op_get_dat_mpi (op_dat dat) {
  if (dat->data_d == NULL) return;
  op_get_dat(dat);
}

void
op_put_dat_mpi (op_dat dat) {
  if (dat->data_d == NULL) return;
  op_put_dat(dat);
}

#ifdef __cplusplus
extern "C" {
#endif
char *scratch = NULL;
long scratch_size = 0;
void prepareScratch(op_arg *args, int nargs, int nthreads) {
  long req_size = 0;
  for (int i = 0; i < nargs; i++) {
    if (args[i].argtype == OP_ARG_GBL && (args[i].acc == OP_INC || args[i].acc == OP_MAX || args[i].acc == OP_MIN))
      req_size += ((args[i].size-1)/8+1)*8*nthreads;
  }
  if (scratch_size < req_size) {
    if (!scratch) cudaFree(scratch);
    cutilSafeCall(cudaMalloc((void**)&scratch, req_size*sizeof(char)));
    scratch_size = req_size;
  }
  req_size = 0;
  for (int i = 0; i < nargs; i++) {
    if (args[i].argtype == OP_ARG_GBL && (args[i].acc == OP_INC || args[i].acc == OP_MAX || args[i].acc == OP_MIN)) {
      args[i].data_d = scratch + req_size;
      req_size += ((args[i].size-1)/8+1)*8*nthreads;
    }
  }
}

void op_decl_const_char(int dim, char const *type, int typeSize, char *data,
                            char const *name) {
    op_decl_const_core(dim, type, typeSize, data, name);
}

#ifdef __cplusplus
}
#endif

