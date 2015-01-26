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

void op_put_dat(op_dat dat) {
  int set_size = dat->set->size + dat->set->exec_size + dat->set->nonexec_size;
  if (strstr( dat->type, ":soa")!= NULL  || (OP_auto_soa && dat->dim > 1)) {
    char *temp_data = (char *)malloc(dat->size*set_size*sizeof(char));
    int element_size = dat->size/dat->dim;
    for (int i = 0; i < dat->dim; i++) {
      for (int j = 0; j < set_size; j++) {
        for (int c = 0; c < element_size; c++) {
          temp_data[element_size*i*set_size + element_size*j + c] = dat->data[dat->size*j+element_size*i+c];
        }
      }
    }
    cutilSafeCall( cudaMemcpy(dat->data_d, temp_data, set_size*dat->size, cudaMemcpyHostToDevice));
    free(temp_data);
  } else {
    cutilSafeCall( cudaMemcpy(dat->data_d, dat->data, set_size*dat->size, cudaMemcpyHostToDevice));
  }
}

void op_get_dat(op_dat dat) {
  int set_size = dat->set->size + dat->set->exec_size + dat->set->nonexec_size;
  if (strstr( dat->type, ":soa")!= NULL  || (OP_auto_soa && dat->dim > 1)) {
    char *temp_data = (char *)malloc(dat->size*set_size*sizeof(char));
    cutilSafeCall( cudaMemcpy(temp_data, dat->data_d, set_size*dat->size, cudaMemcpyDeviceToHost));
    int element_size = dat->size/dat->dim;
    for (int i = 0; i < dat->dim; i++) {
      for (int j = 0; j < set_size; j++) {
        for (int c = 0; c < element_size; c++) {
          dat->data[dat->size*j+element_size*i+c] = temp_data[element_size*i*set_size + element_size*j + c];
        }
      }
    }
    free(temp_data);
  } else {
    cutilSafeCall( cudaMemcpy(dat->data, dat->data_d, set_size*dat->size, cudaMemcpyDeviceToHost));
  }
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
