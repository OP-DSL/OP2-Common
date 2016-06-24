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

#define MPICH_IGNORE_CXX_SEEK
#include <op_lib_c.h>
#include <op_lib_mpi.h>

__global__ void export_halo_gather(int *list, char *dat, int copy_size,
                                   int elem_size, char *export_buffer) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < copy_size) {
    int off = 0;
    if (elem_size % 16 == 0) {
      off += 16 * (elem_size / 16);
      for (int i = 0; i < elem_size / 16; i++) {
        ((double2 *)(export_buffer + id * elem_size))[i] =
            ((double2 *)(dat + list[id] * elem_size))[i];
      }
    } else if (elem_size % 8 == 0) {
      off += 8 * (elem_size / 8);
      for (int i = 0; i < elem_size / 8; i++) {
        ((double *)(export_buffer + id * elem_size))[i] =
            ((double *)(dat + list[id] * elem_size))[i];
      }
    }
    for (int i = off; i < elem_size; i++) {
      export_buffer[id * elem_size + i] = dat[list[id] * elem_size + i];
    }
  }
}

__global__ void export_halo_gather_soa(int *list, char *dat, int copy_size,
                                       int elem_size, char *export_buffer,
                                       int set_size, int dim) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int size_of = elem_size / dim;
  if (id < copy_size) {
    if (size_of == 8) {
      for (int i = 0; i < dim; i++) {
        ((double *)(export_buffer + id * elem_size))[i] =
            ((double *)(dat + list[id] * size_of))[i * set_size];
      }
    } else {
      for (int i = 0; i < dim; i++) {
        for (int j = 0; j < size_of; j++) {
          export_buffer[id * elem_size + i * size_of + j] =
              dat[list[id] * size_of + i * set_size * size_of + j];
        }
      }
    }
  }
}

__global__ void import_halo_scatter_soa(int offset, char *dat, int copy_size,
                                        int elem_size, char *import_buffer,
                                        int set_size, int dim) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int size_of = elem_size / dim;
  if (id < copy_size) {
    if (size_of == 8) {
      for (int i = 0; i < dim; i++) {
        ((double *)(dat + (offset + id) * size_of))[i * set_size] =
            ((double *)(import_buffer + id * elem_size))[i];
      }
    } else {
      for (int i = 0; i < dim; i++) {
        for (int j = 0; j < size_of; j++) {
          dat[(offset + id) * size_of + i * set_size * size_of + j] =
              import_buffer[id * elem_size + i * size_of + j];
        }
      }
    }
  }
}

__global__ void import_halo_scatter_partial_soa(int *list, char *dat,
                                                int copy_size, int elem_size,
                                                char *import_buffer,
                                                int set_size, int dim) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int size_of = elem_size / dim;
  if (id < copy_size) {
    int element = list[id];
    if (size_of == 8) {
      for (int i = 0; i < dim; i++) {
        ((double *)(dat + (element)*size_of))[i * set_size] =
            ((double *)(import_buffer + id * elem_size))[i];
      }
    } else {
      for (int i = 0; i < dim; i++) {
        for (int j = 0; j < size_of; j++) {
          dat[(element)*size_of + i * set_size * size_of + j] =
              import_buffer[id * elem_size + i * size_of + j];
        }
      }
    }
  }
}

__global__ void import_halo_scatter_partial(int *list, char *dat, int copy_size,
                                            int elem_size, char *import_buffer,
                                            int dim) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int size_of = elem_size / dim;
  if (id < copy_size) {
    int element = list[id];
    if (size_of == 8) {
      for (int i = 0; i < dim; i++) {
        ((double *)(dat + element * elem_size))[i] =
            ((double *)(import_buffer + id * elem_size))[i];
      }
    } else {
      for (int i = 0; i < dim; i++) {
        for (int j = 0; j < size_of; j++) {
          dat[element * elem_size + i * size_of + j] =
              import_buffer[id * elem_size + i * size_of + j];
        }
      }
    }
  }
}

void gather_data_to_buffer(op_arg arg, halo_list exp_exec_list,
                           halo_list exp_nonexec_list) {
  int threads = 192;
  int blocks = 1 + ((exp_exec_list->size - 1) / 192);

  if (strstr(arg.dat->type, ":soa") != NULL ||
      (OP_auto_soa && arg.dat->dim > 1)) {

    int set_size = arg.dat->set->size + arg.dat->set->exec_size +
                   arg.dat->set->nonexec_size;

    export_halo_gather_soa<<<blocks, threads>>>(
        export_exec_list_d[arg.dat->set->index], arg.data_d,
        exp_exec_list->size, arg.dat->size, arg.dat->buffer_d, set_size,
        arg.dat->dim);

    int blocks2 = 1 + ((exp_nonexec_list->size - 1) / 192);
    export_halo_gather_soa<<<blocks2, threads>>>(
        export_nonexec_list_d[arg.dat->set->index], arg.data_d,
        exp_nonexec_list->size, arg.dat->size,
        arg.dat->buffer_d + exp_exec_list->size * arg.dat->size, set_size,
        arg.dat->dim);
  } else {
    export_halo_gather<<<blocks, threads>>>(
        export_exec_list_d[arg.dat->set->index], arg.data_d,
        exp_exec_list->size, arg.dat->size, arg.dat->buffer_d);

    int blocks2 = 1 + ((exp_nonexec_list->size - 1) / 192);
    export_halo_gather<<<blocks2, threads>>>(
        export_nonexec_list_d[arg.dat->set->index], arg.data_d,
        exp_nonexec_list->size, arg.dat->size,
        arg.dat->buffer_d + exp_exec_list->size * arg.dat->size);
  }
}

void gather_data_to_buffer_partial(op_arg arg, halo_list exp_nonexec_list) {
  int threads = 192;
  int blocks = 1 + ((exp_nonexec_list->size - 1) / 192);

  if (strstr(arg.dat->type, ":soa") != NULL ||
      (OP_auto_soa && arg.dat->dim > 1)) {

    int set_size = arg.dat->set->size + arg.dat->set->exec_size +
                   arg.dat->set->nonexec_size;

    export_halo_gather_soa<<<blocks, threads>>>(
        export_nonexec_list_partial_d[arg.map->index], arg.data_d,
        exp_nonexec_list->size, arg.dat->size, arg.dat->buffer_d, set_size,
        arg.dat->dim);
  } else {
    export_halo_gather<<<blocks, threads>>>(
        export_nonexec_list_partial_d[arg.map->index], arg.data_d,
        exp_nonexec_list->size, arg.dat->size, arg.dat->buffer_d);
  }
}

void scatter_data_from_buffer(op_arg arg) {
  int threads = 192;
  int blocks = 1 + ((arg.dat->set->exec_size - 1) / 192);

  if (strstr(arg.dat->type, ":soa") != NULL ||
      (OP_auto_soa && arg.dat->dim > 1)) {

    int set_size = arg.dat->set->size + arg.dat->set->exec_size +
                   arg.dat->set->nonexec_size;
    int offset = arg.dat->set->size;
    int copy_size = arg.dat->set->exec_size;

    import_halo_scatter_soa<<<blocks, threads>>>(
        offset, arg.data_d, copy_size, arg.dat->size, arg.dat->buffer_d_r,
        set_size, arg.dat->dim);

    offset += arg.dat->set->exec_size;
    copy_size = arg.dat->set->nonexec_size;

    int blocks2 = 1 + ((arg.dat->set->nonexec_size - 1) / 192);
    import_halo_scatter_soa<<<blocks2, threads>>>(
        offset, arg.data_d, copy_size, arg.dat->size,
        arg.dat->buffer_d_r + arg.dat->set->exec_size * arg.dat->size, set_size,
        arg.dat->dim);
  }
}

void scatter_data_from_buffer_partial(op_arg arg) {
  int threads = 192;
  int blocks = 1 + ((OP_import_nonexec_permap[arg.map->index]->size - 1) / 192);

  if (strstr(arg.dat->type, ":soa") != NULL ||
      (OP_auto_soa && arg.dat->dim > 1)) {

    int set_size = arg.dat->set->size + arg.dat->set->exec_size +
                   arg.dat->set->nonexec_size;
    int init = OP_export_nonexec_permap[arg.map->index]->size;
    int copy_size = OP_import_nonexec_permap[arg.map->index]->size;

    import_halo_scatter_partial_soa<<<blocks, threads>>>(
        import_nonexec_list_partial_d[arg.map->index], arg.data_d, copy_size,
        arg.dat->size, arg.dat->buffer_d + init * arg.dat->size, set_size,
        arg.dat->dim);
  } else {
    int init = OP_export_nonexec_permap[arg.map->index]->size;
    int copy_size = OP_import_nonexec_permap[arg.map->index]->size;

    import_halo_scatter_partial<<<blocks, threads>>>(
        import_nonexec_list_partial_d[arg.map->index], arg.data_d, copy_size,
        arg.dat->size, arg.dat->buffer_d + init * arg.dat->size, arg.dat->dim);
  }
}
