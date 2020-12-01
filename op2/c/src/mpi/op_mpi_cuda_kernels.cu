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

#include <op_lib_mpi.h>

#include <op_lib_c.h>
#include <op_cuda_rt_support.h>
#include <vector>
#include <algorithm>

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

__device__ int lower_bound(int *disps, int count, int value) {
  int *it;
  int *first = disps;
  int step;
  while (count > 0) {
    it = first; 
    step = count / 2; 
    it += step;
    if (*it < value) {
        first = ++it; 
        count -= step + 1; 
    }
    else
        count = step;
  }
  return first-disps;
}

__global__ void gather_data_to_buffer_ptr_cuda_kernel(char *data, char *buffer, int *elem_list, int *disps, 
          unsigned *neigh_to_neigh_offsets, int rank_size, int soa, int type_size, int dim, int set_size) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id > disps[rank_size]) return;
  int neighbour = lower_bound(disps, rank_size, id);
  unsigned buf_pos = neigh_to_neigh_offsets[neighbour];
  unsigned set_elem_index = elem_list[id];
  if (soa) {
    for (int d = 0; d < dim; d++)
      for (int p = 0; p < type_size; p++)
        buffer[buf_pos + (id - disps[neighbour]) * type_size * dim + d * type_size + p] = data[(d*set_size + set_elem_index)*type_size + p];

  } else {
    int dat_size = type_size * dim;
    for (int p = 0; p < dat_size; p++)
      buffer[buf_pos + (id - disps[neighbour]) * dat_size + p] = data[set_elem_index*dat_size + p];
  }
}

__global__ void scatter_data_from_buffer_ptr_cuda_kernel(char *data, char *buffer, int *disps, 
  unsigned *neigh_to_neigh_offsets, int rank_size, int soa, int type_size, int dim, int set_size) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id > disps[rank_size]) return;
  int neighbour = lower_bound(disps, rank_size, id);
  unsigned buf_pos = neigh_to_neigh_offsets[neighbour];
  if (soa) {
    for (int d = 0; d < dim; d++)
      for (int p = 0; p < type_size; p++)
        data[(d*set_size + id)*type_size + p] = buffer[buf_pos + (id - disps[neighbour]) * type_size * dim + d * type_size + p];
  } else {
    int dat_size = type_size * dim;
    for (int p = 0; p < dat_size; p++)
      data[id*dat_size + p] = buffer[buf_pos + (id - disps[neighbour]) * dat_size + p];
  }
}


unsigned *op2_grp_neigh_to_neigh_offsets_h = NULL;
unsigned *op2_grp_neigh_to_neigh_offsets_d = NULL;
int op2_grp_max_gathers = 10;
extern int op2_grp_counter;
int op2_grp_max_neighbours = 0;

void check_realloc_buffer() {
  //Figure out how much space may need at most 
  if (op2_grp_neigh_to_neigh_offsets_h == NULL) {
    for (int i = 0; i < OP_set_index; i++) {
      op2_grp_max_neighbours = MAX(op2_grp_max_neighbours,OP_export_exec_list[i]->ranks_size);
      op2_grp_max_neighbours = MAX(op2_grp_max_neighbours,OP_export_nonexec_list[i]->ranks_size);
      op2_grp_max_neighbours = MAX(op2_grp_max_neighbours,OP_import_exec_list[i]->ranks_size);
      op2_grp_max_neighbours = MAX(op2_grp_max_neighbours,OP_import_nonexec_list[i]->ranks_size);
    }
    //Need host buffers for each dat in flight
    cutilSafeCall(cudaMallocHost(&op2_grp_neigh_to_neigh_offsets_h, op2_grp_max_gathers * op2_grp_max_neighbours * sizeof(unsigned)));
    //But just one device buffer if gather kernels are sequential
    cutilSafeCall(cudaMalloc    (&op2_grp_neigh_to_neigh_offsets_d, op2_grp_max_neighbours * sizeof(unsigned)));
  }
  if (op2_grp_counter >= op2_grp_max_gathers) {
    cutilSafeCall(cudaDeviceSynchronize());
    cutilSafeCall(cudaFreeHost(op2_grp_neigh_to_neigh_offsets_h));
    op2_grp_max_gathers *= 2;
    cutilSafeCall(cudaMallocHost(&op2_grp_neigh_to_neigh_offsets_h, op2_grp_max_gathers * op2_grp_max_neighbours * sizeof(unsigned)));
  }
}

void gather_data_to_buffer_ptr_cuda(op_arg arg, halo_list eel, halo_list enl, char *buffer, 
  std::vector<int>& neigh_list, std::vector<unsigned>& neigh_offsets) {

  check_realloc_buffer();

  int soa = 0;
  if ((OP_auto_soa && arg.dat->dim > 1) || strstr(arg.dat->type, ":soa") != NULL) soa = 1;

  //Exec halo

  //Create op2_grp_neigh_to_neigh_offsets_h into appropriate position
  for (int i = 0; i < eel->ranks_size; i++) {
    int dest_rank = eel->ranks[i];
    int buf_rankpos = std::distance(neigh_list.begin(),std::lower_bound(neigh_list.begin(), neigh_list.end(), dest_rank));
    op2_grp_neigh_to_neigh_offsets_h[op2_grp_counter*op2_grp_max_neighbours+i] = neigh_offsets[buf_rankpos];
    neigh_offsets[buf_rankpos] += eel->sizes[i] * arg.dat->size;
  }
  //Async upload
  cudaMemcpyAsync(op2_grp_neigh_to_neigh_offsets_d,&op2_grp_neigh_to_neigh_offsets_h[op2_grp_counter*op2_grp_max_neighbours],eel->ranks_size * sizeof(unsigned),cudaMemcpyHostToDevice);
  //Launch kernel
  gather_data_to_buffer_ptr_cuda_kernel<<<1 + ((eel->size - 1) / 192),192>>>(arg.dat->data_d, buffer, export_exec_list_d[arg.dat->set->index], export_exec_list_disps_d[arg.dat->set->index], 
    op2_grp_neigh_to_neigh_offsets_d, eel->ranks_size, soa, arg.dat->size/arg.dat->dim, arg.dat->dim, arg.dat->set->size+arg.dat->set->exec_size+arg.dat->set->nonexec_size);
  op2_grp_counter++;

  //Same for nonexec

  //Create op2_grp_neigh_to_neigh_offsets_h into appropriate position
  for (int i = 0; i < enl->ranks_size; i++) {
    int dest_rank = enl->ranks[i];
    int buf_rankpos = std::distance(neigh_list.begin(),std::lower_bound(neigh_list.begin(), neigh_list.end(), dest_rank));
    op2_grp_neigh_to_neigh_offsets_h[op2_grp_counter*op2_grp_max_neighbours+i] = neigh_offsets[buf_rankpos];
    neigh_offsets[buf_rankpos] += enl->sizes[i] * arg.dat->size;
  }
  //Async upload
  cudaMemcpyAsync(op2_grp_neigh_to_neigh_offsets_d,&op2_grp_neigh_to_neigh_offsets_h[op2_grp_counter*op2_grp_max_neighbours],enl->ranks_size * sizeof(unsigned),cudaMemcpyHostToDevice);
  //Launch kernel
  gather_data_to_buffer_ptr_cuda_kernel<<<1 + ((enl->size - 1) / 192),192>>>(arg.dat->data_d, buffer, export_nonexec_list_d[arg.dat->set->index], export_nonexec_list_disps_d[arg.dat->set->index], 
    op2_grp_neigh_to_neigh_offsets_d, enl->ranks_size, soa, arg.dat->size/arg.dat->dim, arg.dat->dim, arg.dat->set->size+arg.dat->set->exec_size+arg.dat->set->nonexec_size);

  op2_grp_counter++;

}

void scatter_data_from_buffer_ptr_cuda(op_arg arg, halo_list iel, halo_list inl, char *buffer, 
  std::vector<int>& neigh_list, std::vector<unsigned>& neigh_offsets) {

  check_realloc_buffer();

  int soa = 0;
  if ((OP_auto_soa && arg.dat->dim > 1) || strstr(arg.dat->type, ":soa") != NULL) soa = 1;

  //Exec halo

  //Create op2_grp_neigh_to_neigh_offsets_h into appropriate position
  for (int i = 0; i < iel->ranks_size; i++) {
    int dest_rank = iel->ranks[i];
    int buf_rankpos = std::distance(neigh_list.begin(),std::lower_bound(neigh_list.begin(), neigh_list.end(), dest_rank));
    op2_grp_neigh_to_neigh_offsets_h[op2_grp_counter*op2_grp_max_neighbours+i] = neigh_offsets[buf_rankpos];
    neigh_offsets[buf_rankpos] += iel->sizes[i] * arg.dat->size;
  }
  //Async upload
  cudaMemcpyAsync(op2_grp_neigh_to_neigh_offsets_d,&op2_grp_neigh_to_neigh_offsets_h[op2_grp_counter*op2_grp_max_neighbours],iel->ranks_size * sizeof(unsigned),cudaMemcpyHostToDevice);
  //Launch kernel
  unsigned offset = arg.dat->set->size * (soa?arg.dat->size/arg.dat->dim:arg.dat->size);
  scatter_data_from_buffer_ptr_cuda_kernel<<<1 + ((iel->size - 1) / 192),192>>>(arg.dat->data_d+offset, buffer, import_exec_list_disps_d[arg.dat->set->index], 
    op2_grp_neigh_to_neigh_offsets_d, iel->ranks_size, soa, arg.dat->size/arg.dat->dim, arg.dat->dim, arg.dat->set->size+arg.dat->set->exec_size+arg.dat->set->nonexec_size);
  op2_grp_counter++;

  //Same for nonexec

  //Create op2_grp_neigh_to_neigh_offsets_h into appropriate position
  for (int i = 0; i < inl->ranks_size; i++) {
    int dest_rank = inl->ranks[i];
    int buf_rankpos = std::distance(neigh_list.begin(),std::lower_bound(neigh_list.begin(), neigh_list.end(), dest_rank));
    op2_grp_neigh_to_neigh_offsets_h[op2_grp_counter*op2_grp_max_neighbours+i] = neigh_offsets[buf_rankpos];
    neigh_offsets[buf_rankpos] += inl->sizes[i] * arg.dat->size;
  }
  //Async upload
  cudaMemcpyAsync(op2_grp_neigh_to_neigh_offsets_d,&op2_grp_neigh_to_neigh_offsets_h[op2_grp_counter*op2_grp_max_neighbours],inl->ranks_size * sizeof(unsigned),cudaMemcpyHostToDevice);
  //Launch kernel
  offset = (arg.dat->set->size + iel->size) * (soa?arg.dat->size/arg.dat->dim:arg.dat->size);
  scatter_data_from_buffer_ptr_cuda_kernel<<<1 + ((inl->size - 1) / 192),192>>>(arg.dat->data_d+offset, buffer, import_nonexec_list_disps_d[arg.dat->set->index], 
    op2_grp_neigh_to_neigh_offsets_d, inl->ranks_size, soa, arg.dat->size/arg.dat->dim, arg.dat->dim, arg.dat->set->size+arg.dat->set->exec_size+arg.dat->set->nonexec_size);

  op2_grp_counter++;

}