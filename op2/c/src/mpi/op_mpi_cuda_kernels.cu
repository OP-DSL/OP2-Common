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
                                   int elem_size, char *export_buffer, int my_rank) {
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
            // printf("opgather my_rank=%d copy_size=%d id=%d set_elem_index=%d\n", my_rank, copy_size, id, list[id]);
      }
    }
    for (int i = off; i < elem_size; i++) {
      export_buffer[id * elem_size + i] = dat[list[id] * elem_size + i];
    }
  }
}

#ifdef COMM_AVOID
__global__ void export_halo_gather_chained(int *list, char *dat, int copy_size,
                                   int elem_size, char *export_buffer, 
                                   int nhalos, int num_levels, int r, int buf_start,
                                   int level_disp, int disp_in_level, int level_disp_in_rank, int my_rank) {
  
 
      
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  //  printf("cagather my_rank=%d id=%d level_disp_in_rank=%d buf_start=%d\n", 
  //     my_rank, id, level_disp_in_rank, level_disp_in_rank * elem_size);

  if (id < copy_size) {

    // printf("export_halo_gather_chained >>>>> copy_size=%d elem_size=%d nhalos=%d num_levels=%d r=%d buf_start=%d level_disp=%d disp_in_level=%d level_disp_in_rank=%d\n", 
    //       copy_size, elem_size, nhalos, num_levels, r, buf_start, level_disp, disp_in_level, level_disp_in_rank);

    // for(int l = 0; l < nhalos; l++){

      // if(id >= sizes[ranks_disps_by_level[l] + r])  return;
      
      // int level_disp = disps_by_level[l];
      // int rank_disp = ranks_disps_by_level[l];
      // int disp_in_level = disps[rank_disp + r];
      int set_elem_index = list[level_disp + disp_in_level + id] * elem_size;
       
      // int level_disp_in_rank = level_disps[r * num_levels + l];
      int buf_elem_index = buf_start + (level_disp_in_rank + id) * elem_size; // buf_start is already mulitplied with prev elem sizes
      // printf("cagather my_rank=%d id=%d set_elem_index=%d buf_elem_index=%d\n", my_rank, id, set_elem_index / elem_size, buf_elem_index / elem_size);


      int arg_disp = 0;
      int rank_disp = 0;
      int level_disp = 0;

      int off = 0;

      if (elem_size % 16 == 0) {
        off += 16 * (elem_size / 16);
        for (int i = 0; i < elem_size / 16; i++) {
          ((double2 *)(export_buffer + buf_elem_index))[i] = ((double2 *)(dat + set_elem_index))[i];
        }
      } else if (elem_size % 8 == 0) {
        off += 8 * (elem_size / 8);
        for (int i = 0; i < elem_size / 8; i++) {
          // printf("export_halo_gather_chained buf_elem_index=%d(buf_start=%d level_disp_in_rank=%d id=%d elem_size=%d) set_elem_index=%d (level_disp=%d disp_in_level=%d id=%d)\n", 
          // buf_elem_index / elem_size, buf_start, level_disp_in_rank, id, elem_size, set_elem_index / elem_size, level_disp, disp_in_level, id);
          ((double *)(export_buffer + buf_elem_index))[i] = ((double *)(dat + set_elem_index))[i];
        }
      }

      for (int i = off; i < elem_size; i++) {
        export_buffer[buf_elem_index + i] = dat[set_elem_index + i];
      }
    // } 
  }
}
#endif

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
                           halo_list exp_nonexec_list, int my_rank) {
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
        exp_exec_list->size, arg.dat->size, arg.dat->buffer_d, my_rank);

    int blocks2 = 1 + ((exp_nonexec_list->size - 1) / 192);
    export_halo_gather<<<blocks2, threads>>>(
        export_nonexec_list_d[arg.dat->set->index], arg.data_d,
        exp_nonexec_list->size, arg.dat->size,
        arg.dat->buffer_d + exp_exec_list->size * arg.dat->size, my_rank);
  }
}

#ifdef COMM_AVOID
int get_nonexec_start_chained(op_arg *arg){
  int max_nhalos = arg->dat->set->halo_info->max_nhalos;
  int nhalos_index = arg->nhalos_index;
  switch (arg->unpack_method)
  {
  case OP_UNPACK_OP2:
    // if(arg->dat->halo_info->max_nhalos > 1){
    //   nhalos_index = arg->dat->halo_info->nhalos_indices[dat->halo_info->max_nhalos];
    // }
    // return max_nhalos + nhalos_index;
    return max_nhalos;
  case OP_UNPACK_SINGLE_HALO:
    return max_nhalos + nhalos_index;
  case OP_UNPACK_ALL_HALOS:
    return max_nhalos;
  
  default:
    return -1;
  }
}

int get_nonexec_end_chained(op_arg *arg){
  int max_nhalos = arg->dat->set->halo_info->max_nhalos;
  int nhalos_index = arg->nhalos_index;

  switch (arg->unpack_method)
  {
  case OP_UNPACK_OP2:
    if(arg->dat->halo_info->max_nhalos > 1){
      nhalos_index = arg->dat->halo_info->nhalos_indices[arg->dat->halo_info->max_nhalos];
    }
    break;
  case OP_UNPACK_SINGLE_HALO:
    break;
  case OP_UNPACK_ALL_HALOS:
    break;
  default:
    return -1;
  }
  return max_nhalos + nhalos_index + 1;
}

int get_nhalos_chained(op_arg *arg){
  switch (arg->unpack_method)
  {
  case OP_UNPACK_OP2:
    if(arg->dat->halo_info->max_nhalos > 1){
      return arg->dat->halo_info->max_nhalos;
    }
  case OP_UNPACK_SINGLE_HALO:
  case OP_UNPACK_ALL_HALOS:
    return arg->nhalos;
  
  default:
    return -1;
  }
}

int is_arg_valid_chained(op_arg* arg, int exec_flag){

  if (arg->opt == 0)
    return 0;

  if (arg->sent == 1) {
    printf("Error: Halo exchange already in flight for dat %s\n", arg->dat->name);
    fflush(stdout);
    MPI_Abort(OP_MPI_WORLD, 2);
  }

  if (exec_flag == 0 && arg->idx == -1)
    return 0;

  arg->sent = 0;

  if (arg->opt && arg->argtype == OP_ARG_DAT && arg->dat->dirtybit == 1 && (arg->acc == OP_READ || arg->acc == OP_RW)) {
    if (arg->idx == -1 && exec_flag == 0){
      return 0;
    }
    return 1;
  }
  return 0;
}

int is_arg_valid_chained_1(op_arg* arg, int exec_flag){

  if (arg->opt == 0)
    return 0;

  if (arg->sent == 1) {
    printf("Error: Halo exchange already in flight for dat %s\n", arg->dat->name);
    fflush(stdout);
    MPI_Abort(OP_MPI_WORLD, 2);
  }

  if (exec_flag == 0 && arg->idx == -1)
    return 0;

  arg->sent = 0;

  // todo: note dirtybit 4 flagged
  if (arg->opt && arg->argtype == OP_ARG_DAT && arg->dat->dirtybit == 4 && (arg->acc == OP_READ || arg->acc == OP_RW)) {
    if (arg->idx == -1 && exec_flag == 0){
      return 0;
    }
    return 1;
  }
  return 0;
}

void gather_data_to_buffer_chained(int nargs, op_arg *args, int exec_flag, 
                                    int exp_rank_count, int* buf_pos, int* send_sizes, int* total_buf_size, int my_rank) {
  int threads = 192;

  int buf_start = 0;
  int arg_size = 0;
  int buf_index = 0;
  int prev_size = 0;
  int user_managed[nargs];
  for(int n = 0; n < nargs; n++){
    user_managed[n] = (&args[n])->dat->user_managed;
    (&args[n])->dat->user_managed = -1;
  }
  
  for (int r = 0; r < exp_rank_count; r++) {
    
    buf_pos[r] = arg_size;
    printf("r=%d bufpos=%d\n", r, buf_pos[r]);

    for(int n = 0; n < nargs; n++){
      buf_start =  arg_size;
      buf_index = 0;
      op_arg* arg = &args[n];
      if(is_arg_valid_chained(arg, exec_flag) == 0 || arg->dat->user_managed == r)
        continue;

      arg->dat->user_managed = r;
      op_dat dat = arg->dat;

      int nhalos = get_nhalos_chained(arg);
      int nonexec_start = get_nonexec_start_chained(arg);
      int nonexec_end = get_nonexec_end_chained(arg);

      halo_list exp_list = OP_merged_export_exec_nonexec_list[dat->set->index];

      for(int l = 0; l < nhalos; l++){

        int level_disp = exp_list->disps_by_level[l];
        int rank_disp = exp_list->ranks_disps_by_level[l];
        int disp_in_level = exp_list->disps[rank_disp + r];

        int level_disp_in_rank = exp_list->level_disps[r * exp_list->num_levels + l];
        int copy_size = exp_list->sizes[exp_list->ranks_disps_by_level[l] + r];

        printf("1gather_data my_rank=%d n=%d set=%s index=%d nhalos=%d copy_size=%d l=%d\n", my_rank, n, arg->dat->set->name, arg->dat->set->index, nhalos, copy_size, l);
        //todo: remove 0, op2_grp_secondary
        int blocks = 1 + ((copy_size - 1) / 192);
        export_halo_gather_chained<<<blocks, threads>>>(export_exec_nonexec_list_d[arg->dat->set->index], arg->data_d, copy_size,
                                   arg->dat->size, grp_send_buffer_d, 
                                   nhalos, exp_list->num_levels, r, buf_start,
                                   level_disp, disp_in_level, 
                                   level_disp_in_rank, my_rank);
        prev_size += copy_size * dat->size;
      }

      for(int l = nonexec_start; l < nonexec_end; l++){
        

        int level_disp = exp_list->disps_by_level[l];
        int rank_disp = exp_list->ranks_disps_by_level[l];
        int disp_in_level = exp_list->disps[rank_disp + r];

        int level_disp_in_rank = exp_list->level_disps[r * exp_list->num_levels + l];
        int copy_size = exp_list->sizes[exp_list->ranks_disps_by_level[l] + r];

        printf("gather_data my_rank=%d n=%d dat=%s set=%s index=%d nhalos=%d copy_size=%d nonexec_start=%d nonexec_end=%d l=%d buf_start=%d level_disp_in_rank=%d\n", 
        my_rank, n, arg->dat->name, arg->dat->set->name, arg->dat->set->index, nhalos, copy_size, nonexec_start, nonexec_end, l, buf_start/arg->dat->size, level_disp_in_rank);

        int blocks = 1 + ((copy_size - 1) / 192);
        //todo: remove 0, op2_grp_secondary
        export_halo_gather_chained<<<blocks, threads>>>(export_exec_nonexec_list_d[arg->dat->set->index], arg->data_d, copy_size,
                                   arg->dat->size, grp_send_buffer_d, 
                                   nhalos, exp_list->num_levels, r, buf_start,
                                   level_disp, disp_in_level, 
                                   level_disp_in_rank, my_rank);
        prev_size += copy_size * dat->size;
      }
      arg_size += prev_size;
      prev_size = 0;

      // send_sizes[r] += arg_size - buf_start;
      // printf("my_rank=%d r=%d rank=%d bufpos=%d send_sizes=%d size=%d\n", my_rank, r, exp_list->ranks[r], buf_pos[r], send_sizes[r], arg_size - buf_start);
      
    }
    send_sizes[r] = arg_size - buf_pos[r];
    *total_buf_size += send_sizes[r];
    printf("my_rank=%d r=%d bufpos=%d send_sizes=%d total=%d\n", my_rank, r, buf_pos[r], send_sizes[r], *total_buf_size);
  }

  for(int n = 0; n < nargs; n++){
    (&args[n])->dat->user_managed = user_managed[n];
  }
}
#endif

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
        exp_nonexec_list->size, arg.dat->size, arg.dat->buffer_d, 0);
  }
}

void scatter_data_from_buffer(op_arg arg) {
  printf("scatter_data_from_buffer\n");
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

__global__ void gather_data_to_buffer_ptr_cuda_kernel(const char *__restrict data, char *__restrict buffer, int *elem_list, int *disps, 
          unsigned *neigh_to_neigh_offsets, int rank_size, int soa, int type_size, int dim, int set_size) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= disps[rank_size]) return;
  int neighbour = lower_bound(disps, rank_size, id);
  if (disps[neighbour]!=id) neighbour--;
  unsigned buf_pos = neigh_to_neigh_offsets[neighbour];
  unsigned set_elem_index = elem_list[id];
  if (soa) {
    for (int d = 0; d < dim; d++)
      if (type_size == 8 && (buf_pos + (id - disps[neighbour]) * type_size * dim + d * type_size)%8==0) 
        *(double*)&buffer[buf_pos + (id - disps[neighbour]) * type_size * dim + d * type_size] = *(double*)&data[(d*set_size + set_elem_index)*type_size];
      else
        for (int p = 0; p < type_size; p++)
          buffer[buf_pos + (id - disps[neighbour]) * type_size * dim + d * type_size + p] = data[(d*set_size + set_elem_index)*type_size + p];

  } else {
    int dat_size = type_size * dim;
    if (type_size == 8 && (buf_pos + (id - disps[neighbour]) * dat_size)%8==0) 
      for (int d = 0; d < dim; d++)
        *(double*)&buffer[buf_pos + (id - disps[neighbour]) * dat_size + d*type_size] = *(double*)&data[set_elem_index*dat_size + d*type_size];
    else
      for (int p = 0; p < dat_size; p++)
        buffer[buf_pos + (id - disps[neighbour]) * dat_size + p] = data[set_elem_index*dat_size + p];
  }
}

#ifdef COMM_AVOID
__global__ void scatter_data_from_buffer_ptr_cuda_kernel_chained(char * __restrict data, const char * __restrict buffer, 
  int data_init, int disp_by_level, int disp_in_level, int buf_init, int type_size, int dim, int copy_size, int my_rank){
  
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id < copy_size){

    int dat_size = type_size * dim;
    int id_size = id * dat_size;

    if (type_size == 8 && (buf_init + id_size) % 8 == 0){
      for (int d = 0; d < dim; d++){
        
        // printf("scatter_data_from_buffer_ptr_cuda_kernel_chained id=%d (copy=%d) data_index=%d(data_init=%d + id_size=%d + d=%d * type_size=%d val=%d) buf_index=%d (buf_init=%d + id_size=%d + d=%d * type_size=%d)\n", 
        //     id, copy_size, data_init + id_size + d * type_size, data_init, id_size , d , type_size, id_size + d * type_size, buf_init + id_size + d * type_size, buf_init, id_size , d , type_size);
        
        // *(double*)&data[id_size + d * type_size] = *(double*)&buffer[buf_init + id_size + d * type_size];
        *(double*)&data[data_init + id_size + d * type_size] = *(double*)&buffer[buf_init + id_size + d * type_size];

        // printf("after123 my_rank=%d id=%d (copy=%d) data_index=%d(data_init=%d + id_size=%d + d=%d * type_size=%d val=%d) buf_index=%d (buf_init=%d + id_size=%d + d=%d * type_size=%d)\n", 
        //     my_rank, id, copy_size, data_init + id_size + d * type_size, data_init, id_size , d , type_size, id_size + d * type_size, buf_init + id_size + d * type_size, buf_init, id_size , d , type_size);
        
        // printf("after123 my_rank=%d id=%d (copy=%d) data_index=%d(data_init=%d + id_size=%d + d=%d * type_size=%d val=%d) buf_index=%d (buf_init=%d + id_size=%d + d=%d * type_size=%d)\n", 
        //     my_rank, id, copy_size, (id_size + d * type_size)/type_size, data_init, id_size , d , type_size, id_size + d * type_size, buf_init + id_size + d * type_size, buf_init, id_size , d , type_size);
        
        // printf("2data[%d]=%f buf[%d]=%f (buf_init=%d id_size=%d d=%d type_size=%d id=%d dim=%d)\n", data_init + id_size + d * type_size, 
        // data[data_init + id_size + d * type_size],
        // buf_init + id_size + d * type_size, 
        // buffer[buf_init + id_size + d * type_size], 
        // buf_init , id_size , d , type_size, id, dim);
      }
    }
    else{
       for (int p = 0; p < dat_size; p++){
        //  printf("here123 >>>>>>>>> p=%d type_size=%d\n", p, type_size);
         data[data_init + id_size + p] = buffer[buf_init + id_size + p];
       }
        
    }
     
  }
}
#endif

__global__ void scatter_data_from_buffer_ptr_cuda_kernel(char * __restrict data, const char * __restrict buffer, int *disps, 
  unsigned *neigh_to_neigh_offsets, int rank_size, int soa, int type_size, int dim, int set_size) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= disps[rank_size]) return;
  int neighbour = lower_bound(disps, rank_size, id);
  if (disps[neighbour]!=id) neighbour--;
  unsigned buf_pos = neigh_to_neigh_offsets[neighbour];
  if (soa) {
    for (int d = 0; d < dim; d++)
      if (type_size == 8 && (buf_pos + (id - disps[neighbour]) * type_size * dim + d * type_size)%8==0) 
        *(double*)&data[(d*set_size + id)*type_size] = *(double*)&buffer[buf_pos + (id - disps[neighbour]) * type_size * dim + d * type_size];
      else
        for (int p = 0; p < type_size; p++)
          data[(d*set_size + id)*type_size + p] = buffer[buf_pos + (id - disps[neighbour]) * type_size * dim + d * type_size + p];
  } else {
    int dat_size = type_size * dim;
    // if (*(double*)&buffer[buf_pos + (id - disps[neighbour]) * dat_size] != *(double*)&data[id*dat_size])
    //   printf("Mismatch\n");
    if (type_size == 8 && (buf_pos + (id - disps[neighbour]) * dat_size)%8==0) 
      for (int d = 0; d < dim; d++)
        *(double*)&data[id*dat_size + d*type_size] =  *(double*)&buffer[buf_pos + (id - disps[neighbour]) * dat_size + d*type_size];
    else
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

#ifdef COMM_AVOID
void scatter_data_from_buffer_ptr_cuda_chained(op_arg* args, int nargs, int rank_count, char *buffer, int exec_flag, int my_rank){
  printf("scatter_data_from_buffer_ptr_cuda_chained  nargs=%d rank_count=%d exec_flag=%d\n", nargs, rank_count, exec_flag);
  int buf_init = 0;
  int user_managed[nargs];
  for(int n = 0; n < nargs; n++){
    user_managed[n] = (&args[n])->dat->user_managed;
    (&args[n])->dat->user_managed = -1;
  }
  for (int r = 0; r < rank_count; r++) {
    int imp_disp = 0;
    for(int n = 0; n < nargs; n++){
      op_arg* arg = &args[n];
      op_dat dat = arg->dat;
      if(is_arg_valid_chained_1(arg, exec_flag) == 0 || arg->dat->user_managed == r)
        continue;
      arg->dat->user_managed = r;

      int nhalos = get_nhalos_chained(arg);
      int nonexec_start = get_nonexec_start_chained(arg);
      int nonexec_end = get_nonexec_end_chained(arg);
      int init = (dat->set->size + 0) * dat->size;
      halo_list imp_list = OP_merged_import_exec_nonexec_list[dat->set->index];

      for(int l = 0; l < nhalos; l++){

        int disp_by_level = imp_list->disps_by_level[l];
        int disp_in_level = imp_list->disps[imp_list->ranks_disps_by_level[l] + r];
        int copy_size = imp_list->sizes[imp_list->ranks_disps_by_level[l] + r];

        int data_init = init + (disp_by_level + disp_in_level) * dat->size;

         printf("scatter exec data my_rank=%d n=%d dat=%s set=%s index=%d nhalos=%d copy_size=%d nonexec_start=%d nonexec_end=%d l=%d data_init=%d buf_init=%d\n",
        my_rank, n, arg->dat->name, arg->dat->set->name, arg->dat->set->index, nhalos, copy_size, nonexec_start, nonexec_end, l, data_init, buf_init);

        scatter_data_from_buffer_ptr_cuda_kernel_chained<<<1 + ((copy_size - 1) / 192),192>>>(dat->data_d, grp_recv_buffer_d, 
          data_init, disp_by_level, disp_in_level, buf_init, arg->dat->size / arg->dat->dim, arg->dat->dim, 
          copy_size, my_rank);

        buf_init += copy_size * dat->size;   
      }

      for(int l = nonexec_start; l < nonexec_end; l++){

        int disp_by_level = imp_list->disps_by_level[l];
        int disp_in_level = imp_list->disps[imp_list->ranks_disps_by_level[l] + r];
        int copy_size = imp_list->sizes[imp_list->ranks_disps_by_level[l] + r];

        int data_init = init + (disp_by_level + disp_in_level) * dat->size;
        // printf("scatter_data_from_buffer_ptr_cuda offset start >>>>>>> n=%d  buf=%d, copy_size=%d\n", n, buf_init, copy_size);
        printf("scatter none data my_rank=%d n=%d dat=%s set=%s index=%d nhalos=%d copy_size=%d nonexec_start=%d nonexec_end=%d l=%d data_init=%d buf_init=%d\n",
        my_rank, n, arg->dat->name, arg->dat->set->name, arg->dat->set->index, nhalos, copy_size, nonexec_start, nonexec_end, l, data_init, buf_init);

        scatter_data_from_buffer_ptr_cuda_kernel_chained<<<1 + ((copy_size - 1) / 192),192>>>(dat->data_d, grp_recv_buffer_d, 
          data_init, disp_by_level, disp_in_level, buf_init, arg->dat->size / arg->dat->dim, arg->dat->dim, 
          copy_size, my_rank);

        buf_init += copy_size * dat->size;
        // printf("scatter_data_from_buffer_ptr_cuda offset end <<<<<<< n=%d buf=%d\n", n, buf_init);
      }
    }
  }

  for(int n = 0; n < nargs; n++){
    (&args[n])->dat->user_managed = user_managed[n];
    if((&args[n])->dat->dirtybit == 4){
      (&args[n])->dat->dirtybit = 0;
      printf("scatter n=%d datname=%s bit=%d\n", n, (&args[n])->dat->name , (&args[n])->dat->dirtybit);
    }
  }
}
#endif

void scatter_data_from_buffer_ptr_cuda(op_arg arg, halo_list iel, halo_list inl, char *buffer, 
  std::vector<int>& neigh_list, std::vector<unsigned>& neigh_offsets) {

  printf("scatter_data_from_buffer_ptr_cuda offset\n");

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
  cudaMemcpyAsync(op2_grp_neigh_to_neigh_offsets_d,&op2_grp_neigh_to_neigh_offsets_h[op2_grp_counter*op2_grp_max_neighbours],iel->ranks_size * sizeof(unsigned),cudaMemcpyHostToDevice,op2_grp_secondary);
  //Launch kernel
  unsigned offset = arg.dat->set->size * (soa?arg.dat->size/arg.dat->dim:arg.dat->size);
  scatter_data_from_buffer_ptr_cuda_kernel<<<1 + ((iel->size - 1) / 192),192,0,op2_grp_secondary>>>(arg.dat->data_d+offset, buffer, import_exec_list_disps_d[arg.dat->set->index], 
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
  cudaMemcpyAsync(op2_grp_neigh_to_neigh_offsets_d,&op2_grp_neigh_to_neigh_offsets_h[op2_grp_counter*op2_grp_max_neighbours],inl->ranks_size * sizeof(unsigned),cudaMemcpyHostToDevice,op2_grp_secondary);
  //Launch kernel
  offset = (arg.dat->set->size + iel->size) * (soa?arg.dat->size/arg.dat->dim:arg.dat->size);
  printf("scatter_data_from_buffer_ptr_cuda offset=%d\n", offset);
  scatter_data_from_buffer_ptr_cuda_kernel<<<1 + ((inl->size - 1) / 192),192,0,op2_grp_secondary>>>(arg.dat->data_d+offset, buffer, import_nonexec_list_disps_d[arg.dat->set->index], 
    op2_grp_neigh_to_neigh_offsets_d, inl->ranks_size, soa, arg.dat->size/arg.dat->dim, arg.dat->dim, arg.dat->set->size+arg.dat->set->exec_size+arg.dat->set->nonexec_size);

  op2_grp_counter++;

}