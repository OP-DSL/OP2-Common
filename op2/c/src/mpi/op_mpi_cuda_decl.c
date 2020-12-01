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

//
// This file implements the OP2 user-level functions for the CUDA backend
//

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <mpi.h>

#include <op_cuda_rt_support.h>
#include <op_lib_core.h>
#include <op_rt_support.h>

#include <op_lib_c.h>
#include <op_lib_mpi.h>
#include <op_util.h>

//
// MPI Communicator for halo creation and exchange
//

MPI_Comm OP_MPI_WORLD;
MPI_Comm OP_MPI_GLOBAL;

//
// CUDA-specific OP2 functions
//

void op_init(int argc, char **argv, int diags) {
  op_init_soa(argc, argv, diags, 0);
}

void op_init_soa(int argc, char **argv, int diags, int soa) {
  int flag = 0;
  OP_auto_soa = soa;
  MPI_Initialized(&flag);
  if (!flag) {
    MPI_Init(&argc, &argv);
  }
  OP_MPI_WORLD = MPI_COMM_WORLD;
  OP_MPI_GLOBAL = MPI_COMM_WORLD;
  op_init_core(argc, argv, diags);

#if CUDART_VERSION < 3020
#error : "must be compiled using CUDA 3.2 or later"
#endif

#ifdef CUDA_NO_SM_13_DOUBLE_INTRINSICS
#warning : " *** no support for double precision arithmetic *** "
#endif

  cutilDeviceInit(argc, argv);

//
// The following call is only made in the C version of OP2,
// as it causes memory trashing when called from Fortran.
// \warning add -DSET_CUDA_CACHE_CONFIG to compiling line
// for this file when implementing C OP2.
//
  if (OP_hybrid_gpu) {
#ifdef SET_CUDA_CACHE_CONFIG
    cutilSafeCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
#endif
    cutilSafeCall(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
    printf("\n 16/48 L1/shared \n");
  }
}

void op_mpi_init(int argc, char **argv, int diags, MPI_Fint global,
                 MPI_Fint local) {
  op_mpi_init_soa(argc, argv, diags, global, local, 0);
}

void op_mpi_init_soa(int argc, char **argv, int diags, MPI_Fint global,
                     MPI_Fint local, int soa) {
  OP_auto_soa = soa;
  int flag = 0;
  MPI_Initialized(&flag);
  if (!flag) {
    printf("Error: MPI has to be initialized when calling op_mpi_init with "
           "communicators\n");
    exit(-1);
  }
  OP_MPI_WORLD = MPI_Comm_f2c(local);
  OP_MPI_GLOBAL = MPI_Comm_f2c(global);
  op_init_core(argc, argv, diags);

#if CUDART_VERSION < 3020
#error : "must be compiled using CUDA 3.2 or later"
#endif

#ifdef CUDA_NO_SM_13_DOUBLE_INTRINSICS
#warning : " *** no support for double precision arithmetic *** "
#endif

  cutilDeviceInit(argc, argv);

//
// The following call is only made in the C version of OP2,
// as it causes memory trashing when called from Fortran.
// \warning add -DSET_CUDA_CACHE_CONFIG to compiling line
// for this file when implementing C OP2.
//

#ifdef SET_CUDA_CACHE_CONFIG
  cutilSafeCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
#endif

  //cutilSafeCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
  cutilSafeCall(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
  printf("\n 16/48 L1/shared \n");
}

op_dat op_decl_dat_char(op_set set, int dim, char const *type, int size,
                        char *data, char const *name) {
  char *d = (char *)xmalloc(set->size * dim * size);
  if (d == NULL && set->size>0) {
    printf(" op_decl_dat_char error -- error allocating memory to dat\n");
    exit(-1);
  }

  memcpy(d, data, set->size * dim * size * sizeof(char));
  op_dat out_dat = op_decl_dat_core(set, dim, type, size, d, name);
  op_dat_entry *item;
  op_dat_entry *tmp_item;
  for (item = TAILQ_FIRST(&OP_dat_list); item != NULL; item = tmp_item) {
    tmp_item = TAILQ_NEXT(item, entries);
    if (item->dat == out_dat) {
      item->orig_ptr = data;
      break;
    }
  }
  out_dat->user_managed = 0;
  return out_dat;
}

op_dat op_decl_dat_temp_char(op_set set, int dim, char const *type, int size,
                             char const *name) {
  char *data = NULL;
  op_dat dat = op_decl_dat_temp_core(set, dim, type, size, data, name);

  // create empty data block to assign to this temporary dat (including the
  // halos)
  int set_size = set->size + OP_import_exec_list[set->index]->size +
                 OP_import_nonexec_list[set->index]->size;

  // initialize data bits to 0
  dat->data = (char *)xcalloc(set_size * dim * size, 1);
  dat->user_managed = 0;

  // transpose
  if (strstr(dat->type, ":soa") != NULL || (OP_auto_soa && dat->dim > 1)) {
    cutilSafeCall(
        cudaMalloc((void **)&(dat->buffer_d_r),
                   dat->size * (OP_import_exec_list[set->index]->size +
                                OP_import_nonexec_list[set->index]->size)));
  }

  op_cpHostToDevice((void **)&(dat->data_d), (void **)&(dat->data),
                    dat->size * set_size);

  // need to allocate mpi_buffers for this new temp_dat
  op_mpi_buffer mpi_buf = (op_mpi_buffer)xmalloc(sizeof(op_mpi_buffer_core));

  halo_list exec_e_list = OP_export_exec_list[set->index];
  halo_list nonexec_e_list = OP_export_nonexec_list[set->index];

  mpi_buf->buf_exec = (char *)xmalloc((exec_e_list->size) * dat->size);
  mpi_buf->buf_nonexec = (char *)xmalloc((nonexec_e_list->size) * dat->size);

  halo_list exec_i_list = OP_import_exec_list[set->index];
  halo_list nonexec_i_list = OP_import_nonexec_list[set->index];

  mpi_buf->s_req = (MPI_Request *)xmalloc(
      sizeof(MPI_Request) *
      (exec_e_list->ranks_size + nonexec_e_list->ranks_size));
  mpi_buf->r_req = (MPI_Request *)xmalloc(
      sizeof(MPI_Request) *
      (exec_i_list->ranks_size + nonexec_i_list->ranks_size));

  mpi_buf->s_num_req = 0;
  mpi_buf->r_num_req = 0;

  dat->mpi_buffer = mpi_buf;

  // need to allocate device buffers for mpi comms for this new temp_dat
  cutilSafeCall(
      cudaMalloc((void **)&(dat->buffer_d),
                 dat->size * (OP_export_exec_list[set->index]->size +
                              OP_export_nonexec_list[set->index]->size)));

  return dat;
}

int op_free_dat_temp_char(op_dat dat) {
  // need to free mpi_buffers use in this op_dat
  free(((op_mpi_buffer)(dat->mpi_buffer))->buf_exec);
  free(((op_mpi_buffer)(dat->mpi_buffer))->buf_nonexec);
  free(((op_mpi_buffer)(dat->mpi_buffer))->s_req);
  free(((op_mpi_buffer)(dat->mpi_buffer))->r_req);
  free(dat->mpi_buffer);

  // need to free device buffers used in mpi comms
  cutilSafeCall(cudaFree(dat->buffer_d));

  if (strstr(dat->type, ":soa") != NULL || (OP_auto_soa && dat->dim > 1)) {
    cutilSafeCall(cudaFree(dat->buffer_d_r));
  }

  // free data on device
  cutilSafeCall(cudaFree(dat->data_d));
  return op_free_dat_temp_core(dat);
}

void op_mv_halo_device(op_set set, op_dat dat) {
  int set_size = set->size + OP_import_exec_list[set->index]->size +
                 OP_import_nonexec_list[set->index]->size;
  if (strstr(dat->type, ":soa") != NULL || (OP_auto_soa && dat->dim > 1)) {
    char *temp_data = (char *)malloc(dat->size * set_size * sizeof(char));
    int element_size = dat->size / dat->dim;
    for (int i = 0; i < dat->dim; i++) {
      for (int j = 0; j < set_size; j++) {
        for (int c = 0; c < element_size; c++) {
          temp_data[element_size * i * set_size + element_size * j + c] =
              dat->data[dat->size * j + element_size * i + c];
        }
      }
    }
    op_cpHostToDevice((void **)&(dat->data_d), (void **)&(temp_data),
                      dat->size * set_size);
    free(temp_data);

    if (dat->buffer_d_r != NULL) cutilSafeCall(cudaFree(dat->buffer_d_r));
    cutilSafeCall(
        cudaMalloc((void **)&(dat->buffer_d_r),
                   dat->size * (OP_import_exec_list[set->index]->size +
                                OP_import_nonexec_list[set->index]->size)));

  } else {
    op_cpHostToDevice((void **)&(dat->data_d), (void **)&(dat->data),
                      dat->size * set_size);
  }
  dat->dirty_hd = 0;
  if (dat->buffer_d != NULL) cutilSafeCall(cudaFree(dat->buffer_d));
  cutilSafeCall(
      cudaMalloc((void **)&(dat->buffer_d),
                 dat->size * (OP_export_exec_list[set->index]->size +
                              OP_export_nonexec_list[set->index]->size +
                              set_import_buffer_size[set->index])));
}

void op_mv_halo_list_device() {
  if (export_exec_list_d != NULL) {
    for (int s = 0; s < OP_set_index; s++)
      if (export_exec_list_d[OP_set_list[s]->index] != NULL)
        cutilSafeCall(cudaFree(export_exec_list_d[OP_set_list[s]->index]));
    free(export_exec_list_d);
  }
  export_exec_list_d = (int **)xmalloc(sizeof(int *) * OP_set_index);

  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];
    export_exec_list_d[set->index] = NULL;

    op_cpHostToDevice((void **)&(export_exec_list_d[set->index]),
                      (void **)&(OP_export_exec_list[set->index]->list),
                      OP_export_exec_list[set->index]->size * sizeof(int));
  }

  if (export_nonexec_list_d != NULL) {
    for (int s = 0; s < OP_set_index; s++)
      if (export_nonexec_list_d[OP_set_list[s]->index] != NULL)
        cutilSafeCall(cudaFree(export_nonexec_list_d[OP_set_list[s]->index]));
    free(export_nonexec_list_d);
  }
  export_nonexec_list_d = (int **)xmalloc(sizeof(int *) * OP_set_index);

  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];
    export_nonexec_list_d[set->index] = NULL;

    op_cpHostToDevice((void **)&(export_nonexec_list_d[set->index]),
                      (void **)&(OP_export_nonexec_list[set->index]->list),
                      OP_export_nonexec_list[set->index]->size * sizeof(int));
  }

  //for grouped, we need the disps array on device too
  if (export_exec_list_disps_d != NULL) {
    for (int s = 0; s < OP_set_index; s++)
      if (export_exec_list_disps_d[OP_set_list[s]->index] != NULL)
        cutilSafeCall(cudaFree(export_exec_list_disps_d[OP_set_list[s]->index]));
    free(export_exec_list_disps_d);
  }
  export_exec_list_disps_d = (int **)xmalloc(sizeof(int *) * OP_set_index);

  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];
    export_exec_list_disps_d[set->index] = NULL;

    //make sure end size is there too
    OP_export_exec_list[set->index]->disps[OP_export_exec_list[set->index]->ranks_size] = 
      OP_export_exec_list[set->index]->ranks_size == 0 ? 0 :
      OP_export_exec_list[set->index]->disps[OP_export_exec_list[set->index]->ranks_size-1] +
      OP_export_exec_list[set->index]->sizes[OP_export_exec_list[set->index]->ranks_size-1];
    op_cpHostToDevice((void **)&(export_exec_list_disps_d[set->index]),
                      (void **)&(OP_export_exec_list[set->index]->disps),
                      (OP_export_exec_list[set->index]->ranks_size+1) * sizeof(int));
  }

  if (export_nonexec_list_disps_d != NULL) {
    for (int s = 0; s < OP_set_index; s++)
      if (export_nonexec_list_disps_d[OP_set_list[s]->index] != NULL)
        cutilSafeCall(cudaFree(export_nonexec_list_disps_d[OP_set_list[s]->index]));
    free(export_nonexec_list_disps_d);
  }
  export_nonexec_list_disps_d = (int **)xmalloc(sizeof(int *) * OP_set_index);

  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];
    export_nonexec_list_disps_d[set->index] = NULL;

    //make sure end size is there too
    OP_export_nonexec_list[set->index]->disps[OP_export_nonexec_list[set->index]->ranks_size] = 
      OP_export_nonexec_list[set->index]->ranks_size == 0 ? 0 :
      OP_export_nonexec_list[set->index]->disps[OP_export_nonexec_list[set->index]->ranks_size-1] +
      OP_export_nonexec_list[set->index]->sizes[OP_export_nonexec_list[set->index]->ranks_size-1];
    op_cpHostToDevice((void **)&(export_nonexec_list_disps_d[set->index]),
                      (void **)&(OP_export_nonexec_list[set->index]->disps),
                      (OP_export_nonexec_list[set->index]->ranks_size+1) * sizeof(int));
  }
  if (import_exec_list_disps_d != NULL) {
    for (int s = 0; s < OP_set_index; s++)
      if (import_exec_list_disps_d[OP_set_list[s]->index] != NULL)
        cutilSafeCall(cudaFree(import_exec_list_disps_d[OP_set_list[s]->index]));
    free(import_exec_list_disps_d);
  }
  import_exec_list_disps_d = (int **)xmalloc(sizeof(int *) * OP_set_index);

  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];
    import_exec_list_disps_d[set->index] = NULL;

    //make sure end size is there too
    OP_import_exec_list[set->index]->disps[OP_import_exec_list[set->index]->ranks_size] = 
      OP_import_exec_list[set->index]->ranks_size == 0 ? 0 :
      OP_import_exec_list[set->index]->disps[OP_import_exec_list[set->index]->ranks_size-1] +
      OP_import_exec_list[set->index]->sizes[OP_import_exec_list[set->index]->ranks_size-1];
    op_cpHostToDevice((void **)&(import_exec_list_disps_d[set->index]),
                      (void **)&(OP_import_exec_list[set->index]->disps),
                      (OP_import_exec_list[set->index]->ranks_size+1) * sizeof(int));
  }

  if (import_nonexec_list_disps_d != NULL) {
    for (int s = 0; s < OP_set_index; s++)
      if (import_nonexec_list_disps_d[OP_set_list[s]->index] != NULL)
        cutilSafeCall(cudaFree(import_nonexec_list_disps_d[OP_set_list[s]->index]));
    free(import_nonexec_list_disps_d);
  }
  import_nonexec_list_disps_d = (int **)xmalloc(sizeof(int *) * OP_set_index);

  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];
    import_nonexec_list_disps_d[set->index] = NULL;

    //make sure end size is there too
    OP_import_nonexec_list[set->index]->disps[OP_import_nonexec_list[set->index]->ranks_size] = 
      OP_import_nonexec_list[set->index]->ranks_size == 0 ? 0 :
      OP_import_nonexec_list[set->index]->disps[OP_import_nonexec_list[set->index]->ranks_size-1] +
      OP_import_nonexec_list[set->index]->sizes[OP_import_nonexec_list[set->index]->ranks_size-1];
    op_cpHostToDevice((void **)&(import_nonexec_list_disps_d[set->index]),
                      (void **)&(OP_import_nonexec_list[set->index]->disps),
                      (OP_import_nonexec_list[set->index]->ranks_size+1) * sizeof(int));
  }

  if ( export_nonexec_list_partial_d!= NULL) {
    for (int s = 0; s < OP_map_index; s++)
      if (OP_map_partial_exchange[s] && export_nonexec_list_partial_d[OP_map_list[s]->index] != NULL)
        cutilSafeCall(cudaFree(export_nonexec_list_partial_d[OP_map_list[s]->index]));
    free(export_nonexec_list_partial_d);
  }
  export_nonexec_list_partial_d = (int **)xmalloc(sizeof(int *) * OP_set_index);

  for (int s = 0; s < OP_map_index; s++) { // for each set
    if (!OP_map_partial_exchange[s])
      continue;
    op_map map = OP_map_list[s];
    export_nonexec_list_partial_d[map->index] = NULL;

    op_cpHostToDevice((void **)&(export_nonexec_list_partial_d[map->index]),
                      (void **)&(OP_export_nonexec_permap[map->index]->list),
                      OP_export_nonexec_permap[map->index]->size * sizeof(int));
  }

  if ( import_nonexec_list_partial_d!= NULL) {
    for (int s = 0; s < OP_map_index; s++)
      if (OP_map_partial_exchange[s] && import_nonexec_list_partial_d[OP_map_list[s]->index] != NULL)
        cutilSafeCall(cudaFree(import_nonexec_list_partial_d[OP_map_list[s]->index]));
    free(import_nonexec_list_partial_d);
  }
  import_nonexec_list_partial_d = (int **)xmalloc(sizeof(int *) * OP_set_index);

  for (int s = 0; s < OP_map_index; s++) { // for each set
    if (!OP_map_partial_exchange[s])
      continue;
    op_map map = OP_map_list[s];
    import_nonexec_list_partial_d[map->index] = NULL;

    op_cpHostToDevice((void **)&(import_nonexec_list_partial_d[map->index]),
                      (void **)&(OP_import_nonexec_permap[map->index]->list),
                      OP_import_nonexec_permap[map->index]->size * sizeof(int));
  }
}

op_set op_decl_set(int size, char const *name) {
  return op_decl_set_core(size, name);
}

op_map op_decl_map(op_set from, op_set to, int dim, int *imap,
                   char const *name) {
  // int *m = (int *)xmalloc(from->size * dim * sizeof(int));
  //  memcpy(m, imap, from->size * dim * sizeof(int));
  op_map out_map = op_decl_map_core(from, to, dim, imap, name);
  out_map->user_managed = 0;
  return out_map;
  // return op_decl_map_core ( from, to, dim, imap, name );
}

op_arg op_arg_dat(op_dat dat, int idx, op_map map, int dim, char const *type,
                  op_access acc) {
  return op_arg_dat_core(dat, idx, map, dim, type, acc);
}

op_arg op_opt_arg_dat(int opt, op_dat dat, int idx, op_map map, int dim,
                      char const *type, op_access acc) {
  return op_opt_arg_dat_core(opt, dat, idx, map, dim, type, acc);
}

op_arg op_arg_gbl_char(char *data, int dim, const char *type, int size,
                       op_access acc) {
  return op_arg_gbl_core(1, data, dim, type, size, acc);
}

op_arg op_opt_arg_gbl_char(int opt, char *data, int dim, const char *type,
                           int size, op_access acc) {
  return op_arg_gbl_core(opt, data, dim, type, size, acc);
}

void op_printf(const char *format, ...) {
  int my_rank;
  MPI_Comm_rank(OP_MPI_WORLD, &my_rank);
  if (my_rank == MPI_ROOT) {
    va_list argptr;
    va_start(argptr, format);
    vprintf(format, argptr);
    va_end(argptr);
  }
}

void op_print(const char *line) {
  int my_rank;
  MPI_Comm_rank(OP_MPI_WORLD, &my_rank);
  if (my_rank == MPI_ROOT) {
    printf("%s\n", line);
  }
}

void op_timers(double *cpu, double *et) {
  MPI_Barrier(OP_MPI_WORLD);
  op_timers_core(cpu, et);
}

//
// This function is defined in the generated master kernel file
// so that it is possible to check on the runtime size of the
// data in cases where it is not known at compile time
//

/*
void
op_decl_const_char ( int dim, char const * type, int size, char * dat,
                     char const * name )
{
  cutilSafeCall ( cudaMemcpyToSymbol ( name, dat, dim * size, 0,
                                       cudaMemcpyHostToDevice ) );
}
*/

void op_exit() {
  // need to free buffer_d used for mpi comms in each op_dat
  if (OP_hybrid_gpu) {
    op_dat_entry *item;
    TAILQ_FOREACH(item, &OP_dat_list, entries) {
      if (strstr(item->dat->type, ":soa") != NULL ||
          (OP_auto_soa && item->dat->dim > 1)) {
        cutilSafeCall(cudaFree((item->dat)->buffer_d_r));
      }
      cutilSafeCall(cudaFree((item->dat)->buffer_d));
    }

    for (int i = 0; i < OP_set_index; i++) {
      if (export_exec_list_d[i] != NULL)
        cutilSafeCall(cudaFree(export_exec_list_d[i]));
      if (export_nonexec_list_d[i] != NULL)
        cutilSafeCall(cudaFree(export_nonexec_list_d[i]));
    }
    for (int i = 0; i < OP_map_index; i++) {
      if (!OP_map_partial_exchange[i])
        continue;
      cutilSafeCall(cudaFree(export_nonexec_list_partial_d[i]));
      cutilSafeCall(cudaFree(import_nonexec_list_partial_d[i]));
    }
  }

  op_mpi_exit();
  op_cuda_exit(); // frees dat_d memory
  op_rt_exit();   // frees plan memory
  op_exit_core(); // frees lib core variables

  int flag = 0;
  MPI_Finalized(&flag);
  if (!flag)
    MPI_Finalize();
}

void op_timing_output() {
  op_timing_output_core();
  printf("Total plan time: %8.4f\n", OP_plan_time);
}

void op_timings_to_csv(const char *outputFileName) {
  int comm_size, comm_rank;
  MPI_Comm_size(OP_MPI_WORLD, &comm_size);
  MPI_Comm_rank(OP_MPI_WORLD, &comm_rank);

  FILE * outputFile = NULL;
  if (op_is_root()) {
    outputFile = fopen(outputFileName, "w");
    if (outputFile == NULL) {
      printf("ERROR: Failed to open file for writing: '%s'\n", outputFileName);
    }
    else {
      fprintf(outputFile, "rank,thread,nranks,nthreads,count,total time,plan time,mpi time,GB used,GB total,kernel name\n");
    }
  }

  bool can_write = (outputFile != NULL);
  MPI_Bcast(&can_write, 1, MPI_INT, MPI_ROOT, OP_MPI_WORLD);

  if (can_write) {
    for (int n = 0; n < OP_kern_max; n++) {
      if (OP_kernels[n].count > 0) {
        if (OP_kernels[n].ntimes == 1 && OP_kernels[n].times[0] == 0.0f &&
            OP_kernels[n].time != 0.0f) {
          // This library is being used by an OP2 translation made with the
          // older
          // translator with older timing logic. Adjust to new logic:
          OP_kernels[n].times[0] = OP_kernels[n].time;
        }

        if (op_is_root()) {
          double times[OP_kernels[n].ntimes*comm_size];
          for (int i=0; i<(OP_kernels[n].ntimes*comm_size); i++) times[i] = 0.0f;
          MPI_Gather(OP_kernels[n].times, OP_kernels[n].ntimes, MPI_DOUBLE, times, OP_kernels[n].ntimes, MPI_DOUBLE, MPI_ROOT, OP_MPI_WORLD);

          float plan_times[comm_size];
          for (int i=0; i<comm_size; i++) plan_times[i] = 0.0f;
          MPI_Gather(&(OP_kernels[n].plan_time), 1, MPI_FLOAT, plan_times, 1, MPI_FLOAT, MPI_ROOT, OP_MPI_WORLD);

          double mpi_times[comm_size];
          for (int i=0; i<comm_size; i++) mpi_times[i] = 0.0f;
          MPI_Gather(&(OP_kernels[n].mpi_time), 1, MPI_DOUBLE, mpi_times, 1, MPI_DOUBLE, MPI_ROOT, OP_MPI_WORLD);

          float transfers[comm_size];
          for (int i=0; i<comm_size; i++) transfers[i] = 0.0f;
          MPI_Gather(&(OP_kernels[n].transfer), 1, MPI_FLOAT, transfers, 1, MPI_FLOAT, MPI_ROOT, OP_MPI_WORLD);

          float transfers2[comm_size];
          for (int i=0; i<comm_size; i++) transfers2[i] = 0.0f;
          MPI_Gather(&(OP_kernels[n].transfer2), 1, MPI_FLOAT, transfers2, 1, MPI_FLOAT, MPI_ROOT, OP_MPI_WORLD);

          // Have data, now write:
          for (int p=0 ; p<comm_size ; p++) {
            for (int thr=0; thr<OP_kernels[n].ntimes; thr++) {
              double kern_time = times[p*OP_kernels[n].ntimes + thr];

              fprintf(outputFile, "%d,%d,%d,%d,%d,%f,%f,%f,%f,%f,%s\n", p, thr,
                      comm_size, OP_kernels[n].ntimes, OP_kernels[n].count,
                      kern_time, plan_times[p], mpi_times[p],
                      transfers[p] / 1e9f, transfers2[p] / 1e9f,
                      OP_kernels[n].name);
            }
          }
        }
        else {
          MPI_Gather(OP_kernels[n].times, OP_kernels[n].ntimes, MPI_DOUBLE, NULL, 0, MPI_DOUBLE, MPI_ROOT, OP_MPI_WORLD);

          MPI_Gather(&(OP_kernels[n].plan_time), 1, MPI_FLOAT, NULL, 0, MPI_FLOAT, MPI_ROOT, OP_MPI_WORLD);

          MPI_Gather(&(OP_kernels[n].mpi_time), 1, MPI_DOUBLE, NULL, 0, MPI_DOUBLE, MPI_ROOT, OP_MPI_WORLD);

          MPI_Gather(&(OP_kernels[n].transfer), 1, MPI_FLOAT, NULL, 0, MPI_FLOAT, MPI_ROOT, OP_MPI_WORLD);

          MPI_Gather(&(OP_kernels[n].transfer2), 1, MPI_FLOAT, NULL, 0, MPI_FLOAT, MPI_ROOT, OP_MPI_WORLD);
        }

        op_mpi_barrier();
      }
    }
  }

  if (op_is_root() && outputFile != NULL) {
    fclose(outputFile);
  }
}

void op_print_dat_to_binfile(op_dat dat, const char *file_name) {
  // need to get data from GPU
  op_cuda_get_data(dat);

  // rearrange data backe to original order in mpi
  op_dat temp = op_mpi_get_data(dat);
  print_dat_to_binfile_mpi(temp, file_name);

  free(temp->data);
  free(temp->set);
  free(temp);
}

void op_print_dat_to_txtfile(op_dat dat, const char *file_name) {
  // need to get data from GPU
  op_cuda_get_data(dat);

  // rearrange data backe to original order in mpi
  op_dat temp = op_mpi_get_data(dat);
  print_dat_to_txtfile_mpi(temp, file_name);

  free(temp->data);
  free(temp->set);
  free(temp);
}

void op_upload_all() {
  op_dat_entry *item;
  TAILQ_FOREACH(item, &OP_dat_list, entries) {
    op_dat dat = item->dat;
    int set_size = dat->set->size + OP_import_exec_list[dat->set->index]->size +
                   OP_import_nonexec_list[dat->set->index]->size;
    if (dat->data_d) {
      if (strstr(dat->type, ":soa") != NULL || (OP_auto_soa && dat->dim > 1)) {
        char *temp_data = (char *)malloc(dat->size * set_size * sizeof(char));
        int element_size = dat->size / dat->dim;
        for (int i = 0; i < dat->dim; i++) {
          for (int j = 0; j < set_size; j++) {
            for (int c = 0; c < element_size; c++) {
              temp_data[element_size * i * set_size + element_size * j + c] =
                  dat->data[dat->size * j + element_size * i + c];
            }
          }
        }
        cutilSafeCall(cudaMemcpy(dat->data_d, temp_data, dat->size * set_size,
                                 cudaMemcpyHostToDevice));
        dat->dirty_hd = 0;
        free(temp_data);
      } else {
        cutilSafeCall(cudaMemcpy(dat->data_d, dat->data, dat->size * set_size,
                                 cudaMemcpyHostToDevice));
        dat->dirty_hd = 0;
      }
    }
  }
}

void op_fetch_data_char(op_dat dat, char *usr_ptr) {
  // need to get data from GPU
  op_cuda_get_data(dat);

  // rearrange data backe to original order in mpi
  op_dat temp = op_mpi_get_data(dat);

  // copy data into usr_ptr
  memcpy((void *)usr_ptr, (void *)temp->data, temp->set->size * temp->size);
  free(temp->data);
  free(temp->set);
  free(temp);
}

op_dat op_fetch_data_file_char(op_dat dat) {
  // need to get data from GPU
  op_cuda_get_data(dat);
  // rearrange data backe to original order in mpi
  return op_mpi_get_data(dat);
}

void op_fetch_data_idx_char(op_dat dat, char *usr_ptr, int low, int high) {
  // need to get data from GPU
  op_cuda_get_data(dat);

  // rearrange data backe to original order in mpi
  op_dat temp = op_mpi_get_data(dat);

  // do allgather on temp->data and copy it to memory block pointed to by
  // use_ptr
  fetch_data_hdf5(temp, usr_ptr, low, high);

  free(temp->data);
  free(temp->set);
  free(temp);
}
