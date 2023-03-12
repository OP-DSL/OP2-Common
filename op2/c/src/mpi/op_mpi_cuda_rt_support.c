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
// This file implements the MPI+CUDA-specific run-time support functions
//

//
// header files
//

#include <mpi.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math_constants.h>

#include <op_cuda_rt_support.h>
#include <op_lib_c.h>
#include <op_lib_core.h>
#include <op_rt_support.h>

#include <op_lib_mpi.h>
#include <op_util.h>

// Small re-declaration to avoid using struct in the C version.
// This is due to the different way in which C and C++ see structs

typedef struct cudaDeviceProp cudaDeviceProp_t;

//
// export lists on the device
//

int **export_exec_list_d = NULL;
int **export_nonexec_list_d = NULL;
int **export_exec_list_disps_d = NULL;
int **export_nonexec_list_disps_d = NULL;
int **export_nonexec_list_partial_d = NULL;
int **import_nonexec_list_partial_d = NULL;
int **import_exec_list_disps_d = NULL;
int **import_nonexec_list_disps_d = NULL;

int **export_exec_nonexec_list_d = NULL;

cudaEvent_t op2_grp_download_event;
cudaStream_t op2_grp_secondary;

void print_array_2(int* arr, int size, const char* name, int my_rank){
  for(int i = 0; i < size; i++){
    printf("array my_rank=%d name=%s size=%d value[%d]=%d\n", my_rank, name, size, i, arr[i]);
  }
}

void cutilDeviceInit(int argc, char **argv) {
  (void)argc;
  (void)argv;
  int deviceCount;
  cutilSafeCall(cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    printf("cutil error: no devices supporting CUDA\n");
    exit(-1);
  }
  printf("Trying to select a device\n");

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // no need to ardcode this following, can be done via numawrap scripts
  /*if (getenv("OMPI_COMM_WORLD_LOCAL_RANK")!=NULL) {
    rank = atoi(getenv("OMPI_COMM_WORLD_LOCAL_RANK"));
  } else if (getenv("MV2_COMM_WORLD_LOCAL_RANK")!=NULL) {
    rank = atoi(getenv("MV2_COMM_WORLD_LOCAL_RANK"));
  } else if (getenv("MPI_LOCALRANKID")!=NULL) {
    rank = atoi(getenv("MPI_LOCALRANKID"));
  } else {
    rank = rank%deviceCount;
  }*/

  // Test we have access to a device

  // This commented out test does not work with CUDA versions above 6.5
  /*float *test;
  cudaError_t err = cudaMalloc((void **)&test, sizeof(float));
  if (err != cudaSuccess) {
    OP_hybrid_gpu = 0;
  } else {
    OP_hybrid_gpu = 1;
  }
  if (OP_hybrid_gpu) {
    cudaFree(test);

    cutilSafeCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

    int deviceId = -1;
    cudaGetDevice(&deviceId);
    cudaDeviceProp_t deviceProp;
    cutilSafeCall ( cudaGetDeviceProperties ( &deviceProp, deviceId ) );
    printf ( "\n Using CUDA device: %d %s on rank %d\n",deviceId,
  deviceProp.name,rank );
  } else {
    printf ( "\n Using CPU on rank %d\n",rank );
  }*/
  //omp_set_default_device(rank);
//  cudaError_t err = cudaSetDevice(rank);
  float *test;
  OP_hybrid_gpu = 0;
  //cudaError_t err = cudaMalloc((void **)&test, sizeof(float));
  for (int i = 0; i < deviceCount; i++) {
    cudaError_t err = cudaSetDevice((i+rank)%deviceCount);
    if (err == cudaSuccess) {
      cudaError_t err = cudaMalloc((void **)&test, sizeof(float));
      if (err == cudaSuccess) {
        OP_hybrid_gpu = 1;
        break;
      }
    }
  }
  if (OP_hybrid_gpu) {
    cutilSafeCall(cudaFree(test));

    cutilSafeCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

    int deviceId = -1;
    cudaGetDevice(&deviceId);
    cudaDeviceProp_t deviceProp;
    cutilSafeCall(cudaGetDeviceProperties(&deviceProp, deviceId));
    printf("\n Using CUDA device: %d %s on rank %d\n", deviceId,
           deviceProp.name, rank);
    cutilSafeCall(cudaStreamCreateWithFlags(&op2_grp_secondary, cudaStreamNonBlocking));
    cutilSafeCall(cudaEventCreateWithFlags(&op2_grp_download_event, cudaEventDisableTiming));
  } else {
    printf("\n Using CPU on rank %d\n", rank);
  }
}

void op_upload_dat(op_dat dat) {
  if (OP_import_exec_list==NULL) return;
  // printf("Uploading newone %s\n", dat->name);
#ifdef COMM_AVOID
  int set_size = dat->set->size + dat->set->total_exec_size + dat->set->total_nonexec_size;
#else
  int set_size = dat->set->size + OP_import_exec_list[dat->set->index]->size +
                 OP_import_nonexec_list[dat->set->index]->size;
  //  printf("Uploading newone %s set_size=%d\n", dat->name, set_size);
#endif
  if (strstr(dat->type, ":soa") != NULL || (OP_auto_soa && dat->dim > 1)) {
    char *temp_data = (char *)xmalloc(dat->size * set_size * sizeof(char));
    int element_size = dat->size / dat->dim;
    for (int i = 0; i < dat->dim; i++) {
      for (int j = 0; j < set_size; j++) {
        for (int c = 0; c < element_size; c++) {
          temp_data[element_size * i * set_size + element_size * j + c] =
              dat->data[dat->size * j + element_size * i + c];
        }
      }
    }
    cutilSafeCall(cudaMemcpy(dat->data_d, temp_data, set_size * dat->size,
                             cudaMemcpyHostToDevice));
    free(temp_data);
  } else {
    cutilSafeCall(cudaMemcpy(dat->data_d, dat->data, set_size * dat->size,
                             cudaMemcpyHostToDevice));
  }
}

#ifdef COMM_AVOID
void op_upload_dat_chained(op_dat dat, int nhalos) {
  if (OP_import_exec_list == NULL) return;
   printf("Uploading 1 %s\n", dat->name);
  int set_size = dat->set->size + dat->set->total_exec_size + dat->set->total_nonexec_size;
  cutilSafeCall(cudaMemcpy(dat->data_d, dat->data, set_size * dat->size,
                            cudaMemcpyHostToDevice));

}
#endif
void op_download_dat(op_dat dat) {
  if (OP_import_exec_list==NULL) return;
   printf("Downloading 1 %s\n", dat->name);
#ifdef COMM_AVOID
  int set_size = dat->set->size + dat->set->total_exec_size + dat->set->total_nonexec_size;
#else
  int set_size = dat->set->size + OP_import_exec_list[dat->set->index]->size +
                 OP_import_nonexec_list[dat->set->index]->size;
#endif
  if (strstr(dat->type, ":soa") != NULL || (OP_auto_soa && dat->dim > 1)) {
    char *temp_data = (char *)xmalloc(dat->size * set_size * sizeof(char));
    cutilSafeCall(cudaMemcpy(temp_data, dat->data_d, set_size * dat->size,
                             cudaMemcpyDeviceToHost));
    int element_size = dat->size / dat->dim;
    for (int i = 0; i < dat->dim; i++) {
      for (int j = 0; j < set_size; j++) {
        for (int c = 0; c < element_size; c++) {
          dat->data[dat->size * j + element_size * i + c] =
              temp_data[element_size * i * set_size + element_size * j + c];
        }
      }
    }
    free(temp_data);
  } else {
    cutilSafeCall(cudaMemcpy(dat->data, dat->data_d, set_size * dat->size,
                             cudaMemcpyDeviceToHost));
  }
}

void op_exchange_halo_cuda(op_arg *arg, int exec_flag) {

  int my_rank = 0;
  // MPI_Comm_rank(OP_MPI_WORLD, &my_rank);

  op_dat dat = arg->dat;

  if (arg->sent == 1) {
    printf("Error: Halo exchange already in flight for dat %s\n", dat->name);
    fflush(stdout);
    MPI_Abort(OP_MPI_WORLD, 2);
  }

  // printf("op_exchange_halo_cuda my_rank=%d dat=%s exec_flag=%d dirtybit=%d db=%d\n", 
  //     my_rank, dat->name, exec_flag, dat->dirtybit, dat->exec_dirtybits[0]);

  // For a directly accessed op_dat do not do halo exchanges if not executing
  // over
  // redundant compute block
  if (exec_flag == 0 && arg->idx == -1)
    return;
  

  // printf("testop_exchange_halo_cuda dat=%s datsize=%d datdim=%d\n", dat->name,dat->size, dat->dim);

  
  arg->sent = 0; // reset flag
  // need to exchange both direct and indirect data sets if they are dirty
  if ((arg->opt) &&
      (arg->acc == OP_READ ||
       arg->acc == OP_RW /* good for debug || arg->acc == OP_INC*/) &&
#ifdef COMM_AVOID
      (dat->exec_dirtybits[0] == 1 || dat->nonexec_dirtybits[0] == 1)){
#else
      (dat->dirtybit == 1)) {
#endif
    halo_list imp_exec_list = OP_import_exec_list[dat->set->index];
    halo_list imp_nonexec_list = OP_import_nonexec_list[dat->set->index];

    halo_list exp_exec_list = OP_export_exec_list[dat->set->index];
    halo_list exp_nonexec_list = OP_export_nonexec_list[dat->set->index];

    //-------first exchange exec elements related to this data array--------

    // sanity checks
    if (compare_sets(imp_exec_list->set, dat->set) == 0) {
      printf("Error: Import list and set mismatch\n");
      MPI_Abort(OP_MPI_WORLD, 2);
    }
    if (compare_sets(exp_exec_list->set, dat->set) == 0) {
      printf("Error: Export list and set mismatch\n");
      MPI_Abort(OP_MPI_WORLD, 2);
    }

    gather_data_to_buffer(*arg, exp_exec_list, exp_nonexec_list, my_rank);

    char *outptr_exec = NULL;
    char *outptr_nonexec = NULL;
    if (OP_gpu_direct) {
      outptr_exec = arg->dat->buffer_d;
      outptr_nonexec =
          arg->dat->buffer_d + exp_exec_list->size * arg->dat->size;
      cutilSafeCall(cudaDeviceSynchronize());
    } else {
      cutilSafeCall(cudaMemcpy(
          ((op_mpi_buffer)(dat->mpi_buffer))->buf_exec, arg->dat->buffer_d,
          exp_exec_list->size * arg->dat->size, cudaMemcpyDeviceToHost));

      cutilSafeCall(cudaMemcpy(
          ((op_mpi_buffer)(dat->mpi_buffer))->buf_nonexec,
          arg->dat->buffer_d + exp_exec_list->size * arg->dat->size,
          exp_nonexec_list->size * arg->dat->size, cudaMemcpyDeviceToHost));

      if (OP_kern_max > 0){
        OP_kernels[OP_kern_curr].halo_data2 += exp_exec_list->size * arg->dat->size;
        OP_kernels[OP_kern_curr].halo_data2 += exp_nonexec_list->size * arg->dat->size;
        // printf("test current=%d dat=%s set=%s exec=%d nonexec=%d\n", 
        // OP_kern_curr, dat->name, dat->set->name, exp_exec_list->size, exp_nonexec_list->size);
      }

      cutilSafeCall(cudaDeviceSynchronize());
      outptr_exec = ((op_mpi_buffer)(dat->mpi_buffer))->buf_exec;
      outptr_nonexec = ((op_mpi_buffer)(dat->mpi_buffer))->buf_nonexec;
    }
    char *ptr = NULL;
#ifdef COMM_AVOID
    if(dat->exec_dirtybits[0] == 1){
#endif
    for (int i = 0; i < exp_exec_list->ranks_size; i++) {

      // printf("ORGEXEC MPI_Isend my_rank=%d to=%d dat=%s bufpos=%d datsize=%d size=%d\n", 
      // my_rank, exp_exec_list->ranks[i], dat->name, exp_exec_list->disps[i] * dat->size, dat->size, exp_exec_list->sizes[i]);

      MPI_Isend(&outptr_exec[exp_exec_list->disps[i] * dat->size],
                dat->size * exp_exec_list->sizes[i], MPI_CHAR,
                exp_exec_list->ranks[i], dat->index, OP_MPI_WORLD,
                &((op_mpi_buffer)(dat->mpi_buffer))
                     ->s_req[((op_mpi_buffer)(dat->mpi_buffer))->s_num_req++]);

      OP_mpi_tx_exec_msg_count++;

      if (OP_kern_max > 0){
        OP_kernels[OP_kern_curr].halo_data += dat->size * exp_exec_list->sizes[i];
      }
      
    }

    int init = dat->set->size * dat->size;
    
    for (int i = 0; i < imp_exec_list->ranks_size; i++) {
      ptr = OP_gpu_direct
                ? &(dat->data_d[init + imp_exec_list->disps[i] * dat->size])
                : &(dat->data[init + imp_exec_list->disps[i] * dat->size]);
      if (OP_gpu_direct && (strstr(arg->dat->type, ":soa") != NULL ||
                            (OP_auto_soa && arg->dat->dim > 1)))
        ptr = dat->buffer_d_r + imp_exec_list->disps[i] * dat->size;

      // printf("ORGEXEC MPI_Irecv my_rank=%d to=%d bufpos=%d size=%d\n", my_rank, imp_exec_list->ranks[i], 
      // imp_exec_list->disps[i] * dat->size, dat->size * imp_exec_list->sizes[i]);

      MPI_Irecv(ptr, dat->size * imp_exec_list->sizes[i], MPI_CHAR,
                imp_exec_list->ranks[i], dat->index, OP_MPI_WORLD,
                &((op_mpi_buffer)(dat->mpi_buffer))
                     ->r_req[((op_mpi_buffer)(dat->mpi_buffer))->r_num_req++]);
      OP_mpi_rx_exec_msg_count++;
    }
#ifdef COMM_AVOID
    }
#endif

    //-----second exchange nonexec elements related to this data array------
    // sanity checks
    if (compare_sets(imp_nonexec_list->set, dat->set) == 0) {
      printf("Error: Non-Import list and set mismatch");
      MPI_Abort(OP_MPI_WORLD, 2);
    }
    if (compare_sets(exp_nonexec_list->set, dat->set) == 0) {
      printf("Error: Non-Export list and set mismatch");
      MPI_Abort(OP_MPI_WORLD, 2);
    }
#ifdef COMM_AVOID
    if(dat->nonexec_dirtybits[0] == 1){
#endif
    for (int i = 0; i < exp_nonexec_list->ranks_size; i++) {

      // printf("ORGNONEXEC test MPI_Isend my_rank=%d to=%d dat=%s bufpos=%d datsize=%d size=%d\n", 
      // my_rank, exp_nonexec_list->ranks[i], dat->name, exp_nonexec_list->disps[i] * dat->size, dat->size, exp_nonexec_list->sizes[i]);
      MPI_Isend(&outptr_nonexec[exp_nonexec_list->disps[i] * dat->size],
                dat->size * exp_nonexec_list->sizes[i], MPI_CHAR,
                exp_nonexec_list->ranks[i], dat->index, OP_MPI_WORLD,
                &((op_mpi_buffer)(dat->mpi_buffer))
                     ->s_req[((op_mpi_buffer)(dat->mpi_buffer))->s_num_req++]);
      OP_mpi_tx_nonexec_msg_count++;

      if (OP_kern_max > 0){
        OP_kernels[OP_kern_curr].halo_data += dat->size * exp_nonexec_list->sizes[i];
      }
    }

    int nonexec_init = (dat->set->size + imp_exec_list->size) * dat->size;
    for (int i = 0; i < imp_nonexec_list->ranks_size; i++) {
      ptr = OP_gpu_direct
                ? &(dat->data_d[nonexec_init +
                                imp_nonexec_list->disps[i] * dat->size])
                : &(dat->data[nonexec_init +
                              imp_nonexec_list->disps[i] * dat->size]);
      if (OP_gpu_direct && (strstr(arg->dat->type, ":soa") != NULL ||
                            (OP_auto_soa && arg->dat->dim > 1)))
        ptr = dat->buffer_d_r +
              (imp_exec_list->size + imp_exec_list->disps[i]) * dat->size;

      // printf("ORG MPI_Irecv my_rank=%d to=%d bufpos=%d size=%d\n", my_rank, imp_nonexec_list->ranks[i], 
      // (imp_exec_list->size + imp_exec_list->disps[i]) * dat->size, dat->size * imp_nonexec_list->sizes[i]);
      MPI_Irecv(ptr, dat->size * imp_nonexec_list->sizes[i], MPI_CHAR,
                imp_nonexec_list->ranks[i], dat->index, OP_MPI_WORLD,
                &((op_mpi_buffer)(dat->mpi_buffer))
                     ->r_req[((op_mpi_buffer)(dat->mpi_buffer))->r_num_req++]);

      OP_mpi_rx_nonexec_msg_count++;
    }
#ifdef COMM_AVOID
    }
#endif

#ifdef COMM_AVOID
    unset_dirtybit(arg);
    if(are_dirtybits_clear(arg) == 1){
      dat->dirtybit = 0;
      // printf("here dat=%s\n", dat->name);
    }
#else
    // clear dirty bit
    dat->dirtybit = 0;
#endif
    arg->sent = 1;
  }
}

#ifdef COMM_AVOID

halo_list imp_common_list;
halo_list exp_common_list;
int exp_rank_count;
int total_halo_size;

void op_exchange_halo_cuda_chained(int nargs, op_arg *args, int exec_flag){

  int my_rank = 0;
  // MPI_Comm_rank(OP_MPI_WORLD, &my_rank);

  op_arg dirty_args[nargs];
  int ndirty_args = get_dirty_args(nargs, args, exec_flag, dirty_args, 1);
  if(ndirty_args < 1){
    // op_printf("op_exchange_halo_cuda_chained ndirty_args< 1\n");
    return;
  }
  // printf("op_exchange_halo_cuda_chained ndirty_args=%d\n", ndirty_args);
  // for(int i = 0; i < ndirty_args; i++){
  //   printf("op_exchange_halo_cuda_chained ndirty_args=%d dat=%s datsize=%d datdim=%d\n", ndirty_args, dirty_args[i].dat->name, dirty_args[i].dat->size, dirty_args[i].dat->dim);
  // }
    
  imp_common_list = OP_merged_import_exec_nonexec_list[dirty_args[0].dat->set->index];  //assumption nargs > 0
  exp_common_list = OP_merged_export_exec_nonexec_list[dirty_args[0].dat->set->index];

  int set_elem_index = 0;
  int buf_index = 0;
  int buf_start = 0;
  int prev_size = 0;

  grp_tag++;

  exp_rank_count = exp_common_list->ranks_size / exp_common_list->num_levels;
  
  int total_buf_size = 0;
  gather_data_to_buffer_chained(ndirty_args, dirty_args, exec_flag, exp_rank_count, 
    ca_buf_pos, ca_send_sizes, &total_buf_size, my_rank);
  
  op_download_buffer_async(grp_send_buffer_d, grp_send_buffer_h, total_buf_size);

  if (OP_kern_max > 0){
    OP_kernels[OP_kern_curr].halo_data2 += total_buf_size;
  }

  int rank_count = imp_common_list->ranks_size / imp_common_list->num_levels;
  int imp_disp = 0;

  for (int i = 0; i < rank_count; i++) {
    int imp_size = 0;
    
    for(int n = 0; n < ndirty_args; n++){
      op_arg* arg = &dirty_args[n];
      op_dat dat = arg->dat;
      int nhalos = get_nhalos(arg);
      halo_list imp_list = OP_merged_import_exec_nonexec_list[dat->set->index];
      int halo_index = 0;

      for(int l1 = 0; l1 < nhalos; l1++){
        for(int l2 = 0; l2 < 2; l2++){ // 2 is for exec and nonexec levels
          if((l2 == 0 && dat->exec_dirtybits[l1] == 1) || (l2 == 1 && dat->nonexec_dirtybits[l1] == 1)){   
            imp_size += imp_list->level_sizes[i * imp_list->num_levels + halo_index] * arg->dat->size;
          }
          if(is_halo_required_for_set(dat->set, l1) == 1 && l2 == 0){
            halo_index++;
          }
          if(is_nonexec_halo_required(arg, nhalos, l1) != 1){
              break;
          }
        }
        halo_index++;
      }
    }

    // printf("halo exchange cuda MPI_Irecv my_rank=%d from=%d bufpos=%d size=%d\n", my_rank, imp_common_list->ranks[i], imp_disp, imp_size);
    MPI_Irecv(&grp_recv_buffer_h[imp_disp],
              imp_size, MPI_CHAR,
              imp_common_list->ranks[i], grp_tag, OP_MPI_WORLD,
              &grp_recv_requests[i]);
    imp_disp += imp_size;

    // printf("rxtxexec merged my_rank=%d dat=%s r=%d recved=%d  imp_disp=%d\n", my_rank, "test", imp_common_list->ranks[i], imp_size, imp_disp);
    OP_mpi_rx_msg_count_chained++;
  }
  total_halo_size = imp_disp;

}
#endif
void op_exchange_halo_partial_cuda(op_arg *arg, int exec_flag) {
  op_dat dat = arg->dat;

  if (arg->sent == 1) {
    printf("Error: Halo exchange already in flight for dat %s\n", dat->name);
    fflush(stdout);
    MPI_Abort(OP_MPI_WORLD, 2);
  }

  // For a directly accessed op_dat do not do halo exchanges if not executing
  // over
  // redundant compute block
  if (exec_flag == 0 && arg->idx == -1)
    return;

  arg->sent = 0; // reset flag
  // need to exchange both direct and indirect data sets if they are dirty
  if ((arg->opt) &&
      (arg->acc == OP_READ ||
       arg->acc == OP_RW /* good for debug || arg->acc == OP_INC*/) &&
      (dat->dirtybit == 1)) {

    halo_list imp_nonexec_list = OP_import_nonexec_permap[arg->map->index];
    halo_list exp_nonexec_list = OP_export_nonexec_permap[arg->map->index];

    //-------first exchange exec elements related to this data array--------

    // sanity checks
    if (compare_sets(imp_nonexec_list->set, dat->set) == 0) {
      printf("Error: Import list and set mismatch\n");
      MPI_Abort(OP_MPI_WORLD, 2);
    }
    if (compare_sets(exp_nonexec_list->set, dat->set) == 0) {
      printf("Error: Export list and set mismatch\n");
      MPI_Abort(OP_MPI_WORLD, 2);
    }

    gather_data_to_buffer_partial(*arg, exp_nonexec_list);

    char *outptr_nonexec = NULL;
    if (OP_gpu_direct) {
      outptr_nonexec = arg->dat->buffer_d;
      cutilSafeCall(cudaDeviceSynchronize());
    } else {
      cutilSafeCall(cudaMemcpy(
          ((op_mpi_buffer)(dat->mpi_buffer))->buf_nonexec, arg->dat->buffer_d,
          exp_nonexec_list->size * arg->dat->size, cudaMemcpyDeviceToHost));

      cutilSafeCall(cudaDeviceSynchronize());
      outptr_nonexec = ((op_mpi_buffer)(dat->mpi_buffer))->buf_nonexec;
    }

    //-----second exchange nonexec elements related to this data array------
    // sanity checks
    if (compare_sets(imp_nonexec_list->set, dat->set) == 0) {
      printf("Error: Non-Import list and set mismatch");
      MPI_Abort(OP_MPI_WORLD, 2);
    }
    if (compare_sets(exp_nonexec_list->set, dat->set) == 0) {
      printf("Error: Non-Export list and set mismatch");
      MPI_Abort(OP_MPI_WORLD, 2);
    }

    for (int i = 0; i < exp_nonexec_list->ranks_size; i++) {
      MPI_Isend(&outptr_nonexec[exp_nonexec_list->disps[i] * dat->size],
                dat->size * exp_nonexec_list->sizes[i], MPI_CHAR,
                exp_nonexec_list->ranks[i], dat->index, OP_MPI_WORLD,
                &((op_mpi_buffer)(dat->mpi_buffer))
                     ->s_req[((op_mpi_buffer)(dat->mpi_buffer))->s_num_req++]);
      OP_mpi_tx_nonexec_msg_count++;
    }

    int nonexec_init = OP_export_nonexec_permap[arg->map->index]->size;
    for (int i = 0; i < imp_nonexec_list->ranks_size; i++) {
      char *ptr =
          OP_gpu_direct
              ? &arg->dat
                     ->buffer_d[(nonexec_init + imp_nonexec_list->disps[i]) *
                                dat->size]
              : &((op_mpi_buffer)(dat->mpi_buffer))
                     ->buf_nonexec[(nonexec_init + imp_nonexec_list->disps[i]) *
                                   dat->size];
      MPI_Irecv(ptr, dat->size * imp_nonexec_list->sizes[i], MPI_CHAR,
                imp_nonexec_list->ranks[i], dat->index, OP_MPI_WORLD,
                &((op_mpi_buffer)(dat->mpi_buffer))
                     ->r_req[((op_mpi_buffer)(dat->mpi_buffer))->r_num_req++]);
      OP_mpi_rx_nonexec_msg_count++;
    }

    arg->sent = 1;
  }
}

void op_exchange_halo(op_arg *arg, int exec_flag) {
  op_dat dat = arg->dat;

  if (exec_flag == 0 && arg->idx == -1)
    return;
  if (arg->opt == 0)
    return;

  if (arg->sent == 1) {
    printf("Error: Halo exchange already in flight for dat %s\n", dat->name);
    fflush(stdout);
    MPI_Abort(OP_MPI_WORLD, 2);
  }

  // need to exchange both direct and indirect data sets if they are dirty
  if ((arg->acc == OP_READ ||
       arg->acc == OP_RW /* good for debug || arg->acc == OP_INC*/) &&
      (dat->dirtybit == 1)) {
    // printf("Exchanging Halo of data array %10s\n",dat->name);
    halo_list imp_exec_list = OP_import_exec_list[dat->set->index];
    halo_list imp_nonexec_list = OP_import_nonexec_list[dat->set->index];

    halo_list exp_exec_list = OP_export_exec_list[dat->set->index];
    halo_list exp_nonexec_list = OP_export_nonexec_list[dat->set->index];

    //-------first exchange exec elements related to this data array--------

    // sanity checks
    if (compare_sets(imp_exec_list->set, dat->set) == 0) {
      printf("Error: Import list and set mismatch\n");
      MPI_Abort(OP_MPI_WORLD, 2);
    }
    if (compare_sets(exp_exec_list->set, dat->set) == 0) {
      printf("Error: Export list and set mismatch\n");
      MPI_Abort(OP_MPI_WORLD, 2);
    }

    int set_elem_index;
    for (int i = 0; i < exp_exec_list->ranks_size; i++) {
      for (int j = 0; j < exp_exec_list->sizes[i]; j++) {
        set_elem_index = exp_exec_list->list[exp_exec_list->disps[i] + j];
        memcpy(&((op_mpi_buffer)(dat->mpi_buffer))
                    ->buf_exec[exp_exec_list->disps[i] * dat->size +
                               j * dat->size],
               (void *)&dat->data[dat->size * (set_elem_index)], dat->size);
      }
      MPI_Isend(&((op_mpi_buffer)(dat->mpi_buffer))
                     ->buf_exec[exp_exec_list->disps[i] * dat->size],
                dat->size * exp_exec_list->sizes[i], MPI_CHAR,
                exp_exec_list->ranks[i], dat->index, OP_MPI_WORLD,
                &((op_mpi_buffer)(dat->mpi_buffer))
                     ->s_req[((op_mpi_buffer)(dat->mpi_buffer))->s_num_req++]);
      OP_mpi_tx_exec_msg_count++;
    }

    int init = dat->set->size * dat->size;
    for (int i = 0; i < imp_exec_list->ranks_size; i++) {
      MPI_Irecv(&(dat->data[init + imp_exec_list->disps[i] * dat->size]),
                dat->size * imp_exec_list->sizes[i], MPI_CHAR,
                imp_exec_list->ranks[i], dat->index, OP_MPI_WORLD,
                &((op_mpi_buffer)(dat->mpi_buffer))
                     ->r_req[((op_mpi_buffer)(dat->mpi_buffer))->r_num_req++]);
      OP_mpi_rx_exec_msg_count++;
    }

    //-----second exchange nonexec elements related to this data array------
    // sanity checks
    if (compare_sets(imp_nonexec_list->set, dat->set) == 0) {
      printf("Error: Non-Import list and set mismatch");
      MPI_Abort(OP_MPI_WORLD, 2);
    }
    if (compare_sets(exp_nonexec_list->set, dat->set) == 0) {
      printf("Error: Non-Export list and set mismatch");
      MPI_Abort(OP_MPI_WORLD, 2);
    }

    for (int i = 0; i < exp_nonexec_list->ranks_size; i++) {
      for (int j = 0; j < exp_nonexec_list->sizes[i]; j++) {
        set_elem_index = exp_nonexec_list->list[exp_nonexec_list->disps[i] + j];
        memcpy(&((op_mpi_buffer)(dat->mpi_buffer))
                    ->buf_nonexec[exp_nonexec_list->disps[i] * dat->size +
                                  j * dat->size],
               (void *)&dat->data[dat->size * (set_elem_index)], dat->size);
      }
      MPI_Isend(&((op_mpi_buffer)(dat->mpi_buffer))
                     ->buf_nonexec[exp_nonexec_list->disps[i] * dat->size],
                dat->size * exp_nonexec_list->sizes[i], MPI_CHAR,
                exp_nonexec_list->ranks[i], dat->index, OP_MPI_WORLD,
                &((op_mpi_buffer)(dat->mpi_buffer))
                     ->s_req[((op_mpi_buffer)(dat->mpi_buffer))->s_num_req++]);
      OP_mpi_tx_nonexec_msg_count++;
    }

    int nonexec_init = (dat->set->size + imp_exec_list->size) * dat->size;
    for (int i = 0; i < imp_nonexec_list->ranks_size; i++) {
      MPI_Irecv(
          &(dat->data[nonexec_init + imp_nonexec_list->disps[i] * dat->size]),
          dat->size * imp_nonexec_list->sizes[i], MPI_CHAR,
          imp_nonexec_list->ranks[i], dat->index, OP_MPI_WORLD,
          &((op_mpi_buffer)(dat->mpi_buffer))
               ->r_req[((op_mpi_buffer)(dat->mpi_buffer))->r_num_req++]);
      OP_mpi_rx_nonexec_msg_count++;
    }
    // clear dirty bit
    dat->dirtybit = 0;
    arg->sent = 1;
  }
}

void op_exchange_halo_partial(op_arg *arg, int exec_flag) {
  op_dat dat = arg->dat;

  if (arg->opt == 0)
    return;

  if (arg->sent == 1) {
    printf("Error: Halo exchange already in flight for dat %s\n", dat->name);
    fflush(stdout);
    MPI_Abort(OP_MPI_WORLD, 2);
  }
  arg->sent = 0; // reset flag

  // need to exchange indirect data sets if they are dirty
  if ((arg->acc == OP_READ ||
       arg->acc == OP_RW /* good for debug || arg->acc == OP_INC*/) &&
      (dat->dirtybit == 1)) {
    // int rank;
    // MPI_Comm_rank(OP_MPI_WORLD, &rank);
    halo_list imp_nonexec_list = OP_import_nonexec_permap[arg->map->index];
    halo_list exp_nonexec_list = OP_export_nonexec_permap[arg->map->index];
    //-------exchange nonexec elements related to this data array and
    // map--------

    // sanity checks
    if (compare_sets(imp_nonexec_list->set, dat->set) == 0) {
      printf("Error: Import list and set mismatch\n");
      MPI_Abort(OP_MPI_WORLD, 2);
    }
    if (compare_sets(exp_nonexec_list->set, dat->set) == 0) {
      printf("Error: Export list and set mismatch\n");
      MPI_Abort(OP_MPI_WORLD, 2);
    }

    int set_elem_index;
    for (int i = 0; i < exp_nonexec_list->ranks_size; i++) {
      for (int j = 0; j < exp_nonexec_list->sizes[i]; j++) {
        set_elem_index = exp_nonexec_list->list[exp_nonexec_list->disps[i] + j];
        memcpy(&((op_mpi_buffer)(dat->mpi_buffer))
                    ->buf_nonexec[exp_nonexec_list->disps[i] * dat->size +
                                  j * dat->size],
               (void *)&dat->data[dat->size * (set_elem_index)], dat->size);
      }
      MPI_Isend(&((op_mpi_buffer)(dat->mpi_buffer))
                     ->buf_nonexec[exp_nonexec_list->disps[i] * dat->size],
                dat->size * exp_nonexec_list->sizes[i], MPI_CHAR,
                exp_nonexec_list->ranks[i], dat->index, OP_MPI_WORLD,
                &((op_mpi_buffer)(dat->mpi_buffer))
                     ->s_req[((op_mpi_buffer)(dat->mpi_buffer))->s_num_req++]);
      OP_mpi_tx_nonexec_msg_count++;
    }

    int init = exp_nonexec_list->size;
    for (int i = 0; i < imp_nonexec_list->ranks_size; i++) {
      MPI_Irecv(
          &((op_mpi_buffer)(dat->mpi_buffer))
               ->buf_nonexec[(init + imp_nonexec_list->disps[i]) * dat->size],
          dat->size * imp_nonexec_list->sizes[i], MPI_CHAR,
          imp_nonexec_list->ranks[i], dat->index, OP_MPI_WORLD,
          &((op_mpi_buffer)(dat->mpi_buffer))
               ->r_req[((op_mpi_buffer)(dat->mpi_buffer))->r_num_req++]);
      OP_mpi_rx_nonexec_msg_count++;
    }
    // note that we are not settinging the dirtybit to 0, since it's not a full
    // exchange
    arg->sent = 1;
  }
}
#ifdef COMM_AVOID
void op_wait_all_cuda_chained(int nargs, op_arg *args){ //, int device){

  int my_rank = 0;
  // MPI_Comm_rank(OP_MPI_WORLD, &my_rank);
  // check if this is a direct loop
  int direct_flag = 1;
  for (int n = 0; n < nargs; n++)
    if (args[n].opt && args[n].argtype == OP_ARG_DAT && args[n].idx != -1)
      direct_flag = 0;
  if (direct_flag == 1)
    return;

  // not a direct loop ...
  int exec_flag = 0;
  for (int n = 0; n < nargs; n++) {
    if (args[n].opt && args[n].idx != -1 && args[n].acc != OP_READ) {
      exec_flag = 1;
    }
  }
  op_arg dirty_args[nargs];
  int ndirty_args = get_dirty_args(nargs, args, exec_flag, dirty_args, 1);
  if(ndirty_args < 1){
    // op_printf("op_exchange_halo_cuda_chained ndirty_args< 1\n");
    return;
  }
  // printf("receive ndirty_args=%d\n", ndirty_args);
  // for(int i = 0; i < ndirty_args; i++){
  //   printf("receive ndirty_args=%d dat=%s\n", ndirty_args, dirty_args[i].dat->name);
  // }

  op_download_buffer_sync();
  for (int r = 0; r < exp_rank_count; r++) {
    // printf("halo exchange cuda MPI_Isend my_rank=%d to=%d bufpos=%d size=%d\n", my_rank, exp_common_list->ranks[r], ca_buf_pos[r], ca_send_sizes[r]);
    MPI_Isend(&grp_send_buffer_h[ca_buf_pos[r]],
              (ca_send_sizes[r]), MPI_CHAR,
              exp_common_list->ranks[r], grp_tag, OP_MPI_WORLD,
              &grp_send_requests[r]);

    if (OP_kern_max > 0){
      OP_kernels[OP_kern_curr].halo_data += ca_send_sizes[r];
    }
    
    ca_send_sizes[r] = 0;
    ca_buf_pos[r] = 0;
    OP_mpi_tx_msg_count_chained++;
  }

  MPI_Waitall(imp_common_list->ranks_size / imp_common_list->num_levels, 
                  &grp_recv_requests[0], MPI_STATUSES_IGNORE);

  int rank_count = imp_common_list->ranks_size / imp_common_list->num_levels;

  op_upload_buffer_async(grp_recv_buffer_d, grp_recv_buffer_h, total_halo_size);

  if (OP_kern_max > 0){
    OP_kernels[OP_kern_curr].halo_data2 += total_halo_size;
  }
  scatter_data_from_buffer_ptr_cuda_chained(dirty_args, ndirty_args, rank_count, grp_recv_buffer_d, exec_flag, my_rank);
  op_scatter_sync();

  for(int n = 0; n < ndirty_args; n++){
    (&dirty_args[n])->sent = 2; // set flag to indicate completed comm
    // printf("here0 dat=%s\n", dirty_args[n].dat->name);
    // print_array_2(dirty_args[n].dat->exec_dirtybits, dirty_args[n].dat->set->halo_info->max_nhalos, "execb", my_rank);
    // print_array_2(dirty_args[n].dat->nonexec_dirtybits, dirty_args[n].dat->set->halo_info->max_nhalos, "nonexecb", my_rank);
    unset_dirtybit(&dirty_args[n]);
    // print_array_2(dirty_args[n].dat->exec_dirtybits, dirty_args[n].dat->set->halo_info->max_nhalos, "execa", my_rank);
    // print_array_2(dirty_args[n].dat->nonexec_dirtybits, dirty_args[n].dat->set->halo_info->max_nhalos, "nonexeca", my_rank);
    if(are_dirtybits_clear(&dirty_args[n]) == 1){
      (&dirty_args[n])->dat->dirtybit = 0;
      // printf("here clear ====> dat=%s\n", dirty_args[n].dat->name);
    }
    (&dirty_args[n])->dat->dirty_hd = 2;
  }

  MPI_Waitall(exp_common_list->ranks_size / exp_common_list->num_levels, 
                &grp_send_requests[0], MPI_STATUSES_IGNORE);
}
#endif

void op_wait_all_cuda(op_arg *arg) {
  if (arg->opt && arg->argtype == OP_ARG_DAT && arg->sent == 1) {
    op_dat dat = arg->dat;
    MPI_Waitall(((op_mpi_buffer)(dat->mpi_buffer))->s_num_req,
                ((op_mpi_buffer)(dat->mpi_buffer))->s_req, MPI_STATUSES_IGNORE);
    MPI_Waitall(((op_mpi_buffer)(dat->mpi_buffer))->r_num_req,
                ((op_mpi_buffer)(dat->mpi_buffer))->r_req, MPI_STATUSES_IGNORE);
    ((op_mpi_buffer)(dat->mpi_buffer))->s_num_req = 0;
    ((op_mpi_buffer)(dat->mpi_buffer))->r_num_req = 0;

    if (arg->map != OP_ID && OP_map_partial_exchange[arg->map->index]) {
      halo_list imp_nonexec_list = OP_import_nonexec_permap[arg->map->index];
      int nonexec_init = OP_export_nonexec_permap[arg->map->index]->size;
      ;
      if (OP_gpu_direct == 0)
        cutilSafeCall(cudaMemcpyAsync(
            dat->buffer_d + nonexec_init * dat->size,
            &((op_mpi_buffer)(dat->mpi_buffer))
                 ->buf_nonexec[nonexec_init * dat->size],
            imp_nonexec_list->size * dat->size, cudaMemcpyHostToDevice, 0));
      scatter_data_from_buffer_partial(*arg);
    } else {
      if (OP_gpu_direct == 0) {
        if (strstr(arg->dat->type, ":soa") != NULL ||
            (OP_auto_soa && arg->dat->dim > 1)) {
          int init = dat->set->size * dat->size;
          int size = (dat->set->exec_size + dat->set->nonexec_size) * dat->size;
          cutilSafeCall(cudaMemcpyAsync(dat->buffer_d_r, dat->data + init, size,
                                        cudaMemcpyHostToDevice, 0));
          scatter_data_from_buffer(*arg);
        } else {
          int init = dat->set->size * dat->size;
          cutilSafeCall(
              cudaMemcpyAsync(dat->data_d + init, dat->data + init,
                              (OP_import_exec_list[dat->set->index]->size +
                               OP_import_nonexec_list[dat->set->index]->size) *
                                  arg->dat->size,
                              cudaMemcpyHostToDevice, 0));
          
          if (OP_kern_max > 0){
            OP_kernels[OP_kern_curr].halo_data2 += (OP_import_exec_list[dat->set->index]->size +
                               OP_import_nonexec_list[dat->set->index]->size) * arg->dat->size;
          }
        }
      } else if (strstr(arg->dat->type, ":soa") != NULL ||
                 (OP_auto_soa && arg->dat->dim > 1))
        scatter_data_from_buffer(*arg);
    }
    arg->sent = 2; // set flag to indicate completed comm
  }
}

void op_wait_all(op_arg *arg) {
  if (arg->opt && arg->argtype == OP_ARG_DAT && arg->sent == 1) {
    op_dat dat = arg->dat;
    MPI_Waitall(((op_mpi_buffer)(dat->mpi_buffer))->s_num_req,
                ((op_mpi_buffer)(dat->mpi_buffer))->s_req, MPI_STATUSES_IGNORE);
    MPI_Waitall(((op_mpi_buffer)(dat->mpi_buffer))->r_num_req,
                ((op_mpi_buffer)(dat->mpi_buffer))->r_req, MPI_STATUSES_IGNORE);
    ((op_mpi_buffer)(dat->mpi_buffer))->s_num_req = 0;
    ((op_mpi_buffer)(dat->mpi_buffer))->r_num_req = 0;
    if (arg->map != OP_ID && OP_map_partial_exchange[arg->map->index]) {
      halo_list imp_nonexec_list = OP_import_nonexec_permap[arg->map->index];
      int init = OP_export_nonexec_permap[arg->map->index]->size;
      char *buffer =
          &((op_mpi_buffer)(dat->mpi_buffer))->buf_nonexec[init * dat->size];
      for (int i = 0; i < imp_nonexec_list->size; i++) {
        int set_elem_index = imp_nonexec_list->list[i];
        memcpy((void *)&dat->data[dat->size * (set_elem_index)],
               &buffer[i * dat->size], dat->size);
      }
    }
  }
  arg->sent = 0;
}

void op_partition(const char *lib_name, const char *lib_routine,
                  op_set prime_set, op_map prime_map, op_dat coords) {
  partition(lib_name, lib_routine, prime_set, prime_map, coords);
  if (!OP_hybrid_gpu)
    return;
  op_move_to_device();
}

void op_move_to_device() {
  for (int s = 0; s < OP_set_index; s++) {
    op_set set = OP_set_list[s];
    op_dat_entry *item;
    TAILQ_FOREACH(item, &OP_dat_list, entries) {
      op_dat dat = item->dat;

      if (dat->set->index == set->index)
        op_mv_halo_device(set, dat);
    }
  }

  for (int m = 0; m < OP_map_index; m++) {
    
    // Upload maps in transposed form
    op_map map = OP_map_list[m];
    int set_size = map->from->size + map->from->exec_size;
    int *temp_map = (int *)xmalloc(map->dim * set_size * sizeof(int));
    for (int i = 0; i < map->dim; i++) {
      for (int j = 0; j < set_size; j++) {
        temp_map[i * set_size + j] = map->map[map->dim * j + i];
      }
    }
    op_cpHostToDevice((void **)&(map->map_d), (void **)&(temp_map),
                      map->dim * set_size * sizeof(int));
    free(temp_map);
  }

#ifdef COMM_AVOID

  int my_rank = 0;
  // MPI_Comm_rank(OP_MPI_WORLD, &my_rank);

  for (int m = 0; m < OP_map_index; m++) {
    op_map map = OP_map_list[m];
    int max_level = map->halo_info->max_calc_nhalos;
    
    map->aug_maps_d = (int **)xmalloc(sizeof(int *) * map->halo_info->max_calc_nhalos);

    for(int el = 0; el < map->halo_info->max_calc_nhalos; el++){
      if(is_halo_required_for_map(map, el) == 1 && is_map_required_for_calc(map, el) == 1){

        int exec_size = map->from->exec_sizes[el];
        // for(int l = 0; l < el + 1; l++){
        //   exec_size += OP_aug_import_exec_lists[l][map->from->index]->size;
        // }
        int nonexec_size = 0;
        for(int l = 0; l < el + 1 - 1; l++){ // last non exec level is not included. non exec mappings are not included in maps
          if(is_halo_required_for_map(map, el) == 1){
            nonexec_size += map->from->nonexec_sizes[l];
          }
        }
        int set_size = map->from->size + exec_size + nonexec_size;

        int *temp_map = (int *)xmalloc(map->dim * set_size * sizeof(int));
        for (int i = 0; i < map->dim; i++) {
          for (int j = 0; j < set_size; j++) {
            temp_map[i * set_size + j] = map->aug_maps[el][map->dim * j + i];
            // printf("augmentednew my_rank=%d map[%d]=%s val[%d][%d]=%d\n", my_rank, el, map->name,
            // el * map->dim * set_size + i * set_size + j, i * set_size + j, temp_map[el * map->dim * set_size + i * set_size + j]);
          }
        }
        map->aug_maps_d[el] = NULL;
        op_cpHostToDevice((void **)&(map->aug_maps_d[el]),
                        (void **)&(temp_map),
                        map->dim * set_size * sizeof(int));

        free(temp_map);
      }else{
        map->aug_maps_d[el] = NULL;
      }
    }
  }
#endif

  op_mv_halo_list_device();
}

int op_is_root() {
  int my_rank;
  MPI_Comm_rank(OP_MPI_WORLD, &my_rank);
  return (my_rank == MPI_ROOT);
}

int op2_grp_size_recv_old = 0;
int op2_grp_size_send_old = 0;
void op_realloc_comm_buffer(char **send_buffer_host, char **recv_buffer_host, 
      char **send_buffer_device, char **recv_buffer_device, int device, 
      unsigned size_send, unsigned size_recv) {
  if (op2_grp_size_recv_old < size_recv) {
    //if (*recv_buffer_host != NULL) cutilSafeCall(cudaFreeHost(*recv_buffer_host));
    if (*recv_buffer_device != NULL) cutilSafeCall(cudaFree(*recv_buffer_device));
    cutilSafeCall(cudaMalloc(recv_buffer_device, size_recv));
    //cutilSafeCall(cudaMallocHost(recv_buffer_host, size_send));
    if (op2_grp_size_recv_old>0) cutilSafeCall(cudaHostUnregister ( *recv_buffer_host ));
    *recv_buffer_host = (char*)op_realloc(*recv_buffer_host, size_recv);
    cutilSafeCall(cudaHostRegister ( *recv_buffer_host, size_recv, cudaHostRegisterDefault ));
    op2_grp_size_recv_old = size_recv;
  }
  if (op2_grp_size_send_old < size_send) {
    //if (*send_buffer_host != NULL) cutilSafeCall(cudaFreeHost(*send_buffer_host));
    if (*send_buffer_device != NULL) cutilSafeCall(cudaFree(*send_buffer_device));
    cutilSafeCall(cudaMalloc(send_buffer_device, size_send));
    //cutilSafeCall(cudaMallocHost(send_buffer_host, size_recv));
    if (op2_grp_size_send_old>0) cutilSafeCall(cudaHostUnregister ( *send_buffer_host ));
    *send_buffer_host = (char*)op_realloc(*send_buffer_host, size_send);
    cutilSafeCall(cudaHostRegister ( *send_buffer_host, size_send, cudaHostRegisterDefault ));
    op2_grp_size_send_old = size_send;
  }
}

void op_download_buffer_async(char *send_buffer_device, char *send_buffer_host, unsigned size_send) {
  //Make sure gather kernels on the 0 stream finished before starting download
  cutilSafeCall(cudaEventRecord(op2_grp_download_event,0));
  cutilSafeCall(cudaStreamWaitEvent(op2_grp_secondary, op2_grp_download_event,0));
  cutilSafeCall(cudaMemcpyAsync(send_buffer_host, send_buffer_device, size_send, cudaMemcpyDeviceToHost, op2_grp_secondary));
}
void op_upload_buffer_async  (char *recv_buffer_device, char *recv_buffer_host, unsigned size_recv) {
  cutilSafeCall(cudaMemcpyAsync(recv_buffer_device, recv_buffer_host, size_recv, cudaMemcpyHostToDevice, op2_grp_secondary));
}

void op_scatter_sync() {
  cutilSafeCall(cudaEventRecord(op2_grp_download_event, op2_grp_secondary));
  cutilSafeCall(cudaStreamWaitEvent(0, op2_grp_download_event,0));
}
void op_download_buffer_sync() {
  cutilSafeCall(cudaStreamSynchronize(op2_grp_secondary));
}
