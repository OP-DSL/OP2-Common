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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math_constants.h>

#include <op_lib_c.h>
#include <op_lib_core.h>
#include <op_rt_support.h>
#include <op_cuda_rt_support.h>

#include <op_lib_mpi.h>
#include <op_util.h>

//
//export lists on the device
//

int** export_exec_list_d;
int** export_nonexec_list_d;

void op_exchange_halo(op_arg* arg)
{
  op_dat dat = arg->dat;

  if((arg->argtype == OP_ARG_DAT) && (arg->idx != -1) &&
    (arg->acc == OP_READ || arg->acc == OP_RW ) &&
      (dat->dirtybit == 1)) {

    //printf("Exchanging Halo of data array %10s\n",dat->name);
    halo_list imp_exec_list = OP_import_exec_list[dat->set->index];
    halo_list imp_nonexec_list = OP_import_nonexec_list[dat->set->index];

    halo_list exp_exec_list = OP_export_exec_list[dat->set->index];
    halo_list exp_nonexec_list = OP_export_nonexec_list[dat->set->index];

    //-------first exchange exec elements related to this data array--------

    //sanity checks
    if(compare_sets(imp_exec_list->set,dat->set) == 0)
    {
      printf("Error: Import list and set mismatch\n");
      MPI_Abort(OP_MPI_WORLD, 2);
    }
    if(compare_sets(exp_exec_list->set,dat->set) == 0)
    {
      printf("Error: Export list and set mismatch\n");
      MPI_Abort(OP_MPI_WORLD, 2);
    }

    gather_data_to_buffer(*arg, exp_exec_list, exp_nonexec_list);

    cutilSafeCall( cudaMemcpy ( OP_mpi_buffer_list[dat->index]-> buf_exec,
          arg->dat->buffer_d, exp_exec_list->size*arg->dat->size, cudaMemcpyDeviceToHost ) );

    cutilSafeCall( cudaMemcpy ( OP_mpi_buffer_list[dat->index]-> buf_nonexec,
          arg->dat->buffer_d+exp_exec_list->size*arg->dat->size,
          exp_nonexec_list->size*arg->dat->size,
          cudaMemcpyDeviceToHost ) );

    cutilSafeCall(cudaThreadSynchronize(  ));

    for(int i=0; i<exp_exec_list->ranks_size; i++) {
      MPI_Isend(&OP_mpi_buffer_list[dat->index]->
          buf_exec[exp_exec_list->disps[i]*dat->size],
          dat->size*exp_exec_list->sizes[i],
          MPI_CHAR, exp_exec_list->ranks[i],
          dat->index, OP_MPI_WORLD,
          &OP_mpi_buffer_list[dat->index]->
          s_req[OP_mpi_buffer_list[dat->index]->s_num_req++]);
    }


    int init = dat->set->size*dat->size;
    for(int i=0; i < imp_exec_list->ranks_size; i++) {
      MPI_Irecv(&(OP_dat_list[dat->index]->
            data[init+imp_exec_list->disps[i]*dat->size]),
          dat->size*imp_exec_list->sizes[i],
          MPI_CHAR, imp_exec_list->ranks[i],
          dat->index, OP_MPI_WORLD,
          &OP_mpi_buffer_list[dat->index]->
          r_req[OP_mpi_buffer_list[dat->index]->r_num_req++]);
    }


    //-----second exchange nonexec elements related to this data array------
    //sanity checks
    if(compare_sets(imp_nonexec_list->set,dat->set) == 0) {
      printf("Error: Non-Import list and set mismatch");
      MPI_Abort(OP_MPI_WORLD, 2);
    }
    if(compare_sets(exp_nonexec_list->set,dat->set)==0) {
      printf("Error: Non-Export list and set mismatch");
      MPI_Abort(OP_MPI_WORLD, 2);
    }

    for(int i=0; i<exp_nonexec_list->ranks_size; i++) {
      MPI_Isend(&OP_mpi_buffer_list[dat->index]->
          buf_nonexec[exp_nonexec_list->disps[i]*dat->size],
          dat->size*exp_nonexec_list->sizes[i],
          MPI_CHAR, exp_nonexec_list->ranks[i],
          dat->index, OP_MPI_WORLD,
          &OP_mpi_buffer_list[dat->index]->
          s_req[OP_mpi_buffer_list[dat->index]->s_num_req++]);
    }

    int nonexec_init = (dat->set->size+imp_exec_list->size)*dat->size;
    for(int i=0; i<imp_nonexec_list->ranks_size; i++) {
      MPI_Irecv(&(OP_dat_list[dat->index]->
            data[nonexec_init+imp_nonexec_list->disps[i]*dat->size]),
          dat->size*imp_nonexec_list->sizes[i],
          MPI_CHAR, imp_nonexec_list->ranks[i],
          dat->index, OP_MPI_WORLD,
          &OP_mpi_buffer_list[dat->index]->
          r_req[OP_mpi_buffer_list[dat->index]->r_num_req++]);
    }

    //clear dirty bit
    dat->dirtybit = 0;
    arg->sent = 1;
  }
}

void op_wait_all(op_arg* arg)
{
  if(arg->argtype == OP_ARG_DAT && arg->sent == 1)
  {
    op_dat dat = arg->dat;
    MPI_Waitall(OP_mpi_buffer_list[dat->index]->s_num_req,
      OP_mpi_buffer_list[dat->index]->s_req,
      MPI_STATUSES_IGNORE );
    MPI_Waitall(OP_mpi_buffer_list[dat->index]->r_num_req,
      OP_mpi_buffer_list[dat->index]->r_req,
      MPI_STATUSES_IGNORE );
    OP_mpi_buffer_list[dat->index]->s_num_req = 0;
    OP_mpi_buffer_list[dat->index]->r_num_req = 0;

    int init = dat->set->size*dat->size;
    cutilSafeCall( cudaMemcpy( dat->data_d + init, dat->data + init,
          OP_import_exec_list[dat->set->index]->size*arg->dat->size,
          cudaMemcpyHostToDevice ) );

    int nonexec_init = (dat->set->size+OP_import_exec_list[dat->set->index]->size)*dat->size;
    cutilSafeCall( cudaMemcpy( dat->data_d + nonexec_init, dat->data + nonexec_init,
          OP_import_nonexec_list[dat->set->index]->size*arg->dat->size,
          cudaMemcpyHostToDevice ) );

    cutilSafeCall(cudaThreadSynchronize ());
  }
}

void op_partition(const char* lib_name, const char* lib_routine,
  op_set prime_set, op_map prime_map, op_dat coords )
{
  partition(lib_name, lib_routine, prime_set, prime_map, coords );

  for(int s = 0; s<OP_set_index; s++)
  {
    op_set set=OP_set_list[s];
    for(int d=0; d<OP_dat_index; d++) { //for each data array
      op_dat dat=OP_dat_list[d];

      if(dat->set->index == set->index)
          op_mv_halo_device(set, dat);
    }
  }

  op_mv_halo_list_device();

}

