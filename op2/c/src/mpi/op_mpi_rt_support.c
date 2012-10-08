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

/*
 * op_mpi_rt_support.c
 *
 * Implements the OP2 Distributed memory (MPI) halo exchange and
 * support routines/functions
 *
 * written by: Gihan R. Mudalige, (Started 01-03-2011)
 */

#include <op_lib_c.h>
#include <op_lib_core.h>
#include <op_util.h>

//mpi header
#include <mpi.h>

#include <op_mpi_core.h>
#include <op_rt_support.h>
#include <op_lib_mpi.h>


// //
// //MPI Halo related global variables
// //
//
// extern halo_list *OP_export_exec_list;//EEH list
// halo_list *OP_import_exec_list;//IEH list
//
// halo_list *OP_import_nonexec_list;//INH list
// halo_list *OP_export_nonexec_list;//ENH list
//
// //
// //global array to hold dirty_bits for op_dats
// //

/*******************************************************************************
 * Main MPI Halo Exchange Function
 *******************************************************************************/

void op_exchange_halo(op_arg* arg)
{
  //int my_rank, comm_size;
  //MPI_Comm_rank(OP_MPI_WORLD, &my_rank);
  //MPI_Comm_size(OP_MPI_WORLD, &comm_size);

  op_dat dat = arg->dat;

  if(arg->sent == 1)
  {
    printf("Error: Halo exchange already in flight for dat %s\n", dat->name);
    fflush(stdout);
    MPI_Abort(OP_MPI_WORLD, 2);
  }

  //need to exchange both direct and indirect data sets if they are dirty
  if((arg->acc == OP_READ || arg->acc == OP_RW /* good for debug || arg->acc == OP_INC*/) &&
     (dat->dirtybit == 1))
  {
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

    int set_elem_index;
    for(int i=0; i<exp_exec_list->ranks_size; i++) {
      for(int j = 0; j < exp_exec_list->sizes[i]; j++)
      {
        set_elem_index = exp_exec_list->list[exp_exec_list->disps[i]+j];
        memcpy(&((op_mpi_buffer)(dat->mpi_buffer))->
          buf_exec[exp_exec_list->disps[i]*dat->size+j*dat->size],
          (void *)&dat->data[dat->size*(set_elem_index)],dat->size);
      }
      //printf("export from %d to %d data %10s, number of elements of size %d | sending:\n ",
      //          my_rank, exp_exec_list->ranks[i], dat->name,exp_exec_list->sizes[i]);
      MPI_Isend(&((op_mpi_buffer)(dat->mpi_buffer))->
          buf_exec[exp_exec_list->disps[i]*dat->size],
          dat->size*exp_exec_list->sizes[i],
          MPI_CHAR, exp_exec_list->ranks[i],
          dat->index, OP_MPI_WORLD,
          &((op_mpi_buffer)(dat->mpi_buffer))->
          s_req[((op_mpi_buffer)(dat->mpi_buffer))->s_num_req++]);
    }


    int init = dat->set->size*dat->size;
    for(int i=0; i < imp_exec_list->ranks_size; i++) {
     // printf("import on to %d from %d data %10s, number of elements of size %d | recieving:\n ",
     //       my_rank, imp_exec_list->ranks[i], dat->name, imp_exec_list->sizes[i]);
      MPI_Irecv(&(dat->data[init+imp_exec_list->disps[i]*dat->size]),
          dat->size*imp_exec_list->sizes[i],
          MPI_CHAR, imp_exec_list->ranks[i],
          dat->index, OP_MPI_WORLD,
          &((op_mpi_buffer)(dat->mpi_buffer))->
          r_req[((op_mpi_buffer)(dat->mpi_buffer))->r_num_req++]);
    }

    //-----second exchange nonexec elements related to this data array------
    //sanity checks
    if(compare_sets(imp_nonexec_list->set,dat->set) == 0)
    {
      printf("Error: Non-Import list and set mismatch");
      MPI_Abort(OP_MPI_WORLD, 2);
    }
    if(compare_sets(exp_nonexec_list->set,dat->set)==0)
    {
      printf("Error: Non-Export list and set mismatch");
      MPI_Abort(OP_MPI_WORLD, 2);
    }


    for(int i=0; i<exp_nonexec_list->ranks_size; i++) {
      for(int j = 0; j < exp_nonexec_list->sizes[i]; j++)
      {
        set_elem_index = exp_nonexec_list->list[exp_nonexec_list->disps[i]+j];
        memcpy(&((op_mpi_buffer)(dat->mpi_buffer))->
            buf_nonexec[exp_nonexec_list->disps[i]*dat->size+j*dat->size],
            (void *)&dat->data[dat->size*(set_elem_index)],dat->size);
      }
      //printf("export from %d to %d data %10s, number of elements of size %d | sending:\n ",
      //          my_rank, exp_nonexec_list->ranks[i], dat->name,exp_nonexec_list->sizes[i]);
      MPI_Isend(&((op_mpi_buffer)(dat->mpi_buffer))->
          buf_nonexec[exp_nonexec_list->disps[i]*dat->size],
          dat->size*exp_nonexec_list->sizes[i],
          MPI_CHAR, exp_nonexec_list->ranks[i],
          dat->index, OP_MPI_WORLD,
          &((op_mpi_buffer)(dat->mpi_buffer))->
          s_req[((op_mpi_buffer)(dat->mpi_buffer))->s_num_req++]);
    }

    int nonexec_init = (dat->set->size+imp_exec_list->size)*dat->size;
    for(int i=0; i<imp_nonexec_list->ranks_size; i++) {
      //printf("import on to %d from %d data %10s, number of elements of size %d | recieving:\n ",
      //      my_rank, imp_nonexec_list->ranks[i], dat->name, imp_nonexec_list->sizes[i]);
      MPI_Irecv(&(dat->data[nonexec_init+imp_nonexec_list->disps[i]*dat->size]),
          dat->size*imp_nonexec_list->sizes[i],
          MPI_CHAR, imp_nonexec_list->ranks[i],
          dat->index, OP_MPI_WORLD,
          &((op_mpi_buffer)(dat->mpi_buffer))->
          r_req[((op_mpi_buffer)(dat->mpi_buffer))->r_num_req++]);
    }
    //clear dirty bit
    dat->dirtybit = 0;
    arg->sent = 1;
  }
}

/*******************************************************************************
 * MPI Halo Exchange Wait-all Function (to complete the non-blocking comms)
 *******************************************************************************/

void op_wait_all(op_arg* arg)
{
  if(arg->argtype == OP_ARG_DAT && arg->sent == 1)
  {
    op_dat dat = arg->dat;
    MPI_Waitall(((op_mpi_buffer)(dat->mpi_buffer))->s_num_req,
      ((op_mpi_buffer)(dat->mpi_buffer))->s_req,
      MPI_STATUSES_IGNORE );
    MPI_Waitall(((op_mpi_buffer)(dat->mpi_buffer))->r_num_req,
      ((op_mpi_buffer)(dat->mpi_buffer))->r_req,
      MPI_STATUSES_IGNORE );
    ((op_mpi_buffer)(dat->mpi_buffer))->s_num_req = 0;
    ((op_mpi_buffer)(dat->mpi_buffer))->r_num_req = 0;
  }

  arg->sent = 0;
}

void op_partition(const char* lib_name, const char* lib_routine,
  op_set prime_set, op_map prime_map, op_dat coords )
{
  partition(lib_name, lib_routine, prime_set, prime_map, coords );

}
