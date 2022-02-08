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

// mpi header
#include <mpi.h>

#include <op_lib_c.h>
#include <op_lib_core.h>
#include <op_util.h>

#include <op_lib_mpi.h>
#include <op_mpi_core.h>
#include <op_rt_support.h>

void op_upload_dat(op_dat dat) {}

void op_download_dat(op_dat dat) {}

/*******************************************************************************
 * Main MPI Halo Exchange Function
 *******************************************************************************/

#ifndef COMM_AVOID
void op_exchange_halo(op_arg *arg, int exec_flag) {
  op_dat dat = arg->dat;

  if (arg->opt == 0)
    return;

  if (arg->sent == 1) {
    printf("Error: Halo exchange already in flight for dat %s\n", dat->name);
    fflush(stdout);
    MPI_Abort(OP_MPI_WORLD, 2);
  }
  if (exec_flag == 0 && arg->idx == -1)
    return;

  // For a directly accessed op_dat do not do halo exchanges if not executing
  // over
  // redundant compute block
  if (exec_flag == 0 && arg->idx == -1)
    return;

  arg->sent = 0; // reset flag

  // need to exchange both direct and indirect data sets if they are dirty
  if ((arg->acc == OP_READ ||
       arg->acc == OP_RW /* good for debug || arg->acc == OP_INC*/) &&
      (dat->dirtybit == 1)) {
    printf("op_exchange_halo dat %s exec_flag=%d exchanged >>>>>>>>>>\n", dat->name, exec_flag);
    //    printf("Exchanging Halo of data array %10s\n",dat->name);
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
      int my_rank;
      MPI_Comm_rank(OP_MPI_WORLD, &my_rank);
      // printf("export exec from %d to %d data %10s, number of elements of size %d | sending:\n ",
      //             my_rank, exp_exec_list->ranks[i],
      //             dat->name,exp_exec_list->sizes[i]);
      // double *b = (double*)&((op_mpi_buffer)(dat->mpi_buffer))
      //                ->buf_exec[exp_exec_list->disps[i] * dat->size];
      // for (int el = 0; el < (dat->size * exp_exec_list->sizes[i])/8; el++)
      //   printf("%g ", b[el]);
      // printf("\n");
      
      MPI_Isend(&((op_mpi_buffer)(dat->mpi_buffer))
                     ->buf_exec[exp_exec_list->disps[i] * dat->size],
                dat->size * exp_exec_list->sizes[i], MPI_CHAR,
                exp_exec_list->ranks[i], dat->index, OP_MPI_WORLD,
                &((op_mpi_buffer)(dat->mpi_buffer))
                     ->s_req[((op_mpi_buffer)(dat->mpi_buffer))->s_num_req++]);
    }

    int init = dat->set->size * dat->size;
    for (int i = 0; i < imp_exec_list->ranks_size; i++) {
      //      printf("import exec on to %d from %d data %10s, number of elements
      //      of size %d | recieving:\n ",
      //           my_rank, imp_exec_list->ranks[i], dat->name,
      //           imp_exec_list->sizes[i]);
      MPI_Irecv(&(dat->data[init + imp_exec_list->disps[i] * dat->size]),
                dat->size * imp_exec_list->sizes[i], MPI_CHAR,
                imp_exec_list->ranks[i], dat->index, OP_MPI_WORLD,
                &((op_mpi_buffer)(dat->mpi_buffer))
                     ->r_req[((op_mpi_buffer)(dat->mpi_buffer))->r_num_req++]);
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

    int rank;
    MPI_Comm_rank(OP_MPI_WORLD, &rank);
    for (int i = 0; i < exp_nonexec_list->ranks_size; i++) {
      for (int j = 0; j < exp_nonexec_list->sizes[i]; j++) {
        set_elem_index = exp_nonexec_list->list[exp_nonexec_list->disps[i] + j];
        memcpy(&((op_mpi_buffer)(dat->mpi_buffer))
                    ->buf_nonexec[exp_nonexec_list->disps[i] * dat->size +
                                  j * dat->size],
               (void *)&dat->data[dat->size * (set_elem_index)], dat->size);
      }
      // printf("export nonexec from %d to %d data %10s, number of elements of size %d | sending:\n ",
      //                 rank, exp_nonexec_list->ranks[i],
      //                 dat->name,exp_nonexec_list->sizes[i]);
      // double *b = (double*)&((op_mpi_buffer)(dat->mpi_buffer))
      //                ->buf_nonexec[exp_nonexec_list->disps[i] * dat->size];
      // for (int el = 0; el < (dat->size * exp_nonexec_list->sizes[i])/8; el++)
      //   printf("%g ", b[el]);
      // printf("\n");
      MPI_Isend(&((op_mpi_buffer)(dat->mpi_buffer))
                     ->buf_nonexec[exp_nonexec_list->disps[i] * dat->size],
                dat->size * exp_nonexec_list->sizes[i], MPI_CHAR,
                exp_nonexec_list->ranks[i], dat->index, OP_MPI_WORLD,
                &((op_mpi_buffer)(dat->mpi_buffer))
                     ->s_req[((op_mpi_buffer)(dat->mpi_buffer))->s_num_req++]);
    }

    int nonexec_init = (dat->set->size + imp_exec_list->size) * dat->size;
    for (int i = 0; i < imp_nonexec_list->ranks_size; i++) {
      //      printf("import on to %d from %d data %10s, number of elements of
      //      size %d | recieving:\n ",
      //            my_rank, imp_nonexec_list->ranks[i], dat->name,
      //            imp_nonexec_list->sizes[i]);
      MPI_Irecv(
          &(dat->data[nonexec_init + imp_nonexec_list->disps[i] * dat->size]),
          dat->size * imp_nonexec_list->sizes[i], MPI_CHAR,
          imp_nonexec_list->ranks[i], dat->index, OP_MPI_WORLD,
          &((op_mpi_buffer)(dat->mpi_buffer))
               ->r_req[((op_mpi_buffer)(dat->mpi_buffer))->r_num_req++]);
    }
    // clear dirty bit
    dat->dirtybit = 0;
    arg->sent = 1;
  }
}

#else

void op_exchange_halo_chained(op_arg *arg, int exec_flag, int h_levels) {
  op_dat dat = arg->dat;
  printf("op_exchange_halo_chained dat=%s set=%s h_levels=%d\n", dat->name, dat->set->name, h_levels);

  int my_rank;
  MPI_Comm_rank(OP_MPI_WORLD, &my_rank);

  if (arg->opt == 0)
    return;

  if (arg->sent == 1) {
    printf("Error: Halo exchange already in flight for dat %s\n", dat->name);
    fflush(stdout);
    MPI_Abort(OP_MPI_WORLD, 2);
  }
  if (exec_flag == 0 && arg->idx == -1)
    return;

  // For a directly accessed op_dat do not do halo exchanges if not executing
  // over
  // redundant compute block
  if (exec_flag == 0 && arg->idx == -1)
    return;

  arg->sent = 0; // reset flag

  // int exec_levels = 2;
  int num_levels = dat->set->dat_to_execlevels->get_val_at(h_levels); //dat->set->dat_to_execlevels->get_count();
  printf("op_exchange_halo_chained dat %s h_levels=%d num_levels=%d\n", dat->name, h_levels, num_levels);
  // need to exchange both direct and indirect data sets if they are dirty
  if ((arg->acc == OP_READ ||
       arg->acc == OP_RW /* good for debug || arg->acc == OP_INC*/) &&
      (dat->dirtybit == 1)) {
    printf("op_exchange_halo_chained dat %s h_levels=%d num_levels=%d exchanged >>>>>>>>>>\n", dat->name, h_levels, num_levels);
    halo_list imp_exec_list = OP_merged_import_exec_list[dat->set->index];
    halo_list imp_nonexec_list = OP_aug_import_nonexec_lists[h_levels][dat->set->index]; //OP_import_nonexec_list[dat->set->index];

    halo_list exp_exec_list = OP_merged_export_exec_list[dat->set->index];
    halo_list exp_nonexec_list = OP_aug_export_nonexec_lists[h_levels][dat->set->index]; //OP_export_nonexec_list[dat->set->index];

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

    int set_elem_index = 0;
    int buf_index = 0;
    int buf_start = 0;

    for (int r = 0; r < exp_exec_list->ranks_size / exp_exec_list->num_levels; r++) {
      buf_start =  buf_index;
      for(int l = 0; l < exp_exec_list->num_levels; l++){
        for (int i = 0; i < exp_exec_list->sizes[exp_exec_list->rank_disps[l] + r]; i++) {
          int level_disp = exp_exec_list->level_disps[l];
          int disp_in_level = exp_exec_list->disps[exp_exec_list->rank_disps[l] + r];
          set_elem_index = exp_exec_list->list[level_disp + disp_in_level + i];

          memcpy(&((op_mpi_buffer)(dat->mpi_buffer))
                    ->buf_exec[buf_index * dat->size],
               (void *)&dat->data[dat->size * (set_elem_index)], dat->size);

          buf_index++;
        }
      }

      MPI_Isend(&((op_mpi_buffer)(dat->mpi_buffer))
                     ->buf_exec[buf_start * dat->size],
                dat->size * (buf_index - buf_start), MPI_CHAR,
                exp_exec_list->ranks[r], dat->index, OP_MPI_WORLD,
                &((op_mpi_buffer)(dat->mpi_buffer))
                     ->s_req[((op_mpi_buffer)(dat->mpi_buffer))->s_num_req++]);
        
    }

    for (int i = 0; i < imp_exec_list->ranks_size / imp_exec_list->num_levels; i++) {

      MPI_Irecv(&(dat->aug_data[imp_exec_list->disps_by_rank[i] * dat->size]),
                dat->size * imp_exec_list->sizes_by_rank[i], MPI_CHAR,
                imp_exec_list->ranks[i], dat->index, OP_MPI_WORLD,
                &((op_mpi_buffer)(dat->mpi_buffer))
                     ->r_req[((op_mpi_buffer)(dat->mpi_buffer))->r_num_req++]);
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
      // printf("export nonexec from %d to %d data %10s, number of elements of size %d | sending:\n ",
      //                 rank, exp_nonexec_list->ranks[i],
      //                 dat->name,exp_nonexec_list->sizes[i]);
      // double *b = (double*)&((op_mpi_buffer)(dat->mpi_buffer))
      //                ->buf_nonexec[exp_nonexec_list->disps[i] * dat->size];
      // for (int el = 0; el < (dat->size * exp_nonexec_list->sizes[i])/8; el++)
      //   printf("%g ", b[el]);
      // printf("\n");
      MPI_Isend(&((op_mpi_buffer)(dat->mpi_buffer))
                     ->buf_nonexec[exp_nonexec_list->disps[i] * dat->size],
                dat->size * exp_nonexec_list->sizes[i], MPI_CHAR,
                exp_nonexec_list->ranks[i], dat->index, OP_MPI_WORLD,
                &((op_mpi_buffer)(dat->mpi_buffer))
                     ->s_req[((op_mpi_buffer)(dat->mpi_buffer))->s_num_req++]);
    }

    int nonexec_init = 0;
    for(int l = 0; l < h_levels; l++){
      nonexec_init += OP_aug_import_nonexec_lists[l][dat->set->index]->size;
    }
    nonexec_init *= dat->size;
    nonexec_init += (dat->set->size + imp_exec_list->size) * dat->size;
    for (int i = 0; i < imp_nonexec_list->ranks_size; i++) {
      //      printf("import on to %d from %d data %10s, number of elements of
      //      size %d | recieving:\n ",
      //            my_rank, imp_nonexec_list->ranks[i], dat->name,
      //            imp_nonexec_list->sizes[i]);
      MPI_Irecv(
          &(dat->data[nonexec_init + imp_nonexec_list->disps[i] * dat->size]),
          dat->size * imp_nonexec_list->sizes[i], MPI_CHAR,
          imp_nonexec_list->ranks[i], dat->index, OP_MPI_WORLD,
          &((op_mpi_buffer)(dat->mpi_buffer))
               ->r_req[((op_mpi_buffer)(dat->mpi_buffer))->r_num_req++]);
    }

    // int rank;
    // MPI_Comm_rank(OP_MPI_WORLD, &rank);

    // set_elem_index = 0;
    // buf_index = 0;
    // buf_start = 0;
    // for (int r = 0; r < exp_nonexec_list->ranks_size / exp_nonexec_list->num_levels; r++) {
    //   buf_start =  buf_index;
    //   for(int l = 0; l < exp_nonexec_list->num_levels; l++){
    //     for (int i = 0; i < exp_nonexec_list->sizes[exp_nonexec_list->rank_disps[l] + r]; i++) {
    //       int level_disp = exp_nonexec_list->level_disps[l];
    //       int disp_in_level = exp_nonexec_list->disps[exp_nonexec_list->rank_disps[l] + r];
    //       set_elem_index = exp_nonexec_list->list[level_disp + disp_in_level + i];

    //       memcpy(&((op_mpi_buffer)(dat->mpi_buffer))
    //                 ->buf_nonexec[buf_index * dat->size],
    //            (void *)&dat->data[dat->size * (set_elem_index)], dat->size);
    //       buf_index++;
    //     }
    //   }

    //   MPI_Isend(&((op_mpi_buffer)(dat->mpi_buffer))
    //                  ->buf_nonexec[buf_start * dat->size],
    //             dat->size * (buf_index - buf_start), MPI_CHAR,
    //             exp_nonexec_list->ranks[r], dat->index, OP_MPI_WORLD,
    //             &((op_mpi_buffer)(dat->mpi_buffer))
    //                  ->s_req[((op_mpi_buffer)(dat->mpi_buffer))->s_num_req++]);
        
    // }

    // int nonexec_init = (dat->set->size + imp_exec_list->size) * dat->size;

    //  for (int i = 0; i < imp_nonexec_list->ranks_size / imp_nonexec_list->num_levels; i++) {
    //   MPI_Irecv(&(dat->data[nonexec_init + imp_nonexec_list->disps_by_rank[i] * dat->size]),
    //             dat->size * imp_nonexec_list->sizes_by_rank[i], MPI_CHAR,
    //             imp_nonexec_list->ranks[i], dat->index, OP_MPI_WORLD,
    //             &((op_mpi_buffer)(dat->mpi_buffer))
    //                  ->r_req[((op_mpi_buffer)(dat->mpi_buffer))->r_num_req++]);
    // }

    // clear dirty bit
    dat->dirtybit = 0;
    arg->sent = 1;
  }
}

void op_unpack(op_arg *arg){
  op_dat dat = arg->dat;
  int my_rank;
  MPI_Comm_rank(OP_MPI_WORLD, &my_rank);

  int init = dat->set->size * dat->size;
  halo_list imp_exec_list = OP_merged_import_exec_list[dat->set->index];

  for (int i = 0; i < imp_exec_list->ranks_size / imp_exec_list->num_levels; i++) {
  
    int prev_exec_size = 0;
    for(int l = 0; l < imp_exec_list->num_levels; l++){ // this has to be changed to dat's levels

      memcpy(&(dat->data[init + (imp_exec_list->level_disps[l] + imp_exec_list->disps[imp_exec_list->rank_disps[l] + i]) * dat->size]), 
            &(dat->aug_data[imp_exec_list->disps_by_rank[i] * dat->size + prev_exec_size * dat->size]),
                            dat->size * imp_exec_list->sizes[imp_exec_list->rank_disps[l] + i]);

      prev_exec_size += imp_exec_list->sizes[imp_exec_list->rank_disps[l] + i];
    }
  }
}

#endif

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
    }

    // note that we are not settinging the dirtybit to 0, since it's not a full
    // exchange
    arg->sent = 1;
  }
}

void op_exchange_halo_cuda(op_arg *arg, int exec_flag) {}

void op_exchange_halo_partial_cuda(op_arg *arg, int exec_flag) {}

/*******************************************************************************
 * MPI Halo Exchange Wait-all Function (to complete the non-blocking comms)
 *******************************************************************************/

void op_wait_all(op_arg *arg) {
  if (arg->opt && arg->argtype == OP_ARG_DAT && arg->sent == 1) {
    op_dat dat = arg->dat;
    MPI_Waitall(((op_mpi_buffer)(dat->mpi_buffer))->s_num_req,
                ((op_mpi_buffer)(dat->mpi_buffer))->s_req, MPI_STATUSES_IGNORE);
    MPI_Waitall(((op_mpi_buffer)(dat->mpi_buffer))->r_num_req,
                ((op_mpi_buffer)(dat->mpi_buffer))->r_req, MPI_STATUSES_IGNORE);
    ((op_mpi_buffer)(dat->mpi_buffer))->s_num_req = 0;
    ((op_mpi_buffer)(dat->mpi_buffer))->r_num_req = 0;
    arg->sent = 2; // set flag to indicate completed comm

    #ifdef COMM_AVOID
    op_unpack(arg);
    #endif

    if (arg->map != OP_ID && OP_map_partial_exchange[arg->map->index]) {
      int my_rank;
      op_rank(&my_rank);
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
}

void op_wait_all_cuda(op_arg *arg) {}

void op_partition(const char *lib_name, const char *lib_routine,
                  op_set prime_set, op_map prime_map, op_dat coords) {
  partition(lib_name, lib_routine, prime_set, prime_map, coords);
}

void op_move_to_device() {}

int op_is_root() {
  int my_rank;
  MPI_Comm_rank(OP_MPI_WORLD, &my_rank);
  return (my_rank == MPI_ROOT);
}

void deviceSync() {}

int op2_grp_size_recv_old = 0;
int op2_grp_size_send_old = 0;
void op_realloc_comm_buffer(char **send_buffer_host, char **recv_buffer_host, 
      char **send_buffer_device, char **recv_buffer_device, int device, 
      unsigned size_send, unsigned size_recv) {
  if (op2_grp_size_recv_old < size_recv) {
    *recv_buffer_host = (char*)op_realloc(*recv_buffer_host, size_recv);
    op2_grp_size_recv_old = size_recv;
  }
  if (op2_grp_size_send_old < size_send) {
    *send_buffer_host = (char*)op_realloc(*send_buffer_host, size_send);
    op2_grp_size_send_old = size_send;
  }
}

void op_download_buffer_async(char *send_buffer_device, char *send_buffer_host, unsigned size_send) {}
void op_upload_buffer_async  (char *recv_buffer_device, char *recv_buffer_host, unsigned size_send) {}
void op_download_buffer_sync() {}
void op_scatter_sync() {}
#include <vector>
void gather_data_to_buffer_ptr_cuda(op_arg arg, halo_list eel, halo_list enl, char *buffer, 
                               std::vector<int>& neigh_list, std::vector<unsigned>& neigh_offsets){}
void scatter_data_from_buffer_ptr_cuda(op_arg arg, halo_list iel, halo_list inl, char *buffer, 
                               std::vector<int>& neigh_list, std::vector<unsigned>& neigh_offsets){}