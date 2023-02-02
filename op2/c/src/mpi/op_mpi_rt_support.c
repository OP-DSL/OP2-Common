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
#include <execinfo.h>
void op_backtrace1(op_dat dat) {
  int c, i;
  void *addresses[40];
  char **strings;

  c = backtrace(addresses, 40);
  strings = backtrace_symbols(addresses,c);
  printf("backtrace returned: %d", c);
  for(i = 0; i < c; i++) {
      // printf("%d: %X ", i, (int)addresses[i]);
      printf("testbt %s dat=%s\n", strings[i], dat->name); 
  }
}

void print_array_1(int* arr, int size, const char* name, int my_rank){
  for(int i = 0; i < size; i++){
    printf("array my_rank=%d name=%s size=%d value[%d]=%d\n", my_rank, name, size, i, arr[i]);
  }
}

double pack_time = 0.0, unpack_time = 0.0, halo_exch_time = 0.0;
double ca_c1,ca_c2,ca_c3,ca_c4,ca_t1,ca_t2,ca_t3,ca_t4;
#ifdef COMM_AVOID

halo_list imp_common_list;
halo_list exp_common_list;

void op_exchange_halo_chained(int nargs, op_arg *args, int exec_flag) {
  // printf("test1new exec my_rank\n");
  int my_rank;
  // MPI_Comm_rank(OP_MPI_WORLD, &my_rank);

  op_arg dirty_args[nargs];
  int ndirty_args = get_dirty_args(nargs, args, exec_flag, dirty_args, 1);
  if(ndirty_args < 1)
    return;

  imp_common_list = OP_merged_import_exec_nonexec_list[dirty_args[0].dat->set->index];  //assumption nargs > 0
  exp_common_list = OP_merged_export_exec_nonexec_list[dirty_args[0].dat->set->index];

  //-------first exchange exec elements related to this data array--------

  // sanity checks
  // if (compare_sets(imp_list->set, dat->set) == 0) {
  //   printf("Error: Import list and set mismatch\n");
  //   MPI_Abort(OP_MPI_WORLD, 2);
  // }
  // if (compare_sets(exp_list->set, dat->set) == 0) {
  //   printf("Error: Export list and set mismatch\n");
  //   MPI_Abort(OP_MPI_WORLD, 2);
  // }
  
  int set_elem_index = 0;
  int buf_index = 0;
  int buf_start = 0;
  int prev_size = 0;
  int arg_size = 0;

  grp_tag++;

  int exp_rank_count = exp_common_list->ranks_size / exp_common_list->num_levels;
  for (int r = 0; r < exp_rank_count; r++) {
    buf_start =  arg_size;

    for(int n = 0; n < ndirty_args; n++){

      buf_index = 0;
      op_arg* arg = &dirty_args[n];
      op_dat dat = arg->dat;
      int nhalos = get_nhalos(arg);
      halo_list exp_list = OP_merged_export_exec_nonexec_list[dat->set->index];
      int halo_index = 0;
      
      for(int l = 0; l < nhalos; l++){
        for(int l1 = 0; l1 < 2; l1++){ // 2 is for exec and nonexec levels
          if((l1 == 0 && dat->exec_dirtybits[l] == 1) || (l1 == 1 && dat->nonexec_dirtybits[l] == 1)){   
            int level_disp = exp_list->disps_by_level[halo_index];
            int rank_disp = exp_list->ranks_disps_by_level[halo_index];
            int disp_in_level = exp_list->disps[rank_disp + r];

            for (int i = 0; i < exp_list->sizes[exp_list->ranks_disps_by_level[halo_index] + r]; i++) {
              set_elem_index = exp_list->list[level_disp + disp_in_level + i];
              buf_index = prev_size + i * dat->size;
              
              printf("test1new exec my_rank=%d arg=%d dat=%s level=%d prev_size=%d buf_index=%d (i=%d size=%d) halo_index=%d exp_list=%p nhalos=%d\n", 
              my_rank, n, dat->name, l, prev_size / dat->size, arg_size / dat->size + buf_index / dat->size, i, exp_list->sizes[exp_list->ranks_disps_by_level[l] + r],
              halo_index, exp_list, nhalos);
              memcpy(&grp_send_buffer[arg_size + buf_index],
                  (void *)&dat->data[dat->size * (set_elem_index)], dat->size);
            }
            prev_size += exp_list->sizes[exp_list->ranks_disps_by_level[halo_index] + r] * dat->size;
          }
          if(is_halo_required_for_set(dat->set, l) == 1 && l1 == 0){
            halo_index++;
          }
          if(is_nonexec_halo_required(arg, nhalos, l) != 1){
              break;
          }
        }
        halo_index++;
      }
      arg_size += prev_size;
      prev_size = 0;
    }

    MPI_Isend(&grp_send_buffer[buf_start],
              (arg_size - buf_start), MPI_CHAR,
              exp_common_list->ranks[r], grp_tag, OP_MPI_WORLD,
              &grp_send_requests[r]);
    OP_mpi_tx_msg_count_chained++;

    if (OP_kern_max > 0)
      OP_kernels[OP_kern_curr].halo_data += arg_size - buf_start;

    // printf("rxtxexec merged my_rank=%d dat=%s r=%d sent=%d buf_start=%d prev_size=%d\n", my_rank, "test", 
    // exp_common_list->ranks[r], arg_size- buf_start, buf_start, arg_size);

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

    MPI_Irecv(&grp_recv_buffer[imp_disp],  //adjust disps_by_rank
              imp_size, MPI_CHAR,
              imp_common_list->ranks[i], grp_tag, OP_MPI_WORLD,
              &grp_recv_requests[i]);
    imp_disp += imp_size;

    // printf("rxtxexec merged my_rank=%d dat=%s r=%d recved=%d  imp_disp=%d\n", my_rank, "test", imp_common_list->ranks[i], imp_size_1, imp_disp_1);
    OP_mpi_rx_msg_count_chained++;
  }
}

#endif

void op_exchange_halo(op_arg *arg, int exec_flag) {
  // op_mpi_barrier();
  // op_timers_core(&ca_c1, &ca_t1);
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

  int my_rank;
  // MPI_Comm_rank(OP_MPI_WORLD, &my_rank);

  arg->sent = 0; // reset flag

  // need to exchange both direct and indirect data sets if they are dirty
  if ((arg->acc == OP_READ ||
       arg->acc == OP_RW /* good for debug || arg->acc == OP_INC*/) &&
// #ifdef COMM_AVOID
//       (dat->exec_dirtybits[0] == 1)) {
// #else
      (dat->dirtybit == 1)) {
// #endif
    //    printf("Exchanging Halo of data array %10s\n",dat->name);
    halo_list imp_exec_list = OP_import_exec_list[dat->set->index];
    halo_list exp_exec_list = OP_export_exec_list[dat->set->index];
   
    halo_list imp_nonexec_list = OP_import_nonexec_list[dat->set->index];
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
#ifdef COMM_AVOID
    if(dat->exec_dirtybits[0] == 1){
#endif
    for (int i = 0; i < exp_exec_list->ranks_size; i++) {
      for (int j = 0; j < exp_exec_list->sizes[i]; j++) {
        set_elem_index = exp_exec_list->list[exp_exec_list->disps[i] + j];
        memcpy(&((op_mpi_buffer)(dat->mpi_buffer))
                    ->buf_exec[exp_exec_list->disps[i] * dat->size +
                               j * dat->size],
               (void *)&dat->data[dat->size * (set_elem_index)], dat->size);
      }
      // int my_rank;
      // MPI_Comm_rank(OP_MPI_WORLD, &my_rank);
      // printf("export exec from %d to %d data %10s, number of elements of size %d | sending:\n ",
      //             my_rank, exp_exec_list->ranks[i],
      //             dat->name,exp_exec_list->sizes[i]);
      // double *b = (double*)&((op_mpi_buffer)(dat->mpi_buffer))
      //                ->buf_exec[exp_exec_list->disps[i] * dat->size];
      // for (int el = 0; el < (dat->size * exp_exec_list->sizes[i])/8; el++)
      //   printf("%g ", b[el]);
      // printf("\n");
      // op_backtrace1(dat);
      MPI_Isend(&((op_mpi_buffer)(dat->mpi_buffer))
                     ->buf_exec[exp_exec_list->disps[i] * dat->size],
                dat->size * exp_exec_list->sizes[i], MPI_CHAR,
                exp_exec_list->ranks[i], dat->index, OP_MPI_WORLD,
                &((op_mpi_buffer)(dat->mpi_buffer))
                     ->s_req[((op_mpi_buffer)(dat->mpi_buffer))->s_num_req++]);
      OP_mpi_tx_exec_msg_count++;
      OP_mpi_tx_exec_msg_count_org++;

      if (OP_kern_max > 0)
        OP_kernels[OP_kern_curr].halo_data += dat->size * exp_exec_list->sizes[i];

      // printf("rxtxexec org my_rank=%d dat=%s r=%d sent=%d\n", my_rank, dat->name, i, exp_exec_list->sizes[i]);
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
      OP_mpi_rx_exec_msg_count++;
      OP_mpi_rx_exec_msg_count_org++;
      // printf("rxtxexec org my_rank=%d dat=%s r=%d recevd=%d\n", my_rank, dat->name, i, imp_exec_list->sizes[i]);
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
      OP_mpi_tx_nonexec_msg_count++;
      OP_mpi_tx_nonexec_msg_count_org++;

      if (OP_kern_max > 0)
        OP_kernels[OP_kern_curr].halo_data += dat->size * exp_nonexec_list->sizes[i];
      // printf("rxtxnonexec org my_rank=%d dat=%s r=%d sent=%d\n", my_rank, dat->name, i, exp_nonexec_list->sizes[i]);
      
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
      OP_mpi_rx_nonexec_msg_count++;
      OP_mpi_rx_nonexec_msg_count_org++;
      // printf("rxtxnonexec org my_rank=%d dat=%s r=%d recevd=%d\n", my_rank, dat->name, i, imp_nonexec_list->sizes[i]);
    }
#ifdef COMM_AVOID
    }
#endif
    // clear dirty bit
#ifdef COMM_AVOID
    // dat->exec_dirtybits[0] = 0;
    // dat->nonexec_dirtybits[0] = 0;
    // printf("here0 dat=%s\n", dat->name);
    // print_array_1(dat->exec_dirtybits, dat->set->halo_info->max_nhalos, "execb", my_rank);
    // print_array_1(dat->nonexec_dirtybits, dat->set->halo_info->max_nhalos, "nonexecb", my_rank);
    unset_dirtybit(arg);
    // printf("here0 dat=%s\n", dat->name);
    // print_array_1(dat->exec_dirtybits, dat->set->halo_info->max_nhalos, "execa", my_rank);
    // print_array_1(dat->nonexec_dirtybits, dat->set->halo_info->max_nhalos, "nonexeca", my_rank);
    if(are_dirtybits_clear(arg) == 1){
      dat->dirtybit = 0;
      // printf("here clear===== dat=%s\n", dat->name);
    }
#else
    dat->dirtybit = 0;
#endif
    arg->sent = 1;
  }

  // op_mpi_barrier();
  // op_timers_core(&ca_c2, &ca_t2);
  // pack_time += ca_t2 - ca_t1;

  // op_timers_core(&ca_c3, &ca_t3);
}

#ifdef COMM_AVOID

void op_unpack_merged_single_dat_chained(int nargs, op_arg *args, int exec_flag){

  // printf("op_unpack_merged_single_dat_chained called dat=%s set=%s num_levels=%d nhalos=%d nhalos_index=%d max_nhalos=%d\n", 
  // arg->dat->name, arg->dat->set->name, imp_list->num_levels, nhalos, nhalos_index, max_nhalos);

  int rank_count = imp_common_list->ranks_size / imp_common_list->num_levels;
  int prev_exec_size = 0;
  for (int i = 0; i < rank_count; i++) {
    int imp_disp = 0;
    
    for(int n = 0; n < nargs; n++){
      op_arg* arg = &args[n];
      op_dat dat = arg->dat;

      int nhalos = get_nhalos(arg);
      int init = dat->set->size * dat->size;
      halo_list imp_list = OP_merged_import_exec_nonexec_list[dat->set->index];
      int halo_index = 0;

      for(int l = 0; l < nhalos; l++){
        for(int l1 = 0; l1 < 2; l1++){  // 2 is for exec and nonexec levels

          if((l1 == 0 && dat->exec_dirtybits[l] == 1) || (l1 == 1 && dat->nonexec_dirtybits[l] == 1)){
            memcpy(&(dat->data[init + (imp_list->disps_by_level[halo_index] + imp_list->disps[imp_list->ranks_disps_by_level[halo_index] + i]) * dat->size]), 
              &(grp_recv_buffer[prev_exec_size]),
                              dat->size * imp_list->sizes[imp_list->ranks_disps_by_level[halo_index] + i]);

            prev_exec_size += imp_list->sizes[imp_list->ranks_disps_by_level[halo_index] + i] * dat->size;
          } 
          

          // printf("op_unpack_merged_single_dat1 called dat=%s set=%s num_levels=%d nhalos=%d imp_disp=%d prev_exec_size=%d\n", 
          //       arg->dat->name, arg->dat->set->name, imp_list->num_levels, nhalos, imp_disp, prev_exec_size);
          if(is_halo_required_for_set(dat->set, l) == 1 && l1 == 0){
            halo_index++;
          }
          if(is_nonexec_halo_required(arg, nhalos, l) != 1){
              break;
          }
        }
        halo_index++;
      }
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

  int my_rank;
  // MPI_Comm_rank(OP_MPI_WORLD, &my_rank);

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
      OP_mpi_tx_nonexec_msg_count++;
      OP_mpi_tx_nonexec_msg_count_partial++;
      // printf("rxtxexec partial my_rank=%d dat=%s r=%d sent=%d\n", my_rank, dat->name, i, exp_nonexec_list->sizes[i]);
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
      OP_mpi_rx_nonexec_msg_count_partial++;
      // printf("rxtxexec partial my_rank=%d dat=%s r=%d receved=%d\n", my_rank, dat->name, i, imp_nonexec_list->sizes[i]);
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

void op_wait_all_chained(int nargs, op_arg *args, int device) {
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
  // double ca_c1,ca_c2,ca_t1,ca_t2;
  // op_timers_core(&ca_c1, &ca_t1);
#ifdef COMM_AVOID
  int my_rank;
  // op_rank(&my_rank);

  op_arg dirty_args[nargs];
  int ndirty_args = get_dirty_args(nargs, args, exec_flag, dirty_args, 1);
  if(ndirty_args < 1)
    return;

  MPI_Waitall(exp_common_list->ranks_size / exp_common_list->num_levels, 
                &grp_send_requests[0], MPI_STATUSES_IGNORE);
  MPI_Waitall(imp_common_list->ranks_size / imp_common_list->num_levels, 
                  &grp_recv_requests[0], MPI_STATUSES_IGNORE);

  op_unpack_merged_single_dat_chained(ndirty_args, dirty_args, exec_flag);
  for (int n = 0; n < ndirty_args; n++) {
      dirty_args[n].sent = 2; // set flag to indicate completed comm
      // printf("here0 dat=%s\n", dirty_args[n].dat->name);
      // print_array_1(dirty_args[n].dat->exec_dirtybits, dirty_args[n].dat->set->halo_info->max_nhalos, dirty_args[n].dat->name, my_rank);
      // print_array_1(dirty_args[n].dat->nonexec_dirtybits, dirty_args[n].dat->set->halo_info->max_nhalos, dirty_args[n].dat->name, my_rank);
      unset_dirtybit(&dirty_args[n]);
      // print_array_1(dirty_args[n].dat->exec_dirtybits, dirty_args[n].dat->set->halo_info->max_nhalos, dirty_args[n].dat->name, my_rank);
      // print_array_1(dirty_args[n].dat->nonexec_dirtybits, dirty_args[n].dat->set->halo_info->max_nhalos, dirty_args[n].dat->name, my_rank);
      if(are_dirtybits_clear(&dirty_args[n]) == 1){
        dirty_args[n].dat->dirtybit = 0;
        // printf("here1 clear dat=%s\n", dirty_args[n].dat->name);
      }
      
      dirty_args[n].dat->dirty_hd = device;
  }
#endif
  // op_timers_core(&ca_c2, &ca_t2);
  // if (OP_kern_max > 0)
  //   OP_kernels[OP_kern_curr].mpi_time += ca_t2 - ca_t1;
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
    arg->sent = 2; // set flag to indicate completed comm
    // op_mpi_barrier();
    // op_timers_core(&ca_c4, &ca_t4);
    // halo_exch_time += ca_t4 - ca_t3;

    if (arg->map != OP_ID && OP_map_partial_exchange[arg->map->index]) {
      int my_rank;
      // op_rank(&my_rank);
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