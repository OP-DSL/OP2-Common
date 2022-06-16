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

double pack_time = 0.0, unpack_time = 0.0, halo_exch_time = 0.0;
double ca_c1,ca_c2,ca_c3,ca_c4,ca_t1,ca_t2,ca_t3,ca_t4;
#ifdef COMM_AVOID

int get_nonexec_start(op_arg *arg){
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

int get_nonexec_end(op_arg *arg){
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

halo_list imp_common_list;
halo_list exp_common_list;

void op_exchange_halo_chained(int nargs, op_arg *args, int exec_flag) {

  // printf("op_exchange_halo_chained >>>>>>>>>\n");
  int my_rank;
  MPI_Comm_rank(OP_MPI_WORLD, &my_rank);

  op_arg dirty_args[nargs];
  int ndirty_args = get_dirty_args(nargs, args, exec_flag, dirty_args, 1);

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
  // printf("op_exchange_halo_merged exchanged <<<<<<< dat=%s set=%s levels=%d datlevels=%d nonexec_start=%d nonexec_end=%d\n",
  //  arg->dat->name, arg->dat->set->name, arg->nhalos, dat->halo_info->max_nhalos, nonexec_start, nonexec_end);
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
      // if(is_arg_valid(arg, exec_flag) == 0)
      //   continue;

      op_dat dat = arg->dat;

      int nhalos = get_nhalos(arg);
      // int nonexec_start = get_nonexec_start(arg);
      // int nonexec_end = get_nonexec_end(arg);

      
      halo_list exp_list = OP_merged_export_exec_nonexec_list[dat->set->index];

      int halo_index = 0;
      int level_disp_in_rank = 0;
      for(int l = 0; l < nhalos; l++){
        for(int l1 = 0; l1 < 2; l1++){ // 2 is for exec and nonexec levels   
          int level_disp = exp_list->disps_by_level[halo_index];
          int rank_disp = exp_list->ranks_disps_by_level[halo_index];
          int disp_in_level = exp_list->disps[rank_disp + r];
          // printf("test1new exec my_rank=%d arg=%d level=%d halo_index=%d\n", my_rank, n, l, halo_index);

          // int level_disp_in_rank = exp_list->level_disps[r * exp_list->num_levels + halo_index];

          for (int i = 0; i < exp_list->sizes[exp_list->ranks_disps_by_level[halo_index] + r]; i++) {
            set_elem_index = exp_list->list[level_disp + disp_in_level + i];
            buf_index = prev_size + i * dat->size;
            memcpy(&grp_send_buffer[arg_size + buf_index],
                (void *)&dat->data[dat->size * (set_elem_index)], dat->size);
            // printf("test1new exec my_rank=%d arg=%d level=%d prev_size=%d buf_index=%d ( disp1=%d i=%d size=%d) halo_index=%d exp_list=%p\n", 
            // my_rank, n, l, prev_size / dat->size, arg_size / dat->size + buf_index / dat->size, level_disp_in_rank / dat->size, i, exp_list->sizes[exp_list->ranks_disps_by_level[l] + r],
            // halo_index, exp_list);
          }
          prev_size += exp_list->sizes[exp_list->ranks_disps_by_level[halo_index] + r] * dat->size;
          level_disp_in_rank += exp_list->sizes[exp_list->ranks_disps_by_level[halo_index] + r] * dat->size;
          if(is_halo_required_for_set(dat->set, l) == 1 && l1 == 0){
            halo_index++;
          }

          if(is_nonexec_halo_required(arg, nhalos, l) != 1){
              break;
          }
        }
        // if(is_halo_required_for_set(dat->set, l) != 1){
          halo_index++;
        // }
        
      }


      arg_size += prev_size;
      prev_size = 0;
    }

    MPI_Isend(&grp_send_buffer[buf_start],
              (arg_size - buf_start), MPI_CHAR,
              exp_common_list->ranks[r], grp_tag, OP_MPI_WORLD,
              &grp_send_requests[r]);
    OP_mpi_tx_exec_msg_count++;
    OP_mpi_tx_exec_msg_count_merged++;

    // printf("rxtxexec merged my_rank=%d dat=%s r=%d sent=%d buf_start=%d prev_size=%d\n", my_rank, "test", 
    // exp_common_list->ranks[r], arg_size_1 - buf_start_1, buf_start_1, arg_size_1);

  }

  int rank_count = imp_common_list->ranks_size / imp_common_list->num_levels;
  int imp_disp = 0;

  for (int i = 0; i < rank_count; i++) {
    int imp_size = 0;
    
    for(int n = 0; n < nargs; n++){
      op_arg* arg = &args[n];
      op_dat dat = arg->dat;
      int nhalos = get_nhalos(arg);
      int nonexec_start = get_nonexec_start(arg);
      int nonexec_end = get_nonexec_end(arg);
      halo_list imp_list = OP_merged_import_exec_nonexec_list[dat->set->index];
      
      int halo_index = 0;
      for(int l1 = 0; l1 < nhalos; l1++){
        for(int l2 = 0; l2 < 2; l2++){ // 2 is for exec and nonexec levels   
          imp_size += imp_list->level_sizes[i * imp_list->num_levels + halo_index] * arg->dat->size;
          if(is_halo_required_for_set(dat->set, l1) == 1 && l2 == 0){
            halo_index++;
          }
          if(is_nonexec_halo_required(arg, nhalos, l1) != 1){
              break;
          }
        }
        // if(is_halo_required_for_set(dat->set, l1) != 1){
          halo_index++;
        // }
        
        // imp_size += imp_list->level_sizes[i * imp_list->num_levels + l1] * arg->dat->size;
      }
      // for(int l1 = nonexec_start; l1 < nonexec_end; l1++){
      //   imp_size += imp_list->level_sizes[i * imp_list->num_levels + l1] * arg->dat->size;
      // }
    }

    MPI_Irecv(&grp_recv_buffer[imp_disp],  //adjust disps_by_rank
              imp_size, MPI_CHAR,
              imp_common_list->ranks[i], grp_tag, OP_MPI_WORLD,
              &grp_recv_requests[i]);
    imp_disp += imp_size;

    // printf("rxtxexec merged my_rank=%d dat=%s r=%d recved=%d  imp_disp=%d\n", my_rank, "test", imp_common_list->ranks[i], imp_size_1, imp_disp_1);
    OP_mpi_rx_exec_msg_count++;
    OP_mpi_rx_exec_msg_count_merged++;
  }
  
}



void op_exchange_halo_merged(op_arg *arg, int exec_flag) {

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
  // For a directly accessed op_dat do not do halo exchanges if not executing
  // over
  // redundant compute block
  if (exec_flag == 0 && arg->idx == -1)
    return;
  
  // int my_rank;
  // MPI_Comm_rank(OP_MPI_WORLD, &my_rank);

  int nhalos = get_nhalos(arg);
  int nonexec_start = get_nonexec_start(arg);
  int nonexec_end = get_nonexec_end(arg);


  arg->sent = 0; // reset flag

  // need to exchange both direct and indirect data sets if they are dirty
  if ((arg->acc == OP_READ ||
       arg->acc == OP_RW /* good for debug || arg->acc == OP_INC*/) &&
      (dat->dirtybit == 1)) {
    //    printf("Exchanging Halo of data array %10s\n",dat->name);
    halo_list imp_list = OP_merged_import_exec_nonexec_list[dat->set->index];
    halo_list exp_list = OP_merged_export_exec_nonexec_list[dat->set->index];

    //-------first exchange exec elements related to this data array--------

    // sanity checks
    if (compare_sets(imp_list->set, dat->set) == 0) {
      printf("Error: Import list and set mismatch\n");
      MPI_Abort(OP_MPI_WORLD, 2);
    }
    if (compare_sets(exp_list->set, dat->set) == 0) {
      printf("Error: Export list and set mismatch\n");
      MPI_Abort(OP_MPI_WORLD, 2);
    }
    // printf("op_exchange_halo_merged exchanged <<<<<<< dat=%s set=%s levels=%d datlevels=%d nonexec_start=%d nonexec_end=%d\n",
    //  arg->dat->name, arg->dat->set->name, arg->nhalos, dat->halo_info->max_nhalos, nonexec_start, nonexec_end);
    int set_elem_index = 0;
    int buf_index = 0;
    int buf_start = 0;

    int exp_rank_count = exp_list->ranks_size / exp_list->num_levels;
    for (int r = 0; r < exp_rank_count; r++) {
      buf_start =  buf_index;
      for(int l = 0; l < nhalos; l++){
        int level_disp = exp_list->disps_by_level[l];
        int rank_disp = exp_list->ranks_disps_by_level[l];
        int disp_in_level = exp_list->disps[rank_disp + r];

        for (int i = 0; i < exp_list->sizes[rank_disp + r]; i++) {
          set_elem_index = exp_list->list[level_disp + disp_in_level + i];
          memcpy(&((op_mpi_buffer)(dat->mpi_buffer))
                    ->buf_merged[buf_index * dat->size],
               (void *)&dat->data[dat->size * (set_elem_index)], dat->size);

          buf_index++;
        }
      }

      for(int l = nonexec_start; l < nonexec_end; l++){
        int level_disp = exp_list->disps_by_level[l];
        int rank_disp = exp_list->ranks_disps_by_level[l];
        int disp_in_level = exp_list->disps[rank_disp + r];

        for (int i = 0; i < exp_list->sizes[rank_disp + r]; i++) {
          set_elem_index = exp_list->list[level_disp + disp_in_level + i];
          memcpy(&((op_mpi_buffer)(dat->mpi_buffer))
                    ->buf_merged[buf_index * dat->size],
               (void *)&dat->data[dat->size * (set_elem_index)], dat->size);

          buf_index++;
        }
      }
      // printf("rxtxexec merged my_rank=%d dat=%s r=%d sent=%d buf_index=%d buf_start=%d\n", my_rank, dat->name, r, buf_index - buf_start, buf_index, buf_start);
      
      MPI_Isend(&((op_mpi_buffer)(dat->mpi_buffer))
                     ->buf_merged[buf_start * dat->size],
                dat->size * (buf_index - buf_start), MPI_CHAR,
                exp_list->ranks[r], dat->index, OP_MPI_WORLD,
                &((op_mpi_buffer)(dat->mpi_buffer))
                     ->s_req[((op_mpi_buffer)(dat->mpi_buffer))->s_num_req++]);
      OP_mpi_tx_exec_msg_count++;
      OP_mpi_tx_exec_msg_count_merged++;
    }

    int init = 0; //dat->set->size * dat->size;
    int rank_count = imp_list->ranks_size / imp_list->num_levels;
    for (int i = 0; i < rank_count; i++) {
      int imp_size = imp_list->sizes_upto_level_by_rank[(nhalos - 1) * rank_count + i] + 
      imp_list->sizes_upto_level_by_rank[(nonexec_end - 1) * rank_count + i] - imp_list->sizes_upto_level_by_rank[(nonexec_start - 1) * rank_count + i];

      MPI_Irecv(&(dat->aug_data[init + imp_list->disps_by_rank[i] * dat->size]),
                dat->size * imp_size, MPI_CHAR,
                imp_list->ranks[i], dat->index, OP_MPI_WORLD,
                &((op_mpi_buffer)(dat->mpi_buffer))
                     ->r_req[((op_mpi_buffer)(dat->mpi_buffer))->r_num_req++]);
      
      // printf("rxtxexec merged my_rank=%d dat=%s r=%d recved=%d\n", my_rank, dat->name, i, imp_exec_list->sizes_by_rank[i]);
      OP_mpi_rx_exec_msg_count++;
      OP_mpi_rx_exec_msg_count_merged++;
    }


    // clear dirty bit
    dat->dirtybit = 0;
    arg->sent = 1;
    // arg->unpack_method = OP_UNPACK_MERGED_SINGLE_DAT;
  }
  // op_mpi_barrier();
  // op_timers_core(&ca_c2, &ca_t2);
  // pack_time += ca_t2 - ca_t1;

  // op_timers_core(&ca_c3, &ca_t3);
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

  // int my_rank;
  // MPI_Comm_rank(OP_MPI_WORLD, &my_rank);

  arg->sent = 0; // reset flag

  // need to exchange both direct and indirect data sets if they are dirty
  if ((arg->acc == OP_READ ||
       arg->acc == OP_RW /* good for debug || arg->acc == OP_INC*/) &&
      (dat->dirtybit == 1)) {
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
      OP_mpi_tx_nonexec_msg_count++;
      OP_mpi_tx_nonexec_msg_count_org++;
      // printf("rxtxnonexec org my_rank=%d dat=%s r=%d sent=%d\n", my_rank, dat->name, i, exp_nonexec_list->sizes[i]);
      
    }

// #ifdef COMM_AVOID
//     int nonexec_init = (dat->set->size + dat->set->exec_sizes[dat->set->halo_info->nhalos_count - 1]) * dat->size;
// #else
    int nonexec_init = (dat->set->size + imp_exec_list->size) * dat->size;
// #endif
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
    // clear dirty bit
    dat->dirtybit = 0;
    arg->sent = 1;
  }

  // op_mpi_barrier();
  // op_timers_core(&ca_c2, &ca_t2);
  // pack_time += ca_t2 - ca_t1;

  // op_timers_core(&ca_c3, &ca_t3);
}

#ifdef COMM_AVOID

void op_exchange_halo_chained_1(op_arg *arg, int exec_flag) {
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

  int nhalos = arg->nhalos;
  int nhalos_index = arg->nhalos_index;

  // printf("op_exchange_halo_chained dat %s nhalos=%d nhalos_index=%d\n", dat->name, nhalos, nhalos_index);
  // need to exchange both direct and indirect data sets if they are dirty
  if ((arg->acc == OP_READ ||
       arg->acc == OP_RW /* good for debug || arg->acc == OP_INC*/) &&
      (dat->dirtybit == 1)) {


    int my_rank;
    MPI_Comm_rank(OP_MPI_WORLD, &my_rank);
    op_set set = arg->dat->set;
    // printf("op_exchange_halo_chained exchanged <<<<<<< my_rank=%d dat=%s set=%s size=%d core=%d(0=%d) exec=%d(0=%d) non=%d(0=%d) max_halo=%d halo_count=%d arg(nhalos=%d index=%d)\n", 
    // my_rank, set->name, arg->dat->name, 
    // set->size, set->core_sizes[nhalos_index], set->core_sizes[0],
    // set->exec_sizes[nhalos_index], set->exec_sizes[0], set->nonexec_sizes[nhalos_index], set->nonexec_sizes[0], set->halo_info->max_nhalos, set->halo_info->nhalos_count,
    // arg->nhalos, arg->nhalos_index);

    halo_list imp_exec_list = OP_merged_import_exec_list[dat->set->index];
    halo_list imp_nonexec_list = OP_merged_import_nonexec_list[dat->set->index];

    halo_list exp_exec_list = OP_merged_export_exec_list[dat->set->index];
    halo_list exp_nonexec_list = OP_merged_export_nonexec_list[dat->set->index];

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
      for(int l = 0; l < nhalos; l++){
        for (int i = 0; i < exp_exec_list->sizes[exp_exec_list->ranks_disps_by_level[l] + r]; i++) {
          int level_disp = exp_exec_list->disps_by_level[l];
          int disp_in_level = exp_exec_list->disps[exp_exec_list->ranks_disps_by_level[l] + r];
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
      OP_mpi_tx_exec_msg_count++;
      OP_mpi_tx_exec_msg_count_chained++;
      // printf("rxtxexec chained my_rank=%d dat=%s r=%d sent=%d buf_index=%d buf_start=%d\n", my_rank, dat->name, r, buf_index - buf_start, buf_index, buf_start);
        
    }

    for (int i = 0; i < imp_exec_list->ranks_size / imp_exec_list->num_levels; i++) {

      MPI_Irecv(&(dat->aug_data[imp_exec_list->disps_by_rank[i] * dat->size]),
                dat->size * imp_exec_list->sizes_by_rank[i], MPI_CHAR,
                imp_exec_list->ranks[i], dat->index, OP_MPI_WORLD,
                &((op_mpi_buffer)(dat->mpi_buffer))
                     ->r_req[((op_mpi_buffer)(dat->mpi_buffer))->r_num_req++]);
      OP_mpi_rx_exec_msg_count++;
      OP_mpi_rx_exec_msg_count_chained++;
      // printf("rxtxexec chained my_rank=%d dat=%s r=%d recved=%d\n", my_rank, dat->name, i, imp_exec_list->sizes_by_rank[i]);
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

 

    set_elem_index = 0;
    buf_index = 0;
    buf_start = 0;
    for (int r = 0; r < exp_nonexec_list->ranks_size / exp_nonexec_list->num_levels; r++) {
      buf_start =  buf_index;
      for(int l = 0; l <= nhalos_index; l++){
        for (int i = 0; i < exp_nonexec_list->sizes[exp_nonexec_list->ranks_disps_by_level[l] + r]; i++) {
          int level_disp = exp_nonexec_list->disps_by_level[l];
          int disp_in_level = exp_nonexec_list->disps[exp_nonexec_list->ranks_disps_by_level[l] + r];
          set_elem_index = exp_nonexec_list->list[level_disp + disp_in_level + i];

          memcpy(&((op_mpi_buffer)(dat->mpi_buffer))
                    ->buf_nonexec[buf_index * dat->size],
               (void *)&dat->data[dat->size * (set_elem_index)], dat->size);
          buf_index++;
        }
      }

      MPI_Isend(&((op_mpi_buffer)(dat->mpi_buffer))
                     ->buf_nonexec[buf_start * dat->size],
                dat->size * (buf_index - buf_start), MPI_CHAR,
                exp_nonexec_list->ranks[r], dat->index, OP_MPI_WORLD,
                &((op_mpi_buffer)(dat->mpi_buffer))
                     ->s_req[((op_mpi_buffer)(dat->mpi_buffer))->s_num_req++]);
      OP_mpi_tx_nonexec_msg_count++;
      OP_mpi_tx_nonexec_msg_count_chained++;
      // printf("rxtxnonexec chained my_rank=%d dat=%s r=%d sent=%d buf_index=%d buf_start=%d\n", my_rank, dat->name, r, buf_index - buf_start, buf_index, buf_start);
        
    }

    int nonexec_init = 0;
    // for(int l = 0; l < 1; l++){
    //   nonexec_init += OP_aug_import_nonexec_lists[l][dat->set->index]->size;
    // }
    // nonexec_init += (dat->set->size + imp_exec_list->size);
    nonexec_init += (imp_exec_list->size);
    nonexec_init *= dat->size;
    // int nonexec_init = (dat->set->size + imp_exec_list->size) * dat->size;

     for (int i = 0; i < imp_nonexec_list->ranks_size / imp_nonexec_list->num_levels; i++) {
      MPI_Irecv(&(dat->aug_data[nonexec_init + imp_nonexec_list->disps_by_rank[i] * dat->size]),
                dat->size * imp_nonexec_list->sizes_by_rank[i], MPI_CHAR,
                imp_nonexec_list->ranks[i], dat->index, OP_MPI_WORLD,
                &((op_mpi_buffer)(dat->mpi_buffer))
                     ->r_req[((op_mpi_buffer)(dat->mpi_buffer))->r_num_req++]);
      OP_mpi_rx_nonexec_msg_count++;
      OP_mpi_rx_nonexec_msg_count_chained++;
      // printf("rxtxexec chained my_rank=%d dat=%s r=%d recved=%d\n", my_rank, dat->name, i, imp_nonexec_list->sizes_by_rank[i]);
    }

    // clear dirty bit
    dat->dirtybit = 0;
    arg->sent = 1;
  }
}

void op_unpack_merged_single_dat_chained(int nargs, op_arg *args, int exec_flag){
  // printf("op_unpack_merged_single_dat_chained\n"); 

  // printf("op_unpack_merged_single_dat_chained called dat=%s set=%s num_levels=%d nhalos=%d nhalos_index=%d max_nhalos=%d\n", 
  // arg->dat->name, arg->dat->set->name, imp_list->num_levels, nhalos, nhalos_index, max_nhalos);
  
  op_arg dirty_args[nargs];
  int ndirty_args = get_dirty_args(nargs, args, exec_flag, dirty_args, 1);

  int rank_count = imp_common_list->ranks_size / imp_common_list->num_levels;
  int prev_exec_size = 0;
  for (int i = 0; i < rank_count; i++) {
    int imp_disp = 0;
    
    for(int n = 0; n < ndirty_args; n++){
      op_arg* arg = &dirty_args[n];
      op_dat dat = arg->dat;
      // if(is_arg_valid(arg, exec_flag) == 0)
      //   continue;

      int nhalos = get_nhalos(arg);
      int nonexec_start = get_nonexec_start(arg);
      int nonexec_end = get_nonexec_end(arg);
      int init = dat->set->size * dat->size;
      halo_list imp_list = OP_merged_import_exec_nonexec_list[dat->set->index];
      // imp_disp += imp_list->disps_by_rank[i] * dat->size;
      
      int halo_index = 0;
      for(int l = 0; l < nhalos; l++){
        for(int l1 = 0; l1 < 2; l1++){  // 2 is for exec and nonexec levels   
          memcpy(&(dat->data[init + (imp_list->disps_by_level[halo_index] + imp_list->disps[imp_list->ranks_disps_by_level[halo_index] + i]) * dat->size]), 
                &(grp_recv_buffer[prev_exec_size]),
                                dat->size * imp_list->sizes[imp_list->ranks_disps_by_level[halo_index] + i]);

          prev_exec_size += imp_list->sizes[imp_list->ranks_disps_by_level[halo_index] + i] * dat->size;

          // printf("op_unpack_merged_single_dat1 called dat=%s set=%s num_levels=%d nhalos=%d imp_disp=%d prev_exec_size=%d\n", 
          //       arg->dat->name, arg->dat->set->name, imp_list->num_levels, nhalos, imp_disp, prev_exec_size);
          if(is_halo_required_for_set(dat->set, l) == 1 && l1 == 0){
            halo_index++;
          }

          if(is_nonexec_halo_required(arg, nhalos, l) != 1){
              break;
          }
        }
        // if(is_halo_required_for_set(dat->set, l) != 1){
          halo_index++;
        // }
      }

      // for(int l = nonexec_start; l < nonexec_end; l++){

      //   memcpy(&(dat->data[init + (imp_list->disps_by_level[l] + imp_list->disps[imp_list->ranks_disps_by_level[l] + i]) * dat->size]), 
      //         &(grp_recv_buffer[prev_exec_size]),
      //                         dat->size * imp_list->sizes[imp_list->ranks_disps_by_level[l] + i]);

      //   prev_exec_size += imp_list->sizes[imp_list->ranks_disps_by_level[l] + i] * dat->size;
      //   // printf("op_unpack_merged_single_dat2 called dat=%s set=%s num_levels=%d nhalos=%d imp_disp=%d prev_exec_size=%d nonexec_start=%d nonexec_end=%d\n", 
      //   //       arg->dat->name, arg->dat->set->name, imp_list->num_levels, nhalos, imp_disp, prev_exec_size, nonexec_start, nonexec_end);
      // }

    }
  }
}

void op_unpack_merged_single_dat(op_arg *arg){

  op_dat dat = arg->dat;
  int nhalos = get_nhalos(arg);
  int nonexec_start = get_nonexec_start(arg);
  int nonexec_end = get_nonexec_end(arg);

  halo_list imp_list = OP_merged_import_exec_nonexec_list[dat->set->index];
  int init = dat->set->size * dat->size;
  // printf("op_unpack_merged_single_dat called dat=%s set=%s num_levels=%d nhalos=%d nhalos_index=%d max_nhalos=%d\n", 
  // arg->dat->name, arg->dat->set->name, imp_list->num_levels, nhalos, nhalos_index, max_nhalos);
  
  int rank_count = imp_list->ranks_size / imp_list->num_levels;
  for (int i = 0; i < rank_count; i++) {
    int prev_size = 0;
    int imp_disp_by_rank = imp_list->disps_by_rank[i] * dat->size;
    for(int l = 0; l < nhalos; l++){
      
      memcpy(&(dat->data[init + (imp_list->disps_by_level[l] + imp_list->disps[imp_list->ranks_disps_by_level[l] + i]) * dat->size]), 
            &(dat->aug_data[imp_disp_by_rank + prev_size]),
                            dat->size * imp_list->sizes[imp_list->ranks_disps_by_level[l] + i]);

      prev_size += imp_list->sizes[imp_list->ranks_disps_by_level[l] + i] * dat->size;

      // printf("op_unpack_merged_single_daca_t1 called dat=%s set=%s num_levels=%d nhalos=%d nhalos_index=%d max_nhalos=%d imp_size=%d prev_exec_size=%d\n", 
      //       arg->dat->name, arg->dat->set->name, imp_list->num_levels, nhalos, nhalos_index, max_nhalos, imp_size, prev_exec_size);
    }


    for(int l = nonexec_start; l < nonexec_end; l++){

      memcpy(&(dat->data[init + (imp_list->disps_by_level[l] + imp_list->disps[imp_list->ranks_disps_by_level[l] + i]) * dat->size]), 
            &(dat->aug_data[imp_disp_by_rank + prev_size]),
                            dat->size * imp_list->sizes[imp_list->ranks_disps_by_level[l] + i]);

      prev_size += imp_list->sizes[imp_list->ranks_disps_by_level[l] + i] * dat->size;
      // printf("op_unpack_merged_single_daca_t2 called dat=%s set=%s num_levels=%d nhalos=%d nhalos_index=%d max_nhalos=%d imp_size=%d prev_exec_size=%d nonexec_start=%d nonexec_end=%d\n", 
      //       arg->dat->name, arg->dat->set->name, imp_list->num_levels, nhalos, nhalos_index, max_nhalos, imp_size, prev_exec_size, nonexec_start, nonexec_end);
    }
  }
}


void op_unpack_exec(op_arg *arg){

  op_dat dat = arg->dat;
  halo_list imp_exec_list = OP_merged_import_exec_list[dat->set->index];
  int init = dat->set->size * dat->size;
  int nhalos = (arg->nhalos > 0) ? arg->nhalos : 
              ((dat->halo_info->nhalos_count > 1) ? dat->halo_info->max_nhalos : -1);
  printf("op_unpack_exec called dat=%s set=%s nhalos=%d\n", arg->dat->name, arg->dat->set->name, nhalos);
  
  for (int i = 0; i < imp_exec_list->ranks_size / imp_exec_list->num_levels; i++) {
    int prev_exec_size = 0;
    for(int l = 0; l < nhalos; l++){
      memcpy(&(dat->data[init + (imp_exec_list->disps_by_level[l] + imp_exec_list->disps[imp_exec_list->ranks_disps_by_level[l] + i]) * dat->size]), 
            &(dat->aug_data[imp_exec_list->disps_by_rank[i] * dat->size + prev_exec_size * dat->size]),
                            dat->size * imp_exec_list->sizes[imp_exec_list->ranks_disps_by_level[l] + i]);

      prev_exec_size += imp_exec_list->sizes[imp_exec_list->ranks_disps_by_level[l] + i];
    }
  }
}

void op_unpack_nonexec(op_arg *arg){
 
  op_dat dat = arg->dat;
  halo_list imp_exec_list = OP_merged_import_exec_list[dat->set->index];
  halo_list imp_nonexec_list = OP_merged_import_nonexec_list[dat->set->index];

  int init = (dat->set->size + imp_exec_list->size ) * dat->size;
  int nhalos_index = (arg->nhalos > 0) ? arg->nhalos_index : 
              ((dat->halo_info->nhalos_count > 1) ? dat->halo_info->nhalos_indices[dat->halo_info->max_nhalos] : -1);

  printf("op_unpack_nonexec called dat=%s set=%s nhalos_index=%d\n", arg->dat->name, arg->dat->set->name, nhalos_index);

  for (int i = 0; i < imp_nonexec_list->ranks_size / imp_nonexec_list->num_levels; i++) {
    int prev_nonexec_size = 0;
    for(int l = 0; l <= nhalos_index; l++){ // this has to be changed to dat's levels
      memcpy(&(dat->data[init + (imp_nonexec_list->disps_by_level[l] + imp_nonexec_list->disps[imp_nonexec_list->ranks_disps_by_level[l] + i]) * dat->size]), 
            &(dat->aug_data[(imp_exec_list->size + imp_nonexec_list->disps_by_rank[i] + prev_nonexec_size) * dat->size]),
                            dat->size * imp_nonexec_list->sizes[imp_nonexec_list->ranks_disps_by_level[l] + i]);

      prev_nonexec_size += imp_nonexec_list->sizes[imp_nonexec_list->ranks_disps_by_level[l] + i];
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
    MPI_Comm_rank(OP_MPI_WORLD, &my_rank);

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

void op_mpi_wait_all_chained(int nargs, op_arg *args, int device) {
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
  MPI_Waitall(exp_common_list->ranks_size / exp_common_list->num_levels, 
                &grp_send_requests[0], MPI_STATUSES_IGNORE);
  MPI_Waitall(imp_common_list->ranks_size / imp_common_list->num_levels, 
                  &grp_recv_requests[0], MPI_STATUSES_IGNORE);

 
  for (int n = 0; n < nargs; n++) {
    if (args[n].opt && args[n].argtype == OP_ARG_DAT && args[n].dat->dirtybit == 1 && (args[n].acc == OP_READ || args[n].acc == OP_RW)) {
      if (args[n].idx == -1 && exec_flag == 0) continue;
      // printf("op_mpi_wait_all_chained n=%d dat=%s\n", n, args[n].dat->name);
      args[n].sent = 2; // set flag to indicate completed comm
      args[n].dat->dirtybit = 0;
      args[n].dat->dirty_hd = device;
    }
  }

  op_unpack_merged_single_dat_chained(nargs, args, exec_flag);
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
    #ifdef COMM_AVOID

    // op_timers_core(&ca_c1, &ca_t1);
    // op_unpack_merged_single_dat(arg);

    // op_mpi_barrier();
    // op_timers_core(&ca_c2, &ca_t2);
    // unpack_time += ca_t2 - ca_t1;
    // if(arg->unpack_method == OP_UNPACK_MERGED_SINGLE_DAT){
    //   op_unpack_merged_single_dat(arg);
    //   // op_unpack_exec_new(arg);
    //   // op_unpack_nonexec_new(arg);
    // }else if(arg->nhalos > 0 || arg->dat->halo_info->nhalos_count > 1){
    //   op_unpack_exec(arg);
    //   op_unpack_nonexec(arg);
    // }
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