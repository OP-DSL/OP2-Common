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
 * This file implements the user-level OP2 functions for the case
 * of the mpi back-end
 */

#include <mpi.h>
#include <op_lib_c.h>
#include <op_lib_core.h>
#include <op_lib_mpi.h>
#include <op_mpi_core.h>
#include <op_rt_support.h>
#include <op_util.h>

//
// MPI Communicator for halo creation and exchange
//

MPI_Comm OP_MPI_WORLD;
MPI_Comm OP_MPI_GLOBAL;

/*
 * Routines called by user code and kernels
 * these wrappers are used by non-CUDA versions
 * op_lib.cu provides wrappers for CUDA version
 */

void op_init_soa(int argc, char **argv, int diags, int soa) {
  OP_auto_soa = soa;
  op_init(argc, argv, diags);
}
void op_init(int argc, char **argv, int diags) {
  int flag = 0;
  MPI_Initialized(&flag);
  if (!flag) {
    MPI_Init(&argc, &argv);
  }
  OP_MPI_WORLD = MPI_COMM_WORLD;
  OP_MPI_GLOBAL = MPI_COMM_WORLD;
  op_init_core(argc, argv, diags);
}

void op_mpi_init_soa(int argc, char **argv, int diags, MPI_Fint global,
                     MPI_Fint local, int soa) {
  OP_auto_soa = soa;
  op_mpi_init(argc, argv, diags, global, local);
}

void op_mpi_init(int argc, char **argv, int diags, MPI_Fint global,
                 MPI_Fint local) {
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
}

op_dat op_decl_dat_char(op_set set, int dim, char const *type, int size,
                        char *data, char const *name) {
  if (set == NULL)
    return NULL;
  /*char *d = (char *)malloc((size_t)set->size * (size_t)dim * (size_t)size);
  if (d == NULL && set->size>0) {
    printf(" op_decl_dat_char error -- error allocating memory to dat\n");
    exit(-1);
  }
  memcpy(d, data, sizeof(char) * set->size * dim * size);
  op_dat out_dat = op_decl_dat_core(set, dim, type, size, d, name);*/
  op_dat out_dat = op_decl_dat_core(set, dim, type, size, data, name);

  op_dat_entry *item;
  op_dat_entry *tmp_item;
  for (item = TAILQ_FIRST(&OP_dat_list); item != NULL; item = tmp_item) {
    tmp_item = TAILQ_NEXT(item, entries);
    if (item->dat == out_dat) {
      item->orig_ptr = data;
      break;
    }
  }
  // free(data); // free user allocated data block ?

  // printf(" op2 pointer for dat %s = %lu  ", name, (unsigned long)d);
  out_dat->user_managed = 0;
  return out_dat;
}

op_dat op_decl_dat_overlay(op_set set, op_dat dat) {
  op_dat overlay_dat = op_decl_dat_overlay_core(set, dat);

  int halo_size = OP_import_exec_list[set->index]->size +
                  OP_import_nonexec_list[set->index]->size;

  op_mpi_buffer mpi_buf = (op_mpi_buffer)xmalloc(sizeof(op_mpi_buffer_core));

  halo_list exec_e_list = OP_export_exec_list[set->index];
  halo_list nonexec_e_list = OP_export_nonexec_list[set->index];

  mpi_buf->buf_exec = (char *)xmalloc((size_t)(exec_e_list->size) * (size_t)overlay_dat->size);

  size_t import_extra = OP_partial_exchange ? set_import_buffer_size[set->index] : 0;
  mpi_buf->buf_nonexec = (char *)xmalloc(((size_t)(nonexec_e_list->size) + import_extra)
                                         * (size_t)overlay_dat->size);

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

  overlay_dat->mpi_buffer = mpi_buf;

  return overlay_dat;
}

op_dat op_decl_dat_overlay_ptr(op_set set, char *dat) {
  op_dat_entry *item;
  op_dat_entry *tmp_item;
  op_dat item_dat = NULL;

  for (item = TAILQ_FIRST(&OP_dat_list); item != NULL; item = tmp_item) {
    tmp_item = TAILQ_NEXT(item, entries);
    if (item->orig_ptr == dat) {
      item_dat = item->dat;
      break;
    }
  }

  if (item_dat == NULL) {
    printf("ERROR: op_dat not found for dat with %p pointer\n", dat);
  }

  return op_decl_dat_overlay(set, item_dat);
}

op_dat op_decl_dat_temp_char(op_set set, int dim, char const *type, int size,
                             char const *name) {
  op_dat dat = op_decl_dat_temp_core(set, dim, type, size, NULL, name);

  // create empty data block to assign to this temporary dat (including the
  // halos)
  size_t set_size = (size_t)set->size + (size_t)OP_import_exec_list[set->index]->size
                                      + (size_t)OP_import_nonexec_list[set->index]->size;

  // need to allocate mpi_buffers for this new temp_dat
  op_mpi_buffer mpi_buf = (op_mpi_buffer)xmalloc(sizeof(op_mpi_buffer_core));

  halo_list exec_e_list = OP_export_exec_list[dat->set->index];
  halo_list nonexec_e_list = OP_export_nonexec_list[dat->set->index];

  mpi_buf->buf_exec = (char *)xmalloc((size_t)(exec_e_list->size) * (size_t)dat->size);

  size_t import_extra = OP_partial_exchange ? set_import_buffer_size[set->index] : 0;
  mpi_buf->buf_nonexec = (char *)xmalloc(((size_t)(nonexec_e_list->size) + import_extra) * (size_t)dat->size);

  halo_list exec_i_list = OP_import_exec_list[dat->set->index];
  halo_list nonexec_i_list = OP_import_nonexec_list[dat->set->index];

  mpi_buf->s_req = (MPI_Request *)xmalloc(
      sizeof(MPI_Request) *
      (exec_e_list->ranks_size + nonexec_e_list->ranks_size));
  mpi_buf->r_req = (MPI_Request *)xmalloc(
      sizeof(MPI_Request) *
      (exec_i_list->ranks_size + nonexec_i_list->ranks_size));

  mpi_buf->s_num_req = 0;
  mpi_buf->r_num_req = 0;
  dat->mpi_buffer = mpi_buf;

  return dat;
}

int op_free_dat_temp_char(op_dat dat) {
  // need to free mpi_buffers used in this op_dat
  free(((op_mpi_buffer)(dat->mpi_buffer))->buf_exec);
  free(((op_mpi_buffer)(dat->mpi_buffer))->buf_nonexec);
  free(((op_mpi_buffer)(dat->mpi_buffer))->s_req);
  free(((op_mpi_buffer)(dat->mpi_buffer))->r_req);
  free(dat->mpi_buffer);
  return op_free_dat_temp_core(dat);
}

void op_upload_all() {}

void op_fetch_data_char(op_dat dat, char *usr_ptr) {
  // rearrange data backe to original order in mpi
  op_dat temp = op_mpi_get_data(dat);

  // copy data into usr_ptr
  memcpy((void *)usr_ptr, (void *)temp->data, temp->set->size * temp->size);

  free(temp->data);
  free(temp->set);
  free(temp);
}

void op_fetch_data_idx_char(op_dat dat, char *usr_ptr, int low, int high) {
  // rearrange data back to original order in mpi
  op_dat temp = op_mpi_get_data(dat);

  // do allgather on temp->data and copy it to memory block pointed to by
  // use_ptr
  fetch_data_hdf5(temp, usr_ptr, low, high);

  free(temp->data);
  free(temp->set);
  free(temp);
}

op_dat op_fetch_data_file_char(op_dat dat) {
  // rearrange data backe to original order in mpi
  return op_mpi_get_data(dat);
}

/*
 * No specific action is required for constants in MPI
 */

void op_decl_const_char(int dim, char const *type, int typeSize, char *data,
                        char const *name) {
  (void)dim;
  (void)type;
  (void)typeSize;
  (void)data;
  (void)name;
}

op_plan *op_plan_get(char const *name, op_set set, int part_size, int nargs,
                     op_arg *args, int ninds, int *inds) {
  return op_plan_get_stage(name, set, part_size, nargs, args, ninds, inds,
                           OP_STAGE_ALL);
}

op_plan *op_plan_get_stage_upload(char const *name, op_set set, int part_size,
                           int nargs, op_arg *args, int ninds, int *inds,
                           int staging, int upload) {
  return op_plan_core(name, set, part_size, nargs, args, ninds, inds, staging);
}

op_plan *op_plan_get_stage(char const *name, op_set set, int part_size,
                           int nargs, op_arg *args, int ninds, int *inds,
                           int staging) {
  return op_plan_core(name, set, part_size, nargs, args, ninds, inds, staging);
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

void op_exit() {

  op_mpi_exit();
  op_rt_exit();
  op_exit_core();

  int flag = 0;
  MPI_Finalized(&flag);
  if (!flag)
    MPI_Finalize();
}

void op_rank(int *rank) { MPI_Comm_rank(OP_MPI_WORLD, rank); }

/*
 * Wrappers of core lib
 */

op_set op_decl_set(int size, char const *name) {
  return op_decl_set_core(size, name);
}

op_map op_decl_map(op_set from, op_set to, int dim, int *imap,
                   char const *name) {

  //  int *m = (int *)malloc(from->size * dim * sizeof(int));
  //  memcpy(m, imap, from->size * dim * sizeof(int));

  op_map out_map = op_decl_map_core(from, to, dim, imap, name);
  out_map->user_managed = 0;
  return out_map;
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

void op_timers(double *cpu, double *et) {
  MPI_Barrier(OP_MPI_WORLD);
  op_timers_core(cpu, et);
}

void op_timing_output() {
  double max_plan_time = 0.0;
  MPI_Reduce(&OP_plan_time, &max_plan_time, 1, MPI_DOUBLE, MPI_MAX, 0,
             OP_MPI_WORLD);
  op_timing_output_core();
  if (op_is_root())
    printf("Total plan time: %8.4f\n", OP_plan_time);
  mpi_timing_output();
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
      op_mpi_barrier();
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
              if (thr==0)
                fprintf(outputFile, "%d,%d,%d,%d,%d,%f,%f,%f,%f,%f,%s\n", p,
                        thr, comm_size, OP_kernels[n].ntimes,
                        OP_kernels[n].count, kern_time, plan_times[p],
                        mpi_times[p], transfers[p] / 1e9f, transfers2[p] / 1e9f,
                        OP_kernels[n].name);
              else
                fprintf(outputFile, "%d,%d,%d,%d,%d,%f,%f,%f,%f,%f,%s\n", p,
                        thr, comm_size, OP_kernels[n].ntimes,
                        OP_kernels[n].count, kern_time, 0.0f, 0.0f, 0.0f, 0.0f,
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
  // rearrange data backe to original order in mpi
  op_dat temp = op_mpi_get_data(dat);
  print_dat_to_binfile_mpi(temp, file_name);

  free(temp->data);
  free(temp->set);
  free(temp);
}

void op_print_dat_to_txtfile(op_dat dat, const char *file_name) {
  // rearrange data backe to original order in mpi
  op_dat temp = op_mpi_get_data(dat);
  print_dat_to_txtfile_mpi(temp, file_name);

  free(temp->data);
  free(temp->set);
  free(temp);
}

void op_debug_arg(int n, op_arg arg) {
  op_dat dat;

  dat = arg.dat;

  int my_rank;
  op_rank(&my_rank);

  if (arg.argtype == OP_ARG_DAT) {
    printf("NJH %i debug %s\n", my_rank, dat->name);
    printf("NJH %i debug %p\n", my_rank,
           ((op_mpi_buffer)(dat->mpi_buffer))->buf_nonexec);

    if (n == 3 && (strcmp(dat->name, "dist") == 0)) {
      printf("NJH %i trying free here...\n", my_rank);
      free(((op_mpi_buffer)(dat->mpi_buffer))->buf_nonexec);
      printf("NJH %i succeeded free here...\n", my_rank);
    };
  }
}
