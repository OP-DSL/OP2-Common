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
 * of the OpenMP back-end
 */

#include <op_lib_c.h>
#include <op_rt_support.h>

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
  op_init_core(argc, argv, diags);
}

#ifdef __cplusplus
extern "C" {
#endif

void op_mpi_init(int argc, char **argv, int diags, int global, int local) {
  op_init_core(argc, argv, diags);
}

void op_mpi_init_soa(int argc, char **argv, int diags, int global, int local,
                     int soa) {
  OP_auto_soa = soa;
  op_mpi_init(argc, argv, diags, global, local);
}

#ifdef __cplusplus
}
#endif

op_dat op_decl_dat_char(op_set set, int dim, char const *type, int size,
                        char *data, char const *name) {
  return op_decl_dat_core(set, dim, type, size, data, name);
}

op_dat op_decl_dat_overlay(op_set set, op_dat dat) {
  return op_decl_dat_overlay_core(set, dat);
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
  return op_decl_dat_temp_core(set, dim, type, size, NULL, name);
}

int op_free_dat_temp_char(op_dat dat) { return op_free_dat_temp_core(dat); }

void op_upload_all() {}

void op_fetch_data_char(op_dat dat, char *usr_ptr) {
  // need to copy data into memory pointed to by usr_ptr
  memcpy((void *)usr_ptr, (void *)dat->data, dat->set->size * dat->size);
}

void op_fetch_data_idx_char(op_dat dat, char *usr_ptr, int low, int high) {
  if (low < 0 || high > dat->set->size - 1) {
    printf("op_fetch_data: Indices not within range of elements held in %s\n",
           dat->name);
    exit(2);
  }
  // need to copy data into memory pointed to by usr_ptr
  memcpy((void *)usr_ptr, (void *)&dat->data[low * dat->size],
         (high + 1) * dat->size);
}

/*
 * No specific action is required for constants in OpenMP
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

int op_get_size(op_set set) { return set->size; }

int op_get_global_set_offset(op_set set) { return 0; }

void op_printf(const char *format, ...) {
  va_list argptr;
  va_start(argptr, format);
  vprintf(format, argptr);
  va_end(argptr);
}

void op_print(const char *line) { printf("%s\n", line); }

void op_exit() {
  op_rt_exit();
  op_exit_core();
}

/*
 * Wrappers of core lib
 */

op_set op_decl_set(int size, char const *name) {
  return op_decl_set_core(size, name);
}

op_map op_decl_map(op_set from, op_set to, int dim, int *imap,
                   char const *name) {
  return op_decl_map_core(from, to, dim, imap, name);
}

op_arg op_arg_dat(op_dat dat, int idx, op_map map, int dim, char const *type,
                  op_access acc) {
  return op_arg_dat_core(dat, idx, map, dim, type, acc);
}

op_arg op_opt_arg_dat(int opt, op_dat dat, int idx, op_map map, int dim,
                      char const *type, op_access acc) {
  return op_opt_arg_dat_core(opt, dat, idx, map, dim, type, acc);
}

void op_timers(double *cpu, double *et) { op_timers_core(cpu, et); }

op_arg op_arg_gbl_char(char *data, int dim, const char *type, int size,
                       op_access acc) {
  return op_arg_gbl_core(1, data, dim, type, size, acc);
}

op_arg op_opt_arg_gbl_char(int opt, char *data, int dim, const char *type,
                           int size, op_access acc) {
  return op_arg_gbl_core(opt, data, dim, type, size, acc);
}

void op_timing_output() {
  op_timing_output_core();
  printf("Total plan time: %8.4f\n", OP_plan_time);
}

void op_timings_to_csv(const char *outputFileName) {
  FILE *outputFile = fopen(outputFileName, "w");
  if (outputFile == NULL) {
    printf("ERROR: Failed to open file for writing: '%s'\n", outputFileName);
  }
  else {
    fprintf(outputFile, "rank,thread,nranks,nthreads,count,total time,plan time,mpi time,GB used,GB total,kernel name\n");
  }

  if (outputFile != NULL) {
    for (int n = 0; n < OP_kern_max; n++) {
      if (OP_kernels[n].count > 0) {
        if (OP_kernels[n].ntimes == 1 && OP_kernels[n].times[0] == 0.0f &&
            OP_kernels[n].time != 0.0f) {
          // This library is being used by an OP2 translation made with the
          // older
          // translator with older timing logic. Adjust to new logic:
          OP_kernels[n].times[0] = OP_kernels[n].time;
        }

        double plan_time = OP_kernels[n].plan_time;
        double mpi_time = OP_kernels[n].mpi_time;
        for (int thr=0; thr<OP_kernels[n].ntimes; thr++) {
          double kern_time = OP_kernels[n].times[thr];
          if (thr > 0 && kern_time == 0.0) {
            continue;
          }
          if (thr==0)
            fprintf(outputFile, "%d,%d,%d,%d,%d,%f,%f,%f,%f,%f,%s\n", 0, thr, 1,
                    OP_kernels[n].ntimes, OP_kernels[n].count, kern_time,
                    plan_time, mpi_time, OP_kernels[n].transfer / 1e9f,
                    OP_kernels[n].transfer2 / 1e9f, OP_kernels[n].name);
          else
            fprintf(outputFile, "%d,%d,%d,%d,%d,%f,%f,%f,%f,%f,%s\n", 0, thr, 1,
                    OP_kernels[n].ntimes, OP_kernels[n].count, kern_time, 0.0f,
                    0.0f, 0.0f, 0.0f, OP_kernels[n].name);
        }
      }
    }

    fclose(outputFile);
  }
}

void op_print_dat_to_binfile(op_dat dat, const char *file_name) {
  op_print_dat_to_binfile_core(dat, file_name);
}

void op_print_dat_to_txtfile(op_dat dat, const char *file_name) {
  op_print_dat_to_txtfile_core(dat, file_name);
}

void op_upload_dat(op_dat dat) {}

void op_download_dat(op_dat dat) {}
