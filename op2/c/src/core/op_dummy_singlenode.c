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
 * This file implements dummy MPI function calls for non-MPI backends
 */

#include "op_lib_core.h"

int op_mpi_halo_exchanges(op_set set, int nargs, op_arg *args) {
  (void)nargs;
  (void)args;
  return set->size;
}

void op_mpi_set_dirtybit(int nargs, op_arg *args) {
  (void)nargs;
  (void)args;
}

void op_mpi_wait_all(int nargs, op_arg *args) {
  (void)nargs;
  (void)args;
}

int op_mpi_halo_exchanges_cuda(op_set set, int nargs, op_arg *args) {
  (void)nargs;
  (void)args;
  return set->size;
}

void op_mpi_set_dirtybit_cuda(int nargs, op_arg *args) {
  (void)nargs;
  (void)args;
}

void op_mpi_wait_all_cuda(int nargs, op_arg *args) {
  (void)nargs;
  (void)args;
}

void op_mpi_reset_halos(int nargs, op_arg *args) {
  (void)nargs;
  (void)args;
}

void op_mpi_barrier() {}

void *op_mpi_perf_time(const char *name, double time) {
  (void)name;
  (void)time;
  return (void *)name;
}

#ifdef COMM_PERF
void op_mpi_perf_comms(void *k_i, int nargs, op_arg *args) {
  (void)k_i;
  (void)nargs;
  (void)args;
}
#endif

void op_mpi_reduce_combined(op_arg *args, int nargs) {
  (void)args;
  (void)nargs;
}

void op_mpi_reduce_float(op_arg *args, float *data) {
  (void)args;
  (void)data;
}

void op_mpi_reduce_double(op_arg *args, double *data) {
  (void)args;
  (void)data;
}

void op_mpi_reduce_int(op_arg *args, int *data) {
  (void)args;
  (void)data;
}

void op_mpi_reduce_bool(op_arg *args, bool *data) {
  (void)args;
  (void)data;
}

void op_partition(const char *lib_name, const char *lib_routine,
                  op_set prime_set, op_map prime_map, op_dat coords) {
  (void)lib_name;
  (void)lib_routine;
  (void)prime_set;
  (void)prime_map;
  (void)coords;
}

void op_renumber(op_map base) { (void)base; }

void op_compute_moment(double t, double *first, double *second) {
  *first = t;
  *second = t * t;
}

void op_partition_reverse() {}

int getSetSizeFromOpArg(op_arg *arg) {
  return arg->opt ? arg->dat->set->size : 0;
}

int op_is_root() { return 1; }

int getHybridGPU() { return OP_hybrid_gpu; }

typedef struct {
} op_export_core;

typedef op_export_core *op_export_handle;

typedef struct {
} op_import_core;

typedef op_import_core *op_import_handle;

void op_theta_init(op_export_handle handle, int *bc_id, double *dtheta_exp,
                   double *dtheta_imp, double *alpha) {

  exit(1);
}

void op_inc_theta(op_export_handle handle, int *bc_id, double *dtheta_exp,
                  double *dtheta_imp) {

  exit(1);
}

op_import_handle op_import_init_size(int nprocs, int *proclist, op_dat mark) {
  exit(1);
  return NULL;
}

op_import_handle op_import_init(op_export_handle exp_handle, op_dat coords,
                                op_dat mark) {
  exit(1);
  return NULL;
}

op_export_handle op_export_init(int nprocs, int *proclist, op_map cellsToNodes,
                                op_set sp_nodes, op_dat coords, op_dat mark) {

  exit(1);
  return NULL;
}

void op_export_data(op_export_handle handle, op_dat dat) { exit(1); }

void op_import_data(op_import_handle handle, op_dat dat) { exit(1); }
