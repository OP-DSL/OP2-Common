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

int op_mpi_halo_exchanges(op_set set, int nargs, op_arg *args)
{
  (void)nargs;
  (void)args;
  return set->size;
}

void op_mpi_set_dirtybit(int nargs, op_arg *args)
{
  (void)nargs;
  (void)args;
}

void op_mpi_wait_all(int nargs, op_arg *args)
{
  (void)nargs;
  (void)args;
}

void op_mpi_global_reduction(int nargs, op_arg *args)
{
  (void)nargs;
  (void)args;
}

void op_mpi_reset_halos(int nargs, op_arg *args)
{
  (void)nargs;
  (void)args;
}

void op_mpi_barrier()
{
}

int op_mpi_perf_time(const char* name, double time)
{
  (void)name;
  (void)time;
  return 0;
}

#ifdef COMM_PERF
void op_mpi_perf_comms(int k_i, int nargs, op_arg *args)
{
  (void)k_i;
  (void)nargs;
  (void)args;
}
#endif

void op_mpi_reduce_float(op_arg* args, float* data)
{
  (void)args;
  (void)data;
}

void op_mpi_reduce_double(op_arg* args, double* data)
{
  (void)args;
  (void)data;
}

void op_mpi_reduce_int(op_arg* args, int* data)
{
  (void)args;
  (void)data;
}

void op_partition(const char* lib_name, const char* lib_routine,
  op_set prime_set, op_map prime_map, op_dat coords ) {
  (void)lib_name;
  (void)lib_routine;
  (void)prime_set;
  (void)prime_map;
  (void)coords;
}

void op_partition_reverse() {
}

