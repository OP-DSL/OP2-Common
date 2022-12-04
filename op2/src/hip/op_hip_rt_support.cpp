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
// This file implements the CUDA-specific run-time support functions
//

//
// header files
//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <op_hip_rt_support.h>
#include <op_lib_c.h>
#include <op_lib_core.h>
#include <op_rt_support.h>

// Small re-declaration to avoid using struct in the C version.
// This is due to the different way in which C and C++ see structs

typedef struct hipDeviceProp_t cudaDeviceProp_t;


// arrays for global constants and reductions

int OP_consts_bytes = 0, OP_reduct_bytes = 0;

char *OP_consts_h, *OP_consts_d, *OP_reduct_h, *OP_reduct_d;

//
// CUDA utility functions
//

#ifdef __cplusplus
extern "C" {
#endif

void __cudaSafeCall(hipError_t err, const char *file, const int line) {
  if (hipSuccess != err) {
    fprintf(stderr, "%s(%i) : cutilSafeCall() Runtime API error : %s.\n", file,
            line, hipGetErrorString(err));
    exit(-1);
  }
}

void __cutilCheckMsg(const char *errorMessage, const char *file,
                     const int line) {
  hipError_t err = hipGetLastError();
  if (hipSuccess != err) {
    fprintf(stderr, "%s(%i) : cutilCheckMsg() error : %s : %s.\n", file, line,
            errorMessage, hipGetErrorString(err));
    exit(-1);
  }
}

//
// routines to move arrays to/from GPU device
//

void op_mvHostToDevice(void **map, int size) {
  if (!OP_hybrid_gpu || size == 0)
    return;
  void *tmp;
  cutilSafeCall(hipMalloc(&tmp, size));
  cutilSafeCall(hipMemcpy(tmp, *map, size, hipMemcpyHostToDevice));
  cutilSafeCall(hipDeviceSynchronize());
  free(*map);
  *map = tmp;
}

void op_cpHostToDevice(void **data_d, void **data_h, int size) {
  if (!OP_hybrid_gpu)
    return;
  if (*data_d != NULL) cutilSafeCall(hipFree(*data_d));
  cutilSafeCall(hipMalloc(data_d, size));
  cutilSafeCall(hipMemcpy(*data_d, *data_h, size, hipMemcpyHostToDevice));
  cutilSafeCall(hipDeviceSynchronize());
}

op_plan *op_plan_get(char const *name, op_set set, int part_size, int nargs,
                     op_arg *args, int ninds, int *inds) {
  return op_plan_get_stage(name, set, part_size, nargs, args, ninds, inds,
                           OP_STAGE_ALL);
}

op_plan *op_plan_get_stage(char const *name, op_set set, int part_size,
                           int nargs, op_arg *args, int ninds, int *inds,
                           int staging) {
  return op_plan_get_stage_upload(name, set, part_size, nargs, args, ninds, inds,
                           staging, 1);
}

op_plan *op_plan_get_stage_upload(char const *name, op_set set, int part_size,
                           int nargs, op_arg *args, int ninds, int *inds,
                           int staging, int upload) {
  op_plan *plan =
      op_plan_core(name, set, part_size, nargs, args, ninds, inds, staging);
  if (!OP_hybrid_gpu || !upload)
    return plan;

  int set_size = set->size;
  for (int i = 0; i < nargs; i++) {
    if (args[i].idx != -1 && args[i].acc != OP_READ) {
      set_size += set->exec_size;
      break;
    }
  }

  if (plan->count == 1) {
    int *offsets = (int *)malloc((plan->ninds_staged + 1) * sizeof(int));
    offsets[0] = 0;
    for (int m = 0; m < plan->ninds_staged; m++) {
      int count = 0;
      for (int m2 = 0; m2 < nargs; m2++)
        if (plan->inds_staged[m2] == m)
          count++;
      offsets[m + 1] = offsets[m] + count;
    }
    op_mvHostToDevice((void **)&(plan->ind_map),
                      offsets[plan->ninds_staged] * set_size * sizeof(int));
    for (int m = 0; m < plan->ninds_staged; m++) {
      plan->ind_maps[m] = &plan->ind_map[set_size * offsets[m]];
    }
    free(offsets);

    int counter = 0;
    for (int m = 0; m < nargs; m++)
      if (plan->loc_maps[m] != NULL)
        counter++;
    op_mvHostToDevice((void **)&(plan->loc_map),
                      sizeof(short) * counter * set_size);
    counter = 0;
    for (int m = 0; m < nargs; m++)
      if (plan->loc_maps[m] != NULL) {
        plan->loc_maps[m] = &plan->loc_map[set_size * counter];
        counter++;
      }

    op_mvHostToDevice((void **)&(plan->ind_sizes),
                      sizeof(int) * plan->nblocks * plan->ninds_staged);
    op_mvHostToDevice((void **)&(plan->ind_offs),
                      sizeof(int) * plan->nblocks * plan->ninds_staged);
    op_mvHostToDevice((void **)&(plan->nthrcol), sizeof(int) * plan->nblocks);
    op_mvHostToDevice((void **)&(plan->thrcol), sizeof(int) * set_size);
    op_mvHostToDevice((void **)&(plan->col_reord), sizeof(int) * set_size);
    op_mvHostToDevice((void **)&(plan->offset), sizeof(int) * plan->nblocks);
    plan->offset_d = plan->offset;
    op_mvHostToDevice((void **)&(plan->nelems), sizeof(int) * plan->nblocks);
    plan->nelems_d = plan->nelems;
    op_mvHostToDevice((void **)&(plan->blkmap), sizeof(int) * plan->nblocks);
    plan->blkmap_d = plan->blkmap;
  }

  return plan;
}

void op_cuda_exit() {
  if (!OP_hybrid_gpu)
    return;
  op_dat_entry *item;
  TAILQ_FOREACH(item, &OP_dat_list, entries) {
    cutilSafeCall(hipFree((item->dat)->data_d));
  }

  for (int ip = 0; ip < OP_plan_index; ip++) {
    OP_plans[ip].ind_map = NULL;
    OP_plans[ip].loc_map = NULL;
    OP_plans[ip].ind_sizes = NULL;
    OP_plans[ip].ind_offs = NULL;
    OP_plans[ip].nthrcol = NULL;
    OP_plans[ip].thrcol = NULL;
    OP_plans[ip].col_reord = NULL;
    OP_plans[ip].offset = NULL;
    OP_plans[ip].nelems = NULL;
    OP_plans[ip].blkmap = NULL;
  }
  // hipDeviceReset ( );
}

//
// routines to resize constant/reduct arrays, if necessary
//

void reallocConstArrays(int consts_bytes) {
  if (consts_bytes > OP_consts_bytes) {
    if (OP_consts_bytes > 0) {
      free(OP_consts_h);
      cutilSafeCall(hipFree(OP_consts_d));
    }
    OP_consts_bytes = 4 * consts_bytes; // 4 is arbitrary, more than needed
    OP_consts_h = (char *)malloc(OP_consts_bytes);
    cutilSafeCall(hipMalloc((void **)&OP_consts_d, OP_consts_bytes));
  }
}

void reallocReductArrays(int reduct_bytes) {
  if (reduct_bytes > OP_reduct_bytes) {
    if (OP_reduct_bytes > 0) {
      free(OP_reduct_h);
      cutilSafeCall(hipFree(OP_reduct_d));
    }
    OP_reduct_bytes = 4 * reduct_bytes; // 4 is arbitrary, more than needed
    OP_reduct_h = (char *)malloc(OP_reduct_bytes);
    cutilSafeCall(hipMalloc((void **)&OP_reduct_d, OP_reduct_bytes));
  }
}

//
// routines to move constant/reduct arrays
//

void mvConstArraysToDevice(int consts_bytes) {
  cutilSafeCall(hipMemcpy(OP_consts_d, OP_consts_h, consts_bytes,
                           hipMemcpyHostToDevice));
  cutilSafeCall(hipDeviceSynchronize());
}

void mvConstArraysToHost(int consts_bytes) {
  cutilSafeCall(hipMemcpy(OP_consts_h, OP_consts_d, consts_bytes,
                           hipMemcpyDeviceToHost));
  cutilSafeCall(hipDeviceSynchronize());
}

void mvReductArraysToDevice(int reduct_bytes) {
  cutilSafeCall(hipMemcpy(OP_reduct_d, OP_reduct_h, reduct_bytes,
                           hipMemcpyHostToDevice));
  cutilSafeCall(hipDeviceSynchronize());
}

void mvReductArraysToHost(int reduct_bytes) {
  cutilSafeCall(hipMemcpy(OP_reduct_h, OP_reduct_d, reduct_bytes,
                           hipMemcpyDeviceToHost));
  cutilSafeCall(hipDeviceSynchronize());
}

//
// routine to fetch data from GPU to CPU (with transposing SoA to AoS if needed)
//

void op_cuda_get_data(op_dat dat) {
  if (!OP_hybrid_gpu)
    return;
  if (dat->dirty_hd == 2)
    dat->dirty_hd = 0;
  else
    return;
  // transpose data
  size_t set_size = dat->set->size + dat->set->exec_size + dat->set->nonexec_size;
  if (strstr(dat->type, ":soa") != NULL || (OP_auto_soa && dat->dim > 1)) {
    char *temp_data = (char *)malloc(dat->size * set_size * sizeof(char));
    cutilSafeCall(hipMemcpy(temp_data, dat->data_d, dat->size * set_size,
                             hipMemcpyDeviceToHost));
    cutilSafeCall(hipDeviceSynchronize());
    int element_size = dat->size / dat->dim;
    for (int i = 0; i < dat->dim; i++) {
      for (int j = 0; j < set_size; j++) {
        for (int c = 0; c < element_size; c++) {
          dat->data[dat->size * j + element_size * i + c] =
              temp_data[element_size * i * set_size + element_size * j +
                        c];
        }
      }
    }
    free(temp_data);
  } else {
    cutilSafeCall(hipMemcpy(dat->data, dat->data_d, dat->size * set_size,
                             hipMemcpyDeviceToHost));
    cutilSafeCall(hipDeviceSynchronize());
  }
}

void deviceSync() {
  cutilSafeCall(hipDeviceSynchronize());
}

#ifndef OPMPI

void cutilDeviceInit(int argc, char **argv) {
  (void)argc;
  (void)argv;
  int deviceCount;
  cutilSafeCall(hipGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    printf("cutil error: no devices supporting CUDA\n");
    exit(-1);
  }

  // Test we have access to a device
  float *test;
  hipError_t err = hipMalloc((void **)&test, sizeof(float));
  if (err != hipSuccess) {
    OP_hybrid_gpu = 0;
  } else {
    OP_hybrid_gpu = 1;
  }
  if (OP_hybrid_gpu) {
    hipFree(test);

    cutilSafeCall(hipDeviceSetCacheConfig(hipFuncCachePreferL1));

    int deviceId = -1;
    hipGetDevice(&deviceId);
    cudaDeviceProp_t deviceProp;
    cutilSafeCall(hipGetDeviceProperties(&deviceProp, deviceId));
    printf("\n Using CUDA device: %d %s\n", deviceId, deviceProp.name);
  } else {
    printf("\n Using CPU\n");
  }
}

void op_upload_dat(op_dat dat) {
  if (!OP_hybrid_gpu)
    return;
  size_t set_size = dat->set->size + dat->set->exec_size + dat->set->nonexec_size;
  if (strstr(dat->type, ":soa") != NULL || (OP_auto_soa && dat->dim > 1)) {
    char *temp_data = (char *)malloc(dat->size * set_size * sizeof(char));
    int element_size = dat->size / dat->dim;
    for (int i = 0; i < dat->dim; i++) {
      for (int j = 0; j < set_size; j++) {
        for (int c = 0; c < element_size; c++) {
          temp_data[element_size * i * set_size + element_size * j + c] =
              dat->data[dat->size * j + element_size * i + c];
        }
      }
    }
    cutilSafeCall(hipMemcpy(dat->data_d, temp_data, set_size * dat->size,
                             hipMemcpyHostToDevice));
    free(temp_data);
  } else {
    cutilSafeCall(hipMemcpy(dat->data_d, dat->data, set_size * dat->size,
                             hipMemcpyHostToDevice));
  }
}

void op_download_dat(op_dat dat) {
  if (!OP_hybrid_gpu)
    return;
  size_t set_size = dat->set->size + dat->set->exec_size + dat->set->nonexec_size;
  if (strstr(dat->type, ":soa") != NULL || (OP_auto_soa && dat->dim > 1)) {
    char *temp_data = (char *)malloc(dat->size * set_size * sizeof(char));
    cutilSafeCall(hipMemcpy(temp_data, dat->data_d, set_size * dat->size,
                             hipMemcpyDeviceToHost));
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
    cutilSafeCall(hipMemcpy(dat->data, dat->data_d, set_size * dat->size,
                             hipMemcpyDeviceToHost));
  }
}

int op_mpi_halo_exchanges(op_set set, int nargs, op_arg *args) {
  for (int n = 0; n < nargs; n++)
    if (args[n].opt && args[n].argtype == OP_ARG_DAT &&
        args[n].dat->dirty_hd == 2) {
      op_download_dat(args[n].dat);
      args[n].dat->dirty_hd = 0;
    }
  return set->size;
}

int op_mpi_halo_exchanges_grouped(op_set set, int nargs, op_arg *args, int device){
  (void)device;
  return device == 1 ? op_mpi_halo_exchanges(set, nargs, args) : op_mpi_halo_exchanges_cuda(set, nargs, args);
}

void op_mpi_set_dirtybit(int nargs, op_arg *args) {
  for (int n = 0; n < nargs; n++) {
    if ((args[n].opt == 1) && (args[n].argtype == OP_ARG_DAT) &&
        (args[n].acc == OP_INC || args[n].acc == OP_WRITE ||
         args[n].acc == OP_RW)) {
      args[n].dat->dirty_hd = 1;
    }
  }
}

void op_mpi_wait_all(int nargs, op_arg *args) {
  (void)nargs;
  (void)args;
}

void op_mpi_wait_all_grouped(int nargs, op_arg *args, int device) {
  (void)device;
  (void)nargs;
  (void)args;
}

void op_mpi_test_all(int nargs, op_arg *args) {
  (void)nargs;
  (void)args;
}

void op_mpi_test_all_grouped(int nargs, op_arg *args) {
  (void)nargs;
  (void)args;
}

int op_mpi_halo_exchanges_cuda(op_set set, int nargs, op_arg *args) {
  for (int n = 0; n < nargs; n++)
    if (args[n].opt && args[n].argtype == OP_ARG_DAT &&
        args[n].dat->dirty_hd == 1) {
      op_upload_dat(args[n].dat);
      args[n].dat->dirty_hd = 0;
    }
  return set->size;
}

void op_mpi_set_dirtybit_cuda(int nargs, op_arg *args) {
  for (int n = 0; n < nargs; n++) {
    if ((args[n].opt == 1) && (args[n].argtype == OP_ARG_DAT) &&
        (args[n].acc == OP_INC || args[n].acc == OP_WRITE ||
         args[n].acc == OP_RW)) {
      args[n].dat->dirty_hd = 2;
    }
  }
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

void op_partition_ptr(const char *lib_name, const char *lib_routine,
                      op_set prime_set, int *prime_map, double *coords) {
  (void)lib_name;
  (void)lib_routine;
  (void)prime_set;
  (void)prime_map;
  (void)coords;
}

void op_partition_reverse() {}

void op_compute_moment(double t, double *first, double *second) {
  *first = t;
  *second = t * t;
}
void op_compute_moment_across_times(double* times, int ntimes, bool ignore_zeros, double *first, double *second) {
  *first = 0.0;
  *second = 0.0f;
  int n = 0;
  for (int i=0; i<ntimes; i++) {
    if (ignore_zeros && (times[i] == 0.0f)) {
      continue;
    }
    *first += times[i];
    *second += times[i] * times[i];
    n++;
  }

  if (n != 0) {
    *first /= (double)n;
    *second /= (double)n;
  }
}

int op_is_root() { return 1; }
#endif

#ifdef __cplusplus
}
#endif
