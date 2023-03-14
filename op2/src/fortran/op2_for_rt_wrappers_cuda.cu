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


#include <op_lib_core.h>
#include <op_lib_c.h>

#include <cuda.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <op_cuda_rt_support.h>

#include <vector>

static char *device_global_reductions = NULL;
static size_t device_global_reductions_size = 0;

static char *device_global_rws = NULL;
static size_t device_global_rws_size = 0;

static void *device_temp_storage = NULL;
static size_t device_temp_storage_size = 0;

static void *device_result = NULL;
static size_t device_result_size = 0;

static void prepareDeviceGblReductions(op_arg *args, int nargs, int max_threads) {
    size_t required_size = 0;

    for (int i = 0; i < nargs; ++i) {
        if (args[i].argtype != OP_ARG_GBL && args[i].argtype != OP_ARG_INFO) continue;
        if (args[i].argtype == OP_ARG_GBL &&
           (args[i].acc != OP_INC && args[i].acc != OP_MIN && args[i].acc != OP_MAX)) continue;

        required_size += args[i].size * max_threads * sizeof(char);
    }

    if (required_size > device_global_reductions_size) {
        printf("required gbl reduction bytes: %zu\n", required_size);
        if (device_global_reductions != NULL) cutilSafeCall(cudaFree(device_global_reductions));
        cutilSafeCall(cudaMalloc((void **) &device_global_reductions, required_size));
        device_global_reductions_size = required_size;
    }

    char *allocated = device_global_reductions;
    for (int i = 0; i < nargs; ++i) {
        if (args[i].argtype != OP_ARG_GBL && args[i].argtype != OP_ARG_INFO) continue;
        if (args[i].argtype == OP_ARG_GBL &&
           (args[i].acc != OP_INC && args[i].acc != OP_MIN && args[i].acc != OP_MAX)) continue;

        args[i].data_d = allocated;
        allocated += args[i].size * max_threads * sizeof(char);
    }
}

static void prepareDeviceGblRWs(op_arg *args, int nargs) {
    size_t required_size = 0;

    for (int i = 0; i < nargs; ++i) {
        if (args[i].argtype != OP_ARG_GBL) continue;
        if (args[i].acc != OP_READ && args[i].acc != OP_WRITE && args[i].acc != OP_RW) continue;

        required_size += args[i].size * sizeof(char);
    }

    if (required_size > device_global_rws_size) {
        printf("required gbl r/w bytes: %zu\n", required_size);
        if (device_global_rws != NULL) cutilSafeCall(cudaFree(device_global_rws));
        cutilSafeCall(cudaMalloc((void **) &device_global_rws, required_size));
        device_global_rws_size = required_size;
    }

    char *allocated = device_global_rws;
    for (int i = 0; i < nargs; ++i) {
        if (args[i].argtype != OP_ARG_GBL) continue;
        if (args[i].acc != OP_READ && args[i].acc != OP_WRITE && args[i].acc != OP_RW) continue;

        args[i].data_d = allocated;
        allocated += args[i].size * sizeof(char);

        if (args[i].acc == OP_WRITE) continue;

        cutilSafeCall(cudaMemcpyAsync(args[i].data_d, args[i].data,
                                      args[i].size * sizeof(char),
                                      cudaMemcpyHostToDevice));
    }
}

template<typename T, typename F, typename HF>
static void reduce_simple(F op, HF host_op, op_arg *arg, int nthreads, int max_threads) {
    if (arg->size * sizeof(char) > device_result_size) {
        cutilSafeCall(cudaMalloc((void **) &device_result, arg->size * sizeof(char)));
        device_result_size = arg->size * sizeof(char);
    }

    size_t required_temp_storage_size = 0;
    op(NULL, required_temp_storage_size, (T *) arg->data_d, (T *) device_result, nthreads, 0, false);

    if (required_temp_storage_size > device_temp_storage_size) {
        if (device_temp_storage != NULL) cutilSafeCall(cudaFree(device_temp_storage));
        cutilSafeCall(cudaMalloc((void **) &device_temp_storage, required_temp_storage_size));
        device_temp_storage_size = required_temp_storage_size;
    }

    for (int d = 0; d < arg->dim; ++d)
        op(device_temp_storage, device_temp_storage_size,
            (T *) arg->data_d + d * max_threads, (T *) device_result + d, nthreads, 0, false);

    std::vector<T> result(arg->dim);
    cutilSafeCall(cudaMemcpy(result.data(), device_result, arg->size * sizeof(char), cudaMemcpyDeviceToHost));

    for (int d = 0; d < arg->dim; ++d)
        host_op(((T *) arg->data) + d, result.data() + d);
}

template<typename T, typename F, typename HF>
static void reduce_info(F op, HF host_op, op_arg *args, int index,
        const std::vector<int> &payload_args, int nthreads, int max_threads) {
    size_t required_result_size = args[index].dim * sizeof(cub::KeyValuePair<int, T>) * sizeof(char);
    if (required_result_size > device_result_size) {
        cutilSafeCall(cudaMalloc((void **) &device_result, required_result_size));
        device_result_size = required_result_size;
    }

    size_t required_temp_storage_size = 0;
    op(NULL, required_temp_storage_size, (T *) args[index].data_d,
       (cub::KeyValuePair<int, T> *) device_result, nthreads, 0, false);

    if (required_temp_storage_size > device_temp_storage_size) {
        if (device_temp_storage != NULL) cutilSafeCall(cudaFree(device_temp_storage));
        cutilSafeCall(cudaMalloc((void **) &device_temp_storage, required_temp_storage_size));
        device_temp_storage_size = required_temp_storage_size;
    }

    for (int d = 0; d < args[index].dim; ++d)
        op(device_temp_storage, device_temp_storage_size, (T *) args[index].data_d + d * max_threads,
           (cub::KeyValuePair<int, T> *) device_result + d, nthreads, 0, false);

    std::vector<cub::KeyValuePair<int, T>> result(args[index].dim);
    cutilSafeCall(cudaMemcpy(result.data(), device_result, required_result_size, cudaMemcpyDeviceToHost));

    for (int d = 0; d < args[index].dim; ++d) {
        if (result[d].value == ((T *) args[index].data)[d])
            continue;

        host_op(((T *) args[index].data) + d, &(result[d].value));

        for (auto payload_index : payload_args) {
            size_t payload_elem_size = sizeof(char) * args[payload_index].size / args[payload_index].dim;

            char *payload_data = args[payload_index].data + d * payload_elem_size;
            char *payload_data_device = args[payload_index].data_d +
                (d * max_threads + result[d].key) * payload_elem_size;

            cutilSafeCall(cudaMemcpy(payload_data, payload_data_device,
                                     payload_elem_size, cudaMemcpyDeviceToHost));
        }
    }
}

template<typename T, typename F, typename F2, typename HF>
static void reduce_arg2(F op, F2 op2, HF host_op,
        op_arg *args, int index, int nargs, int nthreads, int max_threads) {
    std::vector<int> payload_args = {};
    for (int i = 0; i < nargs; ++i) {
        if (args[i].argtype == OP_ARG_INFO && args[i].acc == index)
            payload_args.push_back(i);
    }

    if (payload_args.size() == 0) {
        reduce_simple<T>(op, host_op, &args[index], nthreads, max_threads);
        return;
    }

    reduce_info<T>(op2, host_op, args, index, payload_args, nthreads, max_threads);
}

#define reduce_arg(T, op, op2, host_op) reduce_arg2<T>(cub::DeviceReduce::op<T *, T *>, \
                                                       cub::DeviceReduce::op2<T *, cub::KeyValuePair<int, T> *>, \
                                                       host_op<T>, \
                                                       args, i, nargs, nthreads, max_threads)

template<typename T> static void inc(T *a, T *b) { *a += *b; }
template<typename T> static void min(T *a, T *b) { *a = std::min(*a, *b); }
template<typename T> static void max(T *a, T *b) { *a = std::max(*a, *b); }

static void processDeviceGblReductions(op_arg *args, int nargs, int nthreads, int max_threads) {
    for (int i = 0; i < nargs; ++i) {
        if (args[i].argtype != OP_ARG_GBL) continue;

        if (args[i].acc == OP_INC) {
            if (strcmp(args[i].type, "double") == 0) reduce_arg(double, Sum, ArgMin, inc);
            else if (strcmp(args[i].type, "float") == 0) reduce_arg(float, Sum, ArgMin, inc);
            else if (strcmp(args[i].type, "int") == 0) reduce_arg(int, Sum, ArgMin, inc);
            else {
                fprintf(stderr, "Fatal: unknown type in reduction: %s\n", args[i].type);
                exit(1);
            }
        } else if (args[i].acc == OP_MIN) {
            if (strcmp(args[i].type, "double") == 0) reduce_arg(double, Min, ArgMin, min);
            else if (strcmp(args[i].type, "float") == 0) reduce_arg(float, Min, ArgMin, min);
            else if (strcmp(args[i].type, "int") == 0) reduce_arg(int, Min, ArgMin, min);
            else {
                fprintf(stderr, "Fatal: unknown type in reduction: %s\n", args[i].type);
                exit(1);
            }
        } else if (args[i].acc == OP_MAX) {
            if (strcmp(args[i].type, "double") == 0) reduce_arg(double, Max, ArgMax, max);
            else if (strcmp(args[i].type, "float") == 0) reduce_arg(float, Max, ArgMax, max);
            else if (strcmp(args[i].type, "int") == 0) reduce_arg(int, Max, ArgMax, max);
            else {
                fprintf(stderr, "Fatal: unknown type in reduction: %s\n", args[i].type);
                exit(1);
            }
        }
    }
}

static void processDeviceGblRWs(op_arg *args, int nargs) {
    for (int i = 0; i < nargs; ++i) {
        if (args[i].argtype != OP_ARG_GBL) continue;
        if (args[i].acc != OP_RW && args[i].acc != OP_WRITE) continue;

        cutilSafeCall(cudaMemcpy(args[i].data, args[i].data_d,
                                 args[i].size * sizeof(char),
                                 cudaMemcpyDeviceToHost));
    }
}

#ifdef __cplusplus
extern "C" {
#endif

void op_upload_dat(op_dat dat);
void op_download_dat(op_dat dat);

void op_put_dat(op_dat dat) {
  op_upload_dat(dat);
}

void op_get_dat(op_dat dat) {
  op_download_dat(dat);
}

void
op_get_dat_mpi (op_dat dat) {
  if (dat->data_d == NULL) return;
  op_get_dat(dat);
}

void
op_put_dat_mpi (op_dat dat) {
  if (dat->data_d == NULL) return;
  op_put_dat(dat);
}

void op_get_all_cuda(int nargs, op_arg *args) {
  for (int i = 0; i < nargs; i++) {
    if (args[i].argtype == OP_ARG_DAT && args[i].opt == 1) {
      op_get_dat_mpi(args[i].dat);
    }
  }
}

char *scratch = NULL;
long scratch_size = 0;
void prepareScratch(op_arg *args, int nargs, int nthreads) {
  long req_size = 0;
  for (int i = 0; i < nargs; i++) {
    if ((args[i].argtype == OP_ARG_GBL && (args[i].acc == OP_INC || args[i].acc == OP_MAX || args[i].acc == OP_MIN)) || args[i].argtype == OP_ARG_INFO)
      req_size += ((args[i].size-1)/8+1)*8*nthreads;
  }
  if (scratch_size < req_size) {
    if (!scratch) cutilSafeCall(cudaFree(scratch));
    cutilSafeCall(cudaMalloc((void**)&scratch, req_size*sizeof(char)));
    scratch_size = req_size;
  }
  req_size = 0;
  for (int i = 0; i < nargs; i++) {
    if ((args[i].argtype == OP_ARG_GBL && (args[i].acc == OP_INC || args[i].acc == OP_MAX || args[i].acc == OP_MIN)) || args[i].argtype == OP_ARG_INFO) {
      args[i].data_d = scratch + req_size;
      req_size += ((args[i].size-1)/8+1)*8*nthreads;
    }
  }
}

void prepareDeviceGbls(op_arg *args, int nargs, int max_threads) {
    prepareDeviceGblReductions(args, nargs, max_threads);
    prepareDeviceGblRWs(args, nargs);
}

void processDeviceGbls(op_arg *args, int nargs, int nthreads, int max_threads) {
    processDeviceGblReductions(args, nargs, nthreads, max_threads);
    processDeviceGblRWs(args, nargs);
}

#ifdef __cplusplus
}
#endif

