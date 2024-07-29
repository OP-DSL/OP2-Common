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

// TODO: Make this waste less space?
static constexpr size_t align_size = 8 * sizeof(char);
static constexpr size_t align(size_t x) { return (x + (align_size) - 1) & ~(align_size - 1); }

static constexpr cudaStream_t cuda_stream = 0;

struct device_buffer {
    void *data = nullptr;
    size_t size = 0;

    void ensure_capacity(size_t requested_size) {
        if (requested_size <= size) return;

        if (data != nullptr) cutilSafeCall(cudaFreeAsync(data, cuda_stream));
        cutilSafeCall(cudaMallocAsync((void **) &data, requested_size, cuda_stream));
        size = requested_size;
    }

    device_buffer() = default;
    device_buffer(const device_buffer&) = delete;

    // Causes errors, or maybe exposes API errors from other calls?
    // ~device_buffer() { cutilSafeCall(cudaFree(data)); }
};

static device_buffer device_globals;
static device_buffer device_temp_storage;
static device_buffer device_result;

static bool gbl_inc_atomic = false;

static bool needs_per_thread_storage(const op_arg& arg) {
    if (arg.argtype == OP_ARG_INFO) return true;
    if (arg.argtype != OP_ARG_GBL)  return false;

    if (arg.acc == OP_INC && gbl_inc_atomic) return false;

    return arg.acc == OP_MIN ||
           arg.acc == OP_MAX ||
           arg.acc == OP_INC ||
           arg.acc == OP_WORK;
}

static bool needs_device_storage(const op_arg& arg) {
    if (arg.argtype == OP_ARG_INFO) return true;
    return arg.argtype == OP_ARG_GBL && !(arg.acc == OP_READ && arg.dim == 1);
}

static bool opt_disabled(const op_arg& arg, const op_arg *args) {
    if (arg.argtype == OP_ARG_INFO)
        return args[arg.acc].opt == 0;

    return arg.opt == 0;
}

template<typename T, typename F, typename HF>
static void reduce_simple(F op, HF host_op, op_arg *arg, int nelems, int max_threads) {
    device_result.ensure_capacity(arg->size * sizeof(char));

    size_t required_temp_storage_size = 0;
    op(NULL, required_temp_storage_size, (T *) arg->data_d, (T *) device_result.data, nelems);

    device_temp_storage.ensure_capacity(required_temp_storage_size);

    for (int d = 0; d < arg->dim; ++d)
        op(device_temp_storage.data, device_temp_storage.size,
            (T *) arg->data_d + d * max_threads, (T *) device_result.data + d, nelems);

    std::vector<T> result(arg->dim);
    cutilSafeCall(cudaMemcpyAsync(result.data(), device_result.data, arg->size * sizeof(char), cudaMemcpyDeviceToHost, cuda_stream));
    cutilSafeCall(cudaStreamSynchronize(cuda_stream));

    for (int d = 0; d < arg->dim; ++d)
        host_op(((T *) arg->data) + d, result.data() + d);
}

template<typename T, typename F, typename HF>
static void reduce_info(F op, HF host_op, op_arg *args, int index,
        const std::vector<int> &payload_args, int nelems, int max_threads) {
    size_t required_result_size = args[index].dim * sizeof(cub::KeyValuePair<int, T>) * sizeof(char);
    device_result.ensure_capacity(required_result_size);

    size_t required_temp_storage_size = 0;
    op(NULL, required_temp_storage_size, (T *) args[index].data_d,
       (cub::KeyValuePair<int, T> *) device_result.data, nelems);

    device_temp_storage.ensure_capacity(required_temp_storage_size);

    for (int d = 0; d < args[index].dim; ++d)
        op(device_temp_storage.data, device_temp_storage.size, (T *) args[index].data_d + d * max_threads,
           (cub::KeyValuePair<int, T> *) device_result.data + d, nelems);

    std::vector<cub::KeyValuePair<int, T>> result(args[index].dim);
    cutilSafeCall(cudaMemcpyAsync(result.data(), device_result.data, required_result_size, cudaMemcpyDeviceToHost, cuda_stream));
    cutilSafeCall(cudaStreamSynchronize(cuda_stream));

    for (int d = 0; d < args[index].dim; ++d) {
        if (result[d].value == ((T *) args[index].data)[d])
            continue;

        host_op(((T *) args[index].data) + d, &(result[d].value));

        for (auto payload_index : payload_args) {
            size_t payload_elem_size = sizeof(char) * args[payload_index].size / args[payload_index].dim;

            char *payload_data = args[payload_index].data + d * payload_elem_size;
            char *payload_data_device = args[payload_index].data_d +
                (d * max_threads + result[d].key) * payload_elem_size;

            cutilSafeCall(cudaMemcpyAsync(payload_data, payload_data_device,
                                          payload_elem_size, cudaMemcpyDeviceToHost, cuda_stream));
        }
    }
}

template<typename T, typename F, typename F2, typename HF>
static void reduce_arg2(F op, F2 op2, HF host_op,
        op_arg *args, int index, int nargs, int nelems, int max_threads) {
    std::vector<int> payload_args = {};
    for (int i = 0; i < nargs; ++i) {
        if (args[i].argtype == OP_ARG_INFO && args[i].acc == index)
            payload_args.push_back(i);
    }

    if (payload_args.size() == 0) {
        reduce_simple<T>(op, host_op, &args[index], nelems, max_threads);
        return;
    }

    reduce_info<T>(op2, host_op, args, index, payload_args, nelems, max_threads);
}

template<typename T> static void inc(T *a, T *b) { *a += *b; }
template<typename T> static void min(T *a, T *b) { *a = std::min(*a, *b); }
template<typename T> static void max(T *a, T *b) { *a = std::max(*a, *b); }

#define cub_reduction_wrap(op, out_type) \
    template<typename T> static cudaError_t op(void *d_temp_storage, size_t &temp_storage_bytes, \
                                                T *d_in, out_type *d_out, int num_items) { \
        return cub::DeviceReduce::op(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, cuda_stream); \
    }

cub_reduction_wrap(Sum, T)
cub_reduction_wrap(Min, T)
cub_reduction_wrap(Max, T)

template<typename T>
using KeyValuePair = cub::KeyValuePair<int, T>;

cub_reduction_wrap(ArgMin, KeyValuePair<T>)
cub_reduction_wrap(ArgMax, KeyValuePair<T>)

#define reduce_arg(T, op, op2, host_op) reduce_arg2<T>(op<T>, op2<T>, host_op<T>, \
                                                       args, i, nargs, nelems, max_threads)

static bool processDeviceGblReductions(op_arg *args, int nargs, int nelems, int max_threads) {
    bool needs_sync = false;

    for (int i = 0; i < nargs; ++i) {
        if (opt_disabled(args[i], args)) continue;
        if (args[i].argtype != OP_ARG_GBL) continue;

        if (gbl_inc_atomic && args[i].acc == OP_INC) {
            cutilSafeCall(cudaMemcpyAsync(args[i].data, args[i].data_d,
                          args[i].size * sizeof(char),
                          cudaMemcpyDeviceToHost, cuda_stream));

            needs_sync = true;
            continue;
        }

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

        needs_sync = true;
    }

    return needs_sync;
}

static bool processDeviceGblRWs(op_arg *args, int nargs) {
    bool needs_sync = false;
    for (int i = 0; i < nargs; ++i) {
        if (opt_disabled(args[i], args)) continue;
        if (args[i].argtype != OP_ARG_GBL) continue;
        if (args[i].acc != OP_RW && args[i].acc != OP_WRITE) continue;

        cutilSafeCall(cudaMemcpyAsync(args[i].data, args[i].data_d,
                                      args[i].size * sizeof(char),
                                      cudaMemcpyDeviceToHost, cuda_stream));

        needs_sync = true;
    }

    return needs_sync;
}

#ifdef __cplusplus
extern "C" {
#endif

void op_upload_dat(op_dat dat);
void op_download_dat(op_dat dat);

void op_put_dat(op_dat dat) {
  if (dat->dirty_hd == 1) { //Running on device, but dirty on host
    op_upload_dat(dat);
    dat->dirty_hd = 0;
  }
}

void op_get_dat(op_dat dat) {
  if (dat->dirty_hd == 2) { //Running on host, but dirty on device
    op_download_dat(dat);
    dat->dirty_hd = 0;
  }
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

void op_put_all_cuda(int nargs, op_arg *args) {
  for (int i = 0; i < nargs; i++) {
    if (args[i].argtype == OP_ARG_DAT && args[i].opt == 1) {
      op_put_dat_mpi(args[i].dat);
    }
  }
}

bool prepareDeviceGbls(op_arg *args, int nargs, int max_threads) {
    size_t required_size = 0;

    for (int i = 0; i < nargs; ++i) {
        if (opt_disabled(args[i], args)) continue;
        if (args[i].argtype != OP_ARG_GBL && args[i].argtype != OP_ARG_INFO) continue;

        if (needs_per_thread_storage(args[i])) {
            required_size += align(args[i].size * max_threads * sizeof(char));
        } else if (needs_device_storage(args[i])) {
            required_size += align(args[i].size * sizeof(char));
        }
    }

    bool needs_exit_sync = required_size > 0;
    device_globals.ensure_capacity(required_size);

    char *allocated = (char *) device_globals.data;
    for (int i = 0; i < nargs; ++i) {
        if (opt_disabled(args[i], args)) continue;
        if (args[i].argtype != OP_ARG_GBL && args[i].argtype != OP_ARG_INFO) continue;

        if (needs_per_thread_storage(args[i])) {
            args[i].data_d = allocated;
            allocated += align(args[i].size * max_threads * sizeof(char));
        } else if (needs_device_storage(args[i])) {
            args[i].data_d = allocated;
            allocated += align(args[i].size * sizeof(char));

            if (args[i].acc == OP_WRITE) continue;
            cutilSafeCall(cudaMemcpyAsync(args[i].data_d, args[i].data,
                                          args[i].size * sizeof(char),
                                          cudaMemcpyHostToDevice, cuda_stream));
        }
    }

    return needs_exit_sync;
}

int getBlockLimit(op_arg *args, int nargs, int block_size, char *name) {
    if (OP_cuda_reductions_mib < 0) return INT32_MAX;

    size_t bytes_per_thread = 0;
    for (int i = 0; i < nargs; ++i) {
        if (opt_disabled(args[i], args)) continue;
        if (!needs_per_thread_storage(args[i])) continue;

        bytes_per_thread += args[i].size * sizeof(char);
    }

    if (bytes_per_thread == 0) return INT32_MAX;

    if (bytes_per_thread > 1024)
        printf("Warning: kernel %s needs %zu reduction bytes per thread\n", name, bytes_per_thread);

    size_t max_total_reduction = OP_cuda_reductions_mib * 1024 * 1024;
    return std::max(max_total_reduction / ((size_t) block_size * bytes_per_thread), 1UL);
}

void processDeviceGbls(op_arg *args, int nargs, int nelems, int max_threads) {
    bool needs_red_sync = processDeviceGblReductions(args, nargs, nelems, max_threads);
    bool needs_rw_sync = processDeviceGblRWs(args, nargs);

    if (needs_red_sync || needs_rw_sync)
        cutilSafeCall(cudaStreamSynchronize(cuda_stream));
}

void setGblIncAtomic(bool enable) {
    gbl_inc_atomic = enable;
}

#ifdef __cplusplus
}
#endif

