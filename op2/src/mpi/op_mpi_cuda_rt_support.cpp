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
// This file implements the MPI+CUDA-specific run-time support functions
//

//
// header files
//

#include <mpi.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <op_gpu_shims.h>
#include <op_cuda_rt_support.h>
#include <op_lib_c.h>
#include <op_lib_core.h>
#include <op_rt_support.h>

#include <op_lib_mpi.h>
#include <op_mpi_halo.h>
#include <op_util.h>

//
// halo lists on the device
//

std::vector<op::mpi::DeviceSetHalo> OP_set_halos_d;
std::vector<op::mpi::DeviceMapHalo> OP_map_halos_d;

/* The return code is not checked: a list still alive at static destruction, if
   op_exit was never called, is freed after the runtime has begun unloading. */
void op::mpi::DeviceFree::operator()(idx_l_t *p) const { (void)gpuFree(p); }

void cutilDeviceInit(int argc, char **argv) {
  (void)argc;
  (void)argv;
  int deviceCount;
  cutilSafeCall(gpuGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    printf("cutil error: no devices supporting CUDA\n");
    exit(-1);
  }
  printf("Trying to select a device\n");

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // no need to ardcode this following, can be done via numawrap scripts
  /*if (getenv("OMPI_COMM_WORLD_LOCAL_RANK")!=NULL) {
    rank = atoi(getenv("OMPI_COMM_WORLD_LOCAL_RANK"));
  } else if (getenv("MV2_COMM_WORLD_LOCAL_RANK")!=NULL) {
    rank = atoi(getenv("MV2_COMM_WORLD_LOCAL_RANK"));
  } else if (getenv("MPI_LOCALRANKID")!=NULL) {
    rank = atoi(getenv("MPI_LOCALRANKID"));
  } else {
    rank = rank%deviceCount;
  }*/

  // Test we have access to a device

  // This commented out test does not work with CUDA versions above 6.5
  /*float *test;
  gpuError_t err = gpuMalloc((void **)&test, sizeof(float));
  if (err != gpuSuccess) {
    OP_hybrid_gpu = 0;
  } else {
    OP_hybrid_gpu = 1;
  }
  if (OP_hybrid_gpu) {
    gpuFree(test);

    cutilSafeCall(gpuDeviceSetCacheConfig(gpuFuncCachePreferL1));

    int deviceId = -1;
    gpuGetDevice(&deviceId);
    gpuDeviceProp_t deviceProp;
    cutilSafeCall ( gpuGetDeviceProperties ( &deviceProp, deviceId ) );
    printf ( "\n Using CUDA device: %d %s on rank %d\n",deviceId,
  deviceProp.name,rank );
  } else {
    printf ( "\n Using CPU on rank %d\n",rank );
  }*/
  //omp_set_default_device(rank);
//  gpuError_t err = gpuSetDevice(rank);
  float *test;
  OP_hybrid_gpu = 0;
  //gpuError_t err = gpuMalloc((void **)&test, sizeof(float));
  for (int i = 0; i < deviceCount; i++) {
    gpuError_t err = gpuSetDevice((i+rank)%deviceCount);
    if (err == gpuSuccess) {
      gpuError_t err = op_deviceMalloc((void **)&test, sizeof(float));
      if (err == gpuSuccess) {
        OP_hybrid_gpu = 1;
        break;
      }
    }
  }
  if (OP_hybrid_gpu) {
    cutilSafeCall(gpuFree(test));

    cutilSafeCall(gpuDeviceSetCacheConfig(gpuFuncCachePreferL1));

    int deviceId = -1;
    gpuGetDevice(&deviceId);
    gpuDeviceProp_t deviceProp;
    cutilSafeCall(gpuGetDeviceProperties(&deviceProp, deviceId));
    printf("\n Using CUDA device: %d %s on rank %d\n", deviceId,
           deviceProp.name, rank);
  } else {
    printf("\n Using CPU on rank %d\n", rank);
  }
}

void op_upload_dat(op_dat dat) {
  if (OP_set_halos.empty()) return;
  idx_g_t set_size = dat->set->size + OP_set_halos[dat->set->index].import_exec.size() +
                 OP_set_halos[dat->set->index].import_nonexec.size();
  if (strstr(dat->type, ":soa") != NULL || (OP_auto_soa && dat->dim > 1)) {
    char *temp_data = (char *)xmalloc((size_t)dat->size * round32(set_size) * sizeof(char));
    int element_size = (size_t)dat->size / dat->dim;
    for (int i = 0; i < dat->dim; i++) {
      for (idx_g_t j = 0; j < set_size; j++) {
        for (int c = 0; c < element_size; c++) {
          temp_data[element_size * i * round32(set_size) + element_size * j + c] =
              dat->data[(size_t)dat->size * j + element_size * i + c];
        }
      }
    }
    cutilSafeCall(gpuMemcpy(dat->data_d, temp_data, round32(set_size) * (size_t)dat->size,
                             gpuMemcpyHostToDevice));
    free(temp_data);
  } else {
    cutilSafeCall(gpuMemcpy(dat->data_d, dat->data, set_size * (size_t)dat->size,
                             gpuMemcpyHostToDevice));
  }
}

void op_download_dat(op_dat dat) {
  //Check if partitionig is done
  if (OP_set_halos.empty()) return;
  //  printf("Downloading %s\n", dat->name);
  idx_g_t set_size = dat->set->size + OP_set_halos[dat->set->index].import_exec.size() +
                 OP_set_halos[dat->set->index].import_nonexec.size();
  if (strstr(dat->type, ":soa") != NULL || (OP_auto_soa && dat->dim > 1)) {
    char *temp_data = (char *)xmalloc((size_t)dat->size * round32(set_size) * sizeof(char));
    cutilSafeCall(gpuMemcpy(temp_data, dat->data_d, round32(set_size) * (size_t)dat->size,
                             gpuMemcpyDeviceToHost));
    int element_size = (size_t)dat->size / dat->dim;
    for (int i = 0; i < dat->dim; i++) {
      for (idx_g_t j = 0; j < set_size; j++) {
        for (int c = 0; c < element_size; c++) {
          dat->data[(size_t)dat->size * j + element_size * i + c] =
              temp_data[element_size * i * round32(set_size) + element_size * j + c];
        }
      }
    }
    free(temp_data);
  } else {
    cutilSafeCall(gpuMemcpy(dat->data, dat->data_d, set_size * (size_t)dat->size,
                             gpuMemcpyDeviceToHost));
  }
}

#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif

// Resolve OP_gpu_direct: whether MPI can send and receive straight out of device
// memory. Layered, and only ever promotes to "yes", because passing a device
// pointer to an MPI that cannot take one segfaults inside the transport, while
// staging through host memory is always correct and merely slower.
//
// Called once from op_init, after MPI is up, because the capability queries need
// it. OP2_GPU_DIRECT overrides the detection in either direction.
void op_gpu_direct_init() {
    static bool resolved = false;
    if (resolved) return;
    resolved = true;

    const char *reason;
    const char *override_env = getenv("OP2_GPU_DIRECT");

    if (override_env != NULL) {
        OP_gpu_direct = atoi(override_env) != 0;
        reason = "OP2_GPU_DIRECT";
    } else {
        reason = "not detected, set OP2_GPU_DIRECT=1 to force";

#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
        if (!OP_gpu_direct && MPIX_Query_cuda_support() == 1) {
            OP_gpu_direct = 1;
            reason = "MPIX_Query_cuda_support";
        }
#endif
#if defined(MPIX_ROCM_AWARE_SUPPORT) && MPIX_ROCM_AWARE_SUPPORT
        if (!OP_gpu_direct && MPIX_Query_rocm_support() == 1) {
            OP_gpu_direct = 1;
            reason = "MPIX_Query_rocm_support";
        }
#endif
        // MPIs with no capability query at all: Cray MPICH, Intel MPI, MVAPICH.
        // These say the user asked for GPU support, not that it is present.
        if (!OP_gpu_direct) {
            static const char *hints[] = {"MPICH_GPU_SUPPORT_ENABLED", "I_MPI_OFFLOAD",
                                          "MV2_USE_CUDA"};
            for (const char *hint : hints) {
                const char *value = getenv(hint);

                if (value != NULL && atoi(value) != 0) {
                    OP_gpu_direct = 1;
                    reason = hint;
                    break;
                }
            }
        }
    }

    op_printf("OP2: GPU-direct MPI = %s (%s)\n", OP_gpu_direct ? "yes" : "no", reason);
}

void op_partition(const char *lib_name, const char *lib_routine,
                  op_set prime_set, op_map prime_map, op_dat data) {
  partition(lib_name, lib_routine, prime_set, prime_map, data);
  if (!OP_hybrid_gpu)
    return;
  op_move_to_device();
}

void op_move_to_device() {
  size_t dat_size = 0;
  for (int s = 0; s < OP_set_index; s++) {
    op_set set = OP_set_list[s];
    op_dat_entry *item;
    TAILQ_FOREACH(item, &OP_dat_list, entries) {
      op_dat dat = item->dat;

      if (dat->set->index == set->index)
        dat_size += op_mv_halo_device(set, dat);
    }
  }

  size_t map_size = 0;
  for (int m = 0; m < OP_map_index; m++) {
    // Upload maps in transposed form
    op_map map = OP_map_list[m];
    int set_size = map->from->size + map->from->exec_size;
    int *temp_map = (int *)xmalloc(map->dim * round32(set_size) * sizeof(int));
    for (int i = 0; i < map->dim; i++) {
      for (int j = 0; j < set_size; j++) {
        temp_map[i * round32(set_size) + j] = map->map[map->dim * j + i];
      }
    }
    op_cpHostToDevice((void **)&(map->map_d), (void **)&(temp_map),
                      (size_t)map->dim * round32(set_size) * sizeof(int));
    free(temp_map);

    map_size += map->dim * round32(set_size) * sizeof(int);
  }

  size_t halo_size = op_mv_halo_list_device();

  auto as_mib = [](size_t s) { return (double)s / (1024.0 * 1024.0); };
  op_printf("Total device memory usage: %.1f MiB (dats: %.1f MiB, maps: %.1f MiB, halo lists: %.1f Mib)\n",
          as_mib(dat_size + map_size + halo_size), as_mib(dat_size), as_mib(map_size), as_mib(halo_size));
}
