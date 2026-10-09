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
// This file implements the OP2 user-level functions for the CUDA backend
//

#include <mpi.h>

#include <op_gpu_shims.h>
#include <op_cuda_rt_support.h>
#include <op_autotune.h>
#include <op_hier_plan.h>
#include <op_lib_core.h>
#include <op_rt_support.h>

#include <op_lib_c.h>
#include <op_lib_mpi.h>
#include <op_mpi_halo.h>

using op::mpi::DeviceList;
using op::mpi::DeviceMapHalo;
using op::mpi::DeviceSetHalo;
using op::mpi::HaloList;
#include <op_util.h>
#include <vector>

//
// MPI Communicator for halo creation and exchange
//

MPI_Comm OP_MPI_WORLD;
MPI_Comm OP_MPI_GLOBAL;

//
// CUDA-specific OP2 functions
//

void op_init(int argc, char **argv, int diags) {
  op_init_soa(argc, argv, diags, 0);
}

void op_init_soa(int argc, char **argv, int diags, int soa) {
  int flag = 0;
  OP_auto_soa = soa;
  MPI_Initialized(&flag);
  if (!flag) {
    MPI_Init(&argc, &argv);
  }
  OP_MPI_WORLD = MPI_COMM_WORLD;
  OP_MPI_GLOBAL = MPI_COMM_WORLD;
  op_init_core(argc, argv, diags);

  cutilDeviceInit(argc, argv);
  op_gpu_direct_init();
}

void op_mpi_init(int argc, char **argv, int diags, MPI_Fint global,
                 MPI_Fint local) {
  op_mpi_init_soa(argc, argv, diags, global, local, 0);
}

void op_mpi_init_soa(int argc, char **argv, int diags, MPI_Fint global,
                     MPI_Fint local, int soa) {
  OP_auto_soa = soa;
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

  cutilDeviceInit(argc, argv);
  op_gpu_direct_init();
}

op_dat op_decl_dat_char(op_set set, int dim, char const *type, int size,
                        char *data, char const *name) {
  if (set == NULL)
    return NULL;

  op_dat dat = op_decl_dat_core(set, dim, type, size, data, name);

  op_dat_entry *item;
  op_dat_entry *tmp_item;
  for (item = TAILQ_FIRST(&OP_dat_list); item != NULL; item = tmp_item) {
    tmp_item = TAILQ_NEXT(item, entries);
    if (item->dat == dat) {
      item->orig_ptr = data;
      break;
    }
  }
  dat->user_managed = 0;
  return dat;
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

  if (item_dat == NULL)
    op::mpi::fail("ERROR: op_dat not found for dat with %p pointer\n", (void *)dat);

  return op_decl_dat_overlay(set, item_dat);
}

op_dat op_decl_dat_temp_char(op_set set, int dim, char const *type, int size,
                             char const *name) {
  op_dat dat = op_decl_dat_temp_core(set, dim, type, size, NULL, name);

  // create empty data block to assign to this temporary dat (including the
  // halos, none before partitioning)
  size_t set_size = (size_t)set->size + set->exec_size + set->nonexec_size;

  // transpose
  if (strstr(dat->type, ":soa") != NULL || (OP_auto_soa && dat->dim > 1)) {
    op_deviceMalloc((void **)&(dat->data_d), (size_t)(dat->size) * round32(set_size));
    op_deviceZero(dat->data_d, (size_t)(dat->size) * round32(set_size));
  } else {
    op_deviceMalloc((void **)&(dat->data_d), (size_t)(dat->size) * set_size);
    op_deviceZero(dat->data_d, (size_t)(dat->size) * set_size);
  }

  return dat;
}

int op_free_dat_temp_char(op_dat dat) {
  // free data on device
  cutilSafeCall(gpuFree(dat->data_d));
  return op_free_dat_temp_core(dat);
}

size_t op_mv_halo_device(op_set set, op_dat dat) {
  size_t total_size = 0;

  idx_g_t set_size = set->size + set->exec_size + set->nonexec_size;
  if (strstr(dat->type, ":soa") != NULL || (OP_auto_soa && dat->dim > 1)) {
    char *temp_data = (char *)malloc((size_t)dat->size * round32(set_size) * sizeof(char));
    int element_size = (size_t)dat->size / dat->dim;
    for (int i = 0; i < dat->dim; i++) {
      for (idx_g_t j = 0; j < set_size; j++) {
        for (int c = 0; c < element_size; c++) {
          temp_data[element_size * i * round32(set_size) + element_size * j + c] =
              dat->data[(size_t)dat->size * j + element_size * i + c];
        }
      }
    }
    op_cpHostToDevice((void **)&(dat->data_d), (void **)&(temp_data),
                      (size_t)dat->size * round32(set_size));
    free(temp_data);

    total_size += (size_t)dat->size * round32(set_size) * sizeof(char);
  } else {
    op_cpHostToDevice((void **)&(dat->data_d), (void **)&(dat->data),
                      (size_t)dat->size * set_size);

    total_size += (size_t)dat->size * set_size * sizeof(char);
  }
  dat->dirty_hd = 0;

  return total_size;
}

/* Upload every list the unified exchange reads on the device, replacing (and so
   freeing) any earlier copies: after op_renumber the host lists have changed. */
size_t op_mv_halo_list_device() {
  size_t total_size = 0;
  auto upload = [&](const HaloList &list) {
    /* op_cpHostToDevice takes the host pointer by address: hand it a plain
       pointer, never the address of the owner, which a (void **) cast would
       accept without complaint. */
    idx_l_t *device = nullptr;
    void *host = list.list.get();
    const size_t bytes = list.size() * sizeof(idx_l_t);
    op_cpHostToDevice((void **)&device, &host, bytes);
    total_size += bytes;
    return DeviceList(device);
  };

  OP_set_halos_d = std::vector<DeviceSetHalo>(OP_set_index);
  for (int s = 0; s < OP_set_index; s++) {
    op_set set = OP_set_list[s];
    OP_set_halos_d[set->index].export_exec = upload(OP_set_halos[set->index].export_exec);
    OP_set_halos_d[set->index].export_nonexec = upload(OP_set_halos[set->index].export_nonexec);
  }

  OP_map_halos_d = std::vector<DeviceMapHalo>(OP_map_index);
  for (int m = 0; m < (int)OP_map_halos.size(); m++) {
    if (!OP_map_partial_exchange[m])
      continue;
    OP_map_halos_d[m].export_nonexec = upload(OP_map_halos[m].export_nonexec);
    OP_map_halos_d[m].import_nonexec = upload(OP_map_halos[m].import_nonexec);
  }
  return total_size;
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

void op_timers(double *cpu, double *et) {
  MPI_Barrier(OP_MPI_WORLD);
  op_timers_core(cpu, et);
}

//
// This function is defined in the generated master kernel file
// so that it is possible to check on the runtime size of the
// data in cases where it is not known at compile time
//

/*
void
op_decl_const_char ( int dim, char const * type, int size, char * dat,
                     char const * name )
{
  cutilSafeCall ( gpuMemcpyToSymbol ( name, dat, dim * size, 0,
                                       gpuMemcpyHostToDevice ) );
}
*/

void op_exit() {
  op::f2c::release_hier_plan_device_storage();
  {
    int rank = 0, ranks = 1;
    MPI_Comm_rank(OP_MPI_WORLD, &rank);
    MPI_Comm_size(OP_MPI_WORLD, &ranks);
    op::f2c::autotune_write_report(rank, ranks);
  }

  // free the device halo lists, while the device is still up
  OP_set_halos_d = std::vector<DeviceSetHalo>();
  OP_map_halos_d = std::vector<DeviceMapHalo>();

  op_mpi_exit();
  op_cuda_exit(); // frees dat_d memory
  op_rt_exit();   // frees plan memory
  op_exit_core(); // frees lib core variables

  int flag = 0;
  MPI_Finalized(&flag);
  if (!flag)
    MPI_Finalize();
}

void op_timing_output() {
  double max_plan_time = 0.0;
  MPI_Reduce(&OP_plan_time, &max_plan_time, 1, MPI_DOUBLE, MPI_MAX, 0, OP_MPI_WORLD);
  op_timing_output_core();
  if (op_is_root())
    printf("Total plan time: %8.4f\n", max_plan_time);
  mpi_timing_output();
}

void op_print_dat_to_binfile(op_dat dat, const char *file_name) {
  // need to get data from GPU
  op_cuda_get_data(dat);

  // rearrange data backe to original order in mpi
  op_dat temp = op_mpi_get_data(dat);
  print_dat_to_binfile_mpi(temp, file_name);

  free(temp->data);
  free(temp->set);
  free(temp);
}

void op_print_dat_to_txtfile(op_dat dat, const char *file_name) {
  // need to get data from GPU
  op_cuda_get_data(dat);

  // rearrange data backe to original order in mpi
  op_dat temp = op_mpi_get_data(dat);
  print_dat_to_txtfile_mpi(temp, file_name);

  free(temp->data);
  free(temp->set);
  free(temp);
}

void op_upload_all() {
  op_dat_entry *item;
  TAILQ_FOREACH(item, &OP_dat_list, entries) {
    op_dat dat = item->dat;
    idx_g_t set_size = dat->set->size + dat->set->exec_size + dat->set->nonexec_size;
    if (dat->data_d) {
      if (strstr(dat->type, ":soa") != NULL || (OP_auto_soa && dat->dim > 1)) {
        char *temp_data = (char *)malloc((size_t)dat->size * round32(set_size) * sizeof(char));
        int element_size = (size_t)dat->size / dat->dim;
        for (int i = 0; i < dat->dim; i++) {
          for (idx_g_t j = 0; j < set_size; j++) {
            for (int c = 0; c < element_size; c++) {
              temp_data[element_size * i * round32(set_size) + element_size * j + c] =
                  dat->data[(size_t)dat->size * j + element_size * i + c];
            }
          }
        }
        cutilSafeCall(gpuMemcpy(dat->data_d, temp_data, (size_t)dat->size * round32(set_size),
                                 gpuMemcpyHostToDevice));
        dat->dirty_hd = 0;
        free(temp_data);
      } else {
        cutilSafeCall(gpuMemcpy(dat->data_d, dat->data, (size_t)dat->size * set_size,
                                 gpuMemcpyHostToDevice));
        dat->dirty_hd = 0;
      }
    }
  }
}

void op_fetch_data_char(op_dat dat, char *usr_ptr) {
  // need to get data from GPU
  op_cuda_get_data(dat);

  // rearrange data backe to original order in mpi
  op_dat temp = op_mpi_get_data(dat);

  // copy data into usr_ptr
  memcpy((void *)usr_ptr, (void *)temp->data, temp->set->size * temp->size);
  free(temp->data);
  free(temp->set);
  free(temp);
}

op_dat op_fetch_data_file_char(op_dat dat) {
  // need to get data from GPU
  op_cuda_get_data(dat);
  // rearrange data backe to original order in mpi
  return op_mpi_get_data(dat);
}

void op_fetch_data_idx_char(op_dat dat, char *usr_ptr, int low, int high) {
  // need to get data from GPU
  op_cuda_get_data(dat);

  // rearrange data backe to original order in mpi
  op_dat temp = op_mpi_get_data(dat);

  // do allgather on temp->data and copy it to memory block pointed to by
  // use_ptr
  fetch_data_hdf5(temp, usr_ptr, low, high);

  free(temp->data);
  free(temp->set);
  free(temp);
}
