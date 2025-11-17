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

#include <op_sycl_rt_support.h>
#include <op_lib_c.h>
#include <op_lib_core.h>
#include <op_rt_support.h>

#include <op_lib_mpi.h>
#include <op_util.h>

//
// export lists on the device
//

int **export_exec_list_d = NULL;
int **export_nonexec_list_d = NULL;
int **export_nonexec_list_partial_d = NULL;
int **import_nonexec_list_partial_d = NULL;

void syclDeviceInit(int argc, char **argv) {
  char temp[64];
  const char *pch;

  int OP_sycl_device = 3;
  for (int i = 0; i < argc; ++i) {
    pch = strstr(argv[i], "OP_SYCL_DEVICE=");
    if (pch != NULL) {
      snprintf(temp, 64, "%s", pch);
      if (strcmp(temp + strlen("OP_SYCL_DEVICE="), "host") == 0)
        OP_sycl_device = 0;
      else if (strcmp(temp + strlen("OP_SYCL_DEVICE="), "cpu") == 0)
        OP_sycl_device = 1;
      else if (strcmp(temp + strlen("OP_SYCL_DEVICE="), "gpu") == 0)
        OP_sycl_device = 2;
      else {
        int val = atoi(temp + strlen("OP_SYCL_DEVICE="));
        OP_sycl_device=4+val;
      }
    }
  }
  switch (OP_sycl_device) {
  case 0:
    op2_queue =
        new cl::sycl::queue(cl::sycl::host_selector(), cl::sycl::property::queue::in_order());
    break;
  case 1:
    op2_queue =
        new cl::sycl::queue(cl::sycl::cpu_selector(), cl::sycl::property::queue::in_order());
    break;
  case 2:
    op2_queue =
        new cl::sycl::queue(cl::sycl::gpu_selector(), cl::sycl::property::queue::in_order());
    break;
  case 3:
    op2_queue =
        new cl::sycl::queue(cl::sycl::default_selector(), cl::sycl::property::queue::in_order());
    break;
  default: 
    std::vector<cl::sycl::device> devices;
    devices = cl::sycl::device::get_devices();
    int devid = OP_sycl_device - 4;
    if (devid < 0 || devid >= devices.size()) {
      op_printf("Error, unrecognised SYCL device selection. Available devices (%d)\n",devices.size());
      for (int i = 0; i < devices.size(); i++)
      {
        auto platform = devices[i].get_platform();
        op_printf("%d: [%s] %s\n", i, platform.get_info<cl::sycl::info::platform::name>().c_str(), devices[i].get_info<cl::sycl::info::device::name>().c_str());
      }
      exit(-1);
    }
    OP_hybrid_gpu = 1;
    op2_queue =
        new cl::sycl::queue(devices[devid], cl::sycl::property::queue::in_order());
  }

  OP_hybrid_gpu = 1;
  auto platform = op2_queue->get_device().get_platform();
  std::cout << "Running on " << op2_queue->get_device().get_info<cl::sycl::info::device::name>() << " platform " << platform.get_info<cl::sycl::info::platform::name>() << "\n";
}

void op_upload_dat(op_dat dat) {
  //  printf("Uploading %s\n", dat->name);
  int set_size = dat->set->size + OP_import_exec_list[dat->set->index]->size +
                 OP_import_nonexec_list[dat->set->index]->size;
  if (strstr(dat->type, ":soa") != NULL || (OP_auto_soa && dat->dim > 1)) {
    char *temp_data = (char *)xmalloc(dat->size * set_size * sizeof(char));
    int element_size = dat->size / dat->dim;
    for (int i = 0; i < dat->dim; i++) {
      for (int j = 0; j < set_size; j++) {
        for (int c = 0; c < element_size; c++) {
          temp_data[element_size * i * set_size + element_size * j + c] =
              dat->data[dat->size * j + element_size * i + c];
        }
      }
    }
    op2_queue->memcpy(dat->data_d, temp_data, dat->size * set_size);
    op2_queue->wait();
    free(temp_data);
  } else {
    op2_queue->memcpy(dat->data_d, dat->data, dat->size * set_size);
    op2_queue->wait();
  }
}

void op_download_dat(op_dat dat) {
  //  printf("Downloading %s\n", dat->name);
  int set_size = dat->set->size + OP_import_exec_list[dat->set->index]->size +
                 OP_import_nonexec_list[dat->set->index]->size;
  if (strstr(dat->type, ":soa") != NULL || (OP_auto_soa && dat->dim > 1)) {
    char *temp_data = (char *)xmalloc(dat->size * set_size * sizeof(char));
    op2_queue->memcpy(temp_data, dat->data_d, dat->size * set_size);
    op2_queue->wait();
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
    op2_queue->memcpy(dat->data, dat->data_d, dat->size * set_size);
    op2_queue->wait();
  }
}

void op_exchange_halo_cuda(op_arg *arg, int exec_flag) {
  op_dat dat = arg->dat;

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

  arg->sent = 0; // reset flag
  // need to exchange both direct and indirect data sets if they are dirty
  if ((arg->opt) &&
      (arg->acc == OP_READ ||
       arg->acc == OP_RW /* good for debug || arg->acc == OP_INC*/) &&
      (dat->dirtybit == 1)) {

    halo_list imp_exec_list = OP_import_exec_list[dat->set->index];
    halo_list imp_nonexec_list = OP_import_nonexec_list[dat->set->index];

    halo_list exp_exec_list = OP_export_exec_list[dat->set->index];
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

    op2_queue->wait();
    gather_data_to_buffer(*arg, exp_exec_list, exp_nonexec_list);
    op2_queue->wait();

    char *outptr_exec = NULL;
    char *outptr_nonexec = NULL;
    if (OP_gpu_direct) {
      outptr_exec = arg->dat->buffer_d;
      outptr_nonexec =
          arg->dat->buffer_d + exp_exec_list->size * arg->dat->size;
      op2_queue->wait();
    } else {
      op2_queue->memcpy(
          ((op_mpi_buffer)(dat->mpi_buffer))->buf_exec, arg->dat->buffer_d,
          exp_exec_list->size * arg->dat->size);
      op2_queue->wait();

      op2_queue->memcpy(
          ((op_mpi_buffer)(dat->mpi_buffer))->buf_nonexec,
          arg->dat->buffer_d + exp_exec_list->size * arg->dat->size,
          exp_nonexec_list->size * arg->dat->size);

      op2_queue->wait();
      outptr_exec = ((op_mpi_buffer)(dat->mpi_buffer))->buf_exec;
      outptr_nonexec = ((op_mpi_buffer)(dat->mpi_buffer))->buf_nonexec;
    }

    for (int i = 0; i < exp_exec_list->ranks_size; i++) {
      MPI_Isend(&outptr_exec[exp_exec_list->disps[i] * dat->size],
                dat->size * exp_exec_list->sizes[i], MPI_CHAR,
                exp_exec_list->ranks[i], dat->index, OP_MPI_WORLD,
                &((op_mpi_buffer)(dat->mpi_buffer))
                     ->s_req[((op_mpi_buffer)(dat->mpi_buffer))->s_num_req++]);
    }

    int init = dat->set->size * dat->size;
    char *ptr = NULL;
    for (int i = 0; i < imp_exec_list->ranks_size; i++) {
      ptr = OP_gpu_direct
                ? &(dat->data_d[init + imp_exec_list->disps[i] * dat->size])
                : &(dat->data[init + imp_exec_list->disps[i] * dat->size]);
      if (OP_gpu_direct && (strstr(arg->dat->type, ":soa") != NULL ||
                            (OP_auto_soa && arg->dat->dim > 1)))
        ptr = dat->buffer_d_r + imp_exec_list->disps[i] * dat->size;
      MPI_Irecv(ptr, dat->size * imp_exec_list->sizes[i], MPI_CHAR,
                imp_exec_list->ranks[i], dat->index, OP_MPI_WORLD,
                &((op_mpi_buffer)(dat->mpi_buffer))
                     ->r_req[((op_mpi_buffer)(dat->mpi_buffer))->r_num_req++]);
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
      MPI_Isend(&outptr_nonexec[exp_nonexec_list->disps[i] * dat->size],
                dat->size * exp_nonexec_list->sizes[i], MPI_CHAR,
                exp_nonexec_list->ranks[i], dat->index, OP_MPI_WORLD,
                &((op_mpi_buffer)(dat->mpi_buffer))
                     ->s_req[((op_mpi_buffer)(dat->mpi_buffer))->s_num_req++]);
    }

    int nonexec_init = (dat->set->size + imp_exec_list->size) * dat->size;
    for (int i = 0; i < imp_nonexec_list->ranks_size; i++) {
      ptr = OP_gpu_direct
                ? &(dat->data_d[nonexec_init +
                                imp_nonexec_list->disps[i] * dat->size])
                : &(dat->data[nonexec_init +
                              imp_nonexec_list->disps[i] * dat->size]);
      if (OP_gpu_direct && (strstr(arg->dat->type, ":soa") != NULL ||
                            (OP_auto_soa && arg->dat->dim > 1)))
        ptr = dat->buffer_d_r +
              (imp_exec_list->size + imp_exec_list->disps[i]) * dat->size;
      MPI_Irecv(ptr, dat->size * imp_nonexec_list->sizes[i], MPI_CHAR,
                imp_nonexec_list->ranks[i], dat->index, OP_MPI_WORLD,
                &((op_mpi_buffer)(dat->mpi_buffer))
                     ->r_req[((op_mpi_buffer)(dat->mpi_buffer))->r_num_req++]);
    }

    // clear dirty bit
    dat->dirtybit = 0;
    arg->sent = 1;
  }
}

void op_exchange_halo_partial_cuda(op_arg *arg, int exec_flag) {
  op_dat dat = arg->dat;

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

  arg->sent = 0; // reset flag
  // need to exchange both direct and indirect data sets if they are dirty
  if ((arg->opt) &&
      (arg->acc == OP_READ ||
       arg->acc == OP_RW /* good for debug || arg->acc == OP_INC*/) &&
      (dat->dirtybit == 1)) {

    halo_list imp_nonexec_list = OP_import_nonexec_permap[arg->map->index];
    halo_list exp_nonexec_list = OP_export_nonexec_permap[arg->map->index];

    //-------first exchange exec elements related to this data array--------

    // sanity checks
    if (compare_sets(imp_nonexec_list->set, dat->set) == 0) {
      printf("Error: Import list and set mismatch\n");
      MPI_Abort(OP_MPI_WORLD, 2);
    }
    if (compare_sets(exp_nonexec_list->set, dat->set) == 0) {
      printf("Error: Export list and set mismatch\n");
      MPI_Abort(OP_MPI_WORLD, 2);
    }

    op2_queue->wait();
    gather_data_to_buffer_partial(*arg, exp_nonexec_list);
    op2_queue->wait();

    char *outptr_nonexec = NULL;
    if (OP_gpu_direct) {
      outptr_nonexec = arg->dat->buffer_d;
      op2_queue->wait();
    } else {
      op2_queue->memcpy(
          ((op_mpi_buffer)(dat->mpi_buffer))->buf_nonexec, arg->dat->buffer_d,
          exp_nonexec_list->size * arg->dat->size);

      op2_queue->wait();
      outptr_nonexec = ((op_mpi_buffer)(dat->mpi_buffer))->buf_nonexec;
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
      MPI_Isend(&outptr_nonexec[exp_nonexec_list->disps[i] * dat->size],
                dat->size * exp_nonexec_list->sizes[i], MPI_CHAR,
                exp_nonexec_list->ranks[i], dat->index, OP_MPI_WORLD,
                &((op_mpi_buffer)(dat->mpi_buffer))
                     ->s_req[((op_mpi_buffer)(dat->mpi_buffer))->s_num_req++]);
    }

    int nonexec_init = OP_export_nonexec_permap[arg->map->index]->size;
    for (int i = 0; i < imp_nonexec_list->ranks_size; i++) {
      char *ptr =
          OP_gpu_direct
              ? &arg->dat
                     ->buffer_d[(nonexec_init + imp_nonexec_list->disps[i]) *
                                dat->size]
              : &((op_mpi_buffer)(dat->mpi_buffer))
                     ->buf_nonexec[(nonexec_init + imp_nonexec_list->disps[i]) *
                                   dat->size];
      MPI_Irecv(ptr, dat->size * imp_nonexec_list->sizes[i], MPI_CHAR,
                imp_nonexec_list->ranks[i], dat->index, OP_MPI_WORLD,
                &((op_mpi_buffer)(dat->mpi_buffer))
                     ->r_req[((op_mpi_buffer)(dat->mpi_buffer))->r_num_req++]);
    }

    arg->sent = 1;
  }
}

void op_exchange_halo(op_arg *arg, int exec_flag) {
  op_dat dat = arg->dat;

  if (exec_flag == 0 && arg->idx == -1)
    return;
  if (arg->opt == 0)
    return;

  if (arg->sent == 1) {
    printf("Error: Halo exchange already in flight for dat %s\n", dat->name);
    fflush(stdout);
    MPI_Abort(OP_MPI_WORLD, 2);
  }

  // need to exchange both direct and indirect data sets if they are dirty
  if ((arg->acc == OP_READ ||
       arg->acc == OP_RW /* good for debug || arg->acc == OP_INC*/) &&
      (dat->dirtybit == 1)) {
    // printf("Exchanging Halo of data array %10s\n",dat->name);
    halo_list imp_exec_list = OP_import_exec_list[dat->set->index];
    halo_list imp_nonexec_list = OP_import_nonexec_list[dat->set->index];

    halo_list exp_exec_list = OP_export_exec_list[dat->set->index];
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
      MPI_Isend(&((op_mpi_buffer)(dat->mpi_buffer))
                     ->buf_exec[exp_exec_list->disps[i] * dat->size],
                dat->size * exp_exec_list->sizes[i], MPI_CHAR,
                exp_exec_list->ranks[i], dat->index, OP_MPI_WORLD,
                &((op_mpi_buffer)(dat->mpi_buffer))
                     ->s_req[((op_mpi_buffer)(dat->mpi_buffer))->s_num_req++]);
    }

    int init = dat->set->size * dat->size;
    for (int i = 0; i < imp_exec_list->ranks_size; i++) {
      MPI_Irecv(&(dat->data[init + imp_exec_list->disps[i] * dat->size]),
                dat->size * imp_exec_list->sizes[i], MPI_CHAR,
                imp_exec_list->ranks[i], dat->index, OP_MPI_WORLD,
                &((op_mpi_buffer)(dat->mpi_buffer))
                     ->r_req[((op_mpi_buffer)(dat->mpi_buffer))->r_num_req++]);
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
      MPI_Isend(&((op_mpi_buffer)(dat->mpi_buffer))
                     ->buf_nonexec[exp_nonexec_list->disps[i] * dat->size],
                dat->size * exp_nonexec_list->sizes[i], MPI_CHAR,
                exp_nonexec_list->ranks[i], dat->index, OP_MPI_WORLD,
                &((op_mpi_buffer)(dat->mpi_buffer))
                     ->s_req[((op_mpi_buffer)(dat->mpi_buffer))->s_num_req++]);
    }

    int nonexec_init = (dat->set->size + imp_exec_list->size) * dat->size;
    for (int i = 0; i < imp_nonexec_list->ranks_size; i++) {
      MPI_Irecv(
          &(dat->data[nonexec_init + imp_nonexec_list->disps[i] * dat->size]),
          dat->size * imp_nonexec_list->sizes[i], MPI_CHAR,
          imp_nonexec_list->ranks[i], dat->index, OP_MPI_WORLD,
          &((op_mpi_buffer)(dat->mpi_buffer))
               ->r_req[((op_mpi_buffer)(dat->mpi_buffer))->r_num_req++]);
    }
    // clear dirty bit
    dat->dirtybit = 0;
    arg->sent = 1;
  }
}

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

  // need to exchange indirect data sets if they are dirty
  if ((arg->acc == OP_READ ||
       arg->acc == OP_RW /* good for debug || arg->acc == OP_INC*/) &&
      (dat->dirtybit == 1)) {
    int rank;
    MPI_Comm_rank(OP_MPI_WORLD, &rank);
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
    }
    // note that we are not settinging the dirtybit to 0, since it's not a full
    // exchange
    arg->sent = 1;
  }
}

void op_wait_all_cuda(op_arg *arg) {
  if (arg->opt && arg->argtype == OP_ARG_DAT && arg->sent == 1) {
    op_dat dat = arg->dat;
    MPI_Waitall(((op_mpi_buffer)(dat->mpi_buffer))->s_num_req,
                ((op_mpi_buffer)(dat->mpi_buffer))->s_req, MPI_STATUSES_IGNORE);
    MPI_Waitall(((op_mpi_buffer)(dat->mpi_buffer))->r_num_req,
                ((op_mpi_buffer)(dat->mpi_buffer))->r_req, MPI_STATUSES_IGNORE);
    ((op_mpi_buffer)(dat->mpi_buffer))->s_num_req = 0;
    ((op_mpi_buffer)(dat->mpi_buffer))->r_num_req = 0;

    if (arg->map != OP_ID && OP_map_partial_exchange[arg->map->index]) {
      halo_list imp_nonexec_list = OP_import_nonexec_permap[arg->map->index];
      int nonexec_init = OP_export_nonexec_permap[arg->map->index]->size;
      ;
      if (OP_gpu_direct == 0) {
        op2_queue->wait();
        op2_queue->memcpy(
            dat->buffer_d + nonexec_init * dat->size,
            &((op_mpi_buffer)(dat->mpi_buffer))
                 ->buf_nonexec[nonexec_init * dat->size],
            imp_nonexec_list->size * dat->size);
      }
      op2_queue->wait();
      scatter_data_from_buffer_partial(*arg);
      op2_queue->wait();
    } else {
      if (OP_gpu_direct == 0) {
        if (strstr(arg->dat->type, ":soa") != NULL ||
            (OP_auto_soa && arg->dat->dim > 1)) {
          int init = dat->set->size * dat->size;
          int size = (dat->set->exec_size + dat->set->nonexec_size) * dat->size;
          op2_queue->wait();
          op2_queue->memcpy(dat->buffer_d_r, dat->data + init, size);
          op2_queue->wait();
          scatter_data_from_buffer(*arg);
          op2_queue->wait();
        } else {
          int init = dat->set->size * dat->size;
          op2_queue->wait();
          op2_queue->memcpy(dat->data_d + init, dat->data + init,
                              (OP_import_exec_list[dat->set->index]->size +
                               OP_import_nonexec_list[dat->set->index]->size) *
                                  arg->dat->size);
          op2_queue->wait();
        }
      } else if (strstr(arg->dat->type, ":soa") != NULL ||
                 (OP_auto_soa && arg->dat->dim > 1))
        op2_queue->wait();
        scatter_data_from_buffer(*arg);
        op2_queue->wait();
    }
    arg->sent = 2; // set flag to indicate completed comm
  }
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
    if (arg->map != OP_ID && OP_map_partial_exchange[arg->map->index]) {
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
  arg->sent = 0;
}

void op_partition(const char *lib_name, const char *lib_routine,
                  op_set prime_set, op_map prime_map, op_dat coords) {
  partition(lib_name, lib_routine, prime_set, prime_map, coords);
  if (!OP_hybrid_gpu)
    return;
  op_move_to_device();
}

void op_move_to_device() {
  for (int s = 0; s < OP_set_index; s++) {
    op_set set = OP_set_list[s];
    op_dat_entry *item;
    TAILQ_FOREACH(item, &OP_dat_list, entries) {
      op_dat dat = item->dat;

      if (dat->set->index == set->index)
        op_mv_halo_device(set, dat);
    }
  }

  for (int m = 0; m < OP_map_index; m++) {
    // Upload maps in transposed form
    op_map map = OP_map_list[m];
    int set_size = map->from->size + map->from->exec_size;
    int *temp_map = (int *)xmalloc(map->dim * set_size * sizeof(int));
    for (int i = 0; i < map->dim; i++) {
      for (int j = 0; j < set_size; j++) {
        temp_map[i * set_size + j] = map->map[map->dim * j + i];
      }
    }
    op_cpHostToDevice((void **)&(map->map_d), (void **)&(temp_map),
                      map->dim * set_size * sizeof(int));
    free(temp_map);
  }

  op_mv_halo_list_device();
}

int op_is_root() {
  int my_rank;
  MPI_Comm_rank(OP_MPI_WORLD, &my_rank);
  return (my_rank == MPI_ROOT);
}

