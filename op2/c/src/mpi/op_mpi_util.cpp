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

#include <mpi.h>

#include <op_lib_mpi.h>
#include <op_util.h>
#include <vector>
#include <algorithm>

MPI_Comm OP_MPI_IO_WORLD;

void _mpi_allgather(int *l, int *g, int size, int *recevcnts, int *displs,
                    MPI_Comm comm) {
  MPI_Allgatherv(l, size, MPI_INT, g, recevcnts, displs, MPI_INT, comm);
}

void _mpi_allgather(float *l, float *g, int size, int *recevcnts, int *displs,
                    MPI_Comm comm) {
  MPI_Allgatherv(l, size, MPI_FLOAT, g, recevcnts, displs, MPI_FLOAT, comm);
}

void _mpi_allgather(double *l, double *g, int size, int *recevcnts, int *displs,
                    MPI_Comm comm) {
  MPI_Allgatherv(l, size, MPI_DOUBLE, g, recevcnts, displs, MPI_DOUBLE, comm);
}

void _mpi_gather(int *l, int *g, int size, int *recevcnts, int *displs,
                 MPI_Comm comm) {
  MPI_Gatherv(l, size, MPI_INT, g, recevcnts, displs, MPI_INT, MPI_ROOT, comm);
}

void _mpi_gather(float *l, float *g, int size, int *recevcnts, int *displs,
                 MPI_Comm comm) {
  MPI_Gatherv(l, size, MPI_FLOAT, g, recevcnts, displs, MPI_FLOAT, MPI_ROOT,
              comm);
}

void _mpi_gather(double *l, double *g, int size, int *recevcnts, int *displs,
                 MPI_Comm comm) {
  MPI_Gatherv(l, size, MPI_DOUBLE, g, recevcnts, displs, MPI_DOUBLE, MPI_ROOT,
              comm);
}

template <typename T>
void gather_data_hdf5(op_dat dat, char *usr_ptr, int low, int high) {
  // create new communicator
  int my_rank, comm_size;
  MPI_Comm_dup(OP_MPI_WORLD, &OP_MPI_IO_WORLD);
  MPI_Comm_rank(OP_MPI_IO_WORLD, &my_rank);
  MPI_Comm_size(OP_MPI_IO_WORLD, &comm_size);

  // compute local number of elements in dat
  int count = dat->set->size;

  T *l_array = (T *)xmalloc(dat->dim * (count) * sizeof(T));
  memcpy(l_array, (void *)&(dat->data[0]), dat->size * count);
  int l_size = count;
  size_t elem_size = dat->dim;
  int *recevcnts = (int *)xmalloc(comm_size * sizeof(int));
  int *displs = (int *)xmalloc(comm_size * sizeof(int));
  int disp = 0;
  T *g_array = 0;

  MPI_Allgather(&l_size, 1, MPI_INT, recevcnts, 1, MPI_INT, OP_MPI_IO_WORLD);

  int g_size = 0;
  for (int i = 0; i < comm_size; i++) {
    g_size += recevcnts[i];
    recevcnts[i] = elem_size * recevcnts[i];
  }
  for (int i = 0; i < comm_size; i++) {
    displs[i] = disp;
    disp = disp + recevcnts[i];
  }

  g_array = (T *)xmalloc(elem_size * g_size * sizeof(T));

  // need to all-gather dat->data and copy this to the memory block pointed by
  // usr_ptr
  _mpi_allgather(l_array, g_array, l_size * elem_size, recevcnts, displs,
                 OP_MPI_IO_WORLD);

  if (low < 0 || high > g_size - 1) {
    printf("op_fetch_data: Indices not within range of elements held in %s\n",
           dat->name);
    MPI_Abort(OP_MPI_IO_WORLD, -1);
  }
  memcpy((void *)usr_ptr, (void *)&g_array[low * dat->size],
         (high + 1) * dat->size);

  free(l_array);
  free(recevcnts);
  free(displs);
  free(g_array);
  MPI_Comm_free(&OP_MPI_IO_WORLD);
}

void checked_write(int v, const char *file_name) {
  if (v) {
    printf("error writing to %s\n", file_name);
    MPI_Abort(OP_MPI_IO_WORLD, -1);
  }
}

template <typename T>
void write_bin(FILE *fp, int g_size, int elem_size, T *g_array,
               const char *file_name) {
  checked_write(fwrite(&g_size, sizeof(int), 1, fp) < 1, file_name);
  checked_write(fwrite(&elem_size, sizeof(int), 1, fp) < 1, file_name);

  for (int i = 0; i < g_size; i++)
    checked_write(fwrite(&g_array[i * elem_size], sizeof(T), elem_size, fp) <
                      (size_t)elem_size,
                  file_name);
}

template <typename T, const char *fmt>
void write_txt(FILE *fp, int g_size, int elem_size, T *g_array,
               const char *file_name) {
  checked_write(fprintf(fp, "%d %d\n", g_size, elem_size) < 0, file_name);

  for (int i = 0; i < g_size; i++) {
    for (int j = 0; j < elem_size; j++)
      checked_write(fprintf(fp, fmt, g_array[i * elem_size + j]) < 0,
                    file_name);
    fprintf(fp, "\n");
  }
}

template <typename T, void (*F)(FILE *, int, int, T *, const char *)>
void write_file(op_dat dat, const char *file_name) {
  // create new communicator for output
  int rank, comm_size;
  MPI_Comm_dup(OP_MPI_WORLD, &OP_MPI_IO_WORLD);
  MPI_Comm_rank(OP_MPI_IO_WORLD, &rank);
  MPI_Comm_size(OP_MPI_IO_WORLD, &comm_size);

  // compute local number of elements in dat
  int count = dat->set->size;

  T *l_array = (T *)xmalloc(dat->dim * (count) * sizeof(T));
  memcpy(l_array, (void *)&(dat->data[0]), dat->size * count);

  int l_size = count;
  int elem_size = dat->dim;
  int *recevcnts = (int *)xmalloc(comm_size * sizeof(int));
  int *displs = (int *)xmalloc(comm_size * sizeof(int));
  int disp = 0;
  T *g_array = 0;

  MPI_Allgather(&l_size, 1, MPI_INT, recevcnts, 1, MPI_INT, OP_MPI_IO_WORLD);

  int g_size = 0;
  for (int i = 0; i < comm_size; i++) {
    g_size += recevcnts[i];
    recevcnts[i] = elem_size * recevcnts[i];
  }
  for (int i = 0; i < comm_size; i++) {
    displs[i] = disp;
    disp = disp + recevcnts[i];
  }
  if (rank == MPI_ROOT)
    g_array = (T *)xmalloc(elem_size * g_size * sizeof(T));
  _mpi_gather(l_array, g_array, l_size * elem_size, recevcnts, displs,
              OP_MPI_IO_WORLD);

  if (rank == MPI_ROOT) {
    FILE *fp;
    if ((fp = fopen(file_name, "w")) == NULL) {
      printf("can't open file %s\n", file_name);
      MPI_Abort(OP_MPI_IO_WORLD, -1);
    }

    // Write binary or text as requested by the caller
    F(fp, g_size, elem_size, g_array, file_name);

    fclose(fp);
    free(g_array);
  }

  free(l_array);
  free(recevcnts);
  free(displs);
  MPI_Comm_free(&OP_MPI_IO_WORLD);
}

/*******************************************************************************
* Routine to fetch data from an op_dat to user allocated memory block under hdf5
* -- placed in op_mpi_core.c as this routine does not use any hdf5 functions
*******************************************************************************/

void fetch_data_hdf5(op_dat dat, char *usr_ptr, int low, int high) {
  if (strcmp(dat->type, "double") == 0)
    gather_data_hdf5<double>(dat, usr_ptr, low, high);
  else if (strcmp(dat->type, "float") == 0)
    gather_data_hdf5<float>(dat, usr_ptr, low, high);
  else if (strcmp(dat->type, "int") == 0)
    gather_data_hdf5<int>(dat, usr_ptr, low, high);
  else
    printf("Unknown type %s, cannot error in fetch_data_hdf5() \n", dat->type);
}

/*******************************************************************************
 * Write a op_dat to a named ASCI file
 *******************************************************************************/

extern const char fmt_double[] = "%f ";
extern const char fmt_float[] = "%f ";
extern const char fmt_int[] = "%d ";

void print_dat_to_txtfile_mpi(op_dat dat, const char *file_name) {
  if (strcmp(dat->type, "double") == 0)
    write_file<double, write_txt<double, fmt_double> >(dat, file_name);
  else if (strcmp(dat->type, "float") == 0)
    write_file<float, write_txt<float, fmt_float> >(dat, file_name);
  else if (strcmp(dat->type, "int") == 0)
    write_file<int, write_txt<int, fmt_int> >(dat, file_name);
  else
    printf("Unknown type %s, cannot be written to file %s\n", dat->type,
           file_name);
}

/*******************************************************************************
 * Write a op_dat to a named Binary file
 *******************************************************************************/

void print_dat_to_binfile_mpi(op_dat dat, const char *file_name) {
  if (strcmp(dat->type, "double") == 0)
    write_file<double, write_bin<double> >(dat, file_name);
  else if (strcmp(dat->type, "float") == 0)
    write_file<float, write_bin<float> >(dat, file_name);
  else if (strcmp(dat->type, "int") == 0)
    write_file<int, write_bin<int> >(dat, file_name);
  else
    printf("Unknown type %s, cannot be written to file %s\n", dat->type,
           file_name);
}


int op_mpi_halo_exchanges_grouped(op_set set, int nargs, op_arg *args, int device) {
  int size = set->size;
  int direct_flag = 1;

  if (OP_diags > 0) {
    int dummy;
    for (int n = 0; n < nargs; n++)
      op_arg_check(set, n, args[n], &dummy, "halo_exchange_grouped cuda");
  }

  for (int n = 0; n < nargs; n++) {
    if (device == 2 && args[n].opt && args[n].argtype == OP_ARG_DAT &&
        args[n].dat->dirty_hd == 1) { //Running on device, but dirty on host
      op_upload_dat(args[n].dat);
      args[n].dat->dirty_hd = 0;
    }
    if (device == 1 && args[n].opt && args[n].argtype == OP_ARG_DAT &&
        args[n].dat->dirty_hd == 2) { //Running on host, but dirty on device
      op_download_dat(args[n].dat);
      args[n].dat->dirty_hd = 0;
    }
  }

  // check if this is a direct loop
  for (int n = 0; n < nargs; n++)
    if (args[n].opt && args[n].argtype == OP_ARG_DAT && args[n].idx != -1)
      direct_flag = 0;

  if (direct_flag == 1)
    return size;

  // not a direct loop ...
  int exec_flag = 0;
  for (int n = 0; n < nargs; n++) {
    if (args[n].opt && args[n].idx != -1 && args[n].acc != OP_READ) {
      size = set->size + set->exec_size;
      exec_flag = 1;
    }
  }
  op_timers_core(&c1, &t1);
  std::vector<int> sets();
  sets.resize(0);
  std::vector<unsigned> send_offsets();
  std::vector<unsigned> recv_offsets();
  std::vector<int> send_neigh_list();
  std::vector<int> recv_neigh_list();
  for (int n = 0; n < nargs; n++) {
    if (args[n].opt && args[n].argtype == OP_ARG_DAT && args[n].dat->dirty == 1 && (args[n].acc == OP_READ || args[n].acc == OP_RW)) {
      if (args[n].map == OP_ID && exec_flag == 0) continue; 

      //list of sets on which we have data accessed, and list of MPI neighbors
      if (std::find(sets.begin(), sets.end(), args[n].dat->set->index)!= sets.end()) {
        sets.push_back(args[n].dat->set->index);
        //receive neighbors
        halo_list imp_exec_list = OP_import_exec_list[args[n].dat->set->index];
        for (int i = 0; i < imp_exec_list->ranks_size; i++)
          if (std::find(recv_neigh_list.begin(), recv_neigh_list.end(), imp_exec_list->ranks[i])!= recv_neigh_list.end()) {
            recv_neigh_list.push_back(imp_exec_list->ranks[i]);
            recv_offsets.push_back(0);
          }
        halo_list imp_nonexec_list = OP_import_nonexec_list[args[n].dat->set->index];
        for (int i = 0; i < imp_nonexec_list->ranks_size; i++)
          if (std::find(recv_neigh_list.begin(), recv_neigh_list.end(), imp_nonexec_list->ranks[i])!= recv_neigh_list.end()) {
            recv_neigh_list.push_back(imp_nonexec_list->ranks[i]);
            recv_offsets.push_back(0);
          }

        //send neighbors
        halo_list exp_exec_list = OP_export_exec_list[args[n].dat->set->index];
        for (int i = 0; i < exp_exec_list->ranks_size; i++)
          if (std::find(send_neigh_list.begin(), send_neigh_list.end(), exp_exec_list->ranks[i])!= send_neigh_list.end()) {
            send_neigh_list.push_back(exp_exec_list->ranks[i]);
            send_offsets.push_back(0);
          }
        halo_list exp_nonexec_list = OP_export_nonexec_list[args[n].dat->set->index];
        for (int i = 0; i < exp_nonexec_list->ranks_size; i++)
          if (std::find(send_neigh_list.begin(), send_neigh_list.end(), exp_nonexec_list->ranks[i])!= send_neigh_list.end()) {
            send_neigh_list.push_back(exp_nonexec_list->ranks[i]);
            send_offsets.push_back(0);
          }
      }

      //Amount of memory required for send/recv per neighbor
      halo_list imp_exec_list = OP_import_exec_list[args[n].dat->set->index];
      for (int i = 0; i < imp_exec_list->ranks_size; i++) {
        int idx = std::distance(recv_neigh_list.start(), std::find(recv_neigh_list.begin(), recv_neigh_list.end(), imp_exec_list->ranks[i]));
        recv_offsets[idx] += args[n].dat->size * imp_exec_list->sizes[i];
      }
      halo_list imp_nonexec_list = OP_import_nonexec_list[args[n].dat->set->index];
      for (int i = 0; i < imp_nonexec_list->ranks_size; i++) {
        int idx = std::distance(recv_neigh_list.start(), std::find(recv_neigh_list.begin(), recv_neigh_list.end(), imp_nonexec_list->ranks[i]));
        recv_offsets[idx] += args[n].dat->size * imp_nonexec_list->sizes[i];
      }
      halo_list exp_exec_list = OP_export_exec_list[args[n].dat->set->index];
      for (int i = 0; i < exp_exec_list->ranks_size; i++) {
        int idx = std::distance(send_neigh_list.start(), std::find(send_neigh_list.begin(), send_neigh_list.end(), exp_exec_list->ranks[i]));
        send_offsets[idx] += args[n].dat->size * exp_exec_list->sizes[i];
      }
      halo_list exp_nonexec_list = OP_export_nonexec_list[args[n].dat->set->index];
      for (int i = 0; i < exp_nonexec_list->ranks_size; i++) {
        int idx = std::distance(send_neigh_list.start(), std::find(send_neigh_list.begin(), send_neigh_list.end(), exp_nonexec_list->ranks[i]));
        send_offsets[idx] += args[n].dat->size * exp_nonexec_list->sizes[i];
      }      
    }
  }

  //Realloc buffers
  unsigned size_send = std::accumulate(send_offsets.begin(), send_offsets.end(), 0u);
  unsigned size_recv = std::accumulate(recv_offsets.begin(), recv_offsets.end(), 0u);
  op_realloc_comm_buffer(&send_buffer_host, &recv_buffer_host, &send_buffer_device, &recv_buffer_device, device, size_send, size_recv);

  //Pack buffers
  std::vector<unsigned> send_offsets_fill(send_offsets.size(), 0u);
  for (int n = 0; n < nargs; n++) {
    if (args[n].opt && args[n].argtype == OP_ARG_DAT && args[n].dat->dirty == 1 && (args[n].acc == OP_READ || args[n].acc == OP_RW)) {
      if (args[n].map == OP_ID && exec_flag == 0) continue; 
      halo_list exp_exec_list = OP_export_exec_list[args[n].dat->set->index];
      halo_list exp_nonexec_list = OP_export_nonexec_list[args[n].dat->set->index];
      //need disps and sizes into list to make sure we put them in right place
      if (device==1) gather_data_to_buffer_ptr     (args[n], exp_exec_list, exp_nonexec_list, send_buffer_host, &send_neigh_list[0], );
      if (device==2) gather_data_to_buffer_ptr_cuda(args[n], exp_exec_list, exp_nonexec_list, send_buffer_device);
    }
  }

  //Non-blocking receive
  unsigned curr_offset = 0;
  for (unsigned i = 0; i < recv_neigh_list.size(); i++) {
    char *buf = gpudirect
    MPI_Irecv(buf + curr_offset, recv_offsets[i], MPI_CHAR, recv_neigh_list[i], 0 ,OP_MPI_WORLD, recv_requests[i]);
    curr_offset += recv_offsets[i];
  }

  if (device == 1) {
    unsigned curr_offset = 0;
    for (unsigned i = 0; i < send_neigh_list.size(); i++) {
      char *buf = send_buffer_host;
      MPI_Isend(buf + curr_offset, send_offsets[i], MPI_CHAR, send_neigh_list[i], 0 ,OP_MPI_WORLD, send_requests[i]);
      curr_offset += send_offsets[i];
    }
  } else if (device == 2 && !gpudirect) {
      download_buffer_async(send_buffer_device, send_buffer_host, size_send);
  }
    
  op_timers_core(&c2, &t2);
  if (OP_kern_max > 0)
    OP_kernels[OP_kern_curr].mpi_time += t2 - t1;
  return size;
}

void op_mpi_wait_all_grouped(op_set set, int nargs, op_arg *args, int device) {
  // check if this is a direct loop
  int direct_flag = 1;
  for (int n = 0; n < nargs; n++)
    if (args[n].opt && args[n].argtype == OP_ARG_DAT && args[n].idx != -1)
      direct_flag = 0;
  if (direct_flag == 1)
    return;

  // not a direct loop ...
  int exec_flag = 0;
  for (int n = 0; n < nargs; n++) {
    if (args[n].opt && args[n].idx != -1 && args[n].acc != OP_READ) {
      exec_flag = 1;
    }
  }
  op_timers_core(&c1, &t1);

  //Sends are only started here when running async on the device
  if (device == 2) {
    unsigned curr_offset = 0;
    for (unsigned i = 0; i < send_neigh_list.size(); i++) {
      char *buf = gpudirect
      MPI_Isend(buf + curr_offset, send_offsets[i], MPI_CHAR, send_neigh_list[i], 0 ,OP_MPI_WORLD, send_requests[i]);
      curr_offset += send_offsets[i];
    }
  }

  MPI_Waitall(recv_neigh_list.size(), &recv_requests[0], MPI_STATUSES_IGNORE);
  
  for (int n = 0; n < nargs; n++) {
    if (args[n].opt && args[n].argtype == OP_ARG_DAT && args[n].dat->dirty == 1 && (args[n].acc == OP_READ || args[n].acc == OP_RW)) {
      if (args[n].map == OP_ID && exec_flag == 0) continue; 
      unpack
    }
  }
  op_timers_core(&c2, &t2);
  if (OP_kern_max > 0)
    OP_kernels[OP_kern_curr].mpi_time += t2 - t1;
}