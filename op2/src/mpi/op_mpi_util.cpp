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

#include <op_lib_c.h>
#include <op_lib_mpi.h>
#include <op_util.h>

#include <op_mpi_halo.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

MPI_Comm OP_MPI_IO_WORLD;

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
  memcpy(l_array, (void *)&(dat->data[0]), (size_t)dat->size * count);

  int l_size = count;
  int elem_size = dat->dim;
  int *recevcnts = (int *)xmalloc(comm_size * sizeof(int));
  int *displs = (int *)xmalloc(comm_size * sizeof(int));
  int disp = 0;
  T *g_array = 0;

  MPI_Allgather(&l_size, 1, get_mpi_type(&l_size), recevcnts, 1, get_mpi_type(recevcnts), OP_MPI_IO_WORLD);

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
* Rows [low, high] of a dat in the layout as declared (what op_mpi_get_data
* returns) into usr_ptr, on every rank. Each rank fills in the rows of the range it
* holds and leaves the rest zero, and a bitwise OR over the ranks completes the
* range everywhere: nothing bigger than the range, nothing sized by the number of
* ranks, and no need to know the dat's type.
*******************************************************************************/

void fetch_data_hdf5(op_dat dat, char *usr_ptr, int low, int high) {
  const idx_g_t first = op::mpi::sum_below_rank(dat->set->size, OP_MPI_WORLD);
  const idx_g_t total = op::mpi::sum_over_ranks(dat->set->size, OP_MPI_WORLD);
  if (low < 0 || high > total - 1 || low > high)
    op::mpi::fail("op_fetch_data: indices %d to %d not within the %lld elements of %s\n", low, high, (long long)total,
                  dat->name);

  const std::size_t row = dat->size, bytes = (std::size_t)(high - low + 1) * row;
  std::memset(usr_ptr, 0, bytes);
  const idx_g_t from = std::max<idx_g_t>(low, first), to = std::min<idx_g_t>(high + 1, first + dat->set->size);
  if (from < to)
    std::memcpy(usr_ptr + (from - low) * row, dat->data + (from - first) * row, (to - from) * row);

  // in pieces: an MPI count is an int
  const std::size_t piece = std::numeric_limits<int>::max();
  for (std::size_t done = 0; done < bytes; done += piece)
    MPI_Allreduce(MPI_IN_PLACE, usr_ptr + done, (int)std::min(piece, bytes - done), MPI_BYTE, MPI_BOR, OP_MPI_WORLD);
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

/*******************************************************************************
 * Per-kernel timings of every rank, written as CSV by the root. Shared by the
 * CPU and GPU MPI libraries.
 *******************************************************************************/

void op_timings_to_csv(const char *outputFileName) {
  int comm_size;
  MPI_Comm_size(OP_MPI_WORLD, &comm_size);
  const bool root = op_is_root();

  FILE *outputFile = NULL;
  if (root) {
    outputFile = fopen(outputFileName, "w");
    if (outputFile == NULL)
      printf("ERROR: Failed to open file for writing: '%s'\n", outputFileName);
    else
      fprintf(outputFile, "rank,thread,nranks,nthreads,count,total time,plan time,mpi time,GB used,GB total,kernel name\n");
  }
  int can_write = (outputFile != NULL);
  MPI_Bcast(&can_write, 1, MPI_INT, MPI_ROOT, OP_MPI_WORLD);
  if (!can_write)
    return;

  for (int n = 0; n < OP_kern_max; n++) {
    op_mpi_barrier();
    op_kernel &k = OP_kernels[n];
    if (k.count <= 0)
      continue;
    // A translation made with the older translator keeps only the total time.
    if (k.ntimes == 1 && k.times[0] == 0.0f && k.time != 0.0f)
      k.times[0] = k.time;

    // Only the root's receive buffers are read.
    const int ranks = root ? comm_size : 0;
    std::vector<double> times((std::size_t)ranks * k.ntimes), mpi_times(ranks);
    std::vector<float> plan_times(ranks), transfers(ranks), transfers2(ranks);
    MPI_Gather(k.times, k.ntimes, MPI_DOUBLE, times.data(), k.ntimes, MPI_DOUBLE, MPI_ROOT, OP_MPI_WORLD);
    MPI_Gather(&k.plan_time, 1, MPI_FLOAT, plan_times.data(), 1, MPI_FLOAT, MPI_ROOT, OP_MPI_WORLD);
    MPI_Gather(&k.mpi_time, 1, MPI_DOUBLE, mpi_times.data(), 1, MPI_DOUBLE, MPI_ROOT, OP_MPI_WORLD);
    MPI_Gather(&k.transfer, 1, MPI_FLOAT, transfers.data(), 1, MPI_FLOAT, MPI_ROOT, OP_MPI_WORLD);
    MPI_Gather(&k.transfer2, 1, MPI_FLOAT, transfers2.data(), 1, MPI_FLOAT, MPI_ROOT, OP_MPI_WORLD);

    // The per-rank columns go on each rank's first thread row only.
    for (int p = 0; p < ranks; p++)
      for (int thr = 0; thr < k.ntimes; thr++) {
        const bool first = thr == 0;
        fprintf(outputFile, "%d,%d,%d,%d,%d,%f,%f,%f,%f,%f,%s\n", p, thr, comm_size, k.ntimes, k.count,
                times[(std::size_t)p * k.ntimes + thr], first ? plan_times[p] : 0.0f, first ? mpi_times[p] : 0.0,
                first ? transfers[p] / 1e9f : 0.0f, first ? transfers2[p] / 1e9f : 0.0f, k.name);
      }
    op_mpi_barrier();
  }

  if (root)
    fclose(outputFile);
}

