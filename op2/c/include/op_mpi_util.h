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

#ifndef __OP_MPI_UTIL_H
#define __OP_MPI_UTIL_H

#include <mpi.h>

#include <op_lib_core.h>
#include <op_util.h>

void _mpi_allgather(int *l, int *g, int size, int *recevcnts, int *displs,
    MPI_Comm comm)
{
  MPI_Allgatherv(l, size, MPI_INT, g, recevcnts, displs, MPI_INT, comm);
}

void _mpi_allgather(float *l, float *g, int size, int *recevcnts, int *displs,
    MPI_Comm comm)
{
  MPI_Allgatherv(l, size, MPI_FLOAT, g, recevcnts, displs, MPI_FLOAT, comm);
}

void _mpi_allgather(double *l, double *g, int size, int *recevcnts, int *displs,
    MPI_Comm comm)
{
  MPI_Allgatherv(l, size, MPI_DOUBLE, g, recevcnts, displs, MPI_DOUBLE, comm);
}

void _mpi_gather(int *l, int *g, int size, int *recevcnts, int *displs,
    MPI_Comm comm)
{
  MPI_Gatherv(l, size, MPI_INT, g, recevcnts, displs, MPI_INT, MPI_ROOT, comm);
}

void _mpi_gather(float *l, float *g, int size, int *recevcnts, int *displs,
    MPI_Comm comm)
{
  MPI_Gatherv(l, size, MPI_FLOAT, g, recevcnts, displs, MPI_FLOAT, MPI_ROOT, comm);
}

void _mpi_gather(double *l, double *g, int size, int *recevcnts, int *displs,
    MPI_Comm comm)
{
  MPI_Gatherv(l, size, MPI_DOUBLE, g, recevcnts, displs, MPI_DOUBLE, MPI_ROOT, comm);
}

template < typename T >
void gather_data_hdf5(op_dat dat, char* usr_ptr, int low, int high)
{
  //create new communicator
  int my_rank, comm_size;
  MPI_Comm OP_MPI_IO_WORLD;
  MPI_Comm_dup(MPI_COMM_WORLD, &OP_MPI_IO_WORLD);
  MPI_Comm_rank(OP_MPI_IO_WORLD, &my_rank);
  MPI_Comm_size(OP_MPI_IO_WORLD, &comm_size);

  //compute local number of elements in dat
  int count = dat->set->size;

  T *l_array  = (T *) xmalloc(dat->dim*(count)*sizeof(T));
  memcpy(l_array, (void *)&(dat->data[0]),
      dat->size*count);
  int l_size = count;
  size_t elem_size = dat->dim;
  int* recevcnts = (int *) xmalloc(comm_size*sizeof(int));
  int* displs = (int *) xmalloc(comm_size*sizeof(int));
  int disp = 0;
  T *g_array = 0;

  MPI_Allgather(&l_size, 1, MPI_INT, recevcnts, 1, MPI_INT, OP_MPI_IO_WORLD);

  int g_size = 0;
  for(int i = 0; i<comm_size; i++)
  {
    g_size += recevcnts[i];
    recevcnts[i] =   elem_size*recevcnts[i];
  }
  for(int i = 0; i<comm_size; i++)
  {
    displs[i] =   disp;
    disp = disp + recevcnts[i];
  }

  g_array  = (T *) xmalloc(elem_size*g_size*sizeof(T));

  //need to all-gather dat->data and copy this to the memory block pointed by usr_ptr
  _mpi_gather(l_array, g_array, l_size*elem_size, recevcnts, displs, OP_MPI_IO_WORLD);

  if(low < 0 || high > g_size -1)
  {
    printf("op_fetch_data_hdf5: Indices not within range of elements held in %s\n",
      dat->name);
    MPI_Abort(OP_MPI_IO_WORLD, -1);
  }
  memcpy((void *)usr_ptr, (void *)&g_array[low*dat->size],
  (high+1)*dat->size);

  free(l_array);free(recevcnts);free(displs);free(g_array);
  MPI_Comm_free(&OP_MPI_IO_WORLD);
}

template < typename T, const char *fmt >
void write_txtfile(op_dat dat, const char* file_name)
{
  //create new communicator for output
  int rank, comm_size;
  MPI_Comm OP_MPI_IO_WORLD;
  MPI_Comm_dup(MPI_COMM_WORLD, &OP_MPI_IO_WORLD);
  MPI_Comm_rank(OP_MPI_IO_WORLD, &rank);
  MPI_Comm_size(OP_MPI_IO_WORLD, &comm_size);

  //compute local number of elements in dat
  int count = dat->set->size;

  T *l_array = (T *) xmalloc(dat->dim*(count)*sizeof(T));
  memcpy(l_array, (void *)&(dat->data[0]),
      dat->size*count);

  int l_size = count;
  int elem_size = dat->dim;
  int* recevcnts = (int *) xmalloc(comm_size*sizeof(int));
  int* displs = (int *) xmalloc(comm_size*sizeof(int));
  int disp = 0;
  T *g_array = 0;

  MPI_Allgather(&l_size, 1, MPI_INT, recevcnts, 1, MPI_INT, OP_MPI_IO_WORLD);

  int g_size = 0;
  for(int i = 0; i<comm_size; i++)
  {
    g_size += recevcnts[i];
    recevcnts[i] = elem_size*recevcnts[i];
  }
  for(int i = 0; i<comm_size; i++)
  {
    displs[i] = disp;
    disp = disp + recevcnts[i];
  }
  if(rank==MPI_ROOT) g_array = (T *) xmalloc(elem_size*g_size*sizeof(T));
  _mpi_gather(l_array, g_array, l_size*elem_size, recevcnts, displs, OP_MPI_IO_WORLD);

  if(rank==MPI_ROOT)
  {
    FILE *fp;
    if ( (fp = fopen(file_name,"w")) == NULL) {
      printf("can't open file %s\n",file_name);
      MPI_Abort(OP_MPI_IO_WORLD, -1);
    }

    if (fprintf(fp,"%d %d\n",g_size, elem_size)<0)
    {
      printf("error writing to %s\n",file_name);
      MPI_Abort(OP_MPI_IO_WORLD, -1);
    }

    for(int i = 0; i< g_size; i++)
    {
      for(int j = 0; j < elem_size; j++ )
      {
        if (fprintf(fp,fmt,g_array[i*elem_size+j])<0)
        {
          printf("error writing to %s\n",file_name);
          MPI_Abort(OP_MPI_IO_WORLD, -1);
        }
      }
      fprintf(fp,"\n");
    }
    fclose(fp);
    free(g_array);
  }

  free(l_array);free(recevcnts);free(displs);
  MPI_Comm_free(&OP_MPI_IO_WORLD);
}
#endif /* __OP_MPI_UTIL_H */
