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

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <op_lib_core.h>
#include <op_cuda_rt_support.h>
#include <op_rt_support.h>

#include <op_lib_c.h>
#include <op_util.h>
#include <op_lib_mpi.h>

//
// CUDA-specific OP2 functions
//

void
op_init ( int argc, char ** argv, int diags)
{
  int flag = 0;
  MPI_Initialized(&flag);
  if(!flag)
  {
      MPI_Init(&argc, &argv);
  }

  op_init_core ( argc, argv, diags );

#if CUDART_VERSION < 3020
#error : "must be compiled using CUDA 3.2 or later"
#endif

#ifdef CUDA_NO_SM_13_DOUBLE_INTRINSICS
#warning : " *** no support for double precision arithmetic *** "
#endif

  cutilDeviceInit( argc, argv);

//
// The following call is only made in the C version of OP2,
// as it causes memory trashing when called from Fortran.
// \warning add -DSET_CUDA_CACHE_CONFIG to compiling line
// for this file when implementing C OP2.
//

#ifdef SET_CUDA_CACHE_CONFIG
  cutilSafeCall ( cudaDeviceSetCacheConfig ( cudaFuncCachePreferShared ) );
#endif

  printf ( "\n 16/48 L1/shared \n" );
}

op_dat op_decl_dat_char ( op_set set, int dim, char const *type, int size,
              char * data, char const * name )
{
  char* d = (char*) malloc(set->size*dim*size);
  if (d == NULL) {
    printf ( " op_decl_dat_char error -- error allocating memory to dat\n" );
    exit ( -1 );
  }

  memcpy(d, data, set->size*dim*size*sizeof(char));
  op_dat out_dat = op_decl_dat_core ( set, dim, type, size, d, name );
  out_dat-> user_managed = 0;
  return out_dat;
}


op_dat op_decl_dat_temp_char(op_set set, int dim, char const * type, int size, char const *name )
{
  char* data = NULL;
  op_dat dat = op_decl_dat_temp_core ( set, dim, type, size, data, name );

  //create empty data block to assign to this temporary dat (including the halos)
  int set_size = set->size + OP_import_exec_list[set->index]->size +
  OP_import_nonexec_list[set->index]->size;

  dat->data = (char*) calloc(set_size*dim*size, 1); //initialize data bits to 0
  dat-> user_managed = 0;

  //transpose
  if (strstr( dat->type, ":soa")!= NULL) {
    cutilSafeCall ( cudaMalloc ( ( void ** ) &( dat->buffer_d_r ),
      dat->size * (OP_import_exec_list[set->index]->size +
      OP_import_nonexec_list[set->index]->size) ));
  }

  op_cpHostToDevice ( ( void ** ) &( dat->data_d ),
                    ( void ** ) &( dat->data ), dat->size * set_size );


  //need to allocate mpi_buffers for this new temp_dat
  op_mpi_buffer mpi_buf= (op_mpi_buffer)xmalloc(sizeof(op_mpi_buffer_core));

  halo_list exec_e_list = OP_export_exec_list[set->index];
  halo_list nonexec_e_list = OP_export_nonexec_list[set->index];

  mpi_buf->buf_exec = (char *)xmalloc((exec_e_list->size)*dat->size);
  mpi_buf->buf_nonexec = (char *)xmalloc((nonexec_e_list->size)*dat->size);

  halo_list exec_i_list = OP_import_exec_list[set->index];
  halo_list nonexec_i_list = OP_import_nonexec_list[set->index];

  mpi_buf->s_req = (MPI_Request *)xmalloc(sizeof(MPI_Request)*
      (exec_e_list->ranks_size + nonexec_e_list->ranks_size));
  mpi_buf->r_req = (MPI_Request *)xmalloc(sizeof(MPI_Request)*
      (exec_i_list->ranks_size + nonexec_i_list->ranks_size));

  mpi_buf->s_num_req = 0;
  mpi_buf->r_num_req = 0;

  dat->mpi_buffer = mpi_buf;

  //need to allocate device buffers for mpi comms for this new temp_dat
  cutilSafeCall ( cudaMalloc ( ( void ** ) &( dat->buffer_d ),
      dat->size * (OP_export_exec_list[set->index]->size +
      OP_export_nonexec_list[set->index]->size) ));

  return dat;

}

int op_free_dat_temp_char ( op_dat dat )
{
  //need to free mpi_buffers use in this op_dat
  free(((op_mpi_buffer)(dat->mpi_buffer))->buf_exec);
  free(((op_mpi_buffer)(dat->mpi_buffer))->buf_nonexec);
  free(((op_mpi_buffer)(dat->mpi_buffer))->s_req);
  free(((op_mpi_buffer)(dat->mpi_buffer))->r_req);
  free(dat->mpi_buffer);

  //need to free device buffers used in mpi comms
  cutilSafeCall (cudaFree(dat->buffer_d));

  if (strstr( dat->type, ":soa")!= NULL) {
    cutilSafeCall (cudaFree(dat->buffer_d_r));
  }

  //free data on device
  cutilSafeCall (cudaFree(dat->data_d));
  return op_free_dat_temp_core (dat);
}

void op_mv_halo_device(op_set set, op_dat dat)
{
  int set_size = set->size + OP_import_exec_list[set->index]->size +
  OP_import_nonexec_list[set->index]->size;

  if (strstr( dat->type, ":soa")!= NULL) {
    char *temp_data = (char *)malloc(dat->size*set_size*sizeof(char));
    int element_size = dat->size/dat->dim;
    for (int i = 0; i < dat->dim; i++) {
      for (int j = 0; j < set_size; j++) {
        for (int c = 0; c < element_size; c++) {
          temp_data[element_size*i*set_size + element_size*j + c] = dat->data[dat->size*j+element_size*i+c];
        }
      }
    }
    op_cpHostToDevice ( ( void ** ) &( dat->data_d ),
                        ( void ** ) &( temp_data ), dat->size * set_size );
    free(temp_data);

    cutilSafeCall ( cudaMalloc ( ( void ** ) &( dat->buffer_d_r ),
      dat->size * (OP_import_exec_list[set->index]->size +
      OP_import_nonexec_list[set->index]->size) ));

  } else {
    op_cpHostToDevice ( ( void ** ) &( dat->data_d ),
                        ( void ** ) &( dat->data ), dat->size * set_size );
  }

  cutilSafeCall ( cudaMalloc ( ( void ** ) &( dat->buffer_d ),
      dat->size * (OP_export_exec_list[set->index]->size +
      OP_export_nonexec_list[set->index]->size) ));
}

void op_mv_halo_list_device()
{
  export_exec_list_d = (int **)xmalloc(sizeof(int*)*OP_set_index);

  for(int s=0; s<OP_set_index; s++) { //for each set
      op_set set=OP_set_list[s];

      op_cpHostToDevice ( ( void ** ) &( export_exec_list_d[set->index] ),
                          ( void ** ) &(OP_export_exec_list[set->index]->list),
                          OP_export_exec_list[set->index]->size * sizeof(int) );
  }

  export_nonexec_list_d = (int **)xmalloc(sizeof(int*)*OP_set_index);

  for(int s=0; s<OP_set_index; s++) { //for each set
      op_set set=OP_set_list[s];

      op_cpHostToDevice ( ( void ** ) &( export_nonexec_list_d[set->index] ),
                      ( void ** ) &(OP_export_nonexec_list[set->index]->list),
                      OP_export_nonexec_list[set->index]->size * sizeof(int) );
  }
}

op_set op_decl_set(int size, char const * name )
{
  return op_decl_set_core ( size, name );
}

op_map op_decl_map(op_set from, op_set to, int dim, int * imap, char const * name )
{
  int* m = (int*) malloc(from->size*dim*sizeof(int));
  memcpy(m, imap, from->size*dim*sizeof(int));
  return op_decl_map_core ( from, to, dim, m, name );
  //return op_decl_map_core ( from, to, dim, imap, name );
}

op_arg
op_arg_dat ( op_dat dat, int idx, op_map map, int dim, char const * type,
             op_access acc )
{
  return op_arg_dat_core ( dat, idx, map, dim, type, acc );
}

op_arg
op_opt_arg_dat ( int opt, op_dat dat, int idx, op_map map, int dim, char const * type,
             op_access acc )
{
  return op_opt_arg_dat_core ( opt, dat, idx, map, dim, type, acc );
}

op_arg
op_arg_gbl_char ( char * data, int dim, const char *type, int size, op_access acc )
{
  return op_arg_gbl_core ( data, dim, type, size, acc );
}

void op_printf(const char* format, ...)
{
  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
  if(my_rank==MPI_ROOT)
  {
    va_list argptr;
    va_start(argptr, format);
    vprintf(format, argptr);
    va_end(argptr);
  }
}

void op_timers(double * cpu, double * et)
{
  MPI_Barrier(MPI_COMM_WORLD);
  op_timers_core(cpu,et);
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
  cutilSafeCall ( cudaMemcpyToSymbol ( name, dat, dim * size, 0,
                                       cudaMemcpyHostToDevice ) );
}
*/

void
op_exit (  )
{
  //need to free buffer_d used for mpi comms in each op_dat
  op_dat_entry *item;
  TAILQ_FOREACH(item, &OP_dat_list, entries)
  {
    if (strstr( item->dat->type, ":soa")!= NULL) {
      cutilSafeCall (cudaFree((item->dat)->buffer_d_r));
    }
		cutilSafeCall (cudaFree((item->dat)->buffer_d));
	}

  op_mpi_exit();
  op_cuda_exit();            // frees dat_d memory
  op_rt_exit();              // frees plan memory
  op_exit_core();            // frees lib core variables

  int flag = 0;
  MPI_Finalized(&flag);
  if(!flag)
    MPI_Finalize();
}

void op_timing_output()
{
  op_timing_output_core();
}

void op_print_dat_to_binfile(op_dat dat, const char *file_name)
{
  //need to get data from GPU
  op_cuda_get_data(dat);

  //rearrange data backe to original order in mpi
  op_dat temp = op_mpi_get_data(dat);
  print_dat_to_binfile_mpi(temp, file_name);

  free(temp->data);
  free(temp->set);
  free(temp);
}

void op_print_dat_to_txtfile(op_dat dat, const char *file_name)
{
  //need to get data from GPU
  op_cuda_get_data(dat);

  //rearrange data backe to original order in mpi
  op_dat temp = op_mpi_get_data(dat);
  print_dat_to_txtfile_mpi(temp, file_name);

  free(temp->data);
  free(temp->set);
  free(temp);
}

void op_fetch_data_char ( op_dat dat, char * usr_ptr )
{
  //need to get data from GPU
  op_cuda_get_data(dat);

  //rearrange data backe to original order in mpi
  op_dat temp = op_mpi_get_data(dat);

  //copy data into usr_ptr
  memcpy((void *)usr_ptr, (void *)temp->data, temp->set->size*temp->size);
  free(temp->data);
  free(temp->set);
  free(temp);
}

op_dat op_fetch_data_file_char(op_dat dat)
{
  //need to get data from GPU
  op_cuda_get_data(dat);
  //rearrange data backe to original order in mpi
  return op_mpi_get_data(dat);
}

void op_fetch_data_hdf5_char(op_dat dat, char * usr_ptr, int low, int high)
{
  //need to get data from GPU
  op_cuda_get_data(dat);

  //rearrange data backe to original order in mpi
  op_dat temp = op_mpi_get_data(dat);

  //do allgather on temp->data and copy it to memory block pointed to by use_ptr
  fetch_data_hdf5(dat, usr_ptr, low, high);

  free(temp->data);
  free(temp->set);
  free(temp);
}
