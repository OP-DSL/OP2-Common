/*
  Open source copyright declaration based on BSD open source template:
  http://www.opensource.org/licenses/bsd-license.php

* Copyright (c) 2009-2011, Mike Giles
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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math_constants.h>

#include <op_lib_c.h>
#include <op_lib_core.h>
#include <op_rt_support.h>
#include <op_cuda_rt_support.h>

#include <op_lib_mpi.h>
#include <op_util.h>

// Small re-declaration to avoid using struct in the C version.
// This is due to the different way in which C and C++ see structs
   
typedef struct cudaDeviceProp cudaDeviceProp_t;

// arrays for global constants and reductions

int OP_consts_bytes = 0,
    OP_reduct_bytes = 0;
    
char * OP_consts_h,
     * OP_consts_d,
     * OP_reduct_h,
     * OP_reduct_d;

     
//
//export lists on the device
//
int** export_exec_list_d;
int** export_nonexec_list_d;

//
// CUDA utility functions
//

void
__cudaSafeCall ( cudaError_t err, const char * file, const int line )
{
  if ( cudaSuccess != err )
  {
    printf ( "%s(%i) : cutilSafeCall() Runtime API error : %s.\n",
             file, line, cudaGetErrorString ( err ) );
    exit ( -1 );
  }
}

void
__cutilCheckMsg ( const char * errorMessage, const char * file,
                  const int line )
{
  cudaError_t err = cudaGetLastError (  );
  if ( cudaSuccess != err )
  {
    printf ( "%s(%i) : cutilCheckMsg() error : %s : %s.\n",
             file, line, errorMessage, cudaGetErrorString ( err ) );
    exit ( -1 );
  }
}

/*
void cutilDeviceInit ( int argc, char ** argv)
{
  int deviceCount;
  cutilSafeCall ( cudaGetDeviceCount ( &deviceCount ) );
  if ( deviceCount == 0 )
  {
    printf ( "cutil error: no devices supporting CUDA\n" );
    exit ( -1 );
  }
  cudaError_t error;
  int deviceId = 0;
  for (int i = 0; i< deviceCount;i++) {
	error = cudaSetDevice(i);
	float *test;
	error = cudaMalloc((void **)&test, sizeof(float));
	deviceId = i;
	if (error == cudaSuccess) {		
		cudaFree(test);
		break;
	} else {
	    cudaThreadExit ( );
	}
  }
  if (error != cudaSuccess) {
	printf ( "Could not select CUDA device\n" );
    exit ( -1 );
  }
	
  cudaDeviceProp_t deviceProp;
  cutilSafeCall ( cudaGetDeviceProperties ( &deviceProp, deviceId ) );

  printf ( "\n Using CUDA device: %d %s\n", deviceId, deviceProp.name );
}
*/



//void cutilDeviceInit_mpi( int argc, char ** argv, int my_rank )
void cutilDeviceInit( int argc, char ** argv)
{
  int deviceCount;
  cutilSafeCall ( cudaGetDeviceCount ( &deviceCount ) );
  if ( deviceCount == 0 )
  {
    printf ( "cutil error: no devices supporting CUDA\n" );
    exit ( -1 );
  }
  float *test;
  cudaError_t error = cudaMalloc((void **)&test, sizeof(float));
  if (error != cudaSuccess)  {
      printf ( "Could not select CUDA device\n" );
      exit ( -1 );
  }
  int deviceId = -1;
  cudaGetDevice(&deviceId);
  cudaFree(test);
  cudaDeviceProp_t deviceProp;
  cutilSafeCall ( cudaGetDeviceProperties ( &deviceProp, deviceId ) );

  printf ( "\n Using CUDA device: %d %s\n",deviceId, deviceProp.name );
  //cutilSafeCall ( cudaSetDevice ( my_rank ) );
}


//
// routines to move arrays to/from GPU device
//

void
op_mvHostToDevice ( void ** map, int size )
{
  void *tmp;
  cutilSafeCall ( cudaMalloc ( &tmp, size ) );
  cutilSafeCall ( cudaMemcpy ( tmp, *map, size,
                               cudaMemcpyHostToDevice ) );
  cutilSafeCall ( cudaThreadSynchronize (  ) );
  free ( *map );
  *map = tmp;
}

void
op_cpHostToDevice ( void ** data_d, void ** data_h, int size )
{
  cutilSafeCall ( cudaMalloc ( data_d, size ) );
  cutilSafeCall ( cudaMemcpy ( *data_d, *data_h, size,
                               cudaMemcpyHostToDevice ) );
  cutilSafeCall ( cudaThreadSynchronize (  ) );
}

void
op_fetch_data ( op_dat dat )
{
  cutilSafeCall ( cudaMemcpy ( dat->data, dat->data_d,
                               dat->size * dat->set->size,
                               cudaMemcpyDeviceToHost ) );
  cutilSafeCall ( cudaThreadSynchronize (  ) );
}



op_plan *
op_plan_get ( char const * name, op_set set, int part_size,
              int nargs, op_arg * args, int ninds, int *inds )
{
	return op_plan_get_offset ( name, set, 0, part_size,
		nargs, args, ninds, inds );
}

op_plan *
op_plan_get_offset ( char const * name, op_set set, int set_offset, int part_size,
              int nargs, op_arg * args, int ninds, int *inds )
{
  op_plan *plan = op_plan_core ( name, set, set_offset, part_size,
                                 nargs, args, ninds, inds );
  int set_size = plan->set->size + plan->set->exec_size +  plan->set->nonexec_size;
  
  if ( plan->count == 1 )
  {
    for ( int m = 0; m < ninds; m++ )
      op_mvHostToDevice ( ( void ** ) &( plan->ind_maps[m] ),
                          sizeof ( int ) * plan->nindirect[m] );

    for ( int m = 0; m < nargs; m++ )
      if ( plan->loc_maps[m] != NULL )
        op_mvHostToDevice ( ( void ** ) &( plan->loc_maps[m] ),
                            sizeof ( short ) * set_size );

    op_mvHostToDevice ( ( void ** ) &( plan->ind_sizes ),
                          sizeof ( int ) * plan->nblocks * plan->ninds );
    op_mvHostToDevice ( ( void ** ) &( plan->ind_offs ),
                          sizeof ( int ) * plan->nblocks * plan->ninds );
    op_mvHostToDevice ( ( void ** ) &( plan->nthrcol ),
                          sizeof ( int ) * plan->nblocks );
    op_mvHostToDevice ( ( void ** ) &( plan->thrcol ),
                          sizeof ( int ) * set_size );
    op_mvHostToDevice ( ( void ** ) &( plan->offset ),
                          sizeof ( int ) * plan->nblocks );
    op_mvHostToDevice ( ( void ** ) &( plan->nelems ),
                          sizeof ( int ) * plan->nblocks );
    op_mvHostToDevice ( ( void ** ) &( plan->blkmap ),
                          sizeof ( int ) * plan->nblocks );
  }

  return plan;
}

void
op_cuda_exit ( )
{
  for ( int i = 0; i < OP_dat_index; i++ )
  {
    cutilSafeCall ( cudaFree ( OP_dat_list[i]->data_d ) );
  }

  cudaThreadExit ( );
}


//
// routines to resize constant/reduct arrays, if necessary
//

void
reallocConstArrays ( int consts_bytes )
{
  if ( consts_bytes > OP_consts_bytes )
  {
    if ( OP_consts_bytes > 0 )
    {
      free ( OP_consts_h );
      cutilSafeCall ( cudaFree ( OP_consts_d ) );
    }
    OP_consts_bytes = 4 * consts_bytes;  // 4 is arbitrary, more than needed
    OP_consts_h = ( char * ) malloc ( OP_consts_bytes );
    cutilSafeCall ( cudaMalloc ( ( void ** ) &OP_consts_d,
                                 OP_consts_bytes ) );
  }
}

void
reallocReductArrays ( int reduct_bytes )
{
  if ( reduct_bytes > OP_reduct_bytes )
  {
    if ( OP_reduct_bytes > 0 )
    {
      free ( OP_reduct_h );
      cutilSafeCall ( cudaFree ( OP_reduct_d ) );
    }
    OP_reduct_bytes = 4 * reduct_bytes;  // 4 is arbitrary, more than needed
    OP_reduct_h = ( char * ) malloc ( OP_reduct_bytes );
    cutilSafeCall ( cudaMalloc ( ( void ** ) &OP_reduct_d,
                                 OP_reduct_bytes ) );
  }
}

//
// routines to move constant/reduct arrays
//

void
mvConstArraysToDevice ( int consts_bytes )
{
  cutilSafeCall ( cudaMemcpy ( OP_consts_d, OP_consts_h, consts_bytes,
                               cudaMemcpyHostToDevice ) );
  cutilSafeCall ( cudaThreadSynchronize (  ) );
}

void
mvReductArraysToDevice ( int reduct_bytes )
{
  cutilSafeCall ( cudaMemcpy ( OP_reduct_d, OP_reduct_h, reduct_bytes,
                               cudaMemcpyHostToDevice ) );
  cutilSafeCall ( cudaThreadSynchronize (  ) );
}

void
mvReductArraysToHost ( int reduct_bytes )
{
  cutilSafeCall ( cudaMemcpy ( OP_reduct_h, OP_reduct_d, reduct_bytes,
                               cudaMemcpyDeviceToHost ) );
  cutilSafeCall ( cudaThreadSynchronize (  ) );
}



extern void gather_data_to_buffer(op_arg arg, halo_list exp_exec_list, 
    halo_list exp_nonexec_list);

int exchange_halo_cuda(op_arg arg)
{
    op_dat dat = arg.dat;
	
    if((arg.idx != -1) && (arg.acc == OP_READ || arg.acc == OP_RW ) &&
    	(dirtybit[dat->index] == 1))
    {
    
    	//printf("Exchanging Halo of data array %10s\n",dat->name);
	halo_list imp_exec_list = OP_import_exec_list[dat->set->index];
	halo_list imp_nonexec_list = OP_import_nonexec_list[dat->set->index];
	    
	halo_list exp_exec_list = OP_export_exec_list[dat->set->index];
	halo_list exp_nonexec_list = OP_export_nonexec_list[dat->set->index];

	//-------first exchange exec elements related to this data array--------
	
	//sanity checks
	if(compare_sets(imp_exec_list->set,dat->set) == 0)
	{ 
	    printf("Error: Import list and set mismatch\n"); 
	    MPI_Abort(OP_MPI_WORLD, 2);
	}
	if(compare_sets(exp_exec_list->set,dat->set) == 0)
	{
	    printf("Error: Export list and set mismatch\n"); 
	    MPI_Abort(OP_MPI_WORLD, 2);
	}
	
	
	gather_data_to_buffer(arg, exp_exec_list, exp_nonexec_list);
	
	cutilSafeCall( cudaMemcpy ( OP_mpi_buffer_list[dat->index]-> buf_exec, 
	    arg.dat->buffer_d, exp_exec_list->size*arg.dat->size, cudaMemcpyDeviceToHost ) );
	
	cutilSafeCall( cudaMemcpy ( OP_mpi_buffer_list[dat->index]-> buf_nonexec, 
	    arg.dat->buffer_d+exp_exec_list->size*arg.dat->size, exp_nonexec_list->size*arg.dat->size, 
	    cudaMemcpyDeviceToHost ) );
        
	cutilSafeCall(cudaThreadSynchronize(  ));
	
	for(int i=0; i<exp_exec_list->ranks_size; i++) {
	    //printf("export from %d to %d data %10s, number of elements of size %d | sending:\n ",
	    //  	      my_rank, exp_exec_list->ranks[i], dat->name,exp_exec_list->sizes[i]);
	    MPI_Isend(&OP_mpi_buffer_list[dat->index]->
	    	buf_exec[exp_exec_list->disps[i]*dat->size],
	    	dat->size*exp_exec_list->sizes[i],
	    	MPI_CHAR, exp_exec_list->ranks[i],
	    	dat->index, OP_MPI_WORLD, 
	    	&OP_mpi_buffer_list[dat->index]->
	    	s_req[OP_mpi_buffer_list[dat->index]->s_num_req++]);	    
	}
	
	
	int init = dat->set->size*dat->size;	
	for(int i=0; i < imp_exec_list->ranks_size; i++) {
	    //printf("import on to %d from %d data %10s, number of elements of size %d | recieving:\n ",
	    //  	  my_rank, imp_exec_list.ranks[i], dat.name, imp_exec_list.sizes[i]);
	    MPI_Irecv(&(OP_dat_list[dat->index]->
	    	data[init+imp_exec_list->disps[i]*dat->size]),
	    	dat->size*imp_exec_list->sizes[i], 
	    	MPI_CHAR, imp_exec_list->ranks[i], 
	    	dat->index, OP_MPI_WORLD, 
	    	&OP_mpi_buffer_list[dat->index]->
	    	r_req[OP_mpi_buffer_list[dat->index]->r_num_req++]);
	}
	
	
	//-----second exchange nonexec elements related to this data array------
	//sanity checks
	if(compare_sets(imp_nonexec_list->set,dat->set) == 0)
	{ 
	    printf("Error: Non-Import list and set mismatch");
	    MPI_Abort(OP_MPI_WORLD, 2);   
	}
	if(compare_sets(exp_nonexec_list->set,dat->set)==0)
	{
	    printf("Error: Non-Export list and set mismatch"); 
	    MPI_Abort(OP_MPI_WORLD, 2);
	}
	
	for(int i=0; i<exp_nonexec_list->ranks_size; i++) {
	    MPI_Isend(&OP_mpi_buffer_list[dat->index]->
	    	buf_nonexec[exp_nonexec_list->disps[i]*dat->size],
	    	dat->size*exp_nonexec_list->sizes[i],
	    	MPI_CHAR, exp_nonexec_list->ranks[i],
	    	dat->index, OP_MPI_WORLD, 
	    	&OP_mpi_buffer_list[dat->index]->
	    	s_req[OP_mpi_buffer_list[dat->index]->s_num_req++]);	    	
	}
	
	int nonexec_init = (dat->set->size+imp_exec_list->size)*dat->size;	
	for(int i=0; i<imp_nonexec_list->ranks_size; i++) {
	    MPI_Irecv(&(OP_dat_list[dat->index]->
	    	data[nonexec_init+imp_nonexec_list->disps[i]*dat->size]),
	    	dat->size*imp_nonexec_list->sizes[i], 
	    	MPI_CHAR, imp_nonexec_list->ranks[i], 
	    	dat->index, OP_MPI_WORLD, 
	    	&OP_mpi_buffer_list[dat->index]->
	    	r_req[OP_mpi_buffer_list[dat->index]->r_num_req++]);
	}
	//clear dirty bit
	dirtybit[dat->index] = 0;
	return 1;
    }
    return 0;
    
}


void wait_all_cuda(op_arg arg)
{
    	op_dat dat = arg.dat;
    	MPI_Waitall(OP_mpi_buffer_list[dat->index]->s_num_req,
    	    OP_mpi_buffer_list[dat->index]->s_req,
    	    MPI_STATUSES_IGNORE );
    	MPI_Waitall(OP_mpi_buffer_list[dat->index]->r_num_req,
    	    OP_mpi_buffer_list[dat->index]->r_req,
    	    MPI_STATUSES_IGNORE );
    	OP_mpi_buffer_list[dat->index]->s_num_req = 0;
    	OP_mpi_buffer_list[dat->index]->r_num_req = 0;
    	
    	int init = dat->set->size*dat->size;
    	cutilSafeCall(cudaMemcpy (dat->data_d + init, dat->data + init, 
    	    OP_import_exec_list[dat->set->index]->size*arg.dat->size, cudaMemcpyHostToDevice  ));
    	
    	int nonexec_init = (dat->set->size+OP_import_exec_list[dat->set->index]->size)*dat->size;
    	cutilSafeCall( cudaMemcpy (dat->data_d + nonexec_init, dat->data + nonexec_init, 
    	    OP_import_nonexec_list[dat->set->index]->size*arg.dat->size, cudaMemcpyHostToDevice ));

	cutilSafeCall(cudaThreadSynchronize ());
}
