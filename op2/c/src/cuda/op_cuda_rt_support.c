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

void
cutilDeviceInit ( int argc, char ** argv )
{
  int deviceCount;
  cutilSafeCall ( cudaGetDeviceCount ( &deviceCount ) );
  if ( deviceCount == 0 )
  {
    printf ( "cutil error: no devices supporting CUDA\n" );
    exit ( -1 );
  }

  cudaDeviceProp_t deviceProp;
  cutilSafeCall ( cudaGetDeviceProperties ( &deviceProp, 0 ) );

  printf ( "\n Using CUDA device: %s\n", deviceProp.name );
  cutilSafeCall ( cudaSetDevice ( 0 ) );
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

  if ( plan->count == 1 )
  {
    for ( int m = 0; m < ninds; m++ )
      op_mvHostToDevice ( ( void ** ) &( plan->ind_maps[m] ),
                          sizeof ( int ) * plan->nindirect[m] );

    for ( int m = 0; m < nargs; m++ )
      if ( plan->loc_maps[m] != NULL )
        op_mvHostToDevice ( ( void ** ) &( plan->loc_maps[m] ),
                            sizeof ( short ) * plan->set->size );

    op_mvHostToDevice ( ( void ** ) &( plan->ind_sizes ),
                          sizeof ( int ) * plan->nblocks * plan->ninds );
    op_mvHostToDevice ( ( void ** ) &( plan->ind_offs ),
                          sizeof ( int ) * plan->nblocks * plan->ninds );
    op_mvHostToDevice ( ( void ** ) &( plan->nthrcol ),
                          sizeof ( int ) * plan->nblocks );
    op_mvHostToDevice ( ( void ** ) &( plan->thrcol ),
                          sizeof ( int ) * plan->set->size );
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
