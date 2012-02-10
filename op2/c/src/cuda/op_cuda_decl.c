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
// This file implements the OP2 user-level functions for the CUDA backend
// 

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <op_lib_c.h>
#include <op_cuda_rt_support.h>
#include <op_rt_support.h>

//
// CUDA-specific OP2 functions
//

void
op_init ( int argc, char ** argv, int diags )
{
  op_init_core ( argc, argv, diags );

#if CUDART_VERSION < 3020
#error : "must be compiled using CUDA 3.2 or later"
#endif

#ifdef CUDA_NO_SM_13_DOUBLE_INTRINSICS
#warning : " *** no support for double precision arithmetic *** "
#endif

  cutilDeviceInit ( argc, argv );

// 
// The following call is only made in the C version of OP2,
// as it causes memory trashing when called from Fortran.
// \warning add -DSET_CUDA_CACHE_CONFIG to compiling line
// for this file when implementing C OP2.
//

#ifdef SET_CUDA_CACHE_CONFIG
  cutilSafeCall ( cudaThreadSetCacheConfig ( cudaFuncCachePreferShared ) );
#endif

  printf ( "\n 16/48 L1/shared \n" );
}

op_dat
op_decl_dat ( op_set set, int dim, char const *type, int size,
              char * data, char const * name )
{
  op_dat dat = op_decl_dat_core ( set, dim, type, size, data, name );

  op_cpHostToDevice ( ( void ** ) &( dat->data_d ),
                      ( void ** ) &( dat->data ), dat->size * set->size );

  return dat;
}

op_set
op_decl_set ( int size, char const * name )
{
  return op_decl_set_core ( size, name );
}

op_map
op_decl_map ( op_set from, op_set to, int dim, int * imap, char const * name )
{
  return op_decl_map_core ( from, to, dim, imap, name );
}

op_arg
op_arg_dat ( op_dat dat, int idx, op_map map, int dim, char const * type,
             op_access acc )
{
  return op_arg_dat_core ( dat, idx, map, dim, type, acc );
}

op_arg
op_arg_gbl ( char * data, int dim, const char *type, op_access acc )
{
  return op_arg_gbl ( data, dim, type, acc );
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



void op_exit()
{
  op_cuda_exit();            // frees dat_d memory
  op_rt_exit();              // frees plan memory
  op_exit_core();            // frees lib core variables
}
