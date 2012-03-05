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
op_decl_dat_char ( op_set set, int dim, char const *type, int size,
              char * data, char const * name )
{
  op_dat dat = op_decl_dat_core ( set, dim, type, size, data, name );

  //transpose data
  if (strstr( type, ":soa")!= NULL) {
    char *temp_data = (char *)malloc(dat->size*set->size*sizeof(char));
    int element_size = dat->size/dat->dim;
    for (int i = 0; i < dat->dim; i++) {
      for (int j = 0; j < set->size; j++) {
        for (int c = 0; c < element_size; c++) {
          temp_data[element_size*i*set->size + element_size*j + c] = data[dat->size*j+element_size*i+c];
        }
      }
    }
    op_cpHostToDevice ( ( void ** ) &( dat->data_d ),
                          ( void ** ) &( temp_data ), dat->size * set->size );
    free(temp_data);
  } else {
    op_cpHostToDevice ( ( void ** ) &( dat->data_d ),
                        ( void ** ) &( dat->data ), dat->size * set->size );
  }

  return dat;
}

op_sparsity
op_decl_sparsity ( op_map rowmap, op_map colmap, char const * name )
{
  op_sparsity sparsity = op_decl_sparsity_core ( rowmap, colmap, name );

  op_mvHostToDevice ( (void **)&(sparsity->rowptr),
                      sizeof(int) * (sparsity->nrows + 1) );

  op_mvHostToDevice ( (void **)&(sparsity->colidx),
                      sizeof(int) * (sparsity->total_nz));

  return sparsity;
}

op_mat
op_decl_mat ( op_sparsity sparsity, int dim, char const * type,
              int type_size, char const * name )
{
  op_mat mat = op_decl_mat_core ( sparsity->rowmap->to, sparsity->colmap->to,
                                  dim, type, type_size, name );

  mat->sparsity = sparsity;
  op_callocDevice( (void **)&(mat->data),
                   type_size * mat->sparsity->total_nz );

  return mat;
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
op_arg_mat ( op_mat mat, int idx1, op_map map1, int idx2, op_map map2, int dim,
             char const * type, op_access acc )
{
  return op_arg_mat_core ( mat, idx1, map1, idx2, map2, dim, type, acc );
}

op_arg
op_arg_gbl_char ( char * data, int dim, const char *type, int size, op_access acc )
{
  return op_arg_gbl_core ( data, dim, type, size, acc );
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

int op_get_size(op_set set)
{
  return set->size;
}

void op_printf(const char* format, ...)
{
  va_list argptr;
  va_start(argptr, format);
  vprintf(format, argptr);
  va_end(argptr);
}

void op_timers(double * cpu, double * et)
{
  op_timers_core(cpu,et);
}

void op_exit()
{
  op_cuda_exit();            // frees dat_d memory
  op_rt_exit();              // frees plan memory
  op_exit_core();            // frees lib core variables
}

void op_timing_output()
{
  op_timing_output_core();
}
