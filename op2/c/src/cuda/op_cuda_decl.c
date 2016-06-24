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
op_init_soa ( int argc, char ** argv, int diags, int soa )
{
  OP_auto_soa = soa;
  op_init(argc,argv,diags);
}

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
  cutilSafeCall ( cudaDeviceSetCacheConfig ( cudaFuncCachePreferShared ) );
#endif

  printf ( "\n 16/48 L1/shared \n" );
}

void
op_mpi_init ( int argc, char ** argv, int diags, int global, int local )
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
  cutilSafeCall ( cudaDeviceSetCacheConfig ( cudaFuncCachePreferShared ) );
#endif

  printf ( "\n 16/48 L1/shared \n" );
}

void
op_mpi_init_soa ( int argc, char ** argv, int diags, int global, int local, int soa )
{
  OP_auto_soa = soa;
  op_mpi_init(argc,argv,diags,global,local);
}

op_dat
op_decl_dat_char ( op_set set, int dim, char const *type, int size,
              char * data, char const * name )
{
  op_dat dat = op_decl_dat_core ( set, dim, type, size, data, name );

  //transpose data
  if (strstr( type, ":soa")!= NULL || (OP_auto_soa && dim > 1)) {
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


op_dat
op_decl_dat_temp_char ( op_set set, int dim, char const *type, int size, char const * name )
{
  char* data = NULL;
  op_dat dat = op_decl_dat_temp_core ( set, dim, type, size, data, name );

  dat->data = (char*) calloc(set->size*dim*size, 1); //initialize data bits to 0
  dat-> user_managed = 0;

  //transpose data
  if (strstr( type, ":soa")!= NULL || (OP_auto_soa && dim > 1)) {
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

int op_free_dat_temp_char ( op_dat dat )
{
  //free data on device
  cutilSafeCall (cudaFree(dat->data_d));

  return op_free_dat_temp_core (dat);
}

op_set
op_decl_set ( int size, char const * name )
{
  return op_decl_set_core ( size, name );
}

op_map
op_decl_map ( op_set from, op_set to, int dim, int * imap, char const * name )
{
  op_map map = op_decl_map_core ( from, to, dim, imap, name );
  int set_size = map->from->size;
  int *temp_map = (int *)malloc(map->dim*set_size*sizeof(int));
  for (int i = 0; i < map->dim; i++) {
    for (int j = 0; j < set_size; j++) {
      temp_map[i*set_size + j] = map->map[map->dim*j+i];
    }
  }
  op_cpHostToDevice ( ( void ** ) &( map->map_d ),
                      ( void ** ) &( temp_map ), map->dim * set_size * sizeof(int) );
  free(temp_map);
  return map;
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

void op_print(const char* line)
{
  printf("%s\n",line);
}

void op_timers(double * cpu, double * et)
{
  op_timers_core(cpu,et);
}

int getSetSizeFromOpArg (op_arg * arg)
{
  return arg->opt ? arg->dat->set->size : 0;
}

void op_renumber(op_map base) {
  (void)base;
}

int getHybridGPU() {
  return OP_hybrid_gpu;
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
  printf("Total plan time: %8.4f\n", OP_plan_time);
}

void op_print_dat_to_binfile(op_dat dat, const char *file_name)
{
  //need to get data from GPU
  op_cuda_get_data(dat);
  op_print_dat_to_binfile_core(dat, file_name);
}

void op_print_dat_to_txtfile(op_dat dat, const char *file_name)
{
  //need to get data from GPU
  op_cuda_get_data(dat);
  op_print_dat_to_txtfile_core(dat, file_name);
}

void op_upload_all ()
{
  op_dat_entry *item;
  TAILQ_FOREACH(item, &OP_dat_list, entries) {
    op_dat dat = item->dat;
    int set_size = dat->set->size;
    if (dat->data_d) {
      if (strstr( dat->type, ":soa")!= NULL || (OP_auto_soa && dat->dim > 1)) {
        char *temp_data = (char *)malloc(dat->size*set_size*sizeof(char));
        int element_size = dat->size/dat->dim;
        for (int i = 0; i < dat->dim; i++) {
          for (int j = 0; j < set_size; j++) {
            for (int c = 0; c < element_size; c++) {
              temp_data[element_size*i*set_size + element_size*j + c] = dat->data[dat->size*j+element_size*i+c];
            }
          }
        }
        cutilSafeCall( cudaMemcpy(dat->data_d, temp_data, dat->size * set_size, cudaMemcpyHostToDevice ));
        dat->dirty_hd = 0;
        free(temp_data);
      } else {
        cutilSafeCall( cudaMemcpy(dat->data_d, dat->data, dat->size * set_size, cudaMemcpyHostToDevice ));
        dat->dirty_hd = 0;
      }
    }
  }
}

void op_fetch_data_char ( op_dat dat, char * usr_ptr )
{
  op_cuda_get_data(dat);
  //need to copy data into memory pointed to by usr_ptr
  memcpy((void *)usr_ptr, (void *)dat->data, dat->set->size*dat->size);
}

void
op_fetch_data_idx_char ( op_dat dat, char * usr_ptr, int low, int high)
{
  op_cuda_get_data(dat);
  if(low < 0 || high > dat->set->size -1)
  {
    printf("op_fetch_data: Indices not within range of elements held in %s\n",
      dat->name);
    exit(2);
  }
  //need to copy data into memory pointed to by usr_ptr
  memcpy((void *)usr_ptr, (void *)&dat->data[low*dat->size],
    (high+1)*dat->size);
}


//Dummy for cuda compile

typedef struct {
} op_export_core;

typedef op_export_core *op_export_handle;

typedef struct {
} op_import_core;

typedef op_import_core *op_import_handle;


op_import_handle op_import_init_size(int nprocs, int *proclist, op_dat mark) {

  exit(1);
}

op_import_handle op_import_init(op_export_handle exp_handle, op_dat coords, op_dat mark) {

  exit(1);
}

op_export_handle op_export_init(int nprocs, int *proclist, op_map cellsToNodes, op_set sp_nodes, op_dat coords, op_dat mark) {

  exit(1);
}

void op_theta_init(op_export_handle handle, int *bc_id, double *dtheta_exp, double *dtheta_imp, double *alpha) {

  exit(1);
}

void op_inc_theta(op_export_handle handle, int *bc_id, double *dtheta_exp, double *dtheta_imp) {

  exit(1);
}


void op_export_data(op_export_handle handle, op_dat dat) {

  exit(1);
}

void op_import_data(op_import_handle handle, op_dat dat) {

  exit(1);
}

