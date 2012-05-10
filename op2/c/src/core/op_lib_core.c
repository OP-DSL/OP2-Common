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

/*
 * This file implements the OP2 core library functions used by *any*
 * OP2 implementation
 */

#include <sys/time.h>
#include "op_lib_core.h"

/*
 * OP2 global state variables
 */

int OP_diags = 0,
    OP_part_size = 0,
    OP_block_size = 64,
    OP_cache_line_size = 128;

int OP_set_index = 0, OP_set_max = 0,
    OP_map_index = 0, OP_map_max = 0,
    OP_dat_index = 0, OP_dat_max = 0,
    OP_mat_index = 0, OP_mat_max = 0,
    OP_sparsity_index = 0, OP_sparsity_max = 0,
    OP_kern_max = 0;

/*
 * Lists of sets, maps and dats declared in OP2 programs
 */

op_set * OP_set_list;
op_map * OP_map_list;
op_dat * OP_dat_list;
op_mat * OP_mat_list;
op_sparsity * OP_sparsity_list;
op_kernel * OP_kernels;

static char * copy_str( char const * src )
{
  const size_t len = strlen( src ) + 1;
  char * dest = (char *) calloc ( len, sizeof ( char ) );
  return strncpy ( dest, src, len );
}

/*
 * OP core functions: these must be called by back-end specific functions
 */

void
op_init_core ( int argc, char ** argv, int diags )
{
  OP_diags = diags;

#ifdef OP_BLOCK_SIZE
  OP_block_size = OP_BLOCK_SIZE;
#endif
#ifdef OP_PART_SIZE
  OP_part_size = OP_PART_SIZE;
#endif

  for ( int n = 1; n < argc; n++ )
  {

    if ( strncmp ( argv[n], "OP_BLOCK_SIZE=", 14 ) == 0 )
    {
      OP_block_size = atoi ( argv[n] + 14 );
      printf ( "\n OP_block_size = %d \n", OP_block_size );
    }

    if ( strncmp ( argv[n], "OP_PART_SIZE=", 13 ) == 0 )
    {
      OP_part_size = atoi ( argv[n] + 13 );
      printf ( "\n OP_part_size  = %d \n", OP_part_size );
    }

    if ( strncmp ( argv[n], "OP_CACHE_LINE_SIZE=", 19 ) == 0 )
    {
      OP_cache_line_size = atoi ( argv[n] + 19 );
      printf ( "\n OP_cache_line_size  = %d \n", OP_cache_line_size );
    }

  }
}

op_set
op_decl_set_core ( int size, char const * name )
{
  if ( size < 0 )
  {
    printf ( " op_decl_set error -- negative/zero size for set: %s\n", name );
    exit ( -1 );
  }

  if ( OP_set_index == OP_set_max )
  {
    OP_set_max += 10;
    OP_set_list = ( op_set * ) realloc ( OP_set_list, OP_set_max * sizeof ( op_set ) );

    if ( OP_set_list == NULL )
    {
      printf ( " op_decl_set error -- error reallocating memory\n" );
      exit ( -1 );
    }

  }

  op_set set = ( op_set ) malloc ( sizeof ( op_set_core ) );
  set->index = OP_set_index;
  set->size = size;
  set->core_size = size;
  set->name = copy_str( name );
  set->exec_size = 0;
  set->nonexec_size = 0;
  OP_set_list[OP_set_index++] = set;

  return set;
}

op_map
op_decl_map_core ( op_set from, op_set to, int dim, int * imap, char const * name )
{
  if ( from == NULL )
  {
    printf ( " op_decl_map error -- invalid 'from' set for map %s\n", name );
    exit ( -1 );
  }

  if ( to == NULL )
  {
    printf ( "op_decl_map error -- invalid 'to' set for map %s\n", name );
    exit ( -1 );
  }

  if ( dim <= 0 )
  {
    printf ( "op_decl_map error -- negative/zero dimension for map %s\n", name );
    exit ( -1 );
  }

  /*This check breaks for MPI - need to fix this  */
  /*for ( int d = 0; d < dim; d++ )
  {
    for ( int n = 0; n < from->size; n++ )
    {
      if ( imap[d + n * dim] < 0 || imap[d + n * dim] >= to->size )
      {
        printf ( "op_decl_map error -- invalid data for map %s\n", name );
        printf ( "element = %d, dimension = %d, map = %d\n", n, d, imap[d + n * dim] );
        exit ( -1 );
      }
    }
  }*/

  if ( OP_map_index == OP_map_max )
  {
    OP_map_max += 10;
    OP_map_list = ( op_map * ) realloc ( OP_map_list, OP_map_max * sizeof ( op_map ) );

    if ( OP_map_list == NULL )
    {
      printf ( " op_decl_map error -- error reallocating memory\n" );
      exit ( -1 );
    }
  }

  op_map map = ( op_map ) malloc ( sizeof ( op_map_core ) );
  map->index = OP_map_index;
  map->from = from;
  map->to = to;
  map->dim = dim;
  map->map = imap;
  map->map_d = NULL;
  map->name = copy_str( name );
  map->user_managed = 1;

  OP_map_list[OP_map_index++] = map;

  return map;
}

op_dat
op_decl_dat_core ( op_set set, int dim, char const * type, int size, char * data, char const * name )
{
  if ( set == NULL )
  {
    printf ( "op_decl_dat error -- invalid set for data: %s\n", name );
    exit ( -1 );
  }

  if ( dim <= 0 )
  {
    printf ( "op_decl_dat error -- negative/zero dimension for data: %s\n", name );
    exit ( -1 );
  }

  if ( OP_dat_index == OP_dat_max )
  {
    OP_dat_max += 10;
    OP_dat_list = ( op_dat * ) realloc ( OP_dat_list, OP_dat_max * sizeof ( op_dat ) );
    if ( OP_dat_list == NULL )
    {
      printf ( " op_decl_dat error -- error reallocating memory\n" );
      exit ( -1 );
    }
  }

  op_dat dat = ( op_dat ) malloc ( sizeof ( op_dat_core ) );
  dat->index = OP_dat_index;
  dat->set = set;
  dat->dim = dim;
  dat->data = data;
  dat->data_d = NULL;
  dat->name = copy_str( name );
  dat->type = copy_str( type );
  dat->size = dim * size;
  dat->user_managed = 1;
  OP_dat_list[OP_dat_index++] = dat;

  return dat;
}

op_mat
op_decl_mat_core ( op_set rowset, op_set colset, int dim, char const * type, int size, char const * name )
{

  if ( rowset == NULL )
  {
    printf ( "op_decl_mat error -- invalid rowset for matrix: %s\n", name );
    exit ( -1 );
  }

  if ( colset == NULL )
  {
    printf ( "op_decl_mat error -- invalid colset for matrix: %s\n", name );
    exit ( -1 );
  }

  if ( dim <= 0 )
  {
    printf ( "op_decl_mat error -- negative/zero dimension for matrix: %s\n", name );
    exit ( -1 );
  }

  if ( OP_mat_index == OP_mat_max )
  {
    OP_mat_max += 10;
    OP_mat_list = ( op_mat * ) realloc ( OP_mat_list, OP_mat_max * sizeof ( op_mat ) );
    if ( OP_mat_list == NULL )
    {
      printf ( " op_decl_mat error -- error reallocating memory\n" );
      exit ( -1 );
    }
  }

  op_mat mat = ( op_mat ) malloc ( sizeof ( op_mat_core ) );
  mat->index = OP_mat_index;
  mat->rowset = rowset;
  mat->colset = colset;
  mat->dim = dim;
  mat->name = name;
  mat->type = type;
  mat->size = dim * size;
  mat->mat = 0;
  mat->data = NULL;
  mat->lma_data = NULL;

  OP_mat_list[OP_mat_index++] = mat;

  return mat;
}

op_sparsity
op_decl_sparsity_core ( op_map rowmap, op_map colmap, char const * name )
{
  if ( rowmap == NULL )
  {
    printf ( "op_decl_sparsity error -- invalid rowmap for sparsity: %s\n", name );
    exit ( -1 );
  }

  if ( colmap == NULL )
  {
    printf ( "op_decl_sparsity error -- invalid colmap for sparsity: %s\n", name );
    exit ( -1 );
  }

  if ( rowmap->from != colmap->from ) {
    printf("op_decl_sparsity: row map and col map do not map from the same set for sparsity: %s\n", name );
    exit(1);
  }

  if ( OP_sparsity_index == OP_sparsity_max )
  {
    OP_sparsity_max += 10;
    OP_sparsity_list = ( op_sparsity * ) realloc ( OP_sparsity_list, OP_sparsity_max * sizeof ( op_sparsity ) );
    if ( OP_sparsity_list == NULL )
    {
      printf ( " op_decl_sparsity error -- error reallocating memory\n" );
      exit ( -1 );
    }
  }

  op_sparsity sparsity = ( op_sparsity ) malloc ( sizeof ( op_sparsity_core ) );
  sparsity->rowmap = rowmap;
  sparsity->colmap = colmap;
  sparsity->nrows = rowmap->to->size;
  sparsity->ncols = colmap->to->size;
  sparsity->nnz = NULL;
  sparsity->rowptr = NULL;
  sparsity->colidx = NULL;
  sparsity->max_nonzeros = 0;

  op_build_sparsity_pattern ( rowmap, colmap, sparsity );
  OP_sparsity_list[OP_sparsity_index++] = sparsity;

  return sparsity;
}

void
op_decl_const_core ( int dim, char const * type, int typeSize, char * data, char const * name )
{
  (void)dim;
  (void)type;
  (void)typeSize;
  (void)data;
  (void)name;
}

void
op_exit_core (  )
{
  // free storage and pointers for sets, maps and data

  for ( int i = 0; i < OP_set_index; i++ )
  {
    free ( (char*)OP_set_list[i]->name );
    free ( OP_set_list[i] );
  }
  free ( OP_set_list );
  OP_set_list = NULL;

  for ( int i = 0; i < OP_map_index; i++ )
  {
    if (!OP_map_list[i]->user_managed)
      free ( OP_map_list[i]->map );
    free ( (char*)OP_map_list[i]->name );
    free ( OP_map_list[i] );
  }
  free ( OP_map_list );
  OP_map_list = NULL;

  for ( int i = 0; i < OP_dat_index; i++ )
  {
    if (!OP_dat_list[i]->user_managed)
      free ( OP_dat_list[i]->data );
    free ( (char*)OP_dat_list[i]->name );
    free ( (char*)OP_dat_list[i]->type );
    free ( OP_dat_list[i] );
  }
  free ( OP_dat_list );
  OP_dat_list = NULL;

  for ( int i = 0; i < OP_mat_index; i++ )
  {
    free ( OP_mat_list[i] );
  }
  free ( OP_mat_list );

  for ( int i = 0; i < OP_sparsity_index; i++ )
  {
    free ( OP_sparsity_list[i]->nnz );
    free ( OP_sparsity_list[i]->rowptr );
    free ( OP_sparsity_list[i]->colidx );
    free ( OP_sparsity_list[i] );
  }
  free ( OP_sparsity_list );

  // free storage for timing info

  free ( OP_kernels );
  OP_kernels = NULL;

  // reset initial values

  OP_set_index = 0;
  OP_set_max = 0;
  OP_map_index = 0;
  OP_map_max = 0;
  OP_dat_index = 0;
  OP_dat_max = 0;
  OP_mat_index = 0;
  OP_mat_max = 0;
  OP_sparsity_index = 0;
  OP_sparsity_max = 0;
  OP_kern_max = 0;
}

/*
 * op_arg routines
 */

void
op_err_print ( const char * error_string, int m, const char * name )
{
  printf ( "error: arg %d in kernel \"%s\"\n", m, name );
  printf ( "%s \n", error_string );
  exit ( 1 );
}

void
op_arg_check ( op_set set, int m, op_arg arg, int * ninds, const char * name )
{
  /* error checking for op_arg_dat */

  switch ( arg.argtype  )
  {
    /* error checking for op_arg_dat */
  case OP_ARG_DAT:
    if ( set == NULL )
      op_err_print ( "invalid set", m, name );

    if ( arg.map != NULL && strstr( arg.type, ":soa")!= NULL)
      op_err_print( "SoA dataset accessed indirectly", m, name );

    if ( arg.map == NULL && arg.dat->set != set )
      op_err_print ( "dataset set does not match loop set", m, name );

    if ( arg.map != NULL && ( arg.map->from != set || arg.map->to != arg.dat->set ) )
      op_err_print ( "mapping error", m, name );

    if ( ( arg.map == NULL && arg.idx != -1 ) || ( arg.map != NULL &&
       ( arg.idx >= arg.map->dim || arg.idx < -1*arg.map->dim ) ) )
      op_err_print ( "invalid index", m, name );

    if ( arg.dat->dim != arg.dim )
      op_err_print ( "dataset dim does not match declared dim", m, name );

    if ( strcmp ( arg.dat->type, arg.type ) )
      op_err_print ( "dataset type does not match declared type", m, name );

    if ( arg.idx >= 0 || arg.idx < -1 )
      ( *ninds )++;

    break;

    /* error checking for op_arg_mat */
  case OP_ARG_MAT:
    if ( set == NULL )
      op_err_print ( "invalid set", m, name );

    if ( arg.map  == NULL && arg.mat->rowset != set )
      op_err_print ( "row dataset does not match loop set", m, name );
    if ( arg.map2 == NULL && arg.mat->colset != set )
      op_err_print ( "row dataset does not match loop set", m, name );

    if ( arg.map  != NULL && ( arg.map->from  != set || arg.map->to  != arg.mat->rowset ) )
      op_err_print ( "mapping error in rowmap", m, name );
    if ( arg.map2 != NULL && ( arg.map2->from != set || arg.map2->to != arg.mat->colset ) )
      op_err_print ( "mapping error in colmap", m, name );

    if ( ( arg.map == NULL && arg.idx != -1 ) ||
         ( arg.map != NULL &&
           (arg.idx < OP_I_OFFSET-arg.map->dim || (arg.idx > OP_I_OFFSET && arg.idx < -arg.map->dim) || arg.idx >= arg.map->dim ) ) )
      op_err_print ( "invalid row index", m, name );
    if ( ( arg.map2 == NULL && arg.idx2 != -1 ) ||
         ( arg.map2 != NULL &&
           (arg.idx2 < OP_I_OFFSET-arg.map2->dim || (arg.idx2 > OP_I_OFFSET && arg.idx2 < -arg.map2->dim) || arg.idx2 >= arg.map2->dim ) ) )
      op_err_print ( "invalid col index", m, name );

    if ( arg.mat->dim != arg.dim )
      op_err_print ( "dataset dim does not match declared dim", m, name );

    if ( strcmp ( arg.mat->type, arg.type ) )
      op_err_print ( "dataset type does not match declared type", m, name );

    if ( arg.idx >= 0 || arg.idx < -1 )
      ( *ninds )++;

    break;

    /* error checking for op_arg_gbl */
  case OP_ARG_GBL:
    if ( !strcmp ( arg.type, "error" ) )
      op_err_print ( "datatype does not match declared type", m, name );

    if ( arg.dim <= 0 )
      op_err_print ( "dimension should be strictly positive", m, name );

    if ( arg.data == NULL )
      op_err_print ( "NULL pointer for global data", m, name );

    break;
  }
}

op_arg
op_arg_dat_core ( op_dat dat, int idx, op_map map, int dim, const char * typ, op_access acc )
{
  if ( dat == NULL ) {
    printf ( " op_arg_dat error -- no valid op_dat given\n" );
    exit ( -1 );
  }

  op_arg arg;

  /* index is not used for now */
  arg.index = -1;

  arg.argtype = OP_ARG_DAT;

  arg.dat = dat;
  arg.mat = NULL;
  arg.map = map;
  arg.dim = dim;
  arg.idx = idx;
  arg.idx2 = -1;
  arg.map2 = NULL;
  arg.size = dat->size;
  arg.data = dat->data;
  arg.data_d = dat->data_d;

  arg.type = typ;
  arg.acc = acc;

  return arg;
}

op_arg
op_arg_mat_core ( op_mat mat, int rowidx, op_map rowmap, int colidx, op_map colmap, int dim, const char * typ, op_access acc )
{
  if ( mat == NULL ) {
    printf ( " op_arg_mat error -- no valid op_mat given\n" );
    exit ( -1 );
  }

  op_arg arg;

  /* index is not used for now */
  arg.index = -1;

  arg.argtype = OP_ARG_MAT;

  arg.dat = NULL;
  arg.mat = mat;
  arg.map = rowmap;
  arg.map2 = colmap;
  arg.dim = dim;
  arg.idx = rowidx;
  arg.idx2 = colidx;

  arg.size = mat->size;
  arg.data = NULL;
  arg.data_d = NULL;

  arg.type = typ;
  arg.acc = acc;

  /*initialize to 0 states no-mpi messages inflight for this arg*/
  arg.sent = 0;

  return arg;
}

op_arg
op_arg_gbl_core ( char * data, int dim, const char * typ, int size, op_access acc )
{
  op_arg arg;

  arg.argtype = OP_ARG_GBL;

  arg.dat = NULL;
  arg.mat = NULL;
  arg.map = NULL;
  arg.dim = dim;
  arg.idx = -1;
  arg.size = dim*size;
  arg.idx2 = -1;
  arg.map2 = NULL;
  arg.data = data;
  arg.type = typ;
  arg.acc = acc;

  /* setting default values for remaining fields */
  arg.index = -1;
  arg.data_d = NULL;

  /*not used in global args*/
  arg.sent = 0;

  return arg;
}

/*
 * diagnostic routines
 */

void
op_diagnostic_output (  )
{
  if ( OP_diags > 1 )
  {
    printf ( "\n  OP diagnostic output\n" );
    printf ( "  --------------------\n" );

    printf ( "\n       set       size\n" );
    printf ( "  -------------------\n" );
    for ( int n = 0; n < OP_set_index; n++ )
    {
      printf ( "%10s %10d\n", OP_set_list[n]->name, OP_set_list[n]->size );
    }

    printf ( "\n       map        dim       from         to\n" );
    printf ( "  -----------------------------------------\n" );
    for ( int n = 0; n < OP_map_index; n++ )
    {
      printf ( "%10s %10d %10s %10s\n",
               OP_map_list[n]->name, OP_map_list[n]->dim,
               OP_map_list[n]->from->name, OP_map_list[n]->to->name );
    }

    printf ( "\n       dat        dim        set\n" );
    printf ( "  ------------------------------\n" );
    for ( int n = 0; n < OP_dat_index; n++ )
    {
      printf ( "%10s %10d %10s\n", OP_dat_list[n]->name,
               OP_dat_list[n]->dim, OP_dat_list[n]->set->name );
    }
    printf ( "\n" );
  }
}

void op_timing_output_core()
{
  if ( OP_kern_max > 0 )
  {
    printf ( "\n  count     time     GB/s     GB/s   kernel name " );
    printf ( "\n ----------------------------------------------- \n" );
    for ( int n = 0; n < OP_kern_max; n++ )
    {
      if ( OP_kernels[n].count > 0 )
      {
        if ( OP_kernels[n].transfer2 < 1e-8f )
          printf ( " %6d  %8.4f %8.4f            %s \n",
                   OP_kernels[n].count,
                   OP_kernels[n].time,
                   OP_kernels[n].transfer / ( 1e9f * OP_kernels[n].time ), OP_kernels[n].name );
        else
          printf ( " %6d  %8.4f %8.4f %8.4f   %s \n",
                   OP_kernels[n].count,
                   OP_kernels[n].time,
                   OP_kernels[n].transfer / ( 1e9f * OP_kernels[n].time ),
                   OP_kernels[n].transfer2 / ( 1e9f * OP_kernels[n].time ), OP_kernels[n].name );
      }
    }
  }
}

void
op_timing_output_2_file ( const char * outputFileName )
{
  FILE * outputFile = NULL;
  float totalKernelTime = 0.0f;

  outputFile = fopen ( outputFileName, "w+" );
  if ( outputFile == NULL )
  {
    printf ( "Bad output file\n" );
    exit ( 1 );
  }

  if ( OP_kern_max > 0 )
  {
    fprintf ( outputFile, "\n  count     time     GB/s     GB/s   kernel name " );
    fprintf ( outputFile, "\n ----------------------------------------------- \n" );
    for ( int n = 0; n < OP_kern_max; n++ )
    {
      if ( OP_kernels[n].count > 0 )
      {
        if ( OP_kernels[n].transfer2 < 1e-8f )
        {
          totalKernelTime += OP_kernels[n].time;
          fprintf ( outputFile, " %6d  %8.4f %8.4f            %s \n",
                    OP_kernels[n].count,
                    OP_kernels[n].time,
                    OP_kernels[n].transfer / ( 1e9f * OP_kernels[n].time ), OP_kernels[n].name );
        }
        else
        {
          totalKernelTime += OP_kernels[n].time;
          fprintf ( outputFile, " %6d  %8.4f %8.4f %8.4f   %s \n",
                    OP_kernels[n].count,
                    OP_kernels[n].time,
                    OP_kernels[n].transfer / ( 1e9f * OP_kernels[n].time ),
                    OP_kernels[n].transfer2 / ( 1e9f * OP_kernels[n].time ), OP_kernels[n].name );
        }
      }
    }
    fprintf ( outputFile, "Total kernel time = %f\n", totalKernelTime );
  }

  fclose ( outputFile );
}

void op_timers_core( double * cpu, double * et )
{
  (void)cpu;
  struct timeval t;

  gettimeofday ( &t, ( struct timezone * ) 0 );
  *et = t.tv_sec + t.tv_usec * 1.0e-6;
}

void
op_timing_realloc ( int kernel )
{
  int OP_kern_max_new;

  if ( kernel >= OP_kern_max )
  {
    OP_kern_max_new = kernel + 10;
    OP_kernels = ( op_kernel * ) realloc ( OP_kernels, OP_kern_max_new * sizeof ( op_kernel ) );
    if ( OP_kernels == NULL )
    {
      printf ( " op_timing_realloc error \n" );
      exit ( -1 );
    }

    for ( int n = OP_kern_max; n < OP_kern_max_new; n++ )
    {
      OP_kernels[n].count = 0;
      OP_kernels[n].time = 0.0f;
      OP_kernels[n].transfer = 0.0f;
      OP_kernels[n].transfer2 = 0.0f;
      OP_kernels[n].name = "unused";
    }
    OP_kern_max = OP_kern_max_new;
  }
}

void
op_dump_dat ( op_dat data )
{
  fflush (stdout);

  if ( data != NULL ) {
    if ( strncmp ( "real", data->type, 4 ) == 0 ) {
      for ( int i = 0; i < data->dim * data->set->size; i++ )
        printf ( "%lf\n", ((double *) data->data)[i] );
    } else if ( strncmp ( "integer", data->type, 7 ) == 0 ) {
      for ( int i = 0; i < data->dim * data->set->size; i++ )
        printf ( "%d\n", data->data[i] );
    } else {
      printf ( "Unsupported type for dumping %s\n", data->type );
      exit ( 0 );
    }
  }

  fflush (stdout);
}

