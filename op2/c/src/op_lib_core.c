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

/*
 * This file implements the OP2 core library functions used by *any*
 * OP2 implementation
 */

#include "op_lib_core.h"

//
// global variables
//

int OP_diags          =0,
    OP_part_size      =0,
    OP_block_size     =512,
    OP_cache_line_size=128;

int OP_set_index =0, OP_set_max =0,
    OP_map_index =0, OP_map_max =0,
    OP_dat_index =0, OP_dat_max =0,
  OP_kern_max=0;

op_set    *OP_set_list;
op_map    *OP_map_list;
op_dat    *OP_dat_list;
op_kernel *OP_kernels;

//
// OP core functions
//

void op_init_core(int argc, char **argv, int diags){
  OP_diags = diags;

#ifdef OP_BLOCK_SIZE
  OP_block_size = OP_BLOCK_SIZE;
#endif
#ifdef OP_PART_SIZE
  OP_part_size = OP_PART_SIZE;
#endif

  for (int n=1; n<argc; n++) {
    if (strncmp(argv[n],"OP_BLOCK_SIZE=",14)==0) {
      OP_block_size = atoi(argv[n]+14);
      printf("\n OP_block_size = %d \n", OP_block_size);
    }
    if (strncmp(argv[n],"OP_PART_SIZE=",13)==0) {
      OP_part_size = atoi(argv[n]+13);
      printf("\n OP_part_size  = %d \n", OP_part_size);
    }
    if (strncmp(argv[n],"OP_CACHE_LINE_SIZE=",19)==0) {
      OP_cache_line_size = atoi(argv[n]+19);
      printf("\n OP_cache_line_size  = %d \n", OP_cache_line_size);
    }
  }
}

op_set op_decl_set_core ( int size, char const *name )
{
  if ( size <= 0 ) {
    printf ( " op_decl_set error -- negative/zero size for set: %s\n", name );
    exit ( -1 );
  }

  if ( OP_set_index == OP_set_max )
  {
    OP_set_max += 10;
    OP_set_list = (op_set *)realloc(OP_set_list,
                                    OP_set_max*sizeof(op_set));
    if ( OP_set_list == NULL )
    {
      printf(" op_decl_set error -- error reallocating memory\n");
      exit(-1);
    }
  }

  op_set set = (op_set) malloc(sizeof(op_set_core));
  set->index = OP_set_index;
  set->size  = size;
  set->name  = name;

  OP_set_list[OP_set_index++] = set;

  return set;
}

op_map op_decl_map_core ( op_set from, op_set to, int dim, int *imap,
                          char const * name )
{
  if (from==NULL) {
    printf(" op_decl_map error -- invalid 'from' set for map %s\n",name);
    exit(-1);
  }

  if (to==NULL) {
    printf("op_decl_map error -- invalid 'to' set for map %s\n",name);
    exit(-1);
  }

  if (dim<=0) {
    printf("op_decl_map error -- negative/zero dimension for map %s\n",name);
    exit(-1);
  }

  for (int d=0; d<dim; d++) {
    for (int n=0; n<from->size; n++) {
      if (imap[d+n*dim]<0 || imap[d+n*dim]>=to->size) {
        printf("op_decl_map error -- invalid data for map %s\n",name);
        printf("element = %d, dimension = %d, map = %d\n",n,d,imap[d+n*dim]);
        exit(-1);
      }
    }
  }

  if (OP_map_index==OP_map_max) {
    OP_map_max += 10;
    OP_map_list = (op_map *) realloc(OP_map_list,
                                     OP_map_max*sizeof(op_map));
    if (OP_map_list==NULL) {
      printf(" op_decl_map error -- error reallocating memory\n");
      exit(-1);
    }
  }

  op_map map = (op_map) malloc(sizeof(op_map_core));
  map->index = OP_map_index;
  map->from  = from;
  map->to    = to;
  map->dim   = dim;
  map->map   = imap;
  map->name  = name;

  OP_map_list[OP_map_index++] = map;

  return map;
}

op_dat op_decl_dat_core ( op_set set, int dim, char const * type,
                          int size, char * data, char const * name )
{
  if (set==NULL) {
    printf("op_decl_dat error -- invalid set for data: %s\n",name);
    exit(-1);
  }

  if (dim<=0) {
    printf("op_decl_dat error -- negative/zero dimension for data: %s\n",name);
    exit(-1);
  }

  if (OP_dat_index==OP_dat_max) {
    OP_dat_max += 10;
    OP_dat_list = (op_dat *) realloc(OP_dat_list,
                                   OP_dat_max*sizeof(op_dat));
    if (OP_dat_list==NULL) {
      printf(" op_decl_dat error -- error reallocating memory\n");
      exit(-1);
    }
  }

  op_dat dat  = (op_dat) malloc(sizeof(op_dat_core));
  dat->index  = OP_dat_index;
  dat->set    = set;
  dat->dim    = dim;
  dat->data   = data;
  dat->data_d = NULL;
  dat->name   = name;
  dat->type   = type;
  dat->size   = dim*size;

  OP_dat_list[OP_dat_index++] = dat;

  return dat;
}

void op_decl_const_core ( int dim, char const * type, int typeSize, char * data, char const * name )
{

}

void op_exit_core ()
{
  // free storage and pointers for sets, maps and data

  for(int i=0; i<OP_set_index; i++) {
    free(OP_set_list[i]);
  }
  free(OP_set_list);

  for(int i=0; i<OP_map_index; i++) {
    free(OP_map_list[i]);
  }
  free(OP_map_list);

  for(int i=0; i<OP_dat_index; i++) {
    free(OP_dat_list[i]);
  }
  free(OP_dat_list);

  // free storage for timing info

  free(OP_kernels);

  // reset initial values

  OP_set_index =0; OP_set_max =0;
  OP_map_index =0; OP_map_max =0;
  OP_dat_index =0; OP_dat_max =0;
  OP_kern_max=0;
}

//
// op_arg routines
//

void op_err_print ( const char *error_string, int m, const char * name )
{
  printf("error: arg %d in kernel \"%s\"\n",m,name);
  printf("%s \n", error_string);
  exit(1);
}

void op_arg_check ( op_set set, int m, op_arg arg, int * ninds, const char * name )
{
  // error checking for op_arg_dat

  if (arg.argtype==OP_ARG_DAT) {
    if (set==NULL)
      op_err_print("invalid set",m,name);

    if (arg.map==NULL && arg.dat->set!=set)
      op_err_print("dataset set does not match loop set",m,name);

    if (arg.map!=NULL && (arg.map->from!=set || arg.map->to!=arg.dat->set))
      op_err_print("mapping error",m,name);

    if ( (arg.map==NULL &&  arg.idx!=-1) ||
         (arg.map!=NULL && (arg.idx<0 || arg.idx>=arg.map->dim) ) )
      op_err_print("invalid index",m,name);

    if (arg.dat->dim != arg.dim)
      op_err_print("dataset dim does not match declared dim",m,name);

    if (strcmp(arg.dat->type,arg.type))
      op_err_print("dataset type does not match declared type",m,name);

    if (arg.idx>=0) (*ninds)++;
  }

  // error checking for op_arg_gbl

  if (arg.argtype==OP_ARG_GBL) {
    if (!strcmp(arg.type,"error"))
      op_err_print("datatype does not match declared type",m,name);

    if (arg.dim<=0)
      op_err_print("dimension should be strictly positive",m,name);

    if (arg.data==NULL)
      op_err_print("NULL pointer for global data",m,name);
  }
}

op_arg op_arg_dat_core ( op_dat dat, int idx, op_map map, int dim, const char * typ, op_access acc )
{
  op_arg arg;

  arg.argtype = OP_ARG_DAT;

  arg.dat  = dat;
  arg.map  = map;
  arg.dim  = dim;
  arg.idx  = idx;
  if (dat!=NULL) {
    arg.size   = dat->size;
    arg.data   = dat->data;
    arg.data_d = dat->data_d;
  }
  arg.type = typ;
  arg.acc  = acc;

  return arg;
}

op_arg op_arg_gbl_core ( char * data, int dim, const char * typ, op_access acc )
{
  op_arg arg;

  arg.argtype = OP_ARG_GBL;

  arg.dat  = NULL;
  arg.map  = NULL;
  arg.dim  = dim;
  arg.idx  = -1;
  arg.size = 0;
  arg.data = data;
  arg.type = typ;
  arg.acc  = acc;

  return arg;
}

//
// diagnostic routines
//

void op_diagnostic_output ()
{
  if (OP_diags > 1) {
    printf("\n  OP diagnostic output\n");
    printf(  "  --------------------\n");

    printf("\n       set       size\n");
    printf(  "  -------------------\n");
    for(int n=0; n<OP_set_index; n++) {
      printf("%10s %10d\n",
             OP_set_list[n]->name,OP_set_list[n]->size);
    }

    printf("\n       map        dim       from         to\n");
    printf(  "  -----------------------------------------\n");
    for(int n=0; n<OP_map_index; n++) {
      printf("%10s %10d %10s %10s\n",
             OP_map_list[n]->name,OP_map_list[n]->dim,
             OP_map_list[n]->from->name,OP_map_list[n]->to->name);
    }

    printf("\n       dat        dim        set\n");
    printf(  "  ------------------------------\n");
    for(int n=0; n<OP_dat_index; n++) {
      printf("%10s %10d %10s\n", OP_dat_list[n]->name,
             OP_dat_list[n]->dim,OP_dat_list[n]->set->name);
    }
    printf("\n");
  }
}

void op_timing_output ()
{
  if (OP_kern_max>0) {
    printf("\n  count     time     GB/s     GB/s   kernel name ");
    printf("\n ----------------------------------------------- \n");
    for (int n=0; n<OP_kern_max; n++) {
      if (OP_kernels[n].count>0) {
        if (OP_kernels[n].transfer2==0.0f)
          printf(" %6d  %8.4f %8.4f            %s \n",
         OP_kernels[n].count,
               OP_kernels[n].time,
               OP_kernels[n].transfer/(1e9f*OP_kernels[n].time),
               OP_kernels[n].name);
        else
          printf(" %6d  %8.4f %8.4f %8.4f   %s \n",
         OP_kernels[n].count,
               OP_kernels[n].time,
               OP_kernels[n].transfer/(1e9f*OP_kernels[n].time),
               OP_kernels[n].transfer2/(1e9f*OP_kernels[n].time),
               OP_kernels[n].name);
      }
    }
  }
}

void op_timing_realloc ( int kernel )
{
  int OP_kern_max_new;

  if (kernel>=OP_kern_max) {
    // printf("allocating more memory for OP_kernels \n");
    OP_kern_max_new = kernel + 10;
    OP_kernels = (op_kernel *) realloc(OP_kernels,
                OP_kern_max_new*sizeof(op_kernel));
    if (OP_kernels==NULL) {
      printf(" op_timing_realloc error \n");
      exit(-1);
    }

    for (int n=OP_kern_max; n<OP_kern_max_new; n++) {
      OP_kernels[n].count     = 0;
      OP_kernels[n].time      = 0.0f;
      OP_kernels[n].transfer  = 0.0f;
      OP_kernels[n].transfer2 = 0.0f;
      OP_kernels[n].name      = "unused";
    }
    OP_kern_max = OP_kern_max_new;
  }
}

