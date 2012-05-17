/*
  Open source copyright declaration based on BSD open source template:
  http://www.opensource.org/licenses/bsd-license.php

* Copyright (c) 2009, Mike Giles
* Copyright (c) 2011, Florian Rathgeber
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

#ifndef __OP_SEQ_H
#define __OP_SEQ_H

#include "op_lib_mat.h"
#include <boost/type_traits.hpp>

static inline void op_arg_set(int n, op_arg arg, char **p_arg){
  int n2;
  if (arg.map==NULL)         // identity mapping, or global data
    n2 = n;
  else                       // standard pointers
    n2 = arg.map->map[arg.idx+n*arg.map->dim];
  *p_arg = arg.data + n2*arg.size;
}

static inline void copy_in(int n, op_arg arg, char **p_arg) {
  // For each index in the target dimension of the map, copy the pointer to
  // the data
  for (int i = 0; i < arg.map->dim; ++i)
      p_arg[i] = arg.data + arg.map->map[i+n*arg.map->dim]*arg.size;
}

op_itspace op_iteration_space(op_set set, int i, int j)
{
  op_itspace ret = (op_itspace)malloc(sizeof(op_itspace_core));
  ret->set = set;
  ret->ndims = 2;
  ret->dims = (int *)malloc(ret->ndims * sizeof(int));
  ret->dims[0] = i;
  ret->dims[1] = j;
  return ret;
}

//
// op_par_loop routine for 1 arguments
//

template < class T0 >
void op_par_loop ( void (*kernel)( T0* ),
  char const * name, op_set set,
  op_arg arg0 )
{
  char *p_arg0 = 0;

  // consistency checks

  int ninds=0;

  if (OP_diags>0) {
    op_arg_check(set,0 ,arg0 ,&ninds,name);
  }

  if (OP_diags>2) {
    if (ninds==0)
      printf(" kernel routine w/o indirection:  %s \n",name);
    else
      printf(" kernel routine with indirection: %s \n",name);
  }

  // Allocate memory for vector map indices

  switch ( arg0.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg0 = arg0.data;
      break;
    case OP_ARG_DAT:
      if (arg0.idx  < -1)
        p_arg0 = (char *)malloc(arg0.map->dim*sizeof(T0));
      break;
    case OP_ARG_MAT:
      p_arg0 = (char*) malloc(arg0.map->dim * arg0.map2->dim * arg0.size);
      break;
  }

  // loop over set elements

  for (int n=0; n<set->size; n++) {
    // Copy in of vector map indices

    if (arg0.argtype == OP_ARG_DAT) {
      if (arg0.idx < -1)
        copy_in(n, arg0, (char**)p_arg0);
      else
        op_arg_set(n, arg0, &p_arg0 );
    }

    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_arg0 );
    // Assemble local matrix into global matrix

    if (arg0.argtype == OP_ARG_MAT) {
      const int rows = arg0.map->dim;
      const int cols = arg0.map2->dim;
      op_mat_addto( arg0.mat, p_arg0, rows, arg0.map->map + n*rows, cols, arg0.map2->map + n*cols);
    }

  }

  // Free memory for vector map indices

  if ((arg0.argtype == OP_ARG_DAT && arg0.idx < -1) || arg0.argtype == OP_ARG_MAT) free(p_arg0);

  // Global matrix assembly
  if (arg0.argtype == OP_ARG_MAT) op_mat_assemble(arg0.mat);
}

//
// op_par_loop routine for 1 arguments with op_iteration_space call
//

template < class T0 >
void op_par_loop ( void (*kernel)( T0*, int, int ),
  char const * name, op_itspace itspace,
  op_arg arg0 )
{
  char *p_arg0 = 0;
  op_set set = itspace->set;
  // consistency checks

  int ninds=0;

  if (OP_diags>0) {
    op_arg_check(set,0 ,arg0 ,&ninds,name);
  }

  if (OP_diags>2) {
    if (ninds==0)
      printf(" kernel routine w/o indirection:  %s \n",name);
    else
      printf(" kernel routine with indirection: %s \n",name);
  }

  // Allocate memory for vector map indices

  switch ( arg0.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg0 = arg0.data;
      break;
    case OP_ARG_DAT:
      if (arg0.idx < -1)
        p_arg0 = (char *)malloc(arg0.map->dim*sizeof(T0));
      break;
    case OP_ARG_MAT:
      p_arg0 = (char*) malloc(sizeof(T0));
      break;
  }

  // loop over set elements

  for (int n=0; n<set->size; n++) {
    // Copy in of vector map indices

    if (arg0.argtype == OP_ARG_DAT) {
      if (arg0.idx < -1)
        copy_in(n, arg0, (char**)p_arg0);
      else
        op_arg_set(n, arg0, &p_arg0 );
    }

    // call kernel function, passing in pointers to data
    int ilower = 0;
    int iupper = itspace->dims[0];
    int jlower = 0;
    int jupper = itspace->dims[1];
    int idxs[2];

    int arg0idxs[2];
    if (arg0.argtype == OP_ARG_MAT) {
      int iut;
      int jut;
      arg0idxs[0] = 0;
      arg0idxs[1] = 1;
      if (arg0.idx < -1) {
        iut = arg0.map->dim;
      } else if (arg0.idx < OP_I_OFFSET) {
        iut = itspace->dims[op_i(arg0.idx)-1];
        arg0idxs[0] = op_i(arg0.idx) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 0, aborting\n");
        exit(-1);
      }
      if (arg0.idx2 < -1) {
        jut = arg0.map2->dim;
      } else if (arg0.idx2 < OP_I_OFFSET) {
        jut = itspace->dims[op_i(arg0.idx2)-1];
        arg0idxs[1] = op_i(arg0.idx2) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 0, aborting\n");
        exit(-1);
      }
      if (iut != iupper || jut != jupper) {
        printf("Map dimensions do not match iteration space, aborting\n");
        exit(-1);
      }
    }


    for (idxs[0] = ilower; idxs[0] < iupper; idxs[0]++) {
      for (idxs[1] = jlower; idxs[1] < jupper; idxs[1]++ ) {


        if (arg0.argtype == OP_ARG_MAT) {
          ((T0 *)p_arg0)[0] = (T0)0;
        }

        kernel( (T0 *)p_arg0, idxs[0], idxs[1]);
        // Assemble local matrix into global matrix

        if (arg0.argtype == OP_ARG_MAT) {
          const int rows = arg0.map->dim;
          const int cols = arg0.map2->dim;
          op_mat_addto(arg0.mat, p_arg0,
                       1, arg0.map->map + n*rows + idxs[arg0idxs[0]],
                       1, arg0.map2->map + n*cols + idxs[arg0idxs[1]]);
        }

      }
    }
  }

  // Free memory for vector map indices

  if ((arg0.argtype == OP_ARG_DAT && arg0.idx < -1) || arg0.argtype == OP_ARG_MAT) free(p_arg0);

  free(itspace->dims);
  free(itspace);
  // Global matrix assembly
  if (arg0.argtype == OP_ARG_MAT) op_mat_assemble(arg0.mat);
}

//
// op_par_loop routine for 2 arguments
//

template < class T0, class T1 >
void op_par_loop ( void (*kernel)( T0*, T1* ),
  char const * name, op_set set,
  op_arg arg0, op_arg arg1 )
{
  char *p_arg0 = 0, *p_arg1 = 0;

  // consistency checks

  int ninds=0;

  if (OP_diags>0) {
    op_arg_check(set,0 ,arg0 ,&ninds,name);
    op_arg_check(set,1 ,arg1 ,&ninds,name);
  }

  if (OP_diags>2) {
    if (ninds==0)
      printf(" kernel routine w/o indirection:  %s \n",name);
    else
      printf(" kernel routine with indirection: %s \n",name);
  }

  // Allocate memory for vector map indices

  switch ( arg0.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg0 = arg0.data;
      break;
    case OP_ARG_DAT:
      if (arg0.idx  < -1)
        p_arg0 = (char *)malloc(arg0.map->dim*sizeof(T0));
      break;
    case OP_ARG_MAT:
      p_arg0 = (char*) malloc(arg0.map->dim * arg0.map2->dim * arg0.size);
      break;
  }

  switch ( arg1.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg1 = arg1.data;
      break;
    case OP_ARG_DAT:
      if (arg1.idx  < -1)
        p_arg1 = (char *)malloc(arg1.map->dim*sizeof(T1));
      break;
    case OP_ARG_MAT:
      p_arg1 = (char*) malloc(arg1.map->dim * arg1.map2->dim * arg1.size);
      break;
  }

  // loop over set elements

  for (int n=0; n<set->size; n++) {
    // Copy in of vector map indices

    if (arg0.argtype == OP_ARG_DAT) {
      if (arg0.idx < -1)
        copy_in(n, arg0, (char**)p_arg0);
      else
        op_arg_set(n, arg0, &p_arg0 );
    }

    if (arg1.argtype == OP_ARG_DAT) {
      if (arg1.idx < -1)
        copy_in(n, arg1, (char**)p_arg1);
      else
        op_arg_set(n, arg1, &p_arg1 );
    }

    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_arg0, (T1 *)p_arg1 );
    // Assemble local matrix into global matrix

    if (arg0.argtype == OP_ARG_MAT) {
      const int rows = arg0.map->dim;
      const int cols = arg0.map2->dim;
      op_mat_addto( arg0.mat, p_arg0, rows, arg0.map->map + n*rows, cols, arg0.map2->map + n*cols);
    }

    if (arg1.argtype == OP_ARG_MAT) {
      const int rows = arg1.map->dim;
      const int cols = arg1.map2->dim;
      op_mat_addto( arg1.mat, p_arg1, rows, arg1.map->map + n*rows, cols, arg1.map2->map + n*cols);
    }

  }

  // Free memory for vector map indices

  if ((arg0.argtype == OP_ARG_DAT && arg0.idx < -1) || arg0.argtype == OP_ARG_MAT) free(p_arg0);

  if ((arg1.argtype == OP_ARG_DAT && arg1.idx < -1) || arg1.argtype == OP_ARG_MAT) free(p_arg1);

  // Global matrix assembly
  if (arg0.argtype == OP_ARG_MAT) op_mat_assemble(arg0.mat);
  if (arg1.argtype == OP_ARG_MAT) op_mat_assemble(arg1.mat);
}

//
// op_par_loop routine for 2 arguments with op_iteration_space call
//

template < class T0, class T1 >
void op_par_loop ( void (*kernel)( T0*, T1*, int, int ),
  char const * name, op_itspace itspace,
  op_arg arg0, op_arg arg1 )
{
  char *p_arg0 = 0, *p_arg1 = 0;
  op_set set = itspace->set;
  // consistency checks

  int ninds=0;

  if (OP_diags>0) {
    op_arg_check(set,0 ,arg0 ,&ninds,name);
    op_arg_check(set,1 ,arg1 ,&ninds,name);
  }

  if (OP_diags>2) {
    if (ninds==0)
      printf(" kernel routine w/o indirection:  %s \n",name);
    else
      printf(" kernel routine with indirection: %s \n",name);
  }

  // Allocate memory for vector map indices

  switch ( arg0.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg0 = arg0.data;
      break;
    case OP_ARG_DAT:
      if (arg0.idx < -1)
        p_arg0 = (char *)malloc(arg0.map->dim*sizeof(T0));
      break;
    case OP_ARG_MAT:
      p_arg0 = (char*) malloc(sizeof(T0));
      break;
  }

  switch ( arg1.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg1 = arg1.data;
      break;
    case OP_ARG_DAT:
      if (arg1.idx < -1)
        p_arg1 = (char *)malloc(arg1.map->dim*sizeof(T1));
      break;
    case OP_ARG_MAT:
      p_arg1 = (char*) malloc(sizeof(T1));
      break;
  }

  // loop over set elements

  for (int n=0; n<set->size; n++) {
    // Copy in of vector map indices

    if (arg0.argtype == OP_ARG_DAT) {
      if (arg0.idx < -1)
        copy_in(n, arg0, (char**)p_arg0);
      else
        op_arg_set(n, arg0, &p_arg0 );
    }

    if (arg1.argtype == OP_ARG_DAT) {
      if (arg1.idx < -1)
        copy_in(n, arg1, (char**)p_arg1);
      else
        op_arg_set(n, arg1, &p_arg1 );
    }

    // call kernel function, passing in pointers to data
    int ilower = 0;
    int iupper = itspace->dims[0];
    int jlower = 0;
    int jupper = itspace->dims[1];
    int idxs[2];

    int arg0idxs[2];
    if (arg0.argtype == OP_ARG_MAT) {
      int iut;
      int jut;
      arg0idxs[0] = 0;
      arg0idxs[1] = 1;
      if (arg0.idx < -1) {
        iut = arg0.map->dim;
      } else if (arg0.idx < OP_I_OFFSET) {
        iut = itspace->dims[op_i(arg0.idx)-1];
        arg0idxs[0] = op_i(arg0.idx) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 0, aborting\n");
        exit(-1);
      }
      if (arg0.idx2 < -1) {
        jut = arg0.map2->dim;
      } else if (arg0.idx2 < OP_I_OFFSET) {
        jut = itspace->dims[op_i(arg0.idx2)-1];
        arg0idxs[1] = op_i(arg0.idx2) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 0, aborting\n");
        exit(-1);
      }
      if (iut != iupper || jut != jupper) {
        printf("Map dimensions do not match iteration space, aborting\n");
        exit(-1);
      }
    }

    int arg1idxs[2];
    if (arg1.argtype == OP_ARG_MAT) {
      int iut;
      int jut;
      arg1idxs[0] = 0;
      arg1idxs[1] = 1;
      if (arg1.idx < -1) {
        iut = arg1.map->dim;
      } else if (arg1.idx < OP_I_OFFSET) {
        iut = itspace->dims[op_i(arg1.idx)-1];
        arg1idxs[0] = op_i(arg1.idx) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 1, aborting\n");
        exit(-1);
      }
      if (arg1.idx2 < -1) {
        jut = arg1.map2->dim;
      } else if (arg1.idx2 < OP_I_OFFSET) {
        jut = itspace->dims[op_i(arg1.idx2)-1];
        arg1idxs[1] = op_i(arg1.idx2) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 1, aborting\n");
        exit(-1);
      }
      if (iut != iupper || jut != jupper) {
        printf("Map dimensions do not match iteration space, aborting\n");
        exit(-1);
      }
    }


    for (idxs[0] = ilower; idxs[0] < iupper; idxs[0]++) {
      for (idxs[1] = jlower; idxs[1] < jupper; idxs[1]++ ) {


        if (arg0.argtype == OP_ARG_MAT) {
          ((T0 *)p_arg0)[0] = (T0)0;
        }

        if (arg1.argtype == OP_ARG_MAT) {
          ((T1 *)p_arg1)[0] = (T1)0;
        }

        kernel( (T0 *)p_arg0, (T1 *)p_arg1, idxs[0], idxs[1]);
        // Assemble local matrix into global matrix

        if (arg0.argtype == OP_ARG_MAT) {
          const int rows = arg0.map->dim;
          const int cols = arg0.map2->dim;
          op_mat_addto(arg0.mat, p_arg0,
                       1, arg0.map->map + n*rows + idxs[arg0idxs[0]],
                       1, arg0.map2->map + n*cols + idxs[arg0idxs[1]]);
        }

        if (arg1.argtype == OP_ARG_MAT) {
          const int rows = arg1.map->dim;
          const int cols = arg1.map2->dim;
          op_mat_addto(arg1.mat, p_arg1,
                       1, arg1.map->map + n*rows + idxs[arg1idxs[0]],
                       1, arg1.map2->map + n*cols + idxs[arg1idxs[1]]);
        }

      }
    }
  }

  // Free memory for vector map indices

  if ((arg0.argtype == OP_ARG_DAT && arg0.idx < -1) || arg0.argtype == OP_ARG_MAT) free(p_arg0);

  if ((arg1.argtype == OP_ARG_DAT && arg1.idx < -1) || arg1.argtype == OP_ARG_MAT) free(p_arg1);

  free(itspace->dims);
  free(itspace);
  // Global matrix assembly
  if (arg0.argtype == OP_ARG_MAT) op_mat_assemble(arg0.mat);
  if (arg1.argtype == OP_ARG_MAT) op_mat_assemble(arg1.mat);
}

//
// op_par_loop routine for 3 arguments
//

template < class T0, class T1, class T2 >
void op_par_loop ( void (*kernel)( T0*, T1*, T2* ),
  char const * name, op_set set,
  op_arg arg0, op_arg arg1, op_arg arg2 )
{
  char *p_arg0 = 0, *p_arg1 = 0, *p_arg2 = 0;

  // consistency checks

  int ninds=0;

  if (OP_diags>0) {
    op_arg_check(set,0 ,arg0 ,&ninds,name);
    op_arg_check(set,1 ,arg1 ,&ninds,name);
    op_arg_check(set,2 ,arg2 ,&ninds,name);
  }

  if (OP_diags>2) {
    if (ninds==0)
      printf(" kernel routine w/o indirection:  %s \n",name);
    else
      printf(" kernel routine with indirection: %s \n",name);
  }

  // Allocate memory for vector map indices

  switch ( arg0.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg0 = arg0.data;
      break;
    case OP_ARG_DAT:
      if (arg0.idx  < -1)
        p_arg0 = (char *)malloc(arg0.map->dim*sizeof(T0));
      break;
    case OP_ARG_MAT:
      p_arg0 = (char*) malloc(arg0.map->dim * arg0.map2->dim * arg0.size);
      break;
  }

  switch ( arg1.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg1 = arg1.data;
      break;
    case OP_ARG_DAT:
      if (arg1.idx  < -1)
        p_arg1 = (char *)malloc(arg1.map->dim*sizeof(T1));
      break;
    case OP_ARG_MAT:
      p_arg1 = (char*) malloc(arg1.map->dim * arg1.map2->dim * arg1.size);
      break;
  }

  switch ( arg2.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg2 = arg2.data;
      break;
    case OP_ARG_DAT:
      if (arg2.idx  < -1)
        p_arg2 = (char *)malloc(arg2.map->dim*sizeof(T2));
      break;
    case OP_ARG_MAT:
      p_arg2 = (char*) malloc(arg2.map->dim * arg2.map2->dim * arg2.size);
      break;
  }

  // loop over set elements

  for (int n=0; n<set->size; n++) {
    // Copy in of vector map indices

    if (arg0.argtype == OP_ARG_DAT) {
      if (arg0.idx < -1)
        copy_in(n, arg0, (char**)p_arg0);
      else
        op_arg_set(n, arg0, &p_arg0 );
    }

    if (arg1.argtype == OP_ARG_DAT) {
      if (arg1.idx < -1)
        copy_in(n, arg1, (char**)p_arg1);
      else
        op_arg_set(n, arg1, &p_arg1 );
    }

    if (arg2.argtype == OP_ARG_DAT) {
      if (arg2.idx < -1)
        copy_in(n, arg2, (char**)p_arg2);
      else
        op_arg_set(n, arg2, &p_arg2 );
    }

    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_arg0, (T1 *)p_arg1, (T2 *)p_arg2 );
    // Assemble local matrix into global matrix

    if (arg0.argtype == OP_ARG_MAT) {
      const int rows = arg0.map->dim;
      const int cols = arg0.map2->dim;
      op_mat_addto( arg0.mat, p_arg0, rows, arg0.map->map + n*rows, cols, arg0.map2->map + n*cols);
    }

    if (arg1.argtype == OP_ARG_MAT) {
      const int rows = arg1.map->dim;
      const int cols = arg1.map2->dim;
      op_mat_addto( arg1.mat, p_arg1, rows, arg1.map->map + n*rows, cols, arg1.map2->map + n*cols);
    }

    if (arg2.argtype == OP_ARG_MAT) {
      const int rows = arg2.map->dim;
      const int cols = arg2.map2->dim;
      op_mat_addto( arg2.mat, p_arg2, rows, arg2.map->map + n*rows, cols, arg2.map2->map + n*cols);
    }

  }

  // Free memory for vector map indices

  if ((arg0.argtype == OP_ARG_DAT && arg0.idx < -1) || arg0.argtype == OP_ARG_MAT) free(p_arg0);

  if ((arg1.argtype == OP_ARG_DAT && arg1.idx < -1) || arg1.argtype == OP_ARG_MAT) free(p_arg1);

  if ((arg2.argtype == OP_ARG_DAT && arg2.idx < -1) || arg2.argtype == OP_ARG_MAT) free(p_arg2);

  // Global matrix assembly
  if (arg0.argtype == OP_ARG_MAT) op_mat_assemble(arg0.mat);
  if (arg1.argtype == OP_ARG_MAT) op_mat_assemble(arg1.mat);
  if (arg2.argtype == OP_ARG_MAT) op_mat_assemble(arg2.mat);
}

//
// op_par_loop routine for 3 arguments with op_iteration_space call
//

template < class T0, class T1, class T2 >
void op_par_loop ( void (*kernel)( T0*, T1*, T2*, int, int ),
  char const * name, op_itspace itspace,
  op_arg arg0, op_arg arg1, op_arg arg2 )
{
  char *p_arg0 = 0, *p_arg1 = 0, *p_arg2 = 0;
  op_set set = itspace->set;
  // consistency checks

  int ninds=0;

  if (OP_diags>0) {
    op_arg_check(set,0 ,arg0 ,&ninds,name);
    op_arg_check(set,1 ,arg1 ,&ninds,name);
    op_arg_check(set,2 ,arg2 ,&ninds,name);
  }

  if (OP_diags>2) {
    if (ninds==0)
      printf(" kernel routine w/o indirection:  %s \n",name);
    else
      printf(" kernel routine with indirection: %s \n",name);
  }

  // Allocate memory for vector map indices

  switch ( arg0.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg0 = arg0.data;
      break;
    case OP_ARG_DAT:
      if (arg0.idx < -1)
        p_arg0 = (char *)malloc(arg0.map->dim*sizeof(T0));
      break;
    case OP_ARG_MAT:
      p_arg0 = (char*) malloc(sizeof(T0));
      break;
  }

  switch ( arg1.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg1 = arg1.data;
      break;
    case OP_ARG_DAT:
      if (arg1.idx < -1)
        p_arg1 = (char *)malloc(arg1.map->dim*sizeof(T1));
      break;
    case OP_ARG_MAT:
      p_arg1 = (char*) malloc(sizeof(T1));
      break;
  }

  switch ( arg2.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg2 = arg2.data;
      break;
    case OP_ARG_DAT:
      if (arg2.idx < -1)
        p_arg2 = (char *)malloc(arg2.map->dim*sizeof(T2));
      break;
    case OP_ARG_MAT:
      p_arg2 = (char*) malloc(sizeof(T2));
      break;
  }

  // loop over set elements

  for (int n=0; n<set->size; n++) {
    // Copy in of vector map indices

    if (arg0.argtype == OP_ARG_DAT) {
      if (arg0.idx < -1)
        copy_in(n, arg0, (char**)p_arg0);
      else
        op_arg_set(n, arg0, &p_arg0 );
    }

    if (arg1.argtype == OP_ARG_DAT) {
      if (arg1.idx < -1)
        copy_in(n, arg1, (char**)p_arg1);
      else
        op_arg_set(n, arg1, &p_arg1 );
    }

    if (arg2.argtype == OP_ARG_DAT) {
      if (arg2.idx < -1)
        copy_in(n, arg2, (char**)p_arg2);
      else
        op_arg_set(n, arg2, &p_arg2 );
    }

    // call kernel function, passing in pointers to data
    int ilower = 0;
    int iupper = itspace->dims[0];
    int jlower = 0;
    int jupper = itspace->dims[1];
    int idxs[2];

    int arg0idxs[2];
    if (arg0.argtype == OP_ARG_MAT) {
      int iut;
      int jut;
      arg0idxs[0] = 0;
      arg0idxs[1] = 1;
      if (arg0.idx < -1) {
        iut = arg0.map->dim;
      } else if (arg0.idx < OP_I_OFFSET) {
        iut = itspace->dims[op_i(arg0.idx)-1];
        arg0idxs[0] = op_i(arg0.idx) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 0, aborting\n");
        exit(-1);
      }
      if (arg0.idx2 < -1) {
        jut = arg0.map2->dim;
      } else if (arg0.idx2 < OP_I_OFFSET) {
        jut = itspace->dims[op_i(arg0.idx2)-1];
        arg0idxs[1] = op_i(arg0.idx2) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 0, aborting\n");
        exit(-1);
      }
      if (iut != iupper || jut != jupper) {
        printf("Map dimensions do not match iteration space, aborting\n");
        exit(-1);
      }
    }

    int arg1idxs[2];
    if (arg1.argtype == OP_ARG_MAT) {
      int iut;
      int jut;
      arg1idxs[0] = 0;
      arg1idxs[1] = 1;
      if (arg1.idx < -1) {
        iut = arg1.map->dim;
      } else if (arg1.idx < OP_I_OFFSET) {
        iut = itspace->dims[op_i(arg1.idx)-1];
        arg1idxs[0] = op_i(arg1.idx) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 1, aborting\n");
        exit(-1);
      }
      if (arg1.idx2 < -1) {
        jut = arg1.map2->dim;
      } else if (arg1.idx2 < OP_I_OFFSET) {
        jut = itspace->dims[op_i(arg1.idx2)-1];
        arg1idxs[1] = op_i(arg1.idx2) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 1, aborting\n");
        exit(-1);
      }
      if (iut != iupper || jut != jupper) {
        printf("Map dimensions do not match iteration space, aborting\n");
        exit(-1);
      }
    }

    int arg2idxs[2];
    if (arg2.argtype == OP_ARG_MAT) {
      int iut;
      int jut;
      arg2idxs[0] = 0;
      arg2idxs[1] = 1;
      if (arg2.idx < -1) {
        iut = arg2.map->dim;
      } else if (arg2.idx < OP_I_OFFSET) {
        iut = itspace->dims[op_i(arg2.idx)-1];
        arg2idxs[0] = op_i(arg2.idx) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 2, aborting\n");
        exit(-1);
      }
      if (arg2.idx2 < -1) {
        jut = arg2.map2->dim;
      } else if (arg2.idx2 < OP_I_OFFSET) {
        jut = itspace->dims[op_i(arg2.idx2)-1];
        arg2idxs[1] = op_i(arg2.idx2) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 2, aborting\n");
        exit(-1);
      }
      if (iut != iupper || jut != jupper) {
        printf("Map dimensions do not match iteration space, aborting\n");
        exit(-1);
      }
    }


    for (idxs[0] = ilower; idxs[0] < iupper; idxs[0]++) {
      for (idxs[1] = jlower; idxs[1] < jupper; idxs[1]++ ) {


        if (arg0.argtype == OP_ARG_MAT) {
          ((T0 *)p_arg0)[0] = (T0)0;
        }

        if (arg1.argtype == OP_ARG_MAT) {
          ((T1 *)p_arg1)[0] = (T1)0;
        }

        if (arg2.argtype == OP_ARG_MAT) {
          ((T2 *)p_arg2)[0] = (T2)0;
        }

        kernel( (T0 *)p_arg0, (T1 *)p_arg1, (T2 *)p_arg2, idxs[0], idxs[1]);
        // Assemble local matrix into global matrix

        if (arg0.argtype == OP_ARG_MAT) {
          const int rows = arg0.map->dim;
          const int cols = arg0.map2->dim;
          op_mat_addto(arg0.mat, p_arg0,
                       1, arg0.map->map + n*rows + idxs[arg0idxs[0]],
                       1, arg0.map2->map + n*cols + idxs[arg0idxs[1]]);
        }

        if (arg1.argtype == OP_ARG_MAT) {
          const int rows = arg1.map->dim;
          const int cols = arg1.map2->dim;
          op_mat_addto(arg1.mat, p_arg1,
                       1, arg1.map->map + n*rows + idxs[arg1idxs[0]],
                       1, arg1.map2->map + n*cols + idxs[arg1idxs[1]]);
        }

        if (arg2.argtype == OP_ARG_MAT) {
          const int rows = arg2.map->dim;
          const int cols = arg2.map2->dim;
          op_mat_addto(arg2.mat, p_arg2,
                       1, arg2.map->map + n*rows + idxs[arg2idxs[0]],
                       1, arg2.map2->map + n*cols + idxs[arg2idxs[1]]);
        }

      }
    }
  }

  // Free memory for vector map indices

  if ((arg0.argtype == OP_ARG_DAT && arg0.idx < -1) || arg0.argtype == OP_ARG_MAT) free(p_arg0);

  if ((arg1.argtype == OP_ARG_DAT && arg1.idx < -1) || arg1.argtype == OP_ARG_MAT) free(p_arg1);

  if ((arg2.argtype == OP_ARG_DAT && arg2.idx < -1) || arg2.argtype == OP_ARG_MAT) free(p_arg2);

  free(itspace->dims);
  free(itspace);
  // Global matrix assembly
  if (arg0.argtype == OP_ARG_MAT) op_mat_assemble(arg0.mat);
  if (arg1.argtype == OP_ARG_MAT) op_mat_assemble(arg1.mat);
  if (arg2.argtype == OP_ARG_MAT) op_mat_assemble(arg2.mat);
}

//
// op_par_loop routine for 4 arguments
//

template < class T0, class T1, class T2, class T3 >
void op_par_loop ( void (*kernel)( T0*, T1*, T2*, T3* ),
  char const * name, op_set set,
  op_arg arg0, op_arg arg1, op_arg arg2, op_arg arg3 )
{
  char *p_arg0 = 0, *p_arg1 = 0, *p_arg2 = 0, *p_arg3 = 0;

  // consistency checks

  int ninds=0;

  if (OP_diags>0) {
    op_arg_check(set,0 ,arg0 ,&ninds,name);
    op_arg_check(set,1 ,arg1 ,&ninds,name);
    op_arg_check(set,2 ,arg2 ,&ninds,name);
    op_arg_check(set,3 ,arg3 ,&ninds,name);
  }

  if (OP_diags>2) {
    if (ninds==0)
      printf(" kernel routine w/o indirection:  %s \n",name);
    else
      printf(" kernel routine with indirection: %s \n",name);
  }

  // Allocate memory for vector map indices

  switch ( arg0.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg0 = arg0.data;
      break;
    case OP_ARG_DAT:
      if (arg0.idx  < -1)
        p_arg0 = (char *)malloc(arg0.map->dim*sizeof(T0));
      break;
    case OP_ARG_MAT:
      p_arg0 = (char*) malloc(arg0.map->dim * arg0.map2->dim * arg0.size);
      break;
  }

  switch ( arg1.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg1 = arg1.data;
      break;
    case OP_ARG_DAT:
      if (arg1.idx  < -1)
        p_arg1 = (char *)malloc(arg1.map->dim*sizeof(T1));
      break;
    case OP_ARG_MAT:
      p_arg1 = (char*) malloc(arg1.map->dim * arg1.map2->dim * arg1.size);
      break;
  }

  switch ( arg2.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg2 = arg2.data;
      break;
    case OP_ARG_DAT:
      if (arg2.idx  < -1)
        p_arg2 = (char *)malloc(arg2.map->dim*sizeof(T2));
      break;
    case OP_ARG_MAT:
      p_arg2 = (char*) malloc(arg2.map->dim * arg2.map2->dim * arg2.size);
      break;
  }

  switch ( arg3.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg3 = arg3.data;
      break;
    case OP_ARG_DAT:
      if (arg3.idx  < -1)
        p_arg3 = (char *)malloc(arg3.map->dim*sizeof(T3));
      break;
    case OP_ARG_MAT:
      p_arg3 = (char*) malloc(arg3.map->dim * arg3.map2->dim * arg3.size);
      break;
  }

  // loop over set elements

  for (int n=0; n<set->size; n++) {
    // Copy in of vector map indices

    if (arg0.argtype == OP_ARG_DAT) {
      if (arg0.idx < -1)
        copy_in(n, arg0, (char**)p_arg0);
      else
        op_arg_set(n, arg0, &p_arg0 );
    }

    if (arg1.argtype == OP_ARG_DAT) {
      if (arg1.idx < -1)
        copy_in(n, arg1, (char**)p_arg1);
      else
        op_arg_set(n, arg1, &p_arg1 );
    }

    if (arg2.argtype == OP_ARG_DAT) {
      if (arg2.idx < -1)
        copy_in(n, arg2, (char**)p_arg2);
      else
        op_arg_set(n, arg2, &p_arg2 );
    }

    if (arg3.argtype == OP_ARG_DAT) {
      if (arg3.idx < -1)
        copy_in(n, arg3, (char**)p_arg3);
      else
        op_arg_set(n, arg3, &p_arg3 );
    }

    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_arg0, (T1 *)p_arg1, (T2 *)p_arg2, (T3 *)p_arg3 );
    // Assemble local matrix into global matrix

    if (arg0.argtype == OP_ARG_MAT) {
      const int rows = arg0.map->dim;
      const int cols = arg0.map2->dim;
      op_mat_addto( arg0.mat, p_arg0, rows, arg0.map->map + n*rows, cols, arg0.map2->map + n*cols);
    }

    if (arg1.argtype == OP_ARG_MAT) {
      const int rows = arg1.map->dim;
      const int cols = arg1.map2->dim;
      op_mat_addto( arg1.mat, p_arg1, rows, arg1.map->map + n*rows, cols, arg1.map2->map + n*cols);
    }

    if (arg2.argtype == OP_ARG_MAT) {
      const int rows = arg2.map->dim;
      const int cols = arg2.map2->dim;
      op_mat_addto( arg2.mat, p_arg2, rows, arg2.map->map + n*rows, cols, arg2.map2->map + n*cols);
    }

    if (arg3.argtype == OP_ARG_MAT) {
      const int rows = arg3.map->dim;
      const int cols = arg3.map2->dim;
      op_mat_addto( arg3.mat, p_arg3, rows, arg3.map->map + n*rows, cols, arg3.map2->map + n*cols);
    }

  }

  // Free memory for vector map indices

  if ((arg0.argtype == OP_ARG_DAT && arg0.idx < -1) || arg0.argtype == OP_ARG_MAT) free(p_arg0);

  if ((arg1.argtype == OP_ARG_DAT && arg1.idx < -1) || arg1.argtype == OP_ARG_MAT) free(p_arg1);

  if ((arg2.argtype == OP_ARG_DAT && arg2.idx < -1) || arg2.argtype == OP_ARG_MAT) free(p_arg2);

  if ((arg3.argtype == OP_ARG_DAT && arg3.idx < -1) || arg3.argtype == OP_ARG_MAT) free(p_arg3);

  // Global matrix assembly
  if (arg0.argtype == OP_ARG_MAT) op_mat_assemble(arg0.mat);
  if (arg1.argtype == OP_ARG_MAT) op_mat_assemble(arg1.mat);
  if (arg2.argtype == OP_ARG_MAT) op_mat_assemble(arg2.mat);
  if (arg3.argtype == OP_ARG_MAT) op_mat_assemble(arg3.mat);
}

//
// op_par_loop routine for 4 arguments with op_iteration_space call
//

template < class T0, class T1, class T2, class T3 >
void op_par_loop ( void (*kernel)( T0*, T1*, T2*, T3*, int, int ),
  char const * name, op_itspace itspace,
  op_arg arg0, op_arg arg1, op_arg arg2, op_arg arg3 )
{
  char *p_arg0 = 0, *p_arg1 = 0, *p_arg2 = 0, *p_arg3 = 0;
  op_set set = itspace->set;
  // consistency checks

  int ninds=0;

  if (OP_diags>0) {
    op_arg_check(set,0 ,arg0 ,&ninds,name);
    op_arg_check(set,1 ,arg1 ,&ninds,name);
    op_arg_check(set,2 ,arg2 ,&ninds,name);
    op_arg_check(set,3 ,arg3 ,&ninds,name);
  }

  if (OP_diags>2) {
    if (ninds==0)
      printf(" kernel routine w/o indirection:  %s \n",name);
    else
      printf(" kernel routine with indirection: %s \n",name);
  }

  // Allocate memory for vector map indices

  switch ( arg0.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg0 = arg0.data;
      break;
    case OP_ARG_DAT:
      if (arg0.idx < -1)
        p_arg0 = (char *)malloc(arg0.map->dim*sizeof(T0));
      break;
    case OP_ARG_MAT:
      p_arg0 = (char*) malloc(sizeof(T0));
      break;
  }

  switch ( arg1.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg1 = arg1.data;
      break;
    case OP_ARG_DAT:
      if (arg1.idx < -1)
        p_arg1 = (char *)malloc(arg1.map->dim*sizeof(T1));
      break;
    case OP_ARG_MAT:
      p_arg1 = (char*) malloc(sizeof(T1));
      break;
  }

  switch ( arg2.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg2 = arg2.data;
      break;
    case OP_ARG_DAT:
      if (arg2.idx < -1)
        p_arg2 = (char *)malloc(arg2.map->dim*sizeof(T2));
      break;
    case OP_ARG_MAT:
      p_arg2 = (char*) malloc(sizeof(T2));
      break;
  }

  switch ( arg3.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg3 = arg3.data;
      break;
    case OP_ARG_DAT:
      if (arg3.idx < -1)
        p_arg3 = (char *)malloc(arg3.map->dim*sizeof(T3));
      break;
    case OP_ARG_MAT:
      p_arg3 = (char*) malloc(sizeof(T3));
      break;
  }

  // loop over set elements

  for (int n=0; n<set->size; n++) {
    // Copy in of vector map indices

    if (arg0.argtype == OP_ARG_DAT) {
      if (arg0.idx < -1)
        copy_in(n, arg0, (char**)p_arg0);
      else
        op_arg_set(n, arg0, &p_arg0 );
    }

    if (arg1.argtype == OP_ARG_DAT) {
      if (arg1.idx < -1)
        copy_in(n, arg1, (char**)p_arg1);
      else
        op_arg_set(n, arg1, &p_arg1 );
    }

    if (arg2.argtype == OP_ARG_DAT) {
      if (arg2.idx < -1)
        copy_in(n, arg2, (char**)p_arg2);
      else
        op_arg_set(n, arg2, &p_arg2 );
    }

    if (arg3.argtype == OP_ARG_DAT) {
      if (arg3.idx < -1)
        copy_in(n, arg3, (char**)p_arg3);
      else
        op_arg_set(n, arg3, &p_arg3 );
    }

    // call kernel function, passing in pointers to data
    int ilower = 0;
    int iupper = itspace->dims[0];
    int jlower = 0;
    int jupper = itspace->dims[1];
    int idxs[2];

    int arg0idxs[2];
    if (arg0.argtype == OP_ARG_MAT) {
      int iut;
      int jut;
      arg0idxs[0] = 0;
      arg0idxs[1] = 1;
      if (arg0.idx < -1) {
        iut = arg0.map->dim;
      } else if (arg0.idx < OP_I_OFFSET) {
        iut = itspace->dims[op_i(arg0.idx)-1];
        arg0idxs[0] = op_i(arg0.idx) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 0, aborting\n");
        exit(-1);
      }
      if (arg0.idx2 < -1) {
        jut = arg0.map2->dim;
      } else if (arg0.idx2 < OP_I_OFFSET) {
        jut = itspace->dims[op_i(arg0.idx2)-1];
        arg0idxs[1] = op_i(arg0.idx2) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 0, aborting\n");
        exit(-1);
      }
      if (iut != iupper || jut != jupper) {
        printf("Map dimensions do not match iteration space, aborting\n");
        exit(-1);
      }
    }

    int arg1idxs[2];
    if (arg1.argtype == OP_ARG_MAT) {
      int iut;
      int jut;
      arg1idxs[0] = 0;
      arg1idxs[1] = 1;
      if (arg1.idx < -1) {
        iut = arg1.map->dim;
      } else if (arg1.idx < OP_I_OFFSET) {
        iut = itspace->dims[op_i(arg1.idx)-1];
        arg1idxs[0] = op_i(arg1.idx) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 1, aborting\n");
        exit(-1);
      }
      if (arg1.idx2 < -1) {
        jut = arg1.map2->dim;
      } else if (arg1.idx2 < OP_I_OFFSET) {
        jut = itspace->dims[op_i(arg1.idx2)-1];
        arg1idxs[1] = op_i(arg1.idx2) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 1, aborting\n");
        exit(-1);
      }
      if (iut != iupper || jut != jupper) {
        printf("Map dimensions do not match iteration space, aborting\n");
        exit(-1);
      }
    }

    int arg2idxs[2];
    if (arg2.argtype == OP_ARG_MAT) {
      int iut;
      int jut;
      arg2idxs[0] = 0;
      arg2idxs[1] = 1;
      if (arg2.idx < -1) {
        iut = arg2.map->dim;
      } else if (arg2.idx < OP_I_OFFSET) {
        iut = itspace->dims[op_i(arg2.idx)-1];
        arg2idxs[0] = op_i(arg2.idx) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 2, aborting\n");
        exit(-1);
      }
      if (arg2.idx2 < -1) {
        jut = arg2.map2->dim;
      } else if (arg2.idx2 < OP_I_OFFSET) {
        jut = itspace->dims[op_i(arg2.idx2)-1];
        arg2idxs[1] = op_i(arg2.idx2) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 2, aborting\n");
        exit(-1);
      }
      if (iut != iupper || jut != jupper) {
        printf("Map dimensions do not match iteration space, aborting\n");
        exit(-1);
      }
    }

    int arg3idxs[2];
    if (arg3.argtype == OP_ARG_MAT) {
      int iut;
      int jut;
      arg3idxs[0] = 0;
      arg3idxs[1] = 1;
      if (arg3.idx < -1) {
        iut = arg3.map->dim;
      } else if (arg3.idx < OP_I_OFFSET) {
        iut = itspace->dims[op_i(arg3.idx)-1];
        arg3idxs[0] = op_i(arg3.idx) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 3, aborting\n");
        exit(-1);
      }
      if (arg3.idx2 < -1) {
        jut = arg3.map2->dim;
      } else if (arg3.idx2 < OP_I_OFFSET) {
        jut = itspace->dims[op_i(arg3.idx2)-1];
        arg3idxs[1] = op_i(arg3.idx2) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 3, aborting\n");
        exit(-1);
      }
      if (iut != iupper || jut != jupper) {
        printf("Map dimensions do not match iteration space, aborting\n");
        exit(-1);
      }
    }


    for (idxs[0] = ilower; idxs[0] < iupper; idxs[0]++) {
      for (idxs[1] = jlower; idxs[1] < jupper; idxs[1]++ ) {


        if (arg0.argtype == OP_ARG_MAT) {
          ((T0 *)p_arg0)[0] = (T0)0;
        }

        if (arg1.argtype == OP_ARG_MAT) {
          ((T1 *)p_arg1)[0] = (T1)0;
        }

        if (arg2.argtype == OP_ARG_MAT) {
          ((T2 *)p_arg2)[0] = (T2)0;
        }

        if (arg3.argtype == OP_ARG_MAT) {
          ((T3 *)p_arg3)[0] = (T3)0;
        }

        kernel( (T0 *)p_arg0, (T1 *)p_arg1, (T2 *)p_arg2, (T3 *)p_arg3, idxs[0], idxs[1]);
        // Assemble local matrix into global matrix

        if (arg0.argtype == OP_ARG_MAT) {
          const int rows = arg0.map->dim;
          const int cols = arg0.map2->dim;
          op_mat_addto(arg0.mat, p_arg0,
                       1, arg0.map->map + n*rows + idxs[arg0idxs[0]],
                       1, arg0.map2->map + n*cols + idxs[arg0idxs[1]]);
        }

        if (arg1.argtype == OP_ARG_MAT) {
          const int rows = arg1.map->dim;
          const int cols = arg1.map2->dim;
          op_mat_addto(arg1.mat, p_arg1,
                       1, arg1.map->map + n*rows + idxs[arg1idxs[0]],
                       1, arg1.map2->map + n*cols + idxs[arg1idxs[1]]);
        }

        if (arg2.argtype == OP_ARG_MAT) {
          const int rows = arg2.map->dim;
          const int cols = arg2.map2->dim;
          op_mat_addto(arg2.mat, p_arg2,
                       1, arg2.map->map + n*rows + idxs[arg2idxs[0]],
                       1, arg2.map2->map + n*cols + idxs[arg2idxs[1]]);
        }

        if (arg3.argtype == OP_ARG_MAT) {
          const int rows = arg3.map->dim;
          const int cols = arg3.map2->dim;
          op_mat_addto(arg3.mat, p_arg3,
                       1, arg3.map->map + n*rows + idxs[arg3idxs[0]],
                       1, arg3.map2->map + n*cols + idxs[arg3idxs[1]]);
        }

      }
    }
  }

  // Free memory for vector map indices

  if ((arg0.argtype == OP_ARG_DAT && arg0.idx < -1) || arg0.argtype == OP_ARG_MAT) free(p_arg0);

  if ((arg1.argtype == OP_ARG_DAT && arg1.idx < -1) || arg1.argtype == OP_ARG_MAT) free(p_arg1);

  if ((arg2.argtype == OP_ARG_DAT && arg2.idx < -1) || arg2.argtype == OP_ARG_MAT) free(p_arg2);

  if ((arg3.argtype == OP_ARG_DAT && arg3.idx < -1) || arg3.argtype == OP_ARG_MAT) free(p_arg3);

  free(itspace->dims);
  free(itspace);
  // Global matrix assembly
  if (arg0.argtype == OP_ARG_MAT) op_mat_assemble(arg0.mat);
  if (arg1.argtype == OP_ARG_MAT) op_mat_assemble(arg1.mat);
  if (arg2.argtype == OP_ARG_MAT) op_mat_assemble(arg2.mat);
  if (arg3.argtype == OP_ARG_MAT) op_mat_assemble(arg3.mat);
}

//
// op_par_loop routine for 5 arguments
//

template < class T0, class T1, class T2, class T3,
           class T4 >
void op_par_loop ( void (*kernel)( T0*, T1*, T2*, T3*,
                                   T4* ),
  char const * name, op_set set,
  op_arg arg0, op_arg arg1, op_arg arg2, op_arg arg3,
  op_arg arg4 )
{
  char *p_arg0 = 0, *p_arg1 = 0, *p_arg2 = 0, *p_arg3 = 0,
       *p_arg4 = 0;

  // consistency checks

  int ninds=0;

  if (OP_diags>0) {
    op_arg_check(set,0 ,arg0 ,&ninds,name);
    op_arg_check(set,1 ,arg1 ,&ninds,name);
    op_arg_check(set,2 ,arg2 ,&ninds,name);
    op_arg_check(set,3 ,arg3 ,&ninds,name);
    op_arg_check(set,4 ,arg4 ,&ninds,name);
  }

  if (OP_diags>2) {
    if (ninds==0)
      printf(" kernel routine w/o indirection:  %s \n",name);
    else
      printf(" kernel routine with indirection: %s \n",name);
  }

  // Allocate memory for vector map indices

  switch ( arg0.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg0 = arg0.data;
      break;
    case OP_ARG_DAT:
      if (arg0.idx  < -1)
        p_arg0 = (char *)malloc(arg0.map->dim*sizeof(T0));
      break;
    case OP_ARG_MAT:
      p_arg0 = (char*) malloc(arg0.map->dim * arg0.map2->dim * arg0.size);
      break;
  }

  switch ( arg1.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg1 = arg1.data;
      break;
    case OP_ARG_DAT:
      if (arg1.idx  < -1)
        p_arg1 = (char *)malloc(arg1.map->dim*sizeof(T1));
      break;
    case OP_ARG_MAT:
      p_arg1 = (char*) malloc(arg1.map->dim * arg1.map2->dim * arg1.size);
      break;
  }

  switch ( arg2.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg2 = arg2.data;
      break;
    case OP_ARG_DAT:
      if (arg2.idx  < -1)
        p_arg2 = (char *)malloc(arg2.map->dim*sizeof(T2));
      break;
    case OP_ARG_MAT:
      p_arg2 = (char*) malloc(arg2.map->dim * arg2.map2->dim * arg2.size);
      break;
  }

  switch ( arg3.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg3 = arg3.data;
      break;
    case OP_ARG_DAT:
      if (arg3.idx  < -1)
        p_arg3 = (char *)malloc(arg3.map->dim*sizeof(T3));
      break;
    case OP_ARG_MAT:
      p_arg3 = (char*) malloc(arg3.map->dim * arg3.map2->dim * arg3.size);
      break;
  }

  switch ( arg4.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg4 = arg4.data;
      break;
    case OP_ARG_DAT:
      if (arg4.idx  < -1)
        p_arg4 = (char *)malloc(arg4.map->dim*sizeof(T4));
      break;
    case OP_ARG_MAT:
      p_arg4 = (char*) malloc(arg4.map->dim * arg4.map2->dim * arg4.size);
      break;
  }

  // loop over set elements

  for (int n=0; n<set->size; n++) {
    // Copy in of vector map indices

    if (arg0.argtype == OP_ARG_DAT) {
      if (arg0.idx < -1)
        copy_in(n, arg0, (char**)p_arg0);
      else
        op_arg_set(n, arg0, &p_arg0 );
    }

    if (arg1.argtype == OP_ARG_DAT) {
      if (arg1.idx < -1)
        copy_in(n, arg1, (char**)p_arg1);
      else
        op_arg_set(n, arg1, &p_arg1 );
    }

    if (arg2.argtype == OP_ARG_DAT) {
      if (arg2.idx < -1)
        copy_in(n, arg2, (char**)p_arg2);
      else
        op_arg_set(n, arg2, &p_arg2 );
    }

    if (arg3.argtype == OP_ARG_DAT) {
      if (arg3.idx < -1)
        copy_in(n, arg3, (char**)p_arg3);
      else
        op_arg_set(n, arg3, &p_arg3 );
    }

    if (arg4.argtype == OP_ARG_DAT) {
      if (arg4.idx < -1)
        copy_in(n, arg4, (char**)p_arg4);
      else
        op_arg_set(n, arg4, &p_arg4 );
    }

    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_arg0, (T1 *)p_arg1, (T2 *)p_arg2, (T3 *)p_arg3,
            (T4 *)p_arg4 );
    // Assemble local matrix into global matrix

    if (arg0.argtype == OP_ARG_MAT) {
      const int rows = arg0.map->dim;
      const int cols = arg0.map2->dim;
      op_mat_addto( arg0.mat, p_arg0, rows, arg0.map->map + n*rows, cols, arg0.map2->map + n*cols);
    }

    if (arg1.argtype == OP_ARG_MAT) {
      const int rows = arg1.map->dim;
      const int cols = arg1.map2->dim;
      op_mat_addto( arg1.mat, p_arg1, rows, arg1.map->map + n*rows, cols, arg1.map2->map + n*cols);
    }

    if (arg2.argtype == OP_ARG_MAT) {
      const int rows = arg2.map->dim;
      const int cols = arg2.map2->dim;
      op_mat_addto( arg2.mat, p_arg2, rows, arg2.map->map + n*rows, cols, arg2.map2->map + n*cols);
    }

    if (arg3.argtype == OP_ARG_MAT) {
      const int rows = arg3.map->dim;
      const int cols = arg3.map2->dim;
      op_mat_addto( arg3.mat, p_arg3, rows, arg3.map->map + n*rows, cols, arg3.map2->map + n*cols);
    }

    if (arg4.argtype == OP_ARG_MAT) {
      const int rows = arg4.map->dim;
      const int cols = arg4.map2->dim;
      op_mat_addto( arg4.mat, p_arg4, rows, arg4.map->map + n*rows, cols, arg4.map2->map + n*cols);
    }

  }

  // Free memory for vector map indices

  if ((arg0.argtype == OP_ARG_DAT && arg0.idx < -1) || arg0.argtype == OP_ARG_MAT) free(p_arg0);

  if ((arg1.argtype == OP_ARG_DAT && arg1.idx < -1) || arg1.argtype == OP_ARG_MAT) free(p_arg1);

  if ((arg2.argtype == OP_ARG_DAT && arg2.idx < -1) || arg2.argtype == OP_ARG_MAT) free(p_arg2);

  if ((arg3.argtype == OP_ARG_DAT && arg3.idx < -1) || arg3.argtype == OP_ARG_MAT) free(p_arg3);

  if ((arg4.argtype == OP_ARG_DAT && arg4.idx < -1) || arg4.argtype == OP_ARG_MAT) free(p_arg4);

  // Global matrix assembly
  if (arg0.argtype == OP_ARG_MAT) op_mat_assemble(arg0.mat);
  if (arg1.argtype == OP_ARG_MAT) op_mat_assemble(arg1.mat);
  if (arg2.argtype == OP_ARG_MAT) op_mat_assemble(arg2.mat);
  if (arg3.argtype == OP_ARG_MAT) op_mat_assemble(arg3.mat);
  if (arg4.argtype == OP_ARG_MAT) op_mat_assemble(arg4.mat);
}

//
// op_par_loop routine for 5 arguments with op_iteration_space call
//

template < class T0, class T1, class T2, class T3,
           class T4 >
void op_par_loop ( void (*kernel)( T0*, T1*, T2*, T3*,
                                   T4*, int, int ),
  char const * name, op_itspace itspace,
  op_arg arg0, op_arg arg1, op_arg arg2, op_arg arg3,
  op_arg arg4 )
{
  char *p_arg0 = 0, *p_arg1 = 0, *p_arg2 = 0, *p_arg3 = 0,
       *p_arg4 = 0;
  op_set set = itspace->set;
  // consistency checks

  int ninds=0;

  if (OP_diags>0) {
    op_arg_check(set,0 ,arg0 ,&ninds,name);
    op_arg_check(set,1 ,arg1 ,&ninds,name);
    op_arg_check(set,2 ,arg2 ,&ninds,name);
    op_arg_check(set,3 ,arg3 ,&ninds,name);
    op_arg_check(set,4 ,arg4 ,&ninds,name);
  }

  if (OP_diags>2) {
    if (ninds==0)
      printf(" kernel routine w/o indirection:  %s \n",name);
    else
      printf(" kernel routine with indirection: %s \n",name);
  }

  // Allocate memory for vector map indices

  switch ( arg0.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg0 = arg0.data;
      break;
    case OP_ARG_DAT:
      if (arg0.idx < -1)
        p_arg0 = (char *)malloc(arg0.map->dim*sizeof(T0));
      break;
    case OP_ARG_MAT:
      p_arg0 = (char*) malloc(sizeof(T0));
      break;
  }

  switch ( arg1.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg1 = arg1.data;
      break;
    case OP_ARG_DAT:
      if (arg1.idx < -1)
        p_arg1 = (char *)malloc(arg1.map->dim*sizeof(T1));
      break;
    case OP_ARG_MAT:
      p_arg1 = (char*) malloc(sizeof(T1));
      break;
  }

  switch ( arg2.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg2 = arg2.data;
      break;
    case OP_ARG_DAT:
      if (arg2.idx < -1)
        p_arg2 = (char *)malloc(arg2.map->dim*sizeof(T2));
      break;
    case OP_ARG_MAT:
      p_arg2 = (char*) malloc(sizeof(T2));
      break;
  }

  switch ( arg3.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg3 = arg3.data;
      break;
    case OP_ARG_DAT:
      if (arg3.idx < -1)
        p_arg3 = (char *)malloc(arg3.map->dim*sizeof(T3));
      break;
    case OP_ARG_MAT:
      p_arg3 = (char*) malloc(sizeof(T3));
      break;
  }

  switch ( arg4.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg4 = arg4.data;
      break;
    case OP_ARG_DAT:
      if (arg4.idx < -1)
        p_arg4 = (char *)malloc(arg4.map->dim*sizeof(T4));
      break;
    case OP_ARG_MAT:
      p_arg4 = (char*) malloc(sizeof(T4));
      break;
  }

  // loop over set elements

  for (int n=0; n<set->size; n++) {
    // Copy in of vector map indices

    if (arg0.argtype == OP_ARG_DAT) {
      if (arg0.idx < -1)
        copy_in(n, arg0, (char**)p_arg0);
      else
        op_arg_set(n, arg0, &p_arg0 );
    }

    if (arg1.argtype == OP_ARG_DAT) {
      if (arg1.idx < -1)
        copy_in(n, arg1, (char**)p_arg1);
      else
        op_arg_set(n, arg1, &p_arg1 );
    }

    if (arg2.argtype == OP_ARG_DAT) {
      if (arg2.idx < -1)
        copy_in(n, arg2, (char**)p_arg2);
      else
        op_arg_set(n, arg2, &p_arg2 );
    }

    if (arg3.argtype == OP_ARG_DAT) {
      if (arg3.idx < -1)
        copy_in(n, arg3, (char**)p_arg3);
      else
        op_arg_set(n, arg3, &p_arg3 );
    }

    if (arg4.argtype == OP_ARG_DAT) {
      if (arg4.idx < -1)
        copy_in(n, arg4, (char**)p_arg4);
      else
        op_arg_set(n, arg4, &p_arg4 );
    }

    // call kernel function, passing in pointers to data
    int ilower = 0;
    int iupper = itspace->dims[0];
    int jlower = 0;
    int jupper = itspace->dims[1];
    int idxs[2];

    int arg0idxs[2];
    if (arg0.argtype == OP_ARG_MAT) {
      int iut;
      int jut;
      arg0idxs[0] = 0;
      arg0idxs[1] = 1;
      if (arg0.idx < -1) {
        iut = arg0.map->dim;
      } else if (arg0.idx < OP_I_OFFSET) {
        iut = itspace->dims[op_i(arg0.idx)-1];
        arg0idxs[0] = op_i(arg0.idx) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 0, aborting\n");
        exit(-1);
      }
      if (arg0.idx2 < -1) {
        jut = arg0.map2->dim;
      } else if (arg0.idx2 < OP_I_OFFSET) {
        jut = itspace->dims[op_i(arg0.idx2)-1];
        arg0idxs[1] = op_i(arg0.idx2) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 0, aborting\n");
        exit(-1);
      }
      if (iut != iupper || jut != jupper) {
        printf("Map dimensions do not match iteration space, aborting\n");
        exit(-1);
      }
    }

    int arg1idxs[2];
    if (arg1.argtype == OP_ARG_MAT) {
      int iut;
      int jut;
      arg1idxs[0] = 0;
      arg1idxs[1] = 1;
      if (arg1.idx < -1) {
        iut = arg1.map->dim;
      } else if (arg1.idx < OP_I_OFFSET) {
        iut = itspace->dims[op_i(arg1.idx)-1];
        arg1idxs[0] = op_i(arg1.idx) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 1, aborting\n");
        exit(-1);
      }
      if (arg1.idx2 < -1) {
        jut = arg1.map2->dim;
      } else if (arg1.idx2 < OP_I_OFFSET) {
        jut = itspace->dims[op_i(arg1.idx2)-1];
        arg1idxs[1] = op_i(arg1.idx2) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 1, aborting\n");
        exit(-1);
      }
      if (iut != iupper || jut != jupper) {
        printf("Map dimensions do not match iteration space, aborting\n");
        exit(-1);
      }
    }

    int arg2idxs[2];
    if (arg2.argtype == OP_ARG_MAT) {
      int iut;
      int jut;
      arg2idxs[0] = 0;
      arg2idxs[1] = 1;
      if (arg2.idx < -1) {
        iut = arg2.map->dim;
      } else if (arg2.idx < OP_I_OFFSET) {
        iut = itspace->dims[op_i(arg2.idx)-1];
        arg2idxs[0] = op_i(arg2.idx) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 2, aborting\n");
        exit(-1);
      }
      if (arg2.idx2 < -1) {
        jut = arg2.map2->dim;
      } else if (arg2.idx2 < OP_I_OFFSET) {
        jut = itspace->dims[op_i(arg2.idx2)-1];
        arg2idxs[1] = op_i(arg2.idx2) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 2, aborting\n");
        exit(-1);
      }
      if (iut != iupper || jut != jupper) {
        printf("Map dimensions do not match iteration space, aborting\n");
        exit(-1);
      }
    }

    int arg3idxs[2];
    if (arg3.argtype == OP_ARG_MAT) {
      int iut;
      int jut;
      arg3idxs[0] = 0;
      arg3idxs[1] = 1;
      if (arg3.idx < -1) {
        iut = arg3.map->dim;
      } else if (arg3.idx < OP_I_OFFSET) {
        iut = itspace->dims[op_i(arg3.idx)-1];
        arg3idxs[0] = op_i(arg3.idx) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 3, aborting\n");
        exit(-1);
      }
      if (arg3.idx2 < -1) {
        jut = arg3.map2->dim;
      } else if (arg3.idx2 < OP_I_OFFSET) {
        jut = itspace->dims[op_i(arg3.idx2)-1];
        arg3idxs[1] = op_i(arg3.idx2) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 3, aborting\n");
        exit(-1);
      }
      if (iut != iupper || jut != jupper) {
        printf("Map dimensions do not match iteration space, aborting\n");
        exit(-1);
      }
    }

    int arg4idxs[2];
    if (arg4.argtype == OP_ARG_MAT) {
      int iut;
      int jut;
      arg4idxs[0] = 0;
      arg4idxs[1] = 1;
      if (arg4.idx < -1) {
        iut = arg4.map->dim;
      } else if (arg4.idx < OP_I_OFFSET) {
        iut = itspace->dims[op_i(arg4.idx)-1];
        arg4idxs[0] = op_i(arg4.idx) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 4, aborting\n");
        exit(-1);
      }
      if (arg4.idx2 < -1) {
        jut = arg4.map2->dim;
      } else if (arg4.idx2 < OP_I_OFFSET) {
        jut = itspace->dims[op_i(arg4.idx2)-1];
        arg4idxs[1] = op_i(arg4.idx2) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 4, aborting\n");
        exit(-1);
      }
      if (iut != iupper || jut != jupper) {
        printf("Map dimensions do not match iteration space, aborting\n");
        exit(-1);
      }
    }


    for (idxs[0] = ilower; idxs[0] < iupper; idxs[0]++) {
      for (idxs[1] = jlower; idxs[1] < jupper; idxs[1]++ ) {


        if (arg0.argtype == OP_ARG_MAT) {
          ((T0 *)p_arg0)[0] = (T0)0;
        }

        if (arg1.argtype == OP_ARG_MAT) {
          ((T1 *)p_arg1)[0] = (T1)0;
        }

        if (arg2.argtype == OP_ARG_MAT) {
          ((T2 *)p_arg2)[0] = (T2)0;
        }

        if (arg3.argtype == OP_ARG_MAT) {
          ((T3 *)p_arg3)[0] = (T3)0;
        }

        if (arg4.argtype == OP_ARG_MAT) {
          ((T4 *)p_arg4)[0] = (T4)0;
        }

        kernel( (T0 *)p_arg0, (T1 *)p_arg1, (T2 *)p_arg2, (T3 *)p_arg3,
                (T4 *)p_arg4, idxs[0], idxs[1]);
        // Assemble local matrix into global matrix

        if (arg0.argtype == OP_ARG_MAT) {
          const int rows = arg0.map->dim;
          const int cols = arg0.map2->dim;
          op_mat_addto(arg0.mat, p_arg0,
                       1, arg0.map->map + n*rows + idxs[arg0idxs[0]],
                       1, arg0.map2->map + n*cols + idxs[arg0idxs[1]]);
        }

        if (arg1.argtype == OP_ARG_MAT) {
          const int rows = arg1.map->dim;
          const int cols = arg1.map2->dim;
          op_mat_addto(arg1.mat, p_arg1,
                       1, arg1.map->map + n*rows + idxs[arg1idxs[0]],
                       1, arg1.map2->map + n*cols + idxs[arg1idxs[1]]);
        }

        if (arg2.argtype == OP_ARG_MAT) {
          const int rows = arg2.map->dim;
          const int cols = arg2.map2->dim;
          op_mat_addto(arg2.mat, p_arg2,
                       1, arg2.map->map + n*rows + idxs[arg2idxs[0]],
                       1, arg2.map2->map + n*cols + idxs[arg2idxs[1]]);
        }

        if (arg3.argtype == OP_ARG_MAT) {
          const int rows = arg3.map->dim;
          const int cols = arg3.map2->dim;
          op_mat_addto(arg3.mat, p_arg3,
                       1, arg3.map->map + n*rows + idxs[arg3idxs[0]],
                       1, arg3.map2->map + n*cols + idxs[arg3idxs[1]]);
        }

        if (arg4.argtype == OP_ARG_MAT) {
          const int rows = arg4.map->dim;
          const int cols = arg4.map2->dim;
          op_mat_addto(arg4.mat, p_arg4,
                       1, arg4.map->map + n*rows + idxs[arg4idxs[0]],
                       1, arg4.map2->map + n*cols + idxs[arg4idxs[1]]);
        }

      }
    }
  }

  // Free memory for vector map indices

  if ((arg0.argtype == OP_ARG_DAT && arg0.idx < -1) || arg0.argtype == OP_ARG_MAT) free(p_arg0);

  if ((arg1.argtype == OP_ARG_DAT && arg1.idx < -1) || arg1.argtype == OP_ARG_MAT) free(p_arg1);

  if ((arg2.argtype == OP_ARG_DAT && arg2.idx < -1) || arg2.argtype == OP_ARG_MAT) free(p_arg2);

  if ((arg3.argtype == OP_ARG_DAT && arg3.idx < -1) || arg3.argtype == OP_ARG_MAT) free(p_arg3);

  if ((arg4.argtype == OP_ARG_DAT && arg4.idx < -1) || arg4.argtype == OP_ARG_MAT) free(p_arg4);

  free(itspace->dims);
  free(itspace);
  // Global matrix assembly
  if (arg0.argtype == OP_ARG_MAT) op_mat_assemble(arg0.mat);
  if (arg1.argtype == OP_ARG_MAT) op_mat_assemble(arg1.mat);
  if (arg2.argtype == OP_ARG_MAT) op_mat_assemble(arg2.mat);
  if (arg3.argtype == OP_ARG_MAT) op_mat_assemble(arg3.mat);
  if (arg4.argtype == OP_ARG_MAT) op_mat_assemble(arg4.mat);
}

//
// op_par_loop routine for 6 arguments
//

template < class T0, class T1, class T2, class T3,
           class T4, class T5 >
void op_par_loop ( void (*kernel)( T0*, T1*, T2*, T3*,
                                   T4*, T5* ),
  char const * name, op_set set,
  op_arg arg0, op_arg arg1, op_arg arg2, op_arg arg3,
  op_arg arg4, op_arg arg5 )
{
  char *p_arg0 = 0, *p_arg1 = 0, *p_arg2 = 0, *p_arg3 = 0,
       *p_arg4 = 0, *p_arg5 = 0;

  // consistency checks

  int ninds=0;

  if (OP_diags>0) {
    op_arg_check(set,0 ,arg0 ,&ninds,name);
    op_arg_check(set,1 ,arg1 ,&ninds,name);
    op_arg_check(set,2 ,arg2 ,&ninds,name);
    op_arg_check(set,3 ,arg3 ,&ninds,name);
    op_arg_check(set,4 ,arg4 ,&ninds,name);
    op_arg_check(set,5 ,arg5 ,&ninds,name);
  }

  if (OP_diags>2) {
    if (ninds==0)
      printf(" kernel routine w/o indirection:  %s \n",name);
    else
      printf(" kernel routine with indirection: %s \n",name);
  }

  // Allocate memory for vector map indices

  switch ( arg0.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg0 = arg0.data;
      break;
    case OP_ARG_DAT:
      if (arg0.idx  < -1)
        p_arg0 = (char *)malloc(arg0.map->dim*sizeof(T0));
      break;
    case OP_ARG_MAT:
      p_arg0 = (char*) malloc(arg0.map->dim * arg0.map2->dim * arg0.size);
      break;
  }

  switch ( arg1.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg1 = arg1.data;
      break;
    case OP_ARG_DAT:
      if (arg1.idx  < -1)
        p_arg1 = (char *)malloc(arg1.map->dim*sizeof(T1));
      break;
    case OP_ARG_MAT:
      p_arg1 = (char*) malloc(arg1.map->dim * arg1.map2->dim * arg1.size);
      break;
  }

  switch ( arg2.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg2 = arg2.data;
      break;
    case OP_ARG_DAT:
      if (arg2.idx  < -1)
        p_arg2 = (char *)malloc(arg2.map->dim*sizeof(T2));
      break;
    case OP_ARG_MAT:
      p_arg2 = (char*) malloc(arg2.map->dim * arg2.map2->dim * arg2.size);
      break;
  }

  switch ( arg3.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg3 = arg3.data;
      break;
    case OP_ARG_DAT:
      if (arg3.idx  < -1)
        p_arg3 = (char *)malloc(arg3.map->dim*sizeof(T3));
      break;
    case OP_ARG_MAT:
      p_arg3 = (char*) malloc(arg3.map->dim * arg3.map2->dim * arg3.size);
      break;
  }

  switch ( arg4.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg4 = arg4.data;
      break;
    case OP_ARG_DAT:
      if (arg4.idx  < -1)
        p_arg4 = (char *)malloc(arg4.map->dim*sizeof(T4));
      break;
    case OP_ARG_MAT:
      p_arg4 = (char*) malloc(arg4.map->dim * arg4.map2->dim * arg4.size);
      break;
  }

  switch ( arg5.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg5 = arg5.data;
      break;
    case OP_ARG_DAT:
      if (arg5.idx  < -1)
        p_arg5 = (char *)malloc(arg5.map->dim*sizeof(T5));
      break;
    case OP_ARG_MAT:
      p_arg5 = (char*) malloc(arg5.map->dim * arg5.map2->dim * arg5.size);
      break;
  }

  // loop over set elements

  for (int n=0; n<set->size; n++) {
    // Copy in of vector map indices

    if (arg0.argtype == OP_ARG_DAT) {
      if (arg0.idx < -1)
        copy_in(n, arg0, (char**)p_arg0);
      else
        op_arg_set(n, arg0, &p_arg0 );
    }

    if (arg1.argtype == OP_ARG_DAT) {
      if (arg1.idx < -1)
        copy_in(n, arg1, (char**)p_arg1);
      else
        op_arg_set(n, arg1, &p_arg1 );
    }

    if (arg2.argtype == OP_ARG_DAT) {
      if (arg2.idx < -1)
        copy_in(n, arg2, (char**)p_arg2);
      else
        op_arg_set(n, arg2, &p_arg2 );
    }

    if (arg3.argtype == OP_ARG_DAT) {
      if (arg3.idx < -1)
        copy_in(n, arg3, (char**)p_arg3);
      else
        op_arg_set(n, arg3, &p_arg3 );
    }

    if (arg4.argtype == OP_ARG_DAT) {
      if (arg4.idx < -1)
        copy_in(n, arg4, (char**)p_arg4);
      else
        op_arg_set(n, arg4, &p_arg4 );
    }

    if (arg5.argtype == OP_ARG_DAT) {
      if (arg5.idx < -1)
        copy_in(n, arg5, (char**)p_arg5);
      else
        op_arg_set(n, arg5, &p_arg5 );
    }

    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_arg0, (T1 *)p_arg1, (T2 *)p_arg2, (T3 *)p_arg3,
            (T4 *)p_arg4, (T5 *)p_arg5 );
    // Assemble local matrix into global matrix

    if (arg0.argtype == OP_ARG_MAT) {
      const int rows = arg0.map->dim;
      const int cols = arg0.map2->dim;
      op_mat_addto( arg0.mat, p_arg0, rows, arg0.map->map + n*rows, cols, arg0.map2->map + n*cols);
    }

    if (arg1.argtype == OP_ARG_MAT) {
      const int rows = arg1.map->dim;
      const int cols = arg1.map2->dim;
      op_mat_addto( arg1.mat, p_arg1, rows, arg1.map->map + n*rows, cols, arg1.map2->map + n*cols);
    }

    if (arg2.argtype == OP_ARG_MAT) {
      const int rows = arg2.map->dim;
      const int cols = arg2.map2->dim;
      op_mat_addto( arg2.mat, p_arg2, rows, arg2.map->map + n*rows, cols, arg2.map2->map + n*cols);
    }

    if (arg3.argtype == OP_ARG_MAT) {
      const int rows = arg3.map->dim;
      const int cols = arg3.map2->dim;
      op_mat_addto( arg3.mat, p_arg3, rows, arg3.map->map + n*rows, cols, arg3.map2->map + n*cols);
    }

    if (arg4.argtype == OP_ARG_MAT) {
      const int rows = arg4.map->dim;
      const int cols = arg4.map2->dim;
      op_mat_addto( arg4.mat, p_arg4, rows, arg4.map->map + n*rows, cols, arg4.map2->map + n*cols);
    }

    if (arg5.argtype == OP_ARG_MAT) {
      const int rows = arg5.map->dim;
      const int cols = arg5.map2->dim;
      op_mat_addto( arg5.mat, p_arg5, rows, arg5.map->map + n*rows, cols, arg5.map2->map + n*cols);
    }

  }

  // Free memory for vector map indices

  if ((arg0.argtype == OP_ARG_DAT && arg0.idx < -1) || arg0.argtype == OP_ARG_MAT) free(p_arg0);

  if ((arg1.argtype == OP_ARG_DAT && arg1.idx < -1) || arg1.argtype == OP_ARG_MAT) free(p_arg1);

  if ((arg2.argtype == OP_ARG_DAT && arg2.idx < -1) || arg2.argtype == OP_ARG_MAT) free(p_arg2);

  if ((arg3.argtype == OP_ARG_DAT && arg3.idx < -1) || arg3.argtype == OP_ARG_MAT) free(p_arg3);

  if ((arg4.argtype == OP_ARG_DAT && arg4.idx < -1) || arg4.argtype == OP_ARG_MAT) free(p_arg4);

  if ((arg5.argtype == OP_ARG_DAT && arg5.idx < -1) || arg5.argtype == OP_ARG_MAT) free(p_arg5);

  // Global matrix assembly
  if (arg0.argtype == OP_ARG_MAT) op_mat_assemble(arg0.mat);
  if (arg1.argtype == OP_ARG_MAT) op_mat_assemble(arg1.mat);
  if (arg2.argtype == OP_ARG_MAT) op_mat_assemble(arg2.mat);
  if (arg3.argtype == OP_ARG_MAT) op_mat_assemble(arg3.mat);
  if (arg4.argtype == OP_ARG_MAT) op_mat_assemble(arg4.mat);
  if (arg5.argtype == OP_ARG_MAT) op_mat_assemble(arg5.mat);
}

//
// op_par_loop routine for 6 arguments with op_iteration_space call
//

template < class T0, class T1, class T2, class T3,
           class T4, class T5 >
void op_par_loop ( void (*kernel)( T0*, T1*, T2*, T3*,
                                   T4*, T5*, int, int ),
  char const * name, op_itspace itspace,
  op_arg arg0, op_arg arg1, op_arg arg2, op_arg arg3,
  op_arg arg4, op_arg arg5 )
{
  char *p_arg0 = 0, *p_arg1 = 0, *p_arg2 = 0, *p_arg3 = 0,
       *p_arg4 = 0, *p_arg5 = 0;
  op_set set = itspace->set;
  // consistency checks

  int ninds=0;

  if (OP_diags>0) {
    op_arg_check(set,0 ,arg0 ,&ninds,name);
    op_arg_check(set,1 ,arg1 ,&ninds,name);
    op_arg_check(set,2 ,arg2 ,&ninds,name);
    op_arg_check(set,3 ,arg3 ,&ninds,name);
    op_arg_check(set,4 ,arg4 ,&ninds,name);
    op_arg_check(set,5 ,arg5 ,&ninds,name);
  }

  if (OP_diags>2) {
    if (ninds==0)
      printf(" kernel routine w/o indirection:  %s \n",name);
    else
      printf(" kernel routine with indirection: %s \n",name);
  }

  // Allocate memory for vector map indices

  switch ( arg0.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg0 = arg0.data;
      break;
    case OP_ARG_DAT:
      if (arg0.idx < -1)
        p_arg0 = (char *)malloc(arg0.map->dim*sizeof(T0));
      break;
    case OP_ARG_MAT:
      p_arg0 = (char*) malloc(sizeof(T0));
      break;
  }

  switch ( arg1.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg1 = arg1.data;
      break;
    case OP_ARG_DAT:
      if (arg1.idx < -1)
        p_arg1 = (char *)malloc(arg1.map->dim*sizeof(T1));
      break;
    case OP_ARG_MAT:
      p_arg1 = (char*) malloc(sizeof(T1));
      break;
  }

  switch ( arg2.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg2 = arg2.data;
      break;
    case OP_ARG_DAT:
      if (arg2.idx < -1)
        p_arg2 = (char *)malloc(arg2.map->dim*sizeof(T2));
      break;
    case OP_ARG_MAT:
      p_arg2 = (char*) malloc(sizeof(T2));
      break;
  }

  switch ( arg3.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg3 = arg3.data;
      break;
    case OP_ARG_DAT:
      if (arg3.idx < -1)
        p_arg3 = (char *)malloc(arg3.map->dim*sizeof(T3));
      break;
    case OP_ARG_MAT:
      p_arg3 = (char*) malloc(sizeof(T3));
      break;
  }

  switch ( arg4.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg4 = arg4.data;
      break;
    case OP_ARG_DAT:
      if (arg4.idx < -1)
        p_arg4 = (char *)malloc(arg4.map->dim*sizeof(T4));
      break;
    case OP_ARG_MAT:
      p_arg4 = (char*) malloc(sizeof(T4));
      break;
  }

  switch ( arg5.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg5 = arg5.data;
      break;
    case OP_ARG_DAT:
      if (arg5.idx < -1)
        p_arg5 = (char *)malloc(arg5.map->dim*sizeof(T5));
      break;
    case OP_ARG_MAT:
      p_arg5 = (char*) malloc(sizeof(T5));
      break;
  }

  // loop over set elements

  for (int n=0; n<set->size; n++) {
    // Copy in of vector map indices

    if (arg0.argtype == OP_ARG_DAT) {
      if (arg0.idx < -1)
        copy_in(n, arg0, (char**)p_arg0);
      else
        op_arg_set(n, arg0, &p_arg0 );
    }

    if (arg1.argtype == OP_ARG_DAT) {
      if (arg1.idx < -1)
        copy_in(n, arg1, (char**)p_arg1);
      else
        op_arg_set(n, arg1, &p_arg1 );
    }

    if (arg2.argtype == OP_ARG_DAT) {
      if (arg2.idx < -1)
        copy_in(n, arg2, (char**)p_arg2);
      else
        op_arg_set(n, arg2, &p_arg2 );
    }

    if (arg3.argtype == OP_ARG_DAT) {
      if (arg3.idx < -1)
        copy_in(n, arg3, (char**)p_arg3);
      else
        op_arg_set(n, arg3, &p_arg3 );
    }

    if (arg4.argtype == OP_ARG_DAT) {
      if (arg4.idx < -1)
        copy_in(n, arg4, (char**)p_arg4);
      else
        op_arg_set(n, arg4, &p_arg4 );
    }

    if (arg5.argtype == OP_ARG_DAT) {
      if (arg5.idx < -1)
        copy_in(n, arg5, (char**)p_arg5);
      else
        op_arg_set(n, arg5, &p_arg5 );
    }

    // call kernel function, passing in pointers to data
    int ilower = 0;
    int iupper = itspace->dims[0];
    int jlower = 0;
    int jupper = itspace->dims[1];
    int idxs[2];

    int arg0idxs[2];
    if (arg0.argtype == OP_ARG_MAT) {
      int iut;
      int jut;
      arg0idxs[0] = 0;
      arg0idxs[1] = 1;
      if (arg0.idx < -1) {
        iut = arg0.map->dim;
      } else if (arg0.idx < OP_I_OFFSET) {
        iut = itspace->dims[op_i(arg0.idx)-1];
        arg0idxs[0] = op_i(arg0.idx) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 0, aborting\n");
        exit(-1);
      }
      if (arg0.idx2 < -1) {
        jut = arg0.map2->dim;
      } else if (arg0.idx2 < OP_I_OFFSET) {
        jut = itspace->dims[op_i(arg0.idx2)-1];
        arg0idxs[1] = op_i(arg0.idx2) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 0, aborting\n");
        exit(-1);
      }
      if (iut != iupper || jut != jupper) {
        printf("Map dimensions do not match iteration space, aborting\n");
        exit(-1);
      }
    }

    int arg1idxs[2];
    if (arg1.argtype == OP_ARG_MAT) {
      int iut;
      int jut;
      arg1idxs[0] = 0;
      arg1idxs[1] = 1;
      if (arg1.idx < -1) {
        iut = arg1.map->dim;
      } else if (arg1.idx < OP_I_OFFSET) {
        iut = itspace->dims[op_i(arg1.idx)-1];
        arg1idxs[0] = op_i(arg1.idx) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 1, aborting\n");
        exit(-1);
      }
      if (arg1.idx2 < -1) {
        jut = arg1.map2->dim;
      } else if (arg1.idx2 < OP_I_OFFSET) {
        jut = itspace->dims[op_i(arg1.idx2)-1];
        arg1idxs[1] = op_i(arg1.idx2) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 1, aborting\n");
        exit(-1);
      }
      if (iut != iupper || jut != jupper) {
        printf("Map dimensions do not match iteration space, aborting\n");
        exit(-1);
      }
    }

    int arg2idxs[2];
    if (arg2.argtype == OP_ARG_MAT) {
      int iut;
      int jut;
      arg2idxs[0] = 0;
      arg2idxs[1] = 1;
      if (arg2.idx < -1) {
        iut = arg2.map->dim;
      } else if (arg2.idx < OP_I_OFFSET) {
        iut = itspace->dims[op_i(arg2.idx)-1];
        arg2idxs[0] = op_i(arg2.idx) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 2, aborting\n");
        exit(-1);
      }
      if (arg2.idx2 < -1) {
        jut = arg2.map2->dim;
      } else if (arg2.idx2 < OP_I_OFFSET) {
        jut = itspace->dims[op_i(arg2.idx2)-1];
        arg2idxs[1] = op_i(arg2.idx2) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 2, aborting\n");
        exit(-1);
      }
      if (iut != iupper || jut != jupper) {
        printf("Map dimensions do not match iteration space, aborting\n");
        exit(-1);
      }
    }

    int arg3idxs[2];
    if (arg3.argtype == OP_ARG_MAT) {
      int iut;
      int jut;
      arg3idxs[0] = 0;
      arg3idxs[1] = 1;
      if (arg3.idx < -1) {
        iut = arg3.map->dim;
      } else if (arg3.idx < OP_I_OFFSET) {
        iut = itspace->dims[op_i(arg3.idx)-1];
        arg3idxs[0] = op_i(arg3.idx) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 3, aborting\n");
        exit(-1);
      }
      if (arg3.idx2 < -1) {
        jut = arg3.map2->dim;
      } else if (arg3.idx2 < OP_I_OFFSET) {
        jut = itspace->dims[op_i(arg3.idx2)-1];
        arg3idxs[1] = op_i(arg3.idx2) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 3, aborting\n");
        exit(-1);
      }
      if (iut != iupper || jut != jupper) {
        printf("Map dimensions do not match iteration space, aborting\n");
        exit(-1);
      }
    }

    int arg4idxs[2];
    if (arg4.argtype == OP_ARG_MAT) {
      int iut;
      int jut;
      arg4idxs[0] = 0;
      arg4idxs[1] = 1;
      if (arg4.idx < -1) {
        iut = arg4.map->dim;
      } else if (arg4.idx < OP_I_OFFSET) {
        iut = itspace->dims[op_i(arg4.idx)-1];
        arg4idxs[0] = op_i(arg4.idx) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 4, aborting\n");
        exit(-1);
      }
      if (arg4.idx2 < -1) {
        jut = arg4.map2->dim;
      } else if (arg4.idx2 < OP_I_OFFSET) {
        jut = itspace->dims[op_i(arg4.idx2)-1];
        arg4idxs[1] = op_i(arg4.idx2) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 4, aborting\n");
        exit(-1);
      }
      if (iut != iupper || jut != jupper) {
        printf("Map dimensions do not match iteration space, aborting\n");
        exit(-1);
      }
    }

    int arg5idxs[2];
    if (arg5.argtype == OP_ARG_MAT) {
      int iut;
      int jut;
      arg5idxs[0] = 0;
      arg5idxs[1] = 1;
      if (arg5.idx < -1) {
        iut = arg5.map->dim;
      } else if (arg5.idx < OP_I_OFFSET) {
        iut = itspace->dims[op_i(arg5.idx)-1];
        arg5idxs[0] = op_i(arg5.idx) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 5, aborting\n");
        exit(-1);
      }
      if (arg5.idx2 < -1) {
        jut = arg5.map2->dim;
      } else if (arg5.idx2 < OP_I_OFFSET) {
        jut = itspace->dims[op_i(arg5.idx2)-1];
        arg5idxs[1] = op_i(arg5.idx2) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 5, aborting\n");
        exit(-1);
      }
      if (iut != iupper || jut != jupper) {
        printf("Map dimensions do not match iteration space, aborting\n");
        exit(-1);
      }
    }


    for (idxs[0] = ilower; idxs[0] < iupper; idxs[0]++) {
      for (idxs[1] = jlower; idxs[1] < jupper; idxs[1]++ ) {


        if (arg0.argtype == OP_ARG_MAT) {
          ((T0 *)p_arg0)[0] = (T0)0;
        }

        if (arg1.argtype == OP_ARG_MAT) {
          ((T1 *)p_arg1)[0] = (T1)0;
        }

        if (arg2.argtype == OP_ARG_MAT) {
          ((T2 *)p_arg2)[0] = (T2)0;
        }

        if (arg3.argtype == OP_ARG_MAT) {
          ((T3 *)p_arg3)[0] = (T3)0;
        }

        if (arg4.argtype == OP_ARG_MAT) {
          ((T4 *)p_arg4)[0] = (T4)0;
        }

        if (arg5.argtype == OP_ARG_MAT) {
          ((T5 *)p_arg5)[0] = (T5)0;
        }

        kernel( (T0 *)p_arg0, (T1 *)p_arg1, (T2 *)p_arg2, (T3 *)p_arg3,
                (T4 *)p_arg4, (T5 *)p_arg5, idxs[0], idxs[1]);
        // Assemble local matrix into global matrix

        if (arg0.argtype == OP_ARG_MAT) {
          const int rows = arg0.map->dim;
          const int cols = arg0.map2->dim;
          op_mat_addto(arg0.mat, p_arg0,
                       1, arg0.map->map + n*rows + idxs[arg0idxs[0]],
                       1, arg0.map2->map + n*cols + idxs[arg0idxs[1]]);
        }

        if (arg1.argtype == OP_ARG_MAT) {
          const int rows = arg1.map->dim;
          const int cols = arg1.map2->dim;
          op_mat_addto(arg1.mat, p_arg1,
                       1, arg1.map->map + n*rows + idxs[arg1idxs[0]],
                       1, arg1.map2->map + n*cols + idxs[arg1idxs[1]]);
        }

        if (arg2.argtype == OP_ARG_MAT) {
          const int rows = arg2.map->dim;
          const int cols = arg2.map2->dim;
          op_mat_addto(arg2.mat, p_arg2,
                       1, arg2.map->map + n*rows + idxs[arg2idxs[0]],
                       1, arg2.map2->map + n*cols + idxs[arg2idxs[1]]);
        }

        if (arg3.argtype == OP_ARG_MAT) {
          const int rows = arg3.map->dim;
          const int cols = arg3.map2->dim;
          op_mat_addto(arg3.mat, p_arg3,
                       1, arg3.map->map + n*rows + idxs[arg3idxs[0]],
                       1, arg3.map2->map + n*cols + idxs[arg3idxs[1]]);
        }

        if (arg4.argtype == OP_ARG_MAT) {
          const int rows = arg4.map->dim;
          const int cols = arg4.map2->dim;
          op_mat_addto(arg4.mat, p_arg4,
                       1, arg4.map->map + n*rows + idxs[arg4idxs[0]],
                       1, arg4.map2->map + n*cols + idxs[arg4idxs[1]]);
        }

        if (arg5.argtype == OP_ARG_MAT) {
          const int rows = arg5.map->dim;
          const int cols = arg5.map2->dim;
          op_mat_addto(arg5.mat, p_arg5,
                       1, arg5.map->map + n*rows + idxs[arg5idxs[0]],
                       1, arg5.map2->map + n*cols + idxs[arg5idxs[1]]);
        }

      }
    }
  }

  // Free memory for vector map indices

  if ((arg0.argtype == OP_ARG_DAT && arg0.idx < -1) || arg0.argtype == OP_ARG_MAT) free(p_arg0);

  if ((arg1.argtype == OP_ARG_DAT && arg1.idx < -1) || arg1.argtype == OP_ARG_MAT) free(p_arg1);

  if ((arg2.argtype == OP_ARG_DAT && arg2.idx < -1) || arg2.argtype == OP_ARG_MAT) free(p_arg2);

  if ((arg3.argtype == OP_ARG_DAT && arg3.idx < -1) || arg3.argtype == OP_ARG_MAT) free(p_arg3);

  if ((arg4.argtype == OP_ARG_DAT && arg4.idx < -1) || arg4.argtype == OP_ARG_MAT) free(p_arg4);

  if ((arg5.argtype == OP_ARG_DAT && arg5.idx < -1) || arg5.argtype == OP_ARG_MAT) free(p_arg5);

  free(itspace->dims);
  free(itspace);
  // Global matrix assembly
  if (arg0.argtype == OP_ARG_MAT) op_mat_assemble(arg0.mat);
  if (arg1.argtype == OP_ARG_MAT) op_mat_assemble(arg1.mat);
  if (arg2.argtype == OP_ARG_MAT) op_mat_assemble(arg2.mat);
  if (arg3.argtype == OP_ARG_MAT) op_mat_assemble(arg3.mat);
  if (arg4.argtype == OP_ARG_MAT) op_mat_assemble(arg4.mat);
  if (arg5.argtype == OP_ARG_MAT) op_mat_assemble(arg5.mat);
}

//
// op_par_loop routine for 7 arguments
//

template < class T0, class T1, class T2, class T3,
           class T4, class T5, class T6 >
void op_par_loop ( void (*kernel)( T0*, T1*, T2*, T3*,
                                   T4*, T5*, T6* ),
  char const * name, op_set set,
  op_arg arg0, op_arg arg1, op_arg arg2, op_arg arg3,
  op_arg arg4, op_arg arg5, op_arg arg6 )
{
  char *p_arg0 = 0, *p_arg1 = 0, *p_arg2 = 0, *p_arg3 = 0,
       *p_arg4 = 0, *p_arg5 = 0, *p_arg6 = 0;

  // consistency checks

  int ninds=0;

  if (OP_diags>0) {
    op_arg_check(set,0 ,arg0 ,&ninds,name);
    op_arg_check(set,1 ,arg1 ,&ninds,name);
    op_arg_check(set,2 ,arg2 ,&ninds,name);
    op_arg_check(set,3 ,arg3 ,&ninds,name);
    op_arg_check(set,4 ,arg4 ,&ninds,name);
    op_arg_check(set,5 ,arg5 ,&ninds,name);
    op_arg_check(set,6 ,arg6 ,&ninds,name);
  }

  if (OP_diags>2) {
    if (ninds==0)
      printf(" kernel routine w/o indirection:  %s \n",name);
    else
      printf(" kernel routine with indirection: %s \n",name);
  }

  // Allocate memory for vector map indices

  switch ( arg0.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg0 = arg0.data;
      break;
    case OP_ARG_DAT:
      if (arg0.idx  < -1)
        p_arg0 = (char *)malloc(arg0.map->dim*sizeof(T0));
      break;
    case OP_ARG_MAT:
      p_arg0 = (char*) malloc(arg0.map->dim * arg0.map2->dim * arg0.size);
      break;
  }

  switch ( arg1.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg1 = arg1.data;
      break;
    case OP_ARG_DAT:
      if (arg1.idx  < -1)
        p_arg1 = (char *)malloc(arg1.map->dim*sizeof(T1));
      break;
    case OP_ARG_MAT:
      p_arg1 = (char*) malloc(arg1.map->dim * arg1.map2->dim * arg1.size);
      break;
  }

  switch ( arg2.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg2 = arg2.data;
      break;
    case OP_ARG_DAT:
      if (arg2.idx  < -1)
        p_arg2 = (char *)malloc(arg2.map->dim*sizeof(T2));
      break;
    case OP_ARG_MAT:
      p_arg2 = (char*) malloc(arg2.map->dim * arg2.map2->dim * arg2.size);
      break;
  }

  switch ( arg3.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg3 = arg3.data;
      break;
    case OP_ARG_DAT:
      if (arg3.idx  < -1)
        p_arg3 = (char *)malloc(arg3.map->dim*sizeof(T3));
      break;
    case OP_ARG_MAT:
      p_arg3 = (char*) malloc(arg3.map->dim * arg3.map2->dim * arg3.size);
      break;
  }

  switch ( arg4.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg4 = arg4.data;
      break;
    case OP_ARG_DAT:
      if (arg4.idx  < -1)
        p_arg4 = (char *)malloc(arg4.map->dim*sizeof(T4));
      break;
    case OP_ARG_MAT:
      p_arg4 = (char*) malloc(arg4.map->dim * arg4.map2->dim * arg4.size);
      break;
  }

  switch ( arg5.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg5 = arg5.data;
      break;
    case OP_ARG_DAT:
      if (arg5.idx  < -1)
        p_arg5 = (char *)malloc(arg5.map->dim*sizeof(T5));
      break;
    case OP_ARG_MAT:
      p_arg5 = (char*) malloc(arg5.map->dim * arg5.map2->dim * arg5.size);
      break;
  }

  switch ( arg6.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg6 = arg6.data;
      break;
    case OP_ARG_DAT:
      if (arg6.idx  < -1)
        p_arg6 = (char *)malloc(arg6.map->dim*sizeof(T6));
      break;
    case OP_ARG_MAT:
      p_arg6 = (char*) malloc(arg6.map->dim * arg6.map2->dim * arg6.size);
      break;
  }

  // loop over set elements

  for (int n=0; n<set->size; n++) {
    // Copy in of vector map indices

    if (arg0.argtype == OP_ARG_DAT) {
      if (arg0.idx < -1)
        copy_in(n, arg0, (char**)p_arg0);
      else
        op_arg_set(n, arg0, &p_arg0 );
    }

    if (arg1.argtype == OP_ARG_DAT) {
      if (arg1.idx < -1)
        copy_in(n, arg1, (char**)p_arg1);
      else
        op_arg_set(n, arg1, &p_arg1 );
    }

    if (arg2.argtype == OP_ARG_DAT) {
      if (arg2.idx < -1)
        copy_in(n, arg2, (char**)p_arg2);
      else
        op_arg_set(n, arg2, &p_arg2 );
    }

    if (arg3.argtype == OP_ARG_DAT) {
      if (arg3.idx < -1)
        copy_in(n, arg3, (char**)p_arg3);
      else
        op_arg_set(n, arg3, &p_arg3 );
    }

    if (arg4.argtype == OP_ARG_DAT) {
      if (arg4.idx < -1)
        copy_in(n, arg4, (char**)p_arg4);
      else
        op_arg_set(n, arg4, &p_arg4 );
    }

    if (arg5.argtype == OP_ARG_DAT) {
      if (arg5.idx < -1)
        copy_in(n, arg5, (char**)p_arg5);
      else
        op_arg_set(n, arg5, &p_arg5 );
    }

    if (arg6.argtype == OP_ARG_DAT) {
      if (arg6.idx < -1)
        copy_in(n, arg6, (char**)p_arg6);
      else
        op_arg_set(n, arg6, &p_arg6 );
    }

    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_arg0, (T1 *)p_arg1, (T2 *)p_arg2, (T3 *)p_arg3,
            (T4 *)p_arg4, (T5 *)p_arg5, (T6 *)p_arg6 );
    // Assemble local matrix into global matrix

    if (arg0.argtype == OP_ARG_MAT) {
      const int rows = arg0.map->dim;
      const int cols = arg0.map2->dim;
      op_mat_addto( arg0.mat, p_arg0, rows, arg0.map->map + n*rows, cols, arg0.map2->map + n*cols);
    }

    if (arg1.argtype == OP_ARG_MAT) {
      const int rows = arg1.map->dim;
      const int cols = arg1.map2->dim;
      op_mat_addto( arg1.mat, p_arg1, rows, arg1.map->map + n*rows, cols, arg1.map2->map + n*cols);
    }

    if (arg2.argtype == OP_ARG_MAT) {
      const int rows = arg2.map->dim;
      const int cols = arg2.map2->dim;
      op_mat_addto( arg2.mat, p_arg2, rows, arg2.map->map + n*rows, cols, arg2.map2->map + n*cols);
    }

    if (arg3.argtype == OP_ARG_MAT) {
      const int rows = arg3.map->dim;
      const int cols = arg3.map2->dim;
      op_mat_addto( arg3.mat, p_arg3, rows, arg3.map->map + n*rows, cols, arg3.map2->map + n*cols);
    }

    if (arg4.argtype == OP_ARG_MAT) {
      const int rows = arg4.map->dim;
      const int cols = arg4.map2->dim;
      op_mat_addto( arg4.mat, p_arg4, rows, arg4.map->map + n*rows, cols, arg4.map2->map + n*cols);
    }

    if (arg5.argtype == OP_ARG_MAT) {
      const int rows = arg5.map->dim;
      const int cols = arg5.map2->dim;
      op_mat_addto( arg5.mat, p_arg5, rows, arg5.map->map + n*rows, cols, arg5.map2->map + n*cols);
    }

    if (arg6.argtype == OP_ARG_MAT) {
      const int rows = arg6.map->dim;
      const int cols = arg6.map2->dim;
      op_mat_addto( arg6.mat, p_arg6, rows, arg6.map->map + n*rows, cols, arg6.map2->map + n*cols);
    }

  }

  // Free memory for vector map indices

  if ((arg0.argtype == OP_ARG_DAT && arg0.idx < -1) || arg0.argtype == OP_ARG_MAT) free(p_arg0);

  if ((arg1.argtype == OP_ARG_DAT && arg1.idx < -1) || arg1.argtype == OP_ARG_MAT) free(p_arg1);

  if ((arg2.argtype == OP_ARG_DAT && arg2.idx < -1) || arg2.argtype == OP_ARG_MAT) free(p_arg2);

  if ((arg3.argtype == OP_ARG_DAT && arg3.idx < -1) || arg3.argtype == OP_ARG_MAT) free(p_arg3);

  if ((arg4.argtype == OP_ARG_DAT && arg4.idx < -1) || arg4.argtype == OP_ARG_MAT) free(p_arg4);

  if ((arg5.argtype == OP_ARG_DAT && arg5.idx < -1) || arg5.argtype == OP_ARG_MAT) free(p_arg5);

  if ((arg6.argtype == OP_ARG_DAT && arg6.idx < -1) || arg6.argtype == OP_ARG_MAT) free(p_arg6);

  // Global matrix assembly
  if (arg0.argtype == OP_ARG_MAT) op_mat_assemble(arg0.mat);
  if (arg1.argtype == OP_ARG_MAT) op_mat_assemble(arg1.mat);
  if (arg2.argtype == OP_ARG_MAT) op_mat_assemble(arg2.mat);
  if (arg3.argtype == OP_ARG_MAT) op_mat_assemble(arg3.mat);
  if (arg4.argtype == OP_ARG_MAT) op_mat_assemble(arg4.mat);
  if (arg5.argtype == OP_ARG_MAT) op_mat_assemble(arg5.mat);
  if (arg6.argtype == OP_ARG_MAT) op_mat_assemble(arg6.mat);
}

//
// op_par_loop routine for 7 arguments with op_iteration_space call
//

template < class T0, class T1, class T2, class T3,
           class T4, class T5, class T6 >
void op_par_loop ( void (*kernel)( T0*, T1*, T2*, T3*,
                                   T4*, T5*, T6*, int, int ),
  char const * name, op_itspace itspace,
  op_arg arg0, op_arg arg1, op_arg arg2, op_arg arg3,
  op_arg arg4, op_arg arg5, op_arg arg6 )
{
  char *p_arg0 = 0, *p_arg1 = 0, *p_arg2 = 0, *p_arg3 = 0,
       *p_arg4 = 0, *p_arg5 = 0, *p_arg6 = 0;
  op_set set = itspace->set;
  // consistency checks

  int ninds=0;

  if (OP_diags>0) {
    op_arg_check(set,0 ,arg0 ,&ninds,name);
    op_arg_check(set,1 ,arg1 ,&ninds,name);
    op_arg_check(set,2 ,arg2 ,&ninds,name);
    op_arg_check(set,3 ,arg3 ,&ninds,name);
    op_arg_check(set,4 ,arg4 ,&ninds,name);
    op_arg_check(set,5 ,arg5 ,&ninds,name);
    op_arg_check(set,6 ,arg6 ,&ninds,name);
  }

  if (OP_diags>2) {
    if (ninds==0)
      printf(" kernel routine w/o indirection:  %s \n",name);
    else
      printf(" kernel routine with indirection: %s \n",name);
  }

  // Allocate memory for vector map indices

  switch ( arg0.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg0 = arg0.data;
      break;
    case OP_ARG_DAT:
      if (arg0.idx < -1)
        p_arg0 = (char *)malloc(arg0.map->dim*sizeof(T0));
      break;
    case OP_ARG_MAT:
      p_arg0 = (char*) malloc(sizeof(T0));
      break;
  }

  switch ( arg1.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg1 = arg1.data;
      break;
    case OP_ARG_DAT:
      if (arg1.idx < -1)
        p_arg1 = (char *)malloc(arg1.map->dim*sizeof(T1));
      break;
    case OP_ARG_MAT:
      p_arg1 = (char*) malloc(sizeof(T1));
      break;
  }

  switch ( arg2.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg2 = arg2.data;
      break;
    case OP_ARG_DAT:
      if (arg2.idx < -1)
        p_arg2 = (char *)malloc(arg2.map->dim*sizeof(T2));
      break;
    case OP_ARG_MAT:
      p_arg2 = (char*) malloc(sizeof(T2));
      break;
  }

  switch ( arg3.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg3 = arg3.data;
      break;
    case OP_ARG_DAT:
      if (arg3.idx < -1)
        p_arg3 = (char *)malloc(arg3.map->dim*sizeof(T3));
      break;
    case OP_ARG_MAT:
      p_arg3 = (char*) malloc(sizeof(T3));
      break;
  }

  switch ( arg4.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg4 = arg4.data;
      break;
    case OP_ARG_DAT:
      if (arg4.idx < -1)
        p_arg4 = (char *)malloc(arg4.map->dim*sizeof(T4));
      break;
    case OP_ARG_MAT:
      p_arg4 = (char*) malloc(sizeof(T4));
      break;
  }

  switch ( arg5.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg5 = arg5.data;
      break;
    case OP_ARG_DAT:
      if (arg5.idx < -1)
        p_arg5 = (char *)malloc(arg5.map->dim*sizeof(T5));
      break;
    case OP_ARG_MAT:
      p_arg5 = (char*) malloc(sizeof(T5));
      break;
  }

  switch ( arg6.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg6 = arg6.data;
      break;
    case OP_ARG_DAT:
      if (arg6.idx < -1)
        p_arg6 = (char *)malloc(arg6.map->dim*sizeof(T6));
      break;
    case OP_ARG_MAT:
      p_arg6 = (char*) malloc(sizeof(T6));
      break;
  }

  // loop over set elements

  for (int n=0; n<set->size; n++) {
    // Copy in of vector map indices

    if (arg0.argtype == OP_ARG_DAT) {
      if (arg0.idx < -1)
        copy_in(n, arg0, (char**)p_arg0);
      else
        op_arg_set(n, arg0, &p_arg0 );
    }

    if (arg1.argtype == OP_ARG_DAT) {
      if (arg1.idx < -1)
        copy_in(n, arg1, (char**)p_arg1);
      else
        op_arg_set(n, arg1, &p_arg1 );
    }

    if (arg2.argtype == OP_ARG_DAT) {
      if (arg2.idx < -1)
        copy_in(n, arg2, (char**)p_arg2);
      else
        op_arg_set(n, arg2, &p_arg2 );
    }

    if (arg3.argtype == OP_ARG_DAT) {
      if (arg3.idx < -1)
        copy_in(n, arg3, (char**)p_arg3);
      else
        op_arg_set(n, arg3, &p_arg3 );
    }

    if (arg4.argtype == OP_ARG_DAT) {
      if (arg4.idx < -1)
        copy_in(n, arg4, (char**)p_arg4);
      else
        op_arg_set(n, arg4, &p_arg4 );
    }

    if (arg5.argtype == OP_ARG_DAT) {
      if (arg5.idx < -1)
        copy_in(n, arg5, (char**)p_arg5);
      else
        op_arg_set(n, arg5, &p_arg5 );
    }

    if (arg6.argtype == OP_ARG_DAT) {
      if (arg6.idx < -1)
        copy_in(n, arg6, (char**)p_arg6);
      else
        op_arg_set(n, arg6, &p_arg6 );
    }

    // call kernel function, passing in pointers to data
    int ilower = 0;
    int iupper = itspace->dims[0];
    int jlower = 0;
    int jupper = itspace->dims[1];
    int idxs[2];

    int arg0idxs[2];
    if (arg0.argtype == OP_ARG_MAT) {
      int iut;
      int jut;
      arg0idxs[0] = 0;
      arg0idxs[1] = 1;
      if (arg0.idx < -1) {
        iut = arg0.map->dim;
      } else if (arg0.idx < OP_I_OFFSET) {
        iut = itspace->dims[op_i(arg0.idx)-1];
        arg0idxs[0] = op_i(arg0.idx) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 0, aborting\n");
        exit(-1);
      }
      if (arg0.idx2 < -1) {
        jut = arg0.map2->dim;
      } else if (arg0.idx2 < OP_I_OFFSET) {
        jut = itspace->dims[op_i(arg0.idx2)-1];
        arg0idxs[1] = op_i(arg0.idx2) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 0, aborting\n");
        exit(-1);
      }
      if (iut != iupper || jut != jupper) {
        printf("Map dimensions do not match iteration space, aborting\n");
        exit(-1);
      }
    }

    int arg1idxs[2];
    if (arg1.argtype == OP_ARG_MAT) {
      int iut;
      int jut;
      arg1idxs[0] = 0;
      arg1idxs[1] = 1;
      if (arg1.idx < -1) {
        iut = arg1.map->dim;
      } else if (arg1.idx < OP_I_OFFSET) {
        iut = itspace->dims[op_i(arg1.idx)-1];
        arg1idxs[0] = op_i(arg1.idx) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 1, aborting\n");
        exit(-1);
      }
      if (arg1.idx2 < -1) {
        jut = arg1.map2->dim;
      } else if (arg1.idx2 < OP_I_OFFSET) {
        jut = itspace->dims[op_i(arg1.idx2)-1];
        arg1idxs[1] = op_i(arg1.idx2) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 1, aborting\n");
        exit(-1);
      }
      if (iut != iupper || jut != jupper) {
        printf("Map dimensions do not match iteration space, aborting\n");
        exit(-1);
      }
    }

    int arg2idxs[2];
    if (arg2.argtype == OP_ARG_MAT) {
      int iut;
      int jut;
      arg2idxs[0] = 0;
      arg2idxs[1] = 1;
      if (arg2.idx < -1) {
        iut = arg2.map->dim;
      } else if (arg2.idx < OP_I_OFFSET) {
        iut = itspace->dims[op_i(arg2.idx)-1];
        arg2idxs[0] = op_i(arg2.idx) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 2, aborting\n");
        exit(-1);
      }
      if (arg2.idx2 < -1) {
        jut = arg2.map2->dim;
      } else if (arg2.idx2 < OP_I_OFFSET) {
        jut = itspace->dims[op_i(arg2.idx2)-1];
        arg2idxs[1] = op_i(arg2.idx2) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 2, aborting\n");
        exit(-1);
      }
      if (iut != iupper || jut != jupper) {
        printf("Map dimensions do not match iteration space, aborting\n");
        exit(-1);
      }
    }

    int arg3idxs[2];
    if (arg3.argtype == OP_ARG_MAT) {
      int iut;
      int jut;
      arg3idxs[0] = 0;
      arg3idxs[1] = 1;
      if (arg3.idx < -1) {
        iut = arg3.map->dim;
      } else if (arg3.idx < OP_I_OFFSET) {
        iut = itspace->dims[op_i(arg3.idx)-1];
        arg3idxs[0] = op_i(arg3.idx) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 3, aborting\n");
        exit(-1);
      }
      if (arg3.idx2 < -1) {
        jut = arg3.map2->dim;
      } else if (arg3.idx2 < OP_I_OFFSET) {
        jut = itspace->dims[op_i(arg3.idx2)-1];
        arg3idxs[1] = op_i(arg3.idx2) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 3, aborting\n");
        exit(-1);
      }
      if (iut != iupper || jut != jupper) {
        printf("Map dimensions do not match iteration space, aborting\n");
        exit(-1);
      }
    }

    int arg4idxs[2];
    if (arg4.argtype == OP_ARG_MAT) {
      int iut;
      int jut;
      arg4idxs[0] = 0;
      arg4idxs[1] = 1;
      if (arg4.idx < -1) {
        iut = arg4.map->dim;
      } else if (arg4.idx < OP_I_OFFSET) {
        iut = itspace->dims[op_i(arg4.idx)-1];
        arg4idxs[0] = op_i(arg4.idx) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 4, aborting\n");
        exit(-1);
      }
      if (arg4.idx2 < -1) {
        jut = arg4.map2->dim;
      } else if (arg4.idx2 < OP_I_OFFSET) {
        jut = itspace->dims[op_i(arg4.idx2)-1];
        arg4idxs[1] = op_i(arg4.idx2) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 4, aborting\n");
        exit(-1);
      }
      if (iut != iupper || jut != jupper) {
        printf("Map dimensions do not match iteration space, aborting\n");
        exit(-1);
      }
    }

    int arg5idxs[2];
    if (arg5.argtype == OP_ARG_MAT) {
      int iut;
      int jut;
      arg5idxs[0] = 0;
      arg5idxs[1] = 1;
      if (arg5.idx < -1) {
        iut = arg5.map->dim;
      } else if (arg5.idx < OP_I_OFFSET) {
        iut = itspace->dims[op_i(arg5.idx)-1];
        arg5idxs[0] = op_i(arg5.idx) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 5, aborting\n");
        exit(-1);
      }
      if (arg5.idx2 < -1) {
        jut = arg5.map2->dim;
      } else if (arg5.idx2 < OP_I_OFFSET) {
        jut = itspace->dims[op_i(arg5.idx2)-1];
        arg5idxs[1] = op_i(arg5.idx2) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 5, aborting\n");
        exit(-1);
      }
      if (iut != iupper || jut != jupper) {
        printf("Map dimensions do not match iteration space, aborting\n");
        exit(-1);
      }
    }

    int arg6idxs[2];
    if (arg6.argtype == OP_ARG_MAT) {
      int iut;
      int jut;
      arg6idxs[0] = 0;
      arg6idxs[1] = 1;
      if (arg6.idx < -1) {
        iut = arg6.map->dim;
      } else if (arg6.idx < OP_I_OFFSET) {
        iut = itspace->dims[op_i(arg6.idx)-1];
        arg6idxs[0] = op_i(arg6.idx) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 6, aborting\n");
        exit(-1);
      }
      if (arg6.idx2 < -1) {
        jut = arg6.map2->dim;
      } else if (arg6.idx2 < OP_I_OFFSET) {
        jut = itspace->dims[op_i(arg6.idx2)-1];
        arg6idxs[1] = op_i(arg6.idx2) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 6, aborting\n");
        exit(-1);
      }
      if (iut != iupper || jut != jupper) {
        printf("Map dimensions do not match iteration space, aborting\n");
        exit(-1);
      }
    }


    for (idxs[0] = ilower; idxs[0] < iupper; idxs[0]++) {
      for (idxs[1] = jlower; idxs[1] < jupper; idxs[1]++ ) {


        if (arg0.argtype == OP_ARG_MAT) {
          ((T0 *)p_arg0)[0] = (T0)0;
        }

        if (arg1.argtype == OP_ARG_MAT) {
          ((T1 *)p_arg1)[0] = (T1)0;
        }

        if (arg2.argtype == OP_ARG_MAT) {
          ((T2 *)p_arg2)[0] = (T2)0;
        }

        if (arg3.argtype == OP_ARG_MAT) {
          ((T3 *)p_arg3)[0] = (T3)0;
        }

        if (arg4.argtype == OP_ARG_MAT) {
          ((T4 *)p_arg4)[0] = (T4)0;
        }

        if (arg5.argtype == OP_ARG_MAT) {
          ((T5 *)p_arg5)[0] = (T5)0;
        }

        if (arg6.argtype == OP_ARG_MAT) {
          ((T6 *)p_arg6)[0] = (T6)0;
        }

        kernel( (T0 *)p_arg0, (T1 *)p_arg1, (T2 *)p_arg2, (T3 *)p_arg3,
                (T4 *)p_arg4, (T5 *)p_arg5, (T6 *)p_arg6, idxs[0], idxs[1]);
        // Assemble local matrix into global matrix

        if (arg0.argtype == OP_ARG_MAT) {
          const int rows = arg0.map->dim;
          const int cols = arg0.map2->dim;
          op_mat_addto(arg0.mat, p_arg0,
                       1, arg0.map->map + n*rows + idxs[arg0idxs[0]],
                       1, arg0.map2->map + n*cols + idxs[arg0idxs[1]]);
        }

        if (arg1.argtype == OP_ARG_MAT) {
          const int rows = arg1.map->dim;
          const int cols = arg1.map2->dim;
          op_mat_addto(arg1.mat, p_arg1,
                       1, arg1.map->map + n*rows + idxs[arg1idxs[0]],
                       1, arg1.map2->map + n*cols + idxs[arg1idxs[1]]);
        }

        if (arg2.argtype == OP_ARG_MAT) {
          const int rows = arg2.map->dim;
          const int cols = arg2.map2->dim;
          op_mat_addto(arg2.mat, p_arg2,
                       1, arg2.map->map + n*rows + idxs[arg2idxs[0]],
                       1, arg2.map2->map + n*cols + idxs[arg2idxs[1]]);
        }

        if (arg3.argtype == OP_ARG_MAT) {
          const int rows = arg3.map->dim;
          const int cols = arg3.map2->dim;
          op_mat_addto(arg3.mat, p_arg3,
                       1, arg3.map->map + n*rows + idxs[arg3idxs[0]],
                       1, arg3.map2->map + n*cols + idxs[arg3idxs[1]]);
        }

        if (arg4.argtype == OP_ARG_MAT) {
          const int rows = arg4.map->dim;
          const int cols = arg4.map2->dim;
          op_mat_addto(arg4.mat, p_arg4,
                       1, arg4.map->map + n*rows + idxs[arg4idxs[0]],
                       1, arg4.map2->map + n*cols + idxs[arg4idxs[1]]);
        }

        if (arg5.argtype == OP_ARG_MAT) {
          const int rows = arg5.map->dim;
          const int cols = arg5.map2->dim;
          op_mat_addto(arg5.mat, p_arg5,
                       1, arg5.map->map + n*rows + idxs[arg5idxs[0]],
                       1, arg5.map2->map + n*cols + idxs[arg5idxs[1]]);
        }

        if (arg6.argtype == OP_ARG_MAT) {
          const int rows = arg6.map->dim;
          const int cols = arg6.map2->dim;
          op_mat_addto(arg6.mat, p_arg6,
                       1, arg6.map->map + n*rows + idxs[arg6idxs[0]],
                       1, arg6.map2->map + n*cols + idxs[arg6idxs[1]]);
        }

      }
    }
  }

  // Free memory for vector map indices

  if ((arg0.argtype == OP_ARG_DAT && arg0.idx < -1) || arg0.argtype == OP_ARG_MAT) free(p_arg0);

  if ((arg1.argtype == OP_ARG_DAT && arg1.idx < -1) || arg1.argtype == OP_ARG_MAT) free(p_arg1);

  if ((arg2.argtype == OP_ARG_DAT && arg2.idx < -1) || arg2.argtype == OP_ARG_MAT) free(p_arg2);

  if ((arg3.argtype == OP_ARG_DAT && arg3.idx < -1) || arg3.argtype == OP_ARG_MAT) free(p_arg3);

  if ((arg4.argtype == OP_ARG_DAT && arg4.idx < -1) || arg4.argtype == OP_ARG_MAT) free(p_arg4);

  if ((arg5.argtype == OP_ARG_DAT && arg5.idx < -1) || arg5.argtype == OP_ARG_MAT) free(p_arg5);

  if ((arg6.argtype == OP_ARG_DAT && arg6.idx < -1) || arg6.argtype == OP_ARG_MAT) free(p_arg6);

  free(itspace->dims);
  free(itspace);
  // Global matrix assembly
  if (arg0.argtype == OP_ARG_MAT) op_mat_assemble(arg0.mat);
  if (arg1.argtype == OP_ARG_MAT) op_mat_assemble(arg1.mat);
  if (arg2.argtype == OP_ARG_MAT) op_mat_assemble(arg2.mat);
  if (arg3.argtype == OP_ARG_MAT) op_mat_assemble(arg3.mat);
  if (arg4.argtype == OP_ARG_MAT) op_mat_assemble(arg4.mat);
  if (arg5.argtype == OP_ARG_MAT) op_mat_assemble(arg5.mat);
  if (arg6.argtype == OP_ARG_MAT) op_mat_assemble(arg6.mat);
}

//
// op_par_loop routine for 8 arguments
//

template < class T0, class T1, class T2, class T3,
           class T4, class T5, class T6, class T7 >
void op_par_loop ( void (*kernel)( T0*, T1*, T2*, T3*,
                                   T4*, T5*, T6*, T7* ),
  char const * name, op_set set,
  op_arg arg0, op_arg arg1, op_arg arg2, op_arg arg3,
  op_arg arg4, op_arg arg5, op_arg arg6, op_arg arg7 )
{
  char *p_arg0 = 0, *p_arg1 = 0, *p_arg2 = 0, *p_arg3 = 0,
       *p_arg4 = 0, *p_arg5 = 0, *p_arg6 = 0, *p_arg7 = 0;

  // consistency checks

  int ninds=0;

  if (OP_diags>0) {
    op_arg_check(set,0 ,arg0 ,&ninds,name);
    op_arg_check(set,1 ,arg1 ,&ninds,name);
    op_arg_check(set,2 ,arg2 ,&ninds,name);
    op_arg_check(set,3 ,arg3 ,&ninds,name);
    op_arg_check(set,4 ,arg4 ,&ninds,name);
    op_arg_check(set,5 ,arg5 ,&ninds,name);
    op_arg_check(set,6 ,arg6 ,&ninds,name);
    op_arg_check(set,7 ,arg7 ,&ninds,name);
  }

  if (OP_diags>2) {
    if (ninds==0)
      printf(" kernel routine w/o indirection:  %s \n",name);
    else
      printf(" kernel routine with indirection: %s \n",name);
  }

  // Allocate memory for vector map indices

  switch ( arg0.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg0 = arg0.data;
      break;
    case OP_ARG_DAT:
      if (arg0.idx  < -1)
        p_arg0 = (char *)malloc(arg0.map->dim*sizeof(T0));
      break;
    case OP_ARG_MAT:
      p_arg0 = (char*) malloc(arg0.map->dim * arg0.map2->dim * arg0.size);
      break;
  }

  switch ( arg1.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg1 = arg1.data;
      break;
    case OP_ARG_DAT:
      if (arg1.idx  < -1)
        p_arg1 = (char *)malloc(arg1.map->dim*sizeof(T1));
      break;
    case OP_ARG_MAT:
      p_arg1 = (char*) malloc(arg1.map->dim * arg1.map2->dim * arg1.size);
      break;
  }

  switch ( arg2.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg2 = arg2.data;
      break;
    case OP_ARG_DAT:
      if (arg2.idx  < -1)
        p_arg2 = (char *)malloc(arg2.map->dim*sizeof(T2));
      break;
    case OP_ARG_MAT:
      p_arg2 = (char*) malloc(arg2.map->dim * arg2.map2->dim * arg2.size);
      break;
  }

  switch ( arg3.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg3 = arg3.data;
      break;
    case OP_ARG_DAT:
      if (arg3.idx  < -1)
        p_arg3 = (char *)malloc(arg3.map->dim*sizeof(T3));
      break;
    case OP_ARG_MAT:
      p_arg3 = (char*) malloc(arg3.map->dim * arg3.map2->dim * arg3.size);
      break;
  }

  switch ( arg4.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg4 = arg4.data;
      break;
    case OP_ARG_DAT:
      if (arg4.idx  < -1)
        p_arg4 = (char *)malloc(arg4.map->dim*sizeof(T4));
      break;
    case OP_ARG_MAT:
      p_arg4 = (char*) malloc(arg4.map->dim * arg4.map2->dim * arg4.size);
      break;
  }

  switch ( arg5.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg5 = arg5.data;
      break;
    case OP_ARG_DAT:
      if (arg5.idx  < -1)
        p_arg5 = (char *)malloc(arg5.map->dim*sizeof(T5));
      break;
    case OP_ARG_MAT:
      p_arg5 = (char*) malloc(arg5.map->dim * arg5.map2->dim * arg5.size);
      break;
  }

  switch ( arg6.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg6 = arg6.data;
      break;
    case OP_ARG_DAT:
      if (arg6.idx  < -1)
        p_arg6 = (char *)malloc(arg6.map->dim*sizeof(T6));
      break;
    case OP_ARG_MAT:
      p_arg6 = (char*) malloc(arg6.map->dim * arg6.map2->dim * arg6.size);
      break;
  }

  switch ( arg7.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg7 = arg7.data;
      break;
    case OP_ARG_DAT:
      if (arg7.idx  < -1)
        p_arg7 = (char *)malloc(arg7.map->dim*sizeof(T7));
      break;
    case OP_ARG_MAT:
      p_arg7 = (char*) malloc(arg7.map->dim * arg7.map2->dim * arg7.size);
      break;
  }

  // loop over set elements

  for (int n=0; n<set->size; n++) {
    // Copy in of vector map indices

    if (arg0.argtype == OP_ARG_DAT) {
      if (arg0.idx < -1)
        copy_in(n, arg0, (char**)p_arg0);
      else
        op_arg_set(n, arg0, &p_arg0 );
    }

    if (arg1.argtype == OP_ARG_DAT) {
      if (arg1.idx < -1)
        copy_in(n, arg1, (char**)p_arg1);
      else
        op_arg_set(n, arg1, &p_arg1 );
    }

    if (arg2.argtype == OP_ARG_DAT) {
      if (arg2.idx < -1)
        copy_in(n, arg2, (char**)p_arg2);
      else
        op_arg_set(n, arg2, &p_arg2 );
    }

    if (arg3.argtype == OP_ARG_DAT) {
      if (arg3.idx < -1)
        copy_in(n, arg3, (char**)p_arg3);
      else
        op_arg_set(n, arg3, &p_arg3 );
    }

    if (arg4.argtype == OP_ARG_DAT) {
      if (arg4.idx < -1)
        copy_in(n, arg4, (char**)p_arg4);
      else
        op_arg_set(n, arg4, &p_arg4 );
    }

    if (arg5.argtype == OP_ARG_DAT) {
      if (arg5.idx < -1)
        copy_in(n, arg5, (char**)p_arg5);
      else
        op_arg_set(n, arg5, &p_arg5 );
    }

    if (arg6.argtype == OP_ARG_DAT) {
      if (arg6.idx < -1)
        copy_in(n, arg6, (char**)p_arg6);
      else
        op_arg_set(n, arg6, &p_arg6 );
    }

    if (arg7.argtype == OP_ARG_DAT) {
      if (arg7.idx < -1)
        copy_in(n, arg7, (char**)p_arg7);
      else
        op_arg_set(n, arg7, &p_arg7 );
    }

    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_arg0, (T1 *)p_arg1, (T2 *)p_arg2, (T3 *)p_arg3,
            (T4 *)p_arg4, (T5 *)p_arg5, (T6 *)p_arg6, (T7 *)p_arg7 );
    // Assemble local matrix into global matrix

    if (arg0.argtype == OP_ARG_MAT) {
      const int rows = arg0.map->dim;
      const int cols = arg0.map2->dim;
      op_mat_addto( arg0.mat, p_arg0, rows, arg0.map->map + n*rows, cols, arg0.map2->map + n*cols);
    }

    if (arg1.argtype == OP_ARG_MAT) {
      const int rows = arg1.map->dim;
      const int cols = arg1.map2->dim;
      op_mat_addto( arg1.mat, p_arg1, rows, arg1.map->map + n*rows, cols, arg1.map2->map + n*cols);
    }

    if (arg2.argtype == OP_ARG_MAT) {
      const int rows = arg2.map->dim;
      const int cols = arg2.map2->dim;
      op_mat_addto( arg2.mat, p_arg2, rows, arg2.map->map + n*rows, cols, arg2.map2->map + n*cols);
    }

    if (arg3.argtype == OP_ARG_MAT) {
      const int rows = arg3.map->dim;
      const int cols = arg3.map2->dim;
      op_mat_addto( arg3.mat, p_arg3, rows, arg3.map->map + n*rows, cols, arg3.map2->map + n*cols);
    }

    if (arg4.argtype == OP_ARG_MAT) {
      const int rows = arg4.map->dim;
      const int cols = arg4.map2->dim;
      op_mat_addto( arg4.mat, p_arg4, rows, arg4.map->map + n*rows, cols, arg4.map2->map + n*cols);
    }

    if (arg5.argtype == OP_ARG_MAT) {
      const int rows = arg5.map->dim;
      const int cols = arg5.map2->dim;
      op_mat_addto( arg5.mat, p_arg5, rows, arg5.map->map + n*rows, cols, arg5.map2->map + n*cols);
    }

    if (arg6.argtype == OP_ARG_MAT) {
      const int rows = arg6.map->dim;
      const int cols = arg6.map2->dim;
      op_mat_addto( arg6.mat, p_arg6, rows, arg6.map->map + n*rows, cols, arg6.map2->map + n*cols);
    }

    if (arg7.argtype == OP_ARG_MAT) {
      const int rows = arg7.map->dim;
      const int cols = arg7.map2->dim;
      op_mat_addto( arg7.mat, p_arg7, rows, arg7.map->map + n*rows, cols, arg7.map2->map + n*cols);
    }

  }

  // Free memory for vector map indices

  if ((arg0.argtype == OP_ARG_DAT && arg0.idx < -1) || arg0.argtype == OP_ARG_MAT) free(p_arg0);

  if ((arg1.argtype == OP_ARG_DAT && arg1.idx < -1) || arg1.argtype == OP_ARG_MAT) free(p_arg1);

  if ((arg2.argtype == OP_ARG_DAT && arg2.idx < -1) || arg2.argtype == OP_ARG_MAT) free(p_arg2);

  if ((arg3.argtype == OP_ARG_DAT && arg3.idx < -1) || arg3.argtype == OP_ARG_MAT) free(p_arg3);

  if ((arg4.argtype == OP_ARG_DAT && arg4.idx < -1) || arg4.argtype == OP_ARG_MAT) free(p_arg4);

  if ((arg5.argtype == OP_ARG_DAT && arg5.idx < -1) || arg5.argtype == OP_ARG_MAT) free(p_arg5);

  if ((arg6.argtype == OP_ARG_DAT && arg6.idx < -1) || arg6.argtype == OP_ARG_MAT) free(p_arg6);

  if ((arg7.argtype == OP_ARG_DAT && arg7.idx < -1) || arg7.argtype == OP_ARG_MAT) free(p_arg7);

  // Global matrix assembly
  if (arg0.argtype == OP_ARG_MAT) op_mat_assemble(arg0.mat);
  if (arg1.argtype == OP_ARG_MAT) op_mat_assemble(arg1.mat);
  if (arg2.argtype == OP_ARG_MAT) op_mat_assemble(arg2.mat);
  if (arg3.argtype == OP_ARG_MAT) op_mat_assemble(arg3.mat);
  if (arg4.argtype == OP_ARG_MAT) op_mat_assemble(arg4.mat);
  if (arg5.argtype == OP_ARG_MAT) op_mat_assemble(arg5.mat);
  if (arg6.argtype == OP_ARG_MAT) op_mat_assemble(arg6.mat);
  if (arg7.argtype == OP_ARG_MAT) op_mat_assemble(arg7.mat);
}

//
// op_par_loop routine for 8 arguments with op_iteration_space call
//

template < class T0, class T1, class T2, class T3,
           class T4, class T5, class T6, class T7 >
void op_par_loop ( void (*kernel)( T0*, T1*, T2*, T3*,
                                   T4*, T5*, T6*, T7*, int, int ),
  char const * name, op_itspace itspace,
  op_arg arg0, op_arg arg1, op_arg arg2, op_arg arg3,
  op_arg arg4, op_arg arg5, op_arg arg6, op_arg arg7 )
{
  char *p_arg0 = 0, *p_arg1 = 0, *p_arg2 = 0, *p_arg3 = 0,
       *p_arg4 = 0, *p_arg5 = 0, *p_arg6 = 0, *p_arg7 = 0;
  op_set set = itspace->set;
  // consistency checks

  int ninds=0;

  if (OP_diags>0) {
    op_arg_check(set,0 ,arg0 ,&ninds,name);
    op_arg_check(set,1 ,arg1 ,&ninds,name);
    op_arg_check(set,2 ,arg2 ,&ninds,name);
    op_arg_check(set,3 ,arg3 ,&ninds,name);
    op_arg_check(set,4 ,arg4 ,&ninds,name);
    op_arg_check(set,5 ,arg5 ,&ninds,name);
    op_arg_check(set,6 ,arg6 ,&ninds,name);
    op_arg_check(set,7 ,arg7 ,&ninds,name);
  }

  if (OP_diags>2) {
    if (ninds==0)
      printf(" kernel routine w/o indirection:  %s \n",name);
    else
      printf(" kernel routine with indirection: %s \n",name);
  }

  // Allocate memory for vector map indices

  switch ( arg0.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg0 = arg0.data;
      break;
    case OP_ARG_DAT:
      if (arg0.idx < -1)
        p_arg0 = (char *)malloc(arg0.map->dim*sizeof(T0));
      break;
    case OP_ARG_MAT:
      p_arg0 = (char*) malloc(sizeof(T0));
      break;
  }

  switch ( arg1.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg1 = arg1.data;
      break;
    case OP_ARG_DAT:
      if (arg1.idx < -1)
        p_arg1 = (char *)malloc(arg1.map->dim*sizeof(T1));
      break;
    case OP_ARG_MAT:
      p_arg1 = (char*) malloc(sizeof(T1));
      break;
  }

  switch ( arg2.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg2 = arg2.data;
      break;
    case OP_ARG_DAT:
      if (arg2.idx < -1)
        p_arg2 = (char *)malloc(arg2.map->dim*sizeof(T2));
      break;
    case OP_ARG_MAT:
      p_arg2 = (char*) malloc(sizeof(T2));
      break;
  }

  switch ( arg3.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg3 = arg3.data;
      break;
    case OP_ARG_DAT:
      if (arg3.idx < -1)
        p_arg3 = (char *)malloc(arg3.map->dim*sizeof(T3));
      break;
    case OP_ARG_MAT:
      p_arg3 = (char*) malloc(sizeof(T3));
      break;
  }

  switch ( arg4.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg4 = arg4.data;
      break;
    case OP_ARG_DAT:
      if (arg4.idx < -1)
        p_arg4 = (char *)malloc(arg4.map->dim*sizeof(T4));
      break;
    case OP_ARG_MAT:
      p_arg4 = (char*) malloc(sizeof(T4));
      break;
  }

  switch ( arg5.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg5 = arg5.data;
      break;
    case OP_ARG_DAT:
      if (arg5.idx < -1)
        p_arg5 = (char *)malloc(arg5.map->dim*sizeof(T5));
      break;
    case OP_ARG_MAT:
      p_arg5 = (char*) malloc(sizeof(T5));
      break;
  }

  switch ( arg6.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg6 = arg6.data;
      break;
    case OP_ARG_DAT:
      if (arg6.idx < -1)
        p_arg6 = (char *)malloc(arg6.map->dim*sizeof(T6));
      break;
    case OP_ARG_MAT:
      p_arg6 = (char*) malloc(sizeof(T6));
      break;
  }

  switch ( arg7.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg7 = arg7.data;
      break;
    case OP_ARG_DAT:
      if (arg7.idx < -1)
        p_arg7 = (char *)malloc(arg7.map->dim*sizeof(T7));
      break;
    case OP_ARG_MAT:
      p_arg7 = (char*) malloc(sizeof(T7));
      break;
  }

  // loop over set elements

  for (int n=0; n<set->size; n++) {
    // Copy in of vector map indices

    if (arg0.argtype == OP_ARG_DAT) {
      if (arg0.idx < -1)
        copy_in(n, arg0, (char**)p_arg0);
      else
        op_arg_set(n, arg0, &p_arg0 );
    }

    if (arg1.argtype == OP_ARG_DAT) {
      if (arg1.idx < -1)
        copy_in(n, arg1, (char**)p_arg1);
      else
        op_arg_set(n, arg1, &p_arg1 );
    }

    if (arg2.argtype == OP_ARG_DAT) {
      if (arg2.idx < -1)
        copy_in(n, arg2, (char**)p_arg2);
      else
        op_arg_set(n, arg2, &p_arg2 );
    }

    if (arg3.argtype == OP_ARG_DAT) {
      if (arg3.idx < -1)
        copy_in(n, arg3, (char**)p_arg3);
      else
        op_arg_set(n, arg3, &p_arg3 );
    }

    if (arg4.argtype == OP_ARG_DAT) {
      if (arg4.idx < -1)
        copy_in(n, arg4, (char**)p_arg4);
      else
        op_arg_set(n, arg4, &p_arg4 );
    }

    if (arg5.argtype == OP_ARG_DAT) {
      if (arg5.idx < -1)
        copy_in(n, arg5, (char**)p_arg5);
      else
        op_arg_set(n, arg5, &p_arg5 );
    }

    if (arg6.argtype == OP_ARG_DAT) {
      if (arg6.idx < -1)
        copy_in(n, arg6, (char**)p_arg6);
      else
        op_arg_set(n, arg6, &p_arg6 );
    }

    if (arg7.argtype == OP_ARG_DAT) {
      if (arg7.idx < -1)
        copy_in(n, arg7, (char**)p_arg7);
      else
        op_arg_set(n, arg7, &p_arg7 );
    }

    // call kernel function, passing in pointers to data
    int ilower = 0;
    int iupper = itspace->dims[0];
    int jlower = 0;
    int jupper = itspace->dims[1];
    int idxs[2];

    int arg0idxs[2];
    if (arg0.argtype == OP_ARG_MAT) {
      int iut;
      int jut;
      arg0idxs[0] = 0;
      arg0idxs[1] = 1;
      if (arg0.idx < -1) {
        iut = arg0.map->dim;
      } else if (arg0.idx < OP_I_OFFSET) {
        iut = itspace->dims[op_i(arg0.idx)-1];
        arg0idxs[0] = op_i(arg0.idx) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 0, aborting\n");
        exit(-1);
      }
      if (arg0.idx2 < -1) {
        jut = arg0.map2->dim;
      } else if (arg0.idx2 < OP_I_OFFSET) {
        jut = itspace->dims[op_i(arg0.idx2)-1];
        arg0idxs[1] = op_i(arg0.idx2) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 0, aborting\n");
        exit(-1);
      }
      if (iut != iupper || jut != jupper) {
        printf("Map dimensions do not match iteration space, aborting\n");
        exit(-1);
      }
    }

    int arg1idxs[2];
    if (arg1.argtype == OP_ARG_MAT) {
      int iut;
      int jut;
      arg1idxs[0] = 0;
      arg1idxs[1] = 1;
      if (arg1.idx < -1) {
        iut = arg1.map->dim;
      } else if (arg1.idx < OP_I_OFFSET) {
        iut = itspace->dims[op_i(arg1.idx)-1];
        arg1idxs[0] = op_i(arg1.idx) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 1, aborting\n");
        exit(-1);
      }
      if (arg1.idx2 < -1) {
        jut = arg1.map2->dim;
      } else if (arg1.idx2 < OP_I_OFFSET) {
        jut = itspace->dims[op_i(arg1.idx2)-1];
        arg1idxs[1] = op_i(arg1.idx2) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 1, aborting\n");
        exit(-1);
      }
      if (iut != iupper || jut != jupper) {
        printf("Map dimensions do not match iteration space, aborting\n");
        exit(-1);
      }
    }

    int arg2idxs[2];
    if (arg2.argtype == OP_ARG_MAT) {
      int iut;
      int jut;
      arg2idxs[0] = 0;
      arg2idxs[1] = 1;
      if (arg2.idx < -1) {
        iut = arg2.map->dim;
      } else if (arg2.idx < OP_I_OFFSET) {
        iut = itspace->dims[op_i(arg2.idx)-1];
        arg2idxs[0] = op_i(arg2.idx) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 2, aborting\n");
        exit(-1);
      }
      if (arg2.idx2 < -1) {
        jut = arg2.map2->dim;
      } else if (arg2.idx2 < OP_I_OFFSET) {
        jut = itspace->dims[op_i(arg2.idx2)-1];
        arg2idxs[1] = op_i(arg2.idx2) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 2, aborting\n");
        exit(-1);
      }
      if (iut != iupper || jut != jupper) {
        printf("Map dimensions do not match iteration space, aborting\n");
        exit(-1);
      }
    }

    int arg3idxs[2];
    if (arg3.argtype == OP_ARG_MAT) {
      int iut;
      int jut;
      arg3idxs[0] = 0;
      arg3idxs[1] = 1;
      if (arg3.idx < -1) {
        iut = arg3.map->dim;
      } else if (arg3.idx < OP_I_OFFSET) {
        iut = itspace->dims[op_i(arg3.idx)-1];
        arg3idxs[0] = op_i(arg3.idx) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 3, aborting\n");
        exit(-1);
      }
      if (arg3.idx2 < -1) {
        jut = arg3.map2->dim;
      } else if (arg3.idx2 < OP_I_OFFSET) {
        jut = itspace->dims[op_i(arg3.idx2)-1];
        arg3idxs[1] = op_i(arg3.idx2) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 3, aborting\n");
        exit(-1);
      }
      if (iut != iupper || jut != jupper) {
        printf("Map dimensions do not match iteration space, aborting\n");
        exit(-1);
      }
    }

    int arg4idxs[2];
    if (arg4.argtype == OP_ARG_MAT) {
      int iut;
      int jut;
      arg4idxs[0] = 0;
      arg4idxs[1] = 1;
      if (arg4.idx < -1) {
        iut = arg4.map->dim;
      } else if (arg4.idx < OP_I_OFFSET) {
        iut = itspace->dims[op_i(arg4.idx)-1];
        arg4idxs[0] = op_i(arg4.idx) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 4, aborting\n");
        exit(-1);
      }
      if (arg4.idx2 < -1) {
        jut = arg4.map2->dim;
      } else if (arg4.idx2 < OP_I_OFFSET) {
        jut = itspace->dims[op_i(arg4.idx2)-1];
        arg4idxs[1] = op_i(arg4.idx2) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 4, aborting\n");
        exit(-1);
      }
      if (iut != iupper || jut != jupper) {
        printf("Map dimensions do not match iteration space, aborting\n");
        exit(-1);
      }
    }

    int arg5idxs[2];
    if (arg5.argtype == OP_ARG_MAT) {
      int iut;
      int jut;
      arg5idxs[0] = 0;
      arg5idxs[1] = 1;
      if (arg5.idx < -1) {
        iut = arg5.map->dim;
      } else if (arg5.idx < OP_I_OFFSET) {
        iut = itspace->dims[op_i(arg5.idx)-1];
        arg5idxs[0] = op_i(arg5.idx) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 5, aborting\n");
        exit(-1);
      }
      if (arg5.idx2 < -1) {
        jut = arg5.map2->dim;
      } else if (arg5.idx2 < OP_I_OFFSET) {
        jut = itspace->dims[op_i(arg5.idx2)-1];
        arg5idxs[1] = op_i(arg5.idx2) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 5, aborting\n");
        exit(-1);
      }
      if (iut != iupper || jut != jupper) {
        printf("Map dimensions do not match iteration space, aborting\n");
        exit(-1);
      }
    }

    int arg6idxs[2];
    if (arg6.argtype == OP_ARG_MAT) {
      int iut;
      int jut;
      arg6idxs[0] = 0;
      arg6idxs[1] = 1;
      if (arg6.idx < -1) {
        iut = arg6.map->dim;
      } else if (arg6.idx < OP_I_OFFSET) {
        iut = itspace->dims[op_i(arg6.idx)-1];
        arg6idxs[0] = op_i(arg6.idx) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 6, aborting\n");
        exit(-1);
      }
      if (arg6.idx2 < -1) {
        jut = arg6.map2->dim;
      } else if (arg6.idx2 < OP_I_OFFSET) {
        jut = itspace->dims[op_i(arg6.idx2)-1];
        arg6idxs[1] = op_i(arg6.idx2) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 6, aborting\n");
        exit(-1);
      }
      if (iut != iupper || jut != jupper) {
        printf("Map dimensions do not match iteration space, aborting\n");
        exit(-1);
      }
    }

    int arg7idxs[2];
    if (arg7.argtype == OP_ARG_MAT) {
      int iut;
      int jut;
      arg7idxs[0] = 0;
      arg7idxs[1] = 1;
      if (arg7.idx < -1) {
        iut = arg7.map->dim;
      } else if (arg7.idx < OP_I_OFFSET) {
        iut = itspace->dims[op_i(arg7.idx)-1];
        arg7idxs[0] = op_i(arg7.idx) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 7, aborting\n");
        exit(-1);
      }
      if (arg7.idx2 < -1) {
        jut = arg7.map2->dim;
      } else if (arg7.idx2 < OP_I_OFFSET) {
        jut = itspace->dims[op_i(arg7.idx2)-1];
        arg7idxs[1] = op_i(arg7.idx2) - 1;
      } else {
        printf("Invalid index (not vector index or op_i) for arg 7, aborting\n");
        exit(-1);
      }
      if (iut != iupper || jut != jupper) {
        printf("Map dimensions do not match iteration space, aborting\n");
        exit(-1);
      }
    }


    for (idxs[0] = ilower; idxs[0] < iupper; idxs[0]++) {
      for (idxs[1] = jlower; idxs[1] < jupper; idxs[1]++ ) {


        if (arg0.argtype == OP_ARG_MAT) {
          ((T0 *)p_arg0)[0] = (T0)0;
        }

        if (arg1.argtype == OP_ARG_MAT) {
          ((T1 *)p_arg1)[0] = (T1)0;
        }

        if (arg2.argtype == OP_ARG_MAT) {
          ((T2 *)p_arg2)[0] = (T2)0;
        }

        if (arg3.argtype == OP_ARG_MAT) {
          ((T3 *)p_arg3)[0] = (T3)0;
        }

        if (arg4.argtype == OP_ARG_MAT) {
          ((T4 *)p_arg4)[0] = (T4)0;
        }

        if (arg5.argtype == OP_ARG_MAT) {
          ((T5 *)p_arg5)[0] = (T5)0;
        }

        if (arg6.argtype == OP_ARG_MAT) {
          ((T6 *)p_arg6)[0] = (T6)0;
        }

        if (arg7.argtype == OP_ARG_MAT) {
          ((T7 *)p_arg7)[0] = (T7)0;
        }

        kernel( (T0 *)p_arg0, (T1 *)p_arg1, (T2 *)p_arg2, (T3 *)p_arg3,
                (T4 *)p_arg4, (T5 *)p_arg5, (T6 *)p_arg6, (T7 *)p_arg7, idxs[0], idxs[1]);
        // Assemble local matrix into global matrix

        if (arg0.argtype == OP_ARG_MAT) {
          const int rows = arg0.map->dim;
          const int cols = arg0.map2->dim;
          op_mat_addto(arg0.mat, p_arg0,
                       1, arg0.map->map + n*rows + idxs[arg0idxs[0]],
                       1, arg0.map2->map + n*cols + idxs[arg0idxs[1]]);
        }

        if (arg1.argtype == OP_ARG_MAT) {
          const int rows = arg1.map->dim;
          const int cols = arg1.map2->dim;
          op_mat_addto(arg1.mat, p_arg1,
                       1, arg1.map->map + n*rows + idxs[arg1idxs[0]],
                       1, arg1.map2->map + n*cols + idxs[arg1idxs[1]]);
        }

        if (arg2.argtype == OP_ARG_MAT) {
          const int rows = arg2.map->dim;
          const int cols = arg2.map2->dim;
          op_mat_addto(arg2.mat, p_arg2,
                       1, arg2.map->map + n*rows + idxs[arg2idxs[0]],
                       1, arg2.map2->map + n*cols + idxs[arg2idxs[1]]);
        }

        if (arg3.argtype == OP_ARG_MAT) {
          const int rows = arg3.map->dim;
          const int cols = arg3.map2->dim;
          op_mat_addto(arg3.mat, p_arg3,
                       1, arg3.map->map + n*rows + idxs[arg3idxs[0]],
                       1, arg3.map2->map + n*cols + idxs[arg3idxs[1]]);
        }

        if (arg4.argtype == OP_ARG_MAT) {
          const int rows = arg4.map->dim;
          const int cols = arg4.map2->dim;
          op_mat_addto(arg4.mat, p_arg4,
                       1, arg4.map->map + n*rows + idxs[arg4idxs[0]],
                       1, arg4.map2->map + n*cols + idxs[arg4idxs[1]]);
        }

        if (arg5.argtype == OP_ARG_MAT) {
          const int rows = arg5.map->dim;
          const int cols = arg5.map2->dim;
          op_mat_addto(arg5.mat, p_arg5,
                       1, arg5.map->map + n*rows + idxs[arg5idxs[0]],
                       1, arg5.map2->map + n*cols + idxs[arg5idxs[1]]);
        }

        if (arg6.argtype == OP_ARG_MAT) {
          const int rows = arg6.map->dim;
          const int cols = arg6.map2->dim;
          op_mat_addto(arg6.mat, p_arg6,
                       1, arg6.map->map + n*rows + idxs[arg6idxs[0]],
                       1, arg6.map2->map + n*cols + idxs[arg6idxs[1]]);
        }

        if (arg7.argtype == OP_ARG_MAT) {
          const int rows = arg7.map->dim;
          const int cols = arg7.map2->dim;
          op_mat_addto(arg7.mat, p_arg7,
                       1, arg7.map->map + n*rows + idxs[arg7idxs[0]],
                       1, arg7.map2->map + n*cols + idxs[arg7idxs[1]]);
        }

      }
    }
  }

  // Free memory for vector map indices

  if ((arg0.argtype == OP_ARG_DAT && arg0.idx < -1) || arg0.argtype == OP_ARG_MAT) free(p_arg0);

  if ((arg1.argtype == OP_ARG_DAT && arg1.idx < -1) || arg1.argtype == OP_ARG_MAT) free(p_arg1);

  if ((arg2.argtype == OP_ARG_DAT && arg2.idx < -1) || arg2.argtype == OP_ARG_MAT) free(p_arg2);

  if ((arg3.argtype == OP_ARG_DAT && arg3.idx < -1) || arg3.argtype == OP_ARG_MAT) free(p_arg3);

  if ((arg4.argtype == OP_ARG_DAT && arg4.idx < -1) || arg4.argtype == OP_ARG_MAT) free(p_arg4);

  if ((arg5.argtype == OP_ARG_DAT && arg5.idx < -1) || arg5.argtype == OP_ARG_MAT) free(p_arg5);

  if ((arg6.argtype == OP_ARG_DAT && arg6.idx < -1) || arg6.argtype == OP_ARG_MAT) free(p_arg6);

  if ((arg7.argtype == OP_ARG_DAT && arg7.idx < -1) || arg7.argtype == OP_ARG_MAT) free(p_arg7);

  free(itspace->dims);
  free(itspace);
  // Global matrix assembly
  if (arg0.argtype == OP_ARG_MAT) op_mat_assemble(arg0.mat);
  if (arg1.argtype == OP_ARG_MAT) op_mat_assemble(arg1.mat);
  if (arg2.argtype == OP_ARG_MAT) op_mat_assemble(arg2.mat);
  if (arg3.argtype == OP_ARG_MAT) op_mat_assemble(arg3.mat);
  if (arg4.argtype == OP_ARG_MAT) op_mat_assemble(arg4.mat);
  if (arg5.argtype == OP_ARG_MAT) op_mat_assemble(arg5.mat);
  if (arg6.argtype == OP_ARG_MAT) op_mat_assemble(arg6.mat);
  if (arg7.argtype == OP_ARG_MAT) op_mat_assemble(arg7.mat);
}


#endif
