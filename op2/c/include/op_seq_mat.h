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

#include "op_lib_core.h"

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


//
// op_par_loop routine for 1 arguments
//

template < class T0 >
void op_par_loop ( void (*kernel)( T0* ),
  char const * name, op_set set,
  op_arg arg0 )
{
  char *p_arg0;

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

  if (arg0.idx < -1) {
    p_arg0 = (char *)malloc(arg0.map->dim*sizeof(T0));
  }


  // loop over set elements

  for (int n=0; n<set->size; n++) {
    // Copy in of vector map indices

    if (arg0.idx < -1)
      copy_in(n, arg0, (char**)p_arg0);
    else
      op_arg_set(n, arg0, &p_arg0 );


    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_arg0 );
  }

  // Free memory for vector map indices

  if (arg0.idx < -1) {
    free((char **)p_arg0);
  }

}

//
// op_par_loop routine for 2 arguments
//

template < class T0, class T1 >
void op_par_loop ( void (*kernel)( T0*, T1* ),
  char const * name, op_set set,
  op_arg arg0, op_arg arg1 )
{
  char *p_arg0, *p_arg1;

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

  if (arg0.idx < -1) {
    p_arg0 = (char *)malloc(arg0.map->dim*sizeof(T0));
  }


  if (arg1.idx < -1) {
    p_arg1 = (char *)malloc(arg1.map->dim*sizeof(T1));
  }


  // loop over set elements

  for (int n=0; n<set->size; n++) {
    // Copy in of vector map indices

    if (arg0.idx < -1)
      copy_in(n, arg0, (char**)p_arg0);
    else
      op_arg_set(n, arg0, &p_arg0 );


    if (arg1.idx < -1)
      copy_in(n, arg1, (char**)p_arg1);
    else
      op_arg_set(n, arg1, &p_arg1 );


    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_arg0, (T1 *)p_arg1 );
  }

  // Free memory for vector map indices

  if (arg0.idx < -1) {
    free((char **)p_arg0);
  }


  if (arg1.idx < -1) {
    free((char **)p_arg1);
  }

}

//
// op_par_loop routine for 3 arguments
//

template < class T0, class T1, class T2 >
void op_par_loop ( void (*kernel)( T0*, T1*, T2* ),
  char const * name, op_set set,
  op_arg arg0, op_arg arg1, op_arg arg2 )
{
  char *p_arg0, *p_arg1, *p_arg2;

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

  if (arg0.idx < -1) {
    p_arg0 = (char *)malloc(arg0.map->dim*sizeof(T0));
  }


  if (arg1.idx < -1) {
    p_arg1 = (char *)malloc(arg1.map->dim*sizeof(T1));
  }


  if (arg2.idx < -1) {
    p_arg2 = (char *)malloc(arg2.map->dim*sizeof(T2));
  }


  // loop over set elements

  for (int n=0; n<set->size; n++) {
    // Copy in of vector map indices

    if (arg0.idx < -1)
      copy_in(n, arg0, (char**)p_arg0);
    else
      op_arg_set(n, arg0, &p_arg0 );


    if (arg1.idx < -1)
      copy_in(n, arg1, (char**)p_arg1);
    else
      op_arg_set(n, arg1, &p_arg1 );


    if (arg2.idx < -1)
      copy_in(n, arg2, (char**)p_arg2);
    else
      op_arg_set(n, arg2, &p_arg2 );


    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_arg0, (T1 *)p_arg1, (T2 *)p_arg2 );
  }

  // Free memory for vector map indices

  if (arg0.idx < -1) {
    free((char **)p_arg0);
  }


  if (arg1.idx < -1) {
    free((char **)p_arg1);
  }


  if (arg2.idx < -1) {
    free((char **)p_arg2);
  }

}

//
// op_par_loop routine for 4 arguments
//

template < class T0, class T1, class T2, class T3 >
void op_par_loop ( void (*kernel)( T0*, T1*, T2*, T3* ),
  char const * name, op_set set,
  op_arg arg0, op_arg arg1, op_arg arg2, op_arg arg3 )
{
  char *p_arg0, *p_arg1, *p_arg2, *p_arg3;

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

  if (arg0.idx < -1) {
    p_arg0 = (char *)malloc(arg0.map->dim*sizeof(T0));
  }


  if (arg1.idx < -1) {
    p_arg1 = (char *)malloc(arg1.map->dim*sizeof(T1));
  }


  if (arg2.idx < -1) {
    p_arg2 = (char *)malloc(arg2.map->dim*sizeof(T2));
  }


  if (arg3.idx < -1) {
    p_arg3 = (char *)malloc(arg3.map->dim*sizeof(T3));
  }


  // loop over set elements

  for (int n=0; n<set->size; n++) {
    // Copy in of vector map indices

    if (arg0.idx < -1)
      copy_in(n, arg0, (char**)p_arg0);
    else
      op_arg_set(n, arg0, &p_arg0 );


    if (arg1.idx < -1)
      copy_in(n, arg1, (char**)p_arg1);
    else
      op_arg_set(n, arg1, &p_arg1 );


    if (arg2.idx < -1)
      copy_in(n, arg2, (char**)p_arg2);
    else
      op_arg_set(n, arg2, &p_arg2 );


    if (arg3.idx < -1)
      copy_in(n, arg3, (char**)p_arg3);
    else
      op_arg_set(n, arg3, &p_arg3 );


    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_arg0, (T1 *)p_arg1, (T2 *)p_arg2, (T3 *)p_arg3 );
  }

  // Free memory for vector map indices

  if (arg0.idx < -1) {
    free((char **)p_arg0);
  }


  if (arg1.idx < -1) {
    free((char **)p_arg1);
  }


  if (arg2.idx < -1) {
    free((char **)p_arg2);
  }


  if (arg3.idx < -1) {
    free((char **)p_arg3);
  }

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
  char *p_arg0, *p_arg1, *p_arg2, *p_arg3,
       *p_arg4;

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

  if (arg0.idx < -1) {
    p_arg0 = (char *)malloc(arg0.map->dim*sizeof(T0));
  }


  if (arg1.idx < -1) {
    p_arg1 = (char *)malloc(arg1.map->dim*sizeof(T1));
  }


  if (arg2.idx < -1) {
    p_arg2 = (char *)malloc(arg2.map->dim*sizeof(T2));
  }


  if (arg3.idx < -1) {
    p_arg3 = (char *)malloc(arg3.map->dim*sizeof(T3));
  }


  if (arg4.idx < -1) {
    p_arg4 = (char *)malloc(arg4.map->dim*sizeof(T4));
  }


  // loop over set elements

  for (int n=0; n<set->size; n++) {
    // Copy in of vector map indices

    if (arg0.idx < -1)
      copy_in(n, arg0, (char**)p_arg0);
    else
      op_arg_set(n, arg0, &p_arg0 );


    if (arg1.idx < -1)
      copy_in(n, arg1, (char**)p_arg1);
    else
      op_arg_set(n, arg1, &p_arg1 );


    if (arg2.idx < -1)
      copy_in(n, arg2, (char**)p_arg2);
    else
      op_arg_set(n, arg2, &p_arg2 );


    if (arg3.idx < -1)
      copy_in(n, arg3, (char**)p_arg3);
    else
      op_arg_set(n, arg3, &p_arg3 );


    if (arg4.idx < -1)
      copy_in(n, arg4, (char**)p_arg4);
    else
      op_arg_set(n, arg4, &p_arg4 );


    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_arg0, (T1 *)p_arg1, (T2 *)p_arg2, (T3 *)p_arg3,
            (T4 *)p_arg4 );
  }

  // Free memory for vector map indices

  if (arg0.idx < -1) {
    free((char **)p_arg0);
  }


  if (arg1.idx < -1) {
    free((char **)p_arg1);
  }


  if (arg2.idx < -1) {
    free((char **)p_arg2);
  }


  if (arg3.idx < -1) {
    free((char **)p_arg3);
  }


  if (arg4.idx < -1) {
    free((char **)p_arg4);
  }

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
  char *p_arg0, *p_arg1, *p_arg2, *p_arg3,
       *p_arg4, *p_arg5;

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

  if (arg0.idx < -1) {
    p_arg0 = (char *)malloc(arg0.map->dim*sizeof(T0));
  }


  if (arg1.idx < -1) {
    p_arg1 = (char *)malloc(arg1.map->dim*sizeof(T1));
  }


  if (arg2.idx < -1) {
    p_arg2 = (char *)malloc(arg2.map->dim*sizeof(T2));
  }


  if (arg3.idx < -1) {
    p_arg3 = (char *)malloc(arg3.map->dim*sizeof(T3));
  }


  if (arg4.idx < -1) {
    p_arg4 = (char *)malloc(arg4.map->dim*sizeof(T4));
  }


  if (arg5.idx < -1) {
    p_arg5 = (char *)malloc(arg5.map->dim*sizeof(T5));
  }


  // loop over set elements

  for (int n=0; n<set->size; n++) {
    // Copy in of vector map indices

    if (arg0.idx < -1)
      copy_in(n, arg0, (char**)p_arg0);
    else
      op_arg_set(n, arg0, &p_arg0 );


    if (arg1.idx < -1)
      copy_in(n, arg1, (char**)p_arg1);
    else
      op_arg_set(n, arg1, &p_arg1 );


    if (arg2.idx < -1)
      copy_in(n, arg2, (char**)p_arg2);
    else
      op_arg_set(n, arg2, &p_arg2 );


    if (arg3.idx < -1)
      copy_in(n, arg3, (char**)p_arg3);
    else
      op_arg_set(n, arg3, &p_arg3 );


    if (arg4.idx < -1)
      copy_in(n, arg4, (char**)p_arg4);
    else
      op_arg_set(n, arg4, &p_arg4 );


    if (arg5.idx < -1)
      copy_in(n, arg5, (char**)p_arg5);
    else
      op_arg_set(n, arg5, &p_arg5 );


    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_arg0, (T1 *)p_arg1, (T2 *)p_arg2, (T3 *)p_arg3,
            (T4 *)p_arg4, (T5 *)p_arg5 );
  }

  // Free memory for vector map indices

  if (arg0.idx < -1) {
    free((char **)p_arg0);
  }


  if (arg1.idx < -1) {
    free((char **)p_arg1);
  }


  if (arg2.idx < -1) {
    free((char **)p_arg2);
  }


  if (arg3.idx < -1) {
    free((char **)p_arg3);
  }


  if (arg4.idx < -1) {
    free((char **)p_arg4);
  }


  if (arg5.idx < -1) {
    free((char **)p_arg5);
  }

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
  char *p_arg0, *p_arg1, *p_arg2, *p_arg3,
       *p_arg4, *p_arg5, *p_arg6;

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

  if (arg0.idx < -1) {
    p_arg0 = (char *)malloc(arg0.map->dim*sizeof(T0));
  }


  if (arg1.idx < -1) {
    p_arg1 = (char *)malloc(arg1.map->dim*sizeof(T1));
  }


  if (arg2.idx < -1) {
    p_arg2 = (char *)malloc(arg2.map->dim*sizeof(T2));
  }


  if (arg3.idx < -1) {
    p_arg3 = (char *)malloc(arg3.map->dim*sizeof(T3));
  }


  if (arg4.idx < -1) {
    p_arg4 = (char *)malloc(arg4.map->dim*sizeof(T4));
  }


  if (arg5.idx < -1) {
    p_arg5 = (char *)malloc(arg5.map->dim*sizeof(T5));
  }


  if (arg6.idx < -1) {
    p_arg6 = (char *)malloc(arg6.map->dim*sizeof(T6));
  }


  // loop over set elements

  for (int n=0; n<set->size; n++) {
    // Copy in of vector map indices

    if (arg0.idx < -1)
      copy_in(n, arg0, (char**)p_arg0);
    else
      op_arg_set(n, arg0, &p_arg0 );


    if (arg1.idx < -1)
      copy_in(n, arg1, (char**)p_arg1);
    else
      op_arg_set(n, arg1, &p_arg1 );


    if (arg2.idx < -1)
      copy_in(n, arg2, (char**)p_arg2);
    else
      op_arg_set(n, arg2, &p_arg2 );


    if (arg3.idx < -1)
      copy_in(n, arg3, (char**)p_arg3);
    else
      op_arg_set(n, arg3, &p_arg3 );


    if (arg4.idx < -1)
      copy_in(n, arg4, (char**)p_arg4);
    else
      op_arg_set(n, arg4, &p_arg4 );


    if (arg5.idx < -1)
      copy_in(n, arg5, (char**)p_arg5);
    else
      op_arg_set(n, arg5, &p_arg5 );


    if (arg6.idx < -1)
      copy_in(n, arg6, (char**)p_arg6);
    else
      op_arg_set(n, arg6, &p_arg6 );


    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_arg0, (T1 *)p_arg1, (T2 *)p_arg2, (T3 *)p_arg3,
            (T4 *)p_arg4, (T5 *)p_arg5, (T6 *)p_arg6 );
  }

  // Free memory for vector map indices

  if (arg0.idx < -1) {
    free((char **)p_arg0);
  }


  if (arg1.idx < -1) {
    free((char **)p_arg1);
  }


  if (arg2.idx < -1) {
    free((char **)p_arg2);
  }


  if (arg3.idx < -1) {
    free((char **)p_arg3);
  }


  if (arg4.idx < -1) {
    free((char **)p_arg4);
  }


  if (arg5.idx < -1) {
    free((char **)p_arg5);
  }


  if (arg6.idx < -1) {
    free((char **)p_arg6);
  }

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
  char *p_arg0, *p_arg1, *p_arg2, *p_arg3,
       *p_arg4, *p_arg5, *p_arg6, *p_arg7;

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

  if (arg0.idx < -1) {
    p_arg0 = (char *)malloc(arg0.map->dim*sizeof(T0));
  }


  if (arg1.idx < -1) {
    p_arg1 = (char *)malloc(arg1.map->dim*sizeof(T1));
  }


  if (arg2.idx < -1) {
    p_arg2 = (char *)malloc(arg2.map->dim*sizeof(T2));
  }


  if (arg3.idx < -1) {
    p_arg3 = (char *)malloc(arg3.map->dim*sizeof(T3));
  }


  if (arg4.idx < -1) {
    p_arg4 = (char *)malloc(arg4.map->dim*sizeof(T4));
  }


  if (arg5.idx < -1) {
    p_arg5 = (char *)malloc(arg5.map->dim*sizeof(T5));
  }


  if (arg6.idx < -1) {
    p_arg6 = (char *)malloc(arg6.map->dim*sizeof(T6));
  }


  if (arg7.idx < -1) {
    p_arg7 = (char *)malloc(arg7.map->dim*sizeof(T7));
  }


  // loop over set elements

  for (int n=0; n<set->size; n++) {
    // Copy in of vector map indices

    if (arg0.idx < -1)
      copy_in(n, arg0, (char**)p_arg0);
    else
      op_arg_set(n, arg0, &p_arg0 );


    if (arg1.idx < -1)
      copy_in(n, arg1, (char**)p_arg1);
    else
      op_arg_set(n, arg1, &p_arg1 );


    if (arg2.idx < -1)
      copy_in(n, arg2, (char**)p_arg2);
    else
      op_arg_set(n, arg2, &p_arg2 );


    if (arg3.idx < -1)
      copy_in(n, arg3, (char**)p_arg3);
    else
      op_arg_set(n, arg3, &p_arg3 );


    if (arg4.idx < -1)
      copy_in(n, arg4, (char**)p_arg4);
    else
      op_arg_set(n, arg4, &p_arg4 );


    if (arg5.idx < -1)
      copy_in(n, arg5, (char**)p_arg5);
    else
      op_arg_set(n, arg5, &p_arg5 );


    if (arg6.idx < -1)
      copy_in(n, arg6, (char**)p_arg6);
    else
      op_arg_set(n, arg6, &p_arg6 );


    if (arg7.idx < -1)
      copy_in(n, arg7, (char**)p_arg7);
    else
      op_arg_set(n, arg7, &p_arg7 );


    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_arg0, (T1 *)p_arg1, (T2 *)p_arg2, (T3 *)p_arg3,
            (T4 *)p_arg4, (T5 *)p_arg5, (T6 *)p_arg6, (T7 *)p_arg7 );
  }

  // Free memory for vector map indices

  if (arg0.idx < -1) {
    free((char **)p_arg0);
  }


  if (arg1.idx < -1) {
    free((char **)p_arg1);
  }


  if (arg2.idx < -1) {
    free((char **)p_arg2);
  }


  if (arg3.idx < -1) {
    free((char **)p_arg3);
  }


  if (arg4.idx < -1) {
    free((char **)p_arg4);
  }


  if (arg5.idx < -1) {
    free((char **)p_arg5);
  }


  if (arg6.idx < -1) {
    free((char **)p_arg6);
  }


  if (arg7.idx < -1) {
    free((char **)p_arg7);
  }

}


#endif
