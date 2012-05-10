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
 * This header file defines the user-level OP2 library for
 * the case of C++ programs, integrating the missing part
 * of the OP2 C interface (implemented in op_lib_c.h).
 * The definitions used here are specifically using C++
 * abstractions and cannot be used by Fortran without
 * a special briding library.
 */

#ifndef __OP_LIB_CPP_H
#define __OP_LIB_CPP_H

/*
 * include core definitions and C declarations of op2 user-level routines
 */

#include <op_lib_core.h>
#include <op_lib_c.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/*
 * run-time type-checking routines
 */

inline int type_error (const double * a, const char *type ) {
  (void)a; return (strcmp ( type, "double" ) && strcmp ( type, "double:soa" ));
}
inline int type_error (const float  * a, const char *type ) {
  (void)a; return (strcmp ( type, "float" ) && strcmp ( type, "float:soa" ));
}
inline int type_error (const int    * a, const char *type ) {
  (void)a; return (strcmp ( type, "int"   ) && strcmp ( type, "int:soa"   ));
}
inline int type_error (const uint   * a, const char *type ) {
  (void)a; return (strcmp ( type, "uint"  ) && strcmp ( type, "uint:soa"  ));
}
inline int type_error (const ll     * a, const char *type ) {
  (void)a; return (strcmp ( type, "ll"    ) && strcmp ( type, "ll:soa"    ));
}
inline int type_error (const ull    * a, const char *type ) {
  (void)a; return (strcmp ( type, "ull"   ) && strcmp ( type, "ull:soa"   ));
}
inline int type_error (const bool   * a, const char *type ) {
  (void)a; return (strcmp ( type, "bool"  ) && strcmp ( type, "bool:soa"  ));
}

/*
 * add in user's datatypes
 */

#ifdef OP_USER_DATATYPES
#include "op_user_datatypes.h"
#endif

/*
 * zero constants
 */

#define ZERO_double  0.0;
#define ZERO_float   0.0f;
#define ZERO_int     0;
#define ZERO_uint    0;
#define ZERO_ll      0;
#define ZERO_ull     0;
#define ZERO_bool    0;

/*
 * external variables declared in op_lib_core.cpp
 */

extern int OP_diags, OP_part_size, OP_block_size;

extern int OP_set_index,  OP_set_max,
           OP_map_index,  OP_map_max,
           OP_dat_index,  OP_dat_max,
           OP_plan_index, OP_plan_max,
                          OP_kern_max;

extern op_set    * OP_set_list;
extern op_map    * OP_map_list;
extern op_dat    * OP_dat_list;
extern op_mat    * OP_mat_list;
extern op_sparsity * OP_sparsity_list;
extern op_kernel * OP_kernels;


op_dat op_decl_dat_char (op_set, int, char const *, int, char *, char const * );


/* Implementation */

template < class T >
op_dat op_decl_dat ( op_set set, int dim, char const *type,
                     T * data, char const * name )
{

  if ( type_error ( data, type ) )
  {
    printf ( "incorrect type specified for dataset \"%s\" \n", name );
    exit ( 1 );
  }

  return op_decl_dat_char ( set, dim, type, sizeof(T), (char *) data, name );
}

template < class T >
void op_decl_const2 ( char const * name, int dim, char const *type, T * data )
{
  if ( type_error ( data, type ) )
  {
    printf ( "incorrect type specified for constant \"%s\" \n", name ); exit ( 1 );
  }

  op_decl_const_char ( dim, type, sizeof ( T ), (char *) data, name );
}

template < class T >
void op_decl_const ( int dim, char const * type, T * data )
{
  (void)dim;
  if ( type_error ( data, type ) )
  {
    printf ( "incorrect type specified for constant in op_decl_const" );
    exit ( 1 );
  }
}

template < class T >
op_arg op_arg_gbl ( T * data, int dim, char const * type, op_access acc )
{
  if ( type_error ( data, type ) )
    return op_arg_gbl_char ( ( char *  )data, dim, "error",  sizeof(T), acc );
  else
    return op_arg_gbl_char ( ( char * ) data, dim, type,  sizeof(T), acc );
}


//
// wrapper functions to handle MPI global reductions
//


inline void op_mpi_reduce(op_arg* args, float *data)
{
  op_mpi_reduce_float(args,data);
}

inline void op_mpi_reduce(op_arg* args, double *data)
{
  op_mpi_reduce_double(args,data);
}

inline void op_mpi_reduce(op_arg* args, int *data)
{
  op_mpi_reduce_int(args,data);
}

template <class T>
void op_mpi_reduce(op_arg* args, T* data)
{

}

void op_mat_lma_to_csr(float *dummy, op_arg mat, op_set set);

void op_mat_lma_to_csr(double *dummy, op_arg mat, op_set set);

#endif /* __OP_LIB_CPP_H */

