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

#ifndef __OP_LIB_MAT_H
#define __OP_LIB_MAT_H

#include "op_lib_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Matrix computation function prototypes
 */

op_sparsity op_decl_sparsity ( op_map rowmap, op_map colmap, char const * name );

op_mat op_decl_mat( op_sparsity sparsity, int dim, char const * type, int type_size, char const * name );

op_arg op_arg_mat ( op_mat mat, int rowidx, op_map rowmap, int colidx, op_map colmap, int dim, const char * typ, op_access acc );

void op_mat_addto( op_mat mat, const void* values, int nrows, const int *irows, int ncols, const int *icols );

void op_mat_addto_scalar( op_mat mat, const void* value, int row, int col );

void op_mat_assemble( op_mat mat );

void op_mat_mult ( const op_mat mat, const op_dat v_in, op_dat v_out );

void op_mat_get_values ( const op_mat mat, double **v, int *m, int *n);

void op_solve ( const op_mat mat, const op_dat b, op_dat x );

#ifdef __cplusplus
}
#endif

#endif /* __OP_LIB_MAT_H */
