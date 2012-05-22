/*
  Open source copyright declaration based on BSD open source template:
  http://www.opensource.org/licenses/bsd-license.php

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

#include <assert.h>
#include <algorithm>
#include <iostream>
#include <iterator>

#include <petscksp.h>

#include "op_lib_mat.h"

op_sparsity op_decl_sparsity ( op_map rowmap, op_map colmap, char const * name )
{
  assert(rowmap && colmap);
  op_sparsity sparsity = op_decl_sparsity_core(rowmap, colmap, name);

  return sparsity;
}

op_mat op_decl_mat( op_sparsity sparsity, int dim, char const * type, int type_size, char const * name )
{
  assert( sparsity );
  op_mat mat = op_decl_mat_core ( sparsity->rowmap->to, sparsity->colmap->to, dim, type, type_size, name );

  Mat p_mat;
  // Create a PETSc CSR sparse matrix and pre-allocate storage
  MatCreateSeqAIJ(PETSC_COMM_SELF,
      sparsity->nrows,
      sparsity->ncols,
      sparsity->max_nonzeros,
      (const PetscInt*)sparsity->nnz,
      &p_mat);
  // Set the column indices (FIXME: benchmark if this is worth it)
  MatSeqAIJSetColumnIndices(p_mat, (PetscInt*)sparsity->colidx);

  MatZeroEntries(p_mat);
  mat->mat = p_mat;
  return mat;
}

op_arg op_arg_mat ( op_mat mat, int rowidx, op_map rowmap, int colidx, op_map colmap, int dim, const char * typ, op_access acc )
{
  return op_arg_mat_core(mat, rowidx, rowmap, colidx, colmap, dim, typ, acc);
}

template < typename T >
static inline int is_float(T dat)
{
  return strncmp(dat->type, "float", 5) == 0;
}

template < typename T >
static inline PetscScalar * to_petsc(T dat, const void * values, int size)
{
  PetscScalar * dvalues;
  // If we're passed float data, we have to convert it to double
  if (is_float(dat)) {
    dvalues = (PetscScalar*)malloc(sizeof(PetscScalar) * size);
    for ( int i = 0; i < size; i++ )
      dvalues[i] = (PetscScalar)(((float*)values)[i]);
  } else {
    dvalues = (PetscScalar *) values;
  }
  return dvalues;
}

void op_mat_addto_scalar( op_mat mat, const void* value, int row, int col )
{
  assert( mat && value);
  PetscScalar v[1];
  if (is_float(mat))
    v[0] = (PetscScalar)((float *)value)[0];
  else
    v[0] = ((const PetscScalar *)value)[0];

  MatSetValues( (Mat) mat->mat,
                1, (const PetscInt *)&row,
                1, (const PetscInt *)&col,
                v, ADD_VALUES );
}

void op_mat_addto( op_mat mat, const void* values, int nrows, const int *irows, int ncols, const int *icols )
{
  assert( mat && values && irows && icols );

  PetscScalar * dvalues = to_petsc(mat, values, nrows * ncols);

  MatSetValues( (Mat) mat->mat,
                nrows, (const PetscInt *)irows,
                ncols, (const PetscInt *)icols,
                dvalues, ADD_VALUES);

  if (is_float(mat)) free(dvalues);
}

void op_mat_assemble( op_mat mat )
{
  assert( mat );

  MatAssemblyBegin((Mat) mat->mat,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd((Mat) mat->mat,MAT_FINAL_ASSEMBLY);
}

static Vec op_create_vec ( const op_dat vec ) {
  assert( vec );

  PetscScalar * dvalues = to_petsc(vec, vec->data, vec->dim * vec->set->size);

  Vec p_vec;
  // Create a PETSc vector and pass it the user-allocated storage
  VecCreateSeqWithArray(MPI_COMM_SELF, vec->dim * vec->set->size, dvalues, &p_vec);
  VecAssemblyBegin(p_vec);
  VecAssemblyEnd(p_vec);

  return p_vec;
}

static void op_destroy_vec ( Vec v, op_dat d ) {
  // If the op_dat holds float data we need to copy out
  if (is_float(d)) {
    PetscScalar * a;
    VecGetArray(v, &a);
    for (int i = 0; i < d->dim * d->set->size; i++) {
      ((float *)d->data)[i] = (float)a[i];
    }
  }
  VecDestroy(v);
}

void op_mat_mult ( const op_mat mat, const op_dat v_in, op_dat v_out )
{
  assert( mat && v_in && v_out );

  Vec p_v_in = op_create_vec(v_in);
  Vec p_v_out = op_create_vec(v_out);

  MatMult((Mat) mat->mat, p_v_in, p_v_out);

  op_destroy_vec(p_v_in, v_in);
  op_destroy_vec(p_v_out, v_out);
}

void op_solve ( const op_mat mat, const op_dat b, op_dat x )
{
  assert( mat && b && x );

  Vec p_b = op_create_vec(b);
  Vec p_x = op_create_vec(x);
  Mat A = (Mat) mat->mat;
  KSP ksp;
  PC pc;

  KSPCreate(PETSC_COMM_WORLD,&ksp);
  KSPSetOperators(ksp,A,A,DIFFERENT_NONZERO_PATTERN);
  KSPGetPC(ksp,&pc);
  PCSetType(pc,PCJACOBI);
  KSPSetTolerances(ksp,1.e-7,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);

  KSPSolve(ksp,p_b,p_x);

  op_destroy_vec(p_b, b);
  op_destroy_vec(p_x, x);
  KSPDestroy(ksp);
}
