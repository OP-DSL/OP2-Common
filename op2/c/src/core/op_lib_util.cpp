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

#include <vector>
#include <set>
#include "op_lib_core.h"
void op_build_sparsity_pattern ( op_sparsity sparsity )
{
  // Create and populate auxiliary data structure: for each element of
  // the from set, for each row pointed to by the row map, add all
  // columns pointed to by the col map
  int rmult = sparsity->dim[0];
  int cmult = sparsity->dim[1];
  std::vector< std::set< int > > s(sparsity->nrows);

  for ( int m = 0; m < sparsity->nmaps; m++ ) {
    op_map rowmap = sparsity->rowmaps[m];
    op_map colmap = sparsity->colmaps[m];
    for ( int e = 0; e < rowmap->from->size; ++e ) {
      for ( int i = 0; i < rowmap->dim; ++i ) {
        for ( int r = 0; r < rmult; r++ ) {
          int row = (r + 1) * rowmap->map[i + e*rowmap->dim];
          for ( int c = 0; c < cmult; c++ ) {
            for ( int d = 0; d < colmap->dim; d++ ) {
              s[row].insert((c+1)*colmap->map[d + e * colmap->dim]);
            }
          }
        }
      }
    }
  }

  // Create final sparsity structure
  int *nnz = (int*)malloc(cmult * sparsity->nrows * sizeof(int));
  int *rowptr = (int*)malloc((sparsity->nrows+1) * sizeof(int));
  rowptr[0] = 0;
  for ( size_t row = 0; row < sparsity->nrows; ++row ) {
    nnz[row] = s[row].size();
    rowptr[row+1] = rowptr[row] + nnz[row];
    if ( sparsity->max_nonzeros < s[row].size() )
      sparsity->max_nonzeros = s[row].size();
  }
  int *colidx = (int*)malloc(rowptr[sparsity->nrows] * sizeof(int));
  for ( size_t row = 0; row < sparsity->nrows; ++row ) {
    std::copy(s[row].begin(), s[row].end(), colidx + rowptr[row]);
  }

  sparsity->nnz = nnz;
  sparsity->total_nz = rowptr[sparsity->nrows];
  sparsity->rowptr = rowptr;
  sparsity->colidx = colidx;
}
