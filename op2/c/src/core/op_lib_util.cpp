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
#include <map>
#include "op_lib_core.h"

void op_build_sparsity_pattern ( op_map rowmap, op_map colmap,
                                 op_sparsity sparsity )
{
  // Create and populate auxiliary data structure: for each element of
  // the from set, for each row pointed to by the row map, add all
  // columns pointed to by the col map
  std::vector< std::set< int > > s(sparsity->nrows);
  std::vector< std::map< int, std::set<int> > > lma(sparsity->nrows);
  for ( int e = 0; e < rowmap->from->size; ++e ) {
    for ( int i = 0; i < rowmap->dim; ++i ) {
      int row = rowmap->map[i + e*rowmap->dim];
      s[row].insert( colmap->map + e*colmap->dim, colmap->map + (e+1)*colmap->dim );
      for ( int j = 0; j < colmap->dim; ++j ) {
        int col = colmap->map[j + e*colmap->dim];
        // e element;
        // i row idx in element matrix
        // j col idx in element matrix
        int offset = rowmap->dim * colmap->dim * e + i * colmap->dim + j;
        lma[row][col].insert(offset);
      }
    }
  }

  // Create final sparsity structure
  int *nnz = (int*)malloc(sparsity->nrows * sizeof(int));
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

  int offset = 0;
  sparsity->nlma = rowmap->from->size * rowmap->dim * colmap->dim;
  int *csr_to_lma = (int *)malloc(sparsity->nlma * sizeof(int));
  int *lmaidx = (int *)malloc((rowptr[sparsity->nrows]+1) * sizeof(int));
  int j = 0;
  for ( size_t row = 0; row < sparsity->nrows; row++ ) {
    for ( int i = rowptr[row]; i < rowptr[row+1]; i++ ) {
      int col = colidx[i];
      std::set <int> tmp = lma[row][col];
      lmaidx[i] = offset;
      offset += tmp.size ();
      for ( std::set <int>::iterator it = tmp.begin ();
            it != tmp.end (); it++ ) {
        csr_to_lma[j++] = *it;
      }
    }
  }
  lmaidx[rowptr[sparsity->nrows]] = offset;
  sparsity->nnz = nnz;
  sparsity->total_nz = rowptr[sparsity->nrows];
  sparsity->rowptr = rowptr;
  sparsity->colidx = colidx;
  sparsity->lmaidx = lmaidx;
  sparsity->csr2lma = csr_to_lma;
}
