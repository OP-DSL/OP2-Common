#include <stdio.h>
#include <stdlib.h>
#include <set>
#include <vector>
#include "op_lib_cpp.h"

#define NUM_ELE 2
#define NUM_NODES 4
#define NUM_DIM 2

// Need to pull this into op_lib (can we write a GPU version that means we don't need to do so much copying?)
op_sparsity op_decl_sparsity ( op_map rowmap, op_map colmap, char const * name )
{
  op_sparsity sparsity = op_decl_sparsity_core(rowmap, colmap, name);

  // Create and populate auxiliary data structure: for each element of the from
  // set, for each row pointed to by the row map, add all columns pointed to by
  // the col map
  std::vector< std::set< int > > s(sparsity->nrows);
  for ( int e = 0; e < rowmap->from->size; ++e ) {
    for ( int i = 0; i < rowmap->dim; ++i ) {
      int row = rowmap->map[i + e*rowmap->dim];
      s[row].insert( colmap->map + e*colmap->dim, colmap->map + (e+1)*colmap->dim );
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

  sparsity->nnz = nnz;
  sparsity->rowptr = rowptr;
  sparsity->colidx = colidx;
  return sparsity;
}

void op_par_loop_mass(const char *, op_set, op_sparsity, op_arg);

int main(int argc, char **argv)
{
  int *p_elem_node;
  float *p_xn;
  int i;
  float val;

  op_init(argc, argv, 5);

  p_elem_node = (int *)malloc(3 * NUM_ELE * sizeof(int));
  p_elem_node[0] = 0;
  p_elem_node[1] = 1;
  p_elem_node[2] = 3;
  p_elem_node[3] = 2;
  p_elem_node[4] = 3;
  p_elem_node[5] = 1;

  p_xn = (float *)malloc(2 * NUM_NODES * sizeof(float));
  p_xn[0] = 0.0f;
  p_xn[1] = 0.0f;
  p_xn[2] = 2.0f;
  p_xn[3] = 0.0f;
  p_xn[4] = 1.0f;
  p_xn[5] = 1.0f;
  p_xn[6] = 0.0f;
  p_xn[7] = 1.5f;

  op_set nodes = op_decl_set(NUM_NODES, "nodes");
  op_set elements = op_decl_set(NUM_ELE, "elements");
  op_map elem_node = op_decl_map(elements, nodes, 3, p_elem_node, "elem_node");

  // op_mat and op_sparsity types are not yet supported in the CUDA
  // OP2 runtime, so just ignore them for now.

  op_sparsity sparsity = op_decl_sparsity(elem_node, elem_node, "sparsity");
  // op_mat mat = op_decl_mat(sparsity, 1, "float", sizeof(float), "mat");
  op_dat xn = op_decl_dat(nodes, 2, "float", p_xn, "xn");

  op_par_loop_mass("mass", elements, sparsity,
                   op_arg_dat(xn, -3, elem_node, 2, "float", OP_READ));

  free(p_elem_node);
  free(p_xn);
  op_exit();
  return 0;
}
