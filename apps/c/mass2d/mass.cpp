#include <stdio.h>
#include <stdlib.h>

#include "op_seq_mat.h"
#include "op_lib_cpp.h"

#include "types.h"
#include "mass.h"
#include "rhs.h"

#define NUM_ELE 2
#define NUM_NODES 4
#define NUM_DIM 2

int main(int argc, char **argv)
{
  int *p_elem_node;
  ValueType *p_xn, *p_b, *p_f, *p_x;

  op_init(argc, argv, 5);

  p_elem_node = (int *)malloc(3 * NUM_ELE * sizeof(int));
  p_elem_node[0] = 0;
  p_elem_node[1] = 1;
  p_elem_node[2] = 3;
  p_elem_node[3] = 2;
  p_elem_node[4] = 3;
  p_elem_node[5] = 1;

  p_xn = (ValueType *)malloc(2 * NUM_NODES * sizeof(ValueType));
  p_xn[0] = 0.0f;
  p_xn[1] = 0.0f;
  p_xn[2] = 2.0f;
  p_xn[3] = 0.0f;
  p_xn[4] = 1.0f;
  p_xn[5] = 1.0f;
  p_xn[6] = 0.0f;
  p_xn[7] = 1.5f;

  p_f = (ValueType*)malloc(NUM_NODES * sizeof(ValueType));
  p_f[0] = 1.0f;
  p_f[1] = 2.0f;
  p_f[2] = 3.0f;
  p_f[3] = 4.0f;

  p_b = (ValueType*)malloc(NUM_NODES * sizeof(ValueType));
  memset(p_b, 0, NUM_NODES * sizeof(ValueType));

  p_x = (ValueType*)malloc(NUM_NODES * sizeof(ValueType));
  memset(p_x, 0, NUM_NODES * sizeof(ValueType));

  op_set nodes = op_decl_set(NUM_NODES, "nodes");
  op_set elements = op_decl_set(NUM_ELE, "elements");
  op_map elem_node = op_decl_map(elements, nodes, 3, p_elem_node, "elem_node");

  op_sparsity sparsity = op_decl_sparsity(elem_node, elem_node, "sparsity");
  op_mat mat = op_decl_mat(sparsity, 1, VALUESTR, sizeof(ValueType), "mat");
  op_dat xn = op_decl_dat(nodes, 2, VALUESTR, p_xn, "xn");

  // Dat for the field initial condition
  op_dat f = op_decl_dat(nodes, 1, VALUESTR, p_f, "f");

  // Dat for the RHS vector
  op_dat b = op_decl_dat(nodes, 1, VALUESTR, p_b, "b");

  // Dat for solution
  op_dat x = op_decl_dat(nodes, 1, VALUESTR, p_x, "x");

  op_par_loop(mass, "mass", op_iteration_space(elements, 3, 3),
              op_arg_mat(mat, op_i(1), elem_node, op_i(2), elem_node, 1, VALUESTR, OP_INC),
              op_arg_dat(xn, -3, elem_node, 2, VALUESTR, OP_READ));

  op_par_loop(rhs, "rhs", elements,
              op_arg_dat(b, -3, elem_node, 1, VALUESTR, OP_INC),
              op_arg_dat(xn, -3, elem_node, 2, VALUESTR, OP_READ),
              op_arg_dat(f, -3, elem_node, 1, VALUESTR, OP_READ));

  op_solve(mat, b, x);

  // Check result
  int failed = 0;
  op_fetch_data(x);
  for (int i=0; i<NUM_NODES; ++i) {
    double delta = fabs((double)p_x[i] - (double)p_f[i]);
    if (delta > TOLERANCE) {
      failed = 1;
      printf("Failed: delta = %18.16f for node %d.\n", delta, i);
    }
  }

  op_exit();
  return failed;
}
