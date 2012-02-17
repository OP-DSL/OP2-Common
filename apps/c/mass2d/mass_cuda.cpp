#include <stdio.h>
#include <stdlib.h>
#include "op_seq.h"
#include "op_lib_cpp.h"

#define NUM_ELE 2
#define NUM_NODES 4
#define NUM_DIM 2

void op_par_loop_mass(const char *, op_set, int, op_arg);
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

  op_dat xn = op_decl_dat(nodes, 2, "float", p_xn, "xn");

  int mat_size = elem_node->to->size * elem_node->to->size;
  op_par_loop_mass("mass", elements, mat_size,
                   op_arg_dat(xn, -3, elem_node, 2, "float", OP_READ));

  free(p_elem_node);
  free(p_xn);
  op_exit();
  return 0;
}
