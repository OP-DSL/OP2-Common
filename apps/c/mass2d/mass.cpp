#include <stdio.h>
#include <stdlib.h>
#include "op_seq.h"
#include "op_lib_cpp.h"
#include "mass.h"

#define NUM_ELE 2
#define NUM_NODES 4
#define NUM_DIM 2

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
    p_xn[0] = 0.f;
    p_xn[1] = 0.f;
    p_xn[2] = 1.f;
    p_xn[3] = 0.f;
    p_xn[4] = 1.f;
    p_xn[5] = 1.f;
    p_xn[6] = 0.f;
    p_xn[7] = 1.f;

    op_set nodes = op_decl_set(NUM_NODES, "nodes");
    op_set elements = op_decl_set(NUM_ELE, "elements");
    op_map elem_node = op_decl_map(elements, nodes, 3, p_elem_node, "elem_node");

    op_sparsity sparsity = op_decl_sparsity(elem_node, elem_node);
    op_mat mat = op_decl_mat(sparsity);
    op_dat xn = op_decl_dat(nodes, 2, "float", p_xn, "xn");

    op_zero(mat);
    op_par_loop(mass, "sum", elements,
                op_arg_mat(mat, -3, elem_node, -3, elem_node, OP_INC),
                op_arg_dat(xn, -3, elem_node, 2, "float", OP_READ));

    free(p_elem_node);
    free(p_xn);
    op_exit();
    return 0;
}
