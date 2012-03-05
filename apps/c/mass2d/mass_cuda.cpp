#include <stdio.h>
#include <stdlib.h>
#include <set>
#include <string>
#include <vector>
#include "op_lib_cpp.h"
#include "op_lib_mat.h"

void op_par_loop_mass(const char *, op_set, op_arg, op_arg);

void read_from_triangle_files(std::string basename,
                              int *nnode,
                              int *nele,
                              int **p_elem_node,
                              float **p_xn)
{
  FILE *f;

  f = fopen((basename + ".node").c_str(), "r");
  int dim, tmp, ttmp;

  fscanf(f, "%d %d %d %d", nnode, &dim, &tmp, &ttmp);

  if ( dim != 3 ) {
    fprintf(stderr, "Unknown triangle format\n");
    exit(-1);
  }
  *p_xn = (float *)malloc(*nnode * sizeof(float) * 2);

  float dummy;
  for ( int i = 0; i < *nnode; i++ ) {
    fscanf(f, "%d %g %g %g", &tmp, (*p_xn) + 2*i, (*p_xn) + 2*i + 1, &dummy);
  }
  fclose(f);
  f = fopen((basename + ".ele").c_str(), "r");
  fscanf(f, "%d %d %d", nele, &dim, &tmp);
  if ( dim != 3 ) {
    fprintf(stderr, "Unknown ele format\n");
    exit(-1);
  }
  *p_elem_node = (int *)malloc(*nele * sizeof(int) * 3);
  for ( int i = 0; i < *nele; i++ ) {
    fscanf(f, "%d %d %d %d %d", &tmp,
        (*p_elem_node) + 3*i,
        (*p_elem_node) + 3*i + 1,
        (*p_elem_node) + 3*i + 2,
        &ttmp);
    // correct for fortran numbering
    (*p_elem_node)[3*i]--;
    (*p_elem_node)[3*i+1]--;
    (*p_elem_node)[3*i+2]--;
  }
  fclose(f);
}

int main(int argc, char **argv)
{
  int *p_elem_node;
  float *p_xn;
  int i;
  float val;

  op_init(argc, argv, 5);

  int nnode = 4;
  int nele = 2;
  if ( argc > 1 ) {
    read_from_triangle_files(argv[1], &nnode, &nele, &p_elem_node, &p_xn);
  } else {
    p_elem_node = (int *)malloc(3 * nele * sizeof(int));
    p_elem_node[0] = 0;
    p_elem_node[1] = 1;
    p_elem_node[2] = 3;
    p_elem_node[3] = 2;
    p_elem_node[4] = 3;
    p_elem_node[5] = 1;


    p_xn = (float *)malloc(2 * nnode * sizeof(float));
    p_xn[0] = 0.0f;
    p_xn[1] = 0.0f;
    p_xn[2] = 2.0f;
    p_xn[3] = 0.0f;
    p_xn[4] = 1.0f;
    p_xn[5] = 1.0f;
    p_xn[6] = 0.0f;
    p_xn[7] = 1.5f;
  }
  op_set nodes = op_decl_set(nnode, "nodes");
  op_set elements = op_decl_set(nele, "elements");
  op_map elem_node = op_decl_map(elements, nodes, 3, p_elem_node, "elem_node");

  op_sparsity sparsity = op_decl_sparsity(elem_node, elem_node, "sparsity");
  op_mat mat = op_decl_mat(sparsity, 1, "float", sizeof(float), "mat");
  op_dat xn = op_decl_dat(nodes, 2, "float", p_xn, "xn");

  op_par_loop_mass("mass", elements,
                   op_arg_mat(mat, -3, elem_node, -3, elem_node, 1, "double", OP_INC),
                   op_arg_dat(xn, -3, elem_node, 2, "float", OP_READ));

  op_exit();
  return 0;
}
