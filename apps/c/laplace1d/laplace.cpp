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

// test program demonstrating assembly of op_sparse_matrix for FE
// discretisation of a 1D Laplace operator and a linear solve

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// OP header file

#include "op_lib_cpp.h"
#include "op_seq_mat.h"

typedef double Real;
#define REAL "double"

// kernel routines for parallel loops

#include "laplace.h"

// define problem size

#define NN       6
#define NITER    2

// main program

int main(int argc, char **argv) {

  int   nnode = (NN+1);

  int   *p_elem_node = (int *)malloc(2*sizeof(int)*NN);
  Real  *p_xn = (Real *)malloc(sizeof(Real)*nnode);
  Real  *p_x  = (Real *)malloc(sizeof(Real)*nnode);
  Real  *p_xref = (Real *)malloc(sizeof(Real)*nnode);
  Real  *p_y  = (Real *)malloc(sizeof(Real)*nnode);

  // create element -> node mapping
  for (int i = 0; i < NN; ++i) {
    p_elem_node[2*i] = i;
    p_elem_node[2*i+1] = i+1;
  }

  // create coordinates and populate x with -1/pi^2*sin(pi*x)
  for (int i = 0; i < nnode; ++i) {
    /*p_xn[i] = sin(0.5*M_PI*i/NN);*/
    p_xn[i] = (Real)i/NN;
    p_x[i] = (1./(M_PI*M_PI))*sin(M_PI*p_xn[i]);
    p_xref[i] = sin(M_PI*p_xn[i]);
  }

  // OP initialisation

  op_init(argc,argv,2);

  // declare sets, pointers, and datasets

  op_set nodes = op_decl_set(nnode, "nodes");
  op_set elements = op_decl_set(NN, "elements");

  op_map elem_node = op_decl_map(elements, nodes, 2, p_elem_node, "elem_node");

  op_dat x = op_decl_dat(nodes, 1, REAL, p_x, "x");
  op_dat y = op_decl_dat(nodes, 1, REAL, p_y, "y");
  op_dat xn = op_decl_dat(nodes, 1, REAL, p_xn, "xn");

  op_sparsity sparsity = op_decl_sparsity(elem_node, elem_node, "sparsity");

  op_mat mat = op_decl_mat(sparsity, 1, REAL, sizeof(Real), "matrix");

  op_diagnostic_output();

  // Fix the values of the boundary nodes to get a unique solution
  Real val = 1e308;
  int idx = 0;
  op_mat_addto(mat, &val, 1, &idx, 1, &idx);
  idx = NN;
  op_mat_addto(mat, &val, 1, &idx, 1, &idx);

  // construct the matrix
  op_par_loop(laplace, "laplace", elements,
              op_arg_mat(mat, -2, elem_node, -2, elem_node, 1, REAL, OP_INC),
              op_arg_dat(xn,  -2, elem_node, 1, REAL, OP_READ));

  // solve
  op_solve(mat, x, y);

  op_exit();
}

