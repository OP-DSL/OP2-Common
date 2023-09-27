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

//
// standard headers
//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// global constants

double gm1, gm1i, wtg1[2], xi1[2], Ng1[4], Ng1_xi[4], wtg2[4], Ng2[16],
    Ng2_xi[32], minf, m2, freq, kappa, nmode, mfan;
int op2_stride;

//
// mpi header file - included by user for user level mpi
//

#include <mpi.h>

//
// OP header file
//

#include "op_lib_cpp.h"
#include "op_lib_mpi.h"
#include "op_util.h"

//
// hdf5 header
//

#include "hdf5.h"

//
// kernel routines for parallel loops
//

/*
#include "dirichlet.h"
#include "dotPV.h"
#include "dotR.h"
#include "init_cg.h"
#include "res_calc.h"
#include "spMV.h"
#include "update.h"
#include "updateP.h"
#include "updateUR.h"
*/

// main program

int main(int argc, char **argv) {
  int my_rank;
  int comm_size;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  int *bnode, *cell;
  double *xm; //, *q;

  int nnode, ncell, nbnodes, niter;
  double rms = 1;

  // read in grid

  printf("reading in grid \n");

  FILE *fp;
  if ((fp = fopen("./FE_grid.dat", "r")) == NULL) {
    printf("can't open file new_grid.dat\n");
    exit(-1);
  }

  if (fscanf(fp, "%d %d %d \n", &nnode, &ncell, &nbnodes) != 3) {
    printf("error reading from new_grid.dat\n");
    exit(-1);
  }

  cell = (int *)malloc(4 * ncell * sizeof(int));
  bnode = (int *)malloc(nbnodes * sizeof(int));
  xm = (double *)malloc(2 * nnode * sizeof(double));

  for (int n = 0; n < nnode; n++) {
    if (fscanf(fp, "%lf %lf \n", &xm[2 * n], &xm[2 * n + 1]) != 2) {
      printf("error reading from new_grid.dat\n");
      exit(-1);
    }
  }

  for (int n = 0; n < ncell; n++) {
    if (fscanf(fp, "%d %d %d %d \n", &cell[4 * n], &cell[4 * n + 1],
               &cell[4 * n + 2], &cell[4 * n + 3]) != 4) {
      printf("error reading from new_grid.dat\n");
      exit(-1);
    }
  }

  for (int n = 0; n < nbnodes; n++) {
    if (fscanf(fp, "%d \n", &bnode[n]) != 1) {
      printf("error reading from new_grid.dat\n");
      exit(-1);
    }
  }

  fclose(fp);

  // set constants and initialise flow field and residual

  printf("initialising flow field \n");

  double gam = 1.4;
  gm1 = gam - 1.0;
  gm1i = 1.0 / gm1;

  wtg1[0] = 0.5;
  wtg1[1] = 0.5;
  xi1[0] = 0.211324865405187;
  xi1[1] = 0.788675134594813;
  Ng1[0] = 0.788675134594813;
  Ng1[1] = 0.211324865405187;
  Ng1[2] = 0.211324865405187;
  Ng1[3] = 0.788675134594813;
  Ng1_xi[0] = -1;
  Ng1_xi[1] = -1;
  Ng1_xi[2] = 1;
  Ng1_xi[3] = 1;
  wtg2[0] = 0.25;
  wtg2[1] = 0.25;
  wtg2[2] = 0.25;
  wtg2[3] = 0.25;
  Ng2[0] = 0.622008467928146;
  Ng2[1] = 0.166666666666667;
  Ng2[2] = 0.166666666666667;
  Ng2[3] = 0.044658198738520;
  Ng2[4] = 0.166666666666667;
  Ng2[5] = 0.622008467928146;
  Ng2[6] = 0.044658198738520;
  Ng2[7] = 0.166666666666667;
  Ng2[8] = 0.166666666666667;
  Ng2[9] = 0.044658198738520;
  Ng2[10] = 0.622008467928146;
  Ng2[11] = 0.166666666666667;
  Ng2[12] = 0.044658198738520;
  Ng2[13] = 0.166666666666667;
  Ng2[14] = 0.166666666666667;
  Ng2[15] = 0.622008467928146;
  Ng2_xi[0] = -0.788675134594813;
  Ng2_xi[1] = 0.788675134594813;
  Ng2_xi[2] = -0.211324865405187;
  Ng2_xi[3] = 0.211324865405187;
  Ng2_xi[4] = -0.788675134594813;
  Ng2_xi[5] = 0.788675134594813;
  Ng2_xi[6] = -0.211324865405187;
  Ng2_xi[7] = 0.211324865405187;
  Ng2_xi[8] = -0.211324865405187;
  Ng2_xi[9] = 0.211324865405187;
  Ng2_xi[10] = -0.788675134594813;
  Ng2_xi[11] = 0.788675134594813;
  Ng2_xi[12] = -0.211324865405187;
  Ng2_xi[13] = 0.211324865405187;
  Ng2_xi[14] = -0.788675134594813;
  Ng2_xi[15] = 0.788675134594813;
  Ng2_xi[16] = -0.788675134594813;
  Ng2_xi[17] = -0.211324865405187;
  Ng2_xi[18] = 0.788675134594813;
  Ng2_xi[19] = 0.211324865405187;
  Ng2_xi[20] = -0.211324865405187;
  Ng2_xi[21] = -0.788675134594813;
  Ng2_xi[22] = 0.211324865405187;
  Ng2_xi[23] = 0.788675134594813;
  Ng2_xi[24] = -0.788675134594813;
  Ng2_xi[25] = -0.211324865405187;
  Ng2_xi[26] = 0.788675134594813;
  Ng2_xi[27] = 0.211324865405187;
  Ng2_xi[28] = -0.211324865405187;
  Ng2_xi[29] = -0.788675134594813;
  Ng2_xi[30] = 0.211324865405187;
  Ng2_xi[31] = 0.788675134594813;

  minf = 0.1;
  m2 = minf * minf;
  freq = 1;
  kappa = 1;
  nmode = 0;

  mfan = 1.0;

  double *phim = (double *)malloc(nnode * sizeof(double));
  memset(phim, 0, nnode * sizeof(double));
  for (int i = 0; i < nnode; i++) {
    phim[i] = minf * xm[2 * i];
  }

  double *K = (double *)malloc(4 * 4 * ncell * sizeof(double));
  memset(K, 0, 4 * 4 * ncell * sizeof(double));
  double *resm = (double *)malloc(nnode * sizeof(double));
  memset(resm, 0, nnode * sizeof(double));

  double *V = (double *)malloc(nnode * sizeof(double));
  memset(V, 0, nnode * sizeof(double));
  double *P = (double *)malloc(nnode * sizeof(double));
  memset(P, 0, nnode * sizeof(double));
  double *U = (double *)malloc(nnode * sizeof(double));
  memset(U, 0, nnode * sizeof(double));

  // OP initialisation

  op_init(argc, argv, 2);

  // declare sets, pointers, datasets and global constants

  op_set nodes = op_decl_set(nnode, "nodes");
  op_set bnodes = op_decl_set(nbnodes, "bedges");
  op_set cells = op_decl_set(ncell, "cells");

  op_map pbnodes = op_decl_map(bnodes, nodes, 1, bnode, "pbedge");
  op_map pcell = op_decl_map(cells, nodes, 4, cell, "pcell");

  op_dat p_xm = op_decl_dat(nodes, 2, "double", xm, "p_x");
  op_dat p_phim = op_decl_dat(nodes, 1, "double", phim, "p_phim");
  op_dat p_resm = op_decl_dat(nodes, 1, "double", resm, "p_resm");
  op_dat p_K = op_decl_dat(cells, 16, "double:soa", K, "p_K");

  op_dat p_V = op_decl_dat(nodes, 1, "double", V, "p_V");
  op_dat p_P = op_decl_dat(nodes, 1, "double", P, "p_P");
  op_dat p_U = op_decl_dat(nodes, 1, "double", U, "p_U");

  op_decl_const(1, "double", &gam);
  op_decl_const(1, "double", &gm1);
  op_decl_const(1, "double", &gm1i);
  op_decl_const(1, "double", &m2);
  op_decl_const(2, "double", wtg1);
  op_decl_const(2, "double", xi1);
  op_decl_const(4, "double", Ng1);
  op_decl_const(4, "double", Ng1_xi);
  op_decl_const(4, "double", wtg2);
  op_decl_const(16, "double", Ng2);
  op_decl_const(32, "double", Ng2_xi);
  op_decl_const(1, "double", &minf);
  op_decl_const(1, "double", &freq);
  op_decl_const(1, "double", &kappa);
  op_decl_const(1, "double", &nmode);
  op_decl_const(1, "double", &mfan);
#ifdef CUDA
  op2_stride = cells->size;
#else
  op2_stride = 1;
#endif
  op_decl_const(1, "int", &op2_stride);

  op_dump_to_hdf5("FE_grid_out.h5");

  op_diagnostic_output();
  // create halos - for sanity check
  op_halo_create();

  op_timing_output();
  op_exit();
}
