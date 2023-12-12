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
#include <unistd.h>  // For getopt

// global constants

float gm1, gm1i, wtg1[2], xi1[2], Ng1[4], Ng1_xi[4], wtg2[4], Ng2[16],
    Ng2_xi[32], minf, m2, freq, kappa, nmode, mfan;

//
// OP header file
//

#include "op_seq.h"

//
// kernel routines for parallel loops
//

#include "dirichlet.h"
#include "dotPV.h"
#include "dotR.h"
#include "init_cg.h"
#include "res_calc.h"
#include "spMV.h"
#include "update.h"
#include "updateP.h"
#include "updateUR.h"

// main program

int main(int argc, char **argv) {


  // Define variables to store input values
  int niter = 20, maxiter = 200;
  float cd_cond = 0.1f;

  int c;

  // Use getopt to parse command line arguments
  while ((c = getopt(argc, argv, "n:m:c:")) != -1) {
    switch (c) {
      case 'n':
        niter = atoi(optarg);
        break;
      case 'm':
        maxiter = atoi(optarg);
        break;
      case 'c':
        cd_cond = atof(optarg);
        break;
      case '?':
        if (optopt == 'n' || optopt == 'm' || optopt == 'c') {
          fprintf(stderr, "Option -%c requires an argument.\n", optopt);
        } else {
          fprintf(stderr, "Unknown option -%c.\n", optopt);
        }
        return 1;
    }
  }

  op_printf("niter = %d, maxiter = %d, cd_cond = %lf\n", niter, maxiter, cd_cond);



  
  // OP initialisation

  op_init(argc, argv, 2);

  int *bnode, *cell;
  float *xm; //, *q;

  int nnode, ncell, nbnodes;//, niter;
  float rms = 1.0f;

  // set constants and initialise flow field and residual

  op_printf("initialising flow field \n");

  float gam = 1.4f;
  gm1 = gam - 1.0f;
  gm1i = 1.0f / gm1;

  wtg1[0] = 0.5f;
  wtg1[1] = 0.5f;
  xi1[0] = 0.211324865405187f;
  xi1[1] = 0.788675134594813f;
  Ng1[0] = 0.788675134594813f;
  Ng1[1] = 0.211324865405187f;
  Ng1[2] = 0.211324865405187f;
  Ng1[3] = 0.788675134594813f;
  Ng1_xi[0] = -1.0f;
  Ng1_xi[1] = -1.0f;
  Ng1_xi[2] = 1.0f;
  Ng1_xi[3] = 1.0f;
  wtg2[0] = 0.25f;
  wtg2[1] = 0.25f;
  wtg2[2] = 0.25f;
  wtg2[3] = 0.25f;
  Ng2[0] = 0.622008467928146f;
  Ng2[1] = 0.166666666666667f;
  Ng2[2] = 0.166666666666667f;
  Ng2[3] = 0.044658198738520f;
  Ng2[4] = 0.166666666666667f;
  Ng2[5] = 0.622008467928146f;
  Ng2[6] = 0.044658198738520f;
  Ng2[7] = 0.166666666666667f;
  Ng2[8] = 0.166666666666667f;
  Ng2[9] = 0.044658198738520f;
  Ng2[10] = 0.622008467928146f;
  Ng2[11] = 0.166666666666667f;
  Ng2[12] = 0.044658198738520f;
  Ng2[13] = 0.166666666666667f;
  Ng2[14] = 0.166666666666667f;
  Ng2[15] = 0.622008467928146f;
  Ng2_xi[0] = -0.788675134594813f;
  Ng2_xi[1] = 0.788675134594813f;
  Ng2_xi[2] = -0.211324865405187f;
  Ng2_xi[3] = 0.211324865405187f;
  Ng2_xi[4] = -0.788675134594813f;
  Ng2_xi[5] = 0.788675134594813f;
  Ng2_xi[6] = -0.211324865405187f;
  Ng2_xi[7] = 0.211324865405187f;
  Ng2_xi[8] = -0.211324865405187f;
  Ng2_xi[9] = 0.211324865405187f;
  Ng2_xi[10] = -0.788675134594813f;
  Ng2_xi[11] = 0.788675134594813f;
  Ng2_xi[12] = -0.211324865405187f;
  Ng2_xi[13] = 0.211324865405187f;
  Ng2_xi[14] = -0.788675134594813f;
  Ng2_xi[15] = 0.788675134594813f;
  Ng2_xi[16] = -0.788675134594813f;
  Ng2_xi[17] = -0.211324865405187f;
  Ng2_xi[18] = 0.788675134594813f;
  Ng2_xi[19] = 0.211324865405187f;
  Ng2_xi[20] = -0.211324865405187f;
  Ng2_xi[21] = -0.788675134594813f;
  Ng2_xi[22] = 0.211324865405187f;
  Ng2_xi[23] = 0.788675134594813f;
  Ng2_xi[24] = -0.788675134594813f;
  Ng2_xi[25] = -0.211324865405187f;
  Ng2_xi[26] = 0.788675134594813f;
  Ng2_xi[27] = 0.211324865405187f;
  Ng2_xi[28] = -0.211324865405187f;
  Ng2_xi[29] = -0.788675134594813f;
  Ng2_xi[30] = 0.211324865405187f;
  Ng2_xi[31] = 0.788675134594813f;

  minf = 0.1f;
  m2 = minf * minf;
  freq = 1.0f;
  kappa = 1.0f;
  nmode = 0.0f;

  mfan = 1.0f;

  char file[] = "FE_grid.h5";

  // declare sets, pointers, datasets and global constants

  op_set nodes = op_decl_set_hdf5(file, "nodes");
  op_set bnodes = op_decl_set_hdf5(file, "bedges");
  op_set cells = op_decl_set_hdf5(file, "cells");

  op_map pbnodes = op_decl_map_hdf5(bnodes, nodes, 1, file, "pbedge");
  op_map pcell = op_decl_map_hdf5(cells, nodes, 4, file, "pcell");

  op_dat p_xm = op_decl_dat_hdf5(nodes, 2, "float", file, "p_x");
  op_dat p_phim = op_decl_dat_hdf5(nodes, 1, "float", file, "p_phim");
  op_dat p_resm = op_decl_dat_hdf5(nodes, 1, "float", file, "p_resm");
  op_dat p_none = op_decl_dat_hdf5(nodes, 4, "float", file, "p_none");
  op_dat p_K = op_decl_dat_hdf5(cells, 16, "float", file, "p_K");
  op_dat p_V = op_decl_dat_hdf5(nodes, 1, "float", file, "p_V");
  op_dat p_P = op_decl_dat_hdf5(nodes, 1, "float", file, "p_P");
  op_dat p_U = op_decl_dat_hdf5(nodes, 1, "float", file, "p_U");

  op_decl_const(1, "float", &gam);
  op_decl_const(1, "float", &gm1);
  op_decl_const(1, "float", &gm1i);
  op_decl_const(1, "float", &m2);
  op_decl_const(2, "float", wtg1);
  op_decl_const(2, "float", xi1);
  op_decl_const(4, "float", Ng1);
  op_decl_const(4, "float", Ng1_xi);
  op_decl_const(4, "float", wtg2);
  op_decl_const(16, "float", Ng2);
  op_decl_const(32, "float", Ng2_xi);
  op_decl_const(1, "float", &minf);
  op_decl_const(1, "float", &freq);
  op_decl_const(1, "float", &kappa);
  op_decl_const(1, "float", &nmode);
  op_decl_const(1, "float", &mfan);

  op_diagnostic_output();

  op_partition("PTSCOTCH", "KWAY", cells, pcell, p_xm);

  op_printf("nodes: %d cells: %d bnodes: %d\n", nodes->size, cells->size,
            bnodes->size);
  nnode = op_get_size(nodes);
  ncell = op_get_size(cells);
  nbnodes = op_get_size(bnodes);

  double cpu_t1, cpu_t2, cpu_tm, wall_t1, wall_t2, wall_tm;
  op_timers(&cpu_t1, &wall_t1);

  // main time-marching loop

  //niter = 500;

  for (int iter = 1; iter <= niter; iter++) {

    op_par_loop(res_calc, "res_calc", cells,
                op_arg_dat(p_xm, -4, pcell, 2, "float", OP_READ),
                op_arg_dat(p_phim, -4, pcell, 1, "float", OP_READ),
                op_opt_arg_dat(1,p_K, -1, OP_ID, 16, "float", OP_WRITE),
                op_opt_arg_dat(1,p_resm, -4, pcell, 1, "float", OP_RW),
                op_opt_arg_dat(0,p_none, -4, pcell, 2, "float", OP_INC));

    op_par_loop(dirichlet, "dirichlet", bnodes,
                op_arg_dat(p_resm, 0, pbnodes, 1, "float", OP_WRITE));

    float c1 = 0.0f;
    float c2 = 0.0f;
    float c3 = 0.0f;
    float alpha = 0.0f;
    float beta = 0.0f;

    // c1 = R'*R;
    op_par_loop(init_cg, "init_cg", nodes,
                op_arg_dat(p_resm, -1, OP_ID, 1, "float", OP_READ),
                op_arg_gbl(&c1, 1, "float", OP_INC),
                op_arg_dat(p_U, -1, OP_ID, 1, "float", OP_WRITE),
                op_arg_dat(p_V, -1, OP_ID, 1, "float", OP_WRITE),
                op_arg_dat(p_P, -1, OP_ID, 1, "float", OP_WRITE));

    // set up stopping conditions
    float res0 = sqrt(c1);
    float res = res0;
    int inner_iter = 0;
    //int maxiter = 200;
    while (res > cd_cond * res0 && inner_iter < maxiter) {
      // V = Stiffness*P
      op_par_loop(spMV, "spMV", cells,
                  op_arg_dat(p_V, -4, pcell, 1, "float", OP_INC),
                  op_arg_dat(p_K, -1, OP_ID, 16, "float", OP_READ),
                  op_arg_dat(p_P, -4, pcell, 1, "float", OP_READ));

      op_par_loop(dirichlet, "dirichlet", bnodes,
                  op_arg_dat(p_V, 0, pbnodes, 1, "float", OP_WRITE));

      c2 = 0.0f;

      // c2 = P'*V;
      op_par_loop(dotPV, "dotPV", nodes,
                  op_arg_dat(p_P, -1, OP_ID, 1, "float", OP_READ),
                  op_arg_dat(p_V, -1, OP_ID, 1, "float", OP_READ),
                  op_arg_gbl(&c2, 1, "float", OP_INC));

      alpha = c1 / c2;

      // U = U + alpha*P;
      // resm = resm-alpha*V;
      op_par_loop(updateUR, "updateUR", nodes,
                  op_arg_dat(p_U, -1, OP_ID, 1, "float", OP_INC),
                  op_arg_dat(p_resm, -1, OP_ID, 1, "float", OP_INC),
                  op_arg_dat(p_P, -1, OP_ID, 1, "float", OP_READ),
                  op_arg_dat(p_V, -1, OP_ID, 1, "float", OP_RW),
                  op_arg_gbl(&alpha, 1, "float", OP_READ));

      c3 = 0.0f;

      // c3 = resm'*resm;
      op_par_loop(dotR, "dotR", nodes,
                  op_arg_dat(p_resm, -1, OP_ID, 1, "float", OP_READ),
                  op_arg_gbl(&c3, 1, "float", OP_INC));
      beta = c3 / c1;
      // P = beta*P+resm;
      op_par_loop(updateP, "updateP", nodes,
                  op_arg_dat(p_resm, -1, OP_ID, 1, "float", OP_READ),
                  op_arg_dat(p_P, -1, OP_ID, 1, "float", OP_RW),
                  op_arg_gbl(&beta, 1, "float", OP_READ));
      c1 = c3;
      res = sqrt(c1);
      inner_iter++;
    }
    rms = 0.0f;
    // phim = phim - Stiffness\Load;
    op_par_loop(update, "update", nodes,
                op_arg_dat(p_phim, -1, OP_ID, 1, "float", OP_RW),
                op_arg_dat(p_resm, -1, OP_ID, 1, "float", OP_WRITE),
                op_arg_dat(p_U, -1, OP_ID, 1, "float", OP_READ),
                op_arg_gbl(&rms, 1, "float", OP_INC));
    // op_printf("rms = %10.5e iter: %d\n", sqrt(rms) / sqrt(nnode), iter);
    // print iteration history
    rms = sqrt(rms / (float)op_get_size(nodes));

    
    op_timers(&cpu_tm, &wall_tm);
    op_printf("%d %d %f %3.15E\n", iter, inner_iter, wall_tm - wall_t1, rms);

    if (rms<1e-12) break;

    if (iter % niter ==
        0) { //&& ncell == 720000) { // defailt mesh -- for validation testing
      float diff = fabs((100.0f * (rms / 0.0000005644214176463586f)) - 100.0f);
      op_printf("\n\nTest problem with %d nodes is within %3.15E %% of the "
                "expected solution\n",
                op_get_size(nodes), diff);
      if (diff < 0.02f) {
        op_printf("This test is considered PASSED\n");
      } else {
        op_printf("This test is considered FAILED\n");
      }
    }
  }

  op_timing_output();
  op_timers(&cpu_t2, &wall_t2);
  op_printf("Max total runtime = %f\n", wall_t2 - wall_t1);
  op_exit();
}
