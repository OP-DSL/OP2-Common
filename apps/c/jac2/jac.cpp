/*
 * Open source copyright declaration based on BSD open source template:
 * http://www.opensource.org/licenses/bsd-license.php
 *
 * This file is part of the OP2 distribution.
 *
 * Copyright (c) 2011, Mike Giles and others. Please see the AUTHORS file in
 * the main source directory for a full list of copyright holders.
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
// test program for new OPlus2 development
//

//
// standard headers
//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// global constants

float alpha;

//
// OP header file
//

#include "op_seq.h"

// jac header file

#define STRIDE 2
#include "check_result.h"

//
// kernel routines for parallel loops
//

#include "res.h"
#include "update.h"

// Error tolerance in checking correctness

#define TOLERANCE 1e-6

// define problem size

#define NN 6
#define NITER 2

// main program

int main(int argc, char **argv) {
  // OP initialisation
  op_init(argc, argv, 5);

  // timer
  double cpu_t1, cpu_t2, wall_t1, wall_t2;

  int nnode, nedge, n, e;
  float dx;

  nnode = (NN - 1) * (NN - 1);
  nedge = (NN - 1) * (NN - 1) + 4 * (NN - 1) * (NN - 2);
  dx = 1.0f / ((float)NN);

  int *pp = (int *)malloc(sizeof(int) * 2 * nedge);
  int *p1 = (int *)malloc(sizeof(int) * nedge);
  int *p2 = (int *)malloc(sizeof(int) * nedge);

  float *xe = (float *)malloc(sizeof(float) * 2 * nedge);
  float *xn = (float *)malloc(sizeof(float) * 2 * nnode);

  double *A = (double *)malloc(sizeof(double) * 3 * nedge);
  float *r = (float *)malloc(sizeof(float) * 2 * nnode);
  float *u = (float *)malloc(sizeof(float) * 2 * nnode);
  float *du = (float *)malloc(sizeof(float) * 3 * nnode);

  // create matrix and r.h.s., and set coordinates needed for renumbering /
  // partitioning

  e = 0;

  for (int i = 1; i < NN; i++) {
    for (int j = 1; j < NN; j++) {
      n = i - 1 + (j - 1) * (NN - 1);
      r[2 * n] = 0.0f;
      u[2 * n] = 0.0f;
      du[3 * n] = 0.0f;
      xn[2 * n] = i * dx;
      xn[2 * n + 1] = j * dx;

      p1[e] = n;
      p2[e] = n;
      pp[2 * e] = p1[e];
      pp[2 * e + 1] = p2[e];
      A[3 * e] = -1.0f;
      xe[2 * e] = i * dx;
      xe[2 * e + 1] = j * dx;
      e++;

      for (int pass = 0; pass < 4; pass++) {
        int i2 = i;
        int j2 = j;
        if (pass == 0)
          i2 += -1;
        if (pass == 1)
          i2 += 1;
        if (pass == 2)
          j2 += -1;
        if (pass == 3)
          j2 += 1;

        if ((i2 == 0) || (i2 == NN) || (j2 == 0) || (j2 == NN)) {
          r[2 * n] += 0.25f;
        } else {
          p1[e] = n;
          p2[e] = i2 - 1 + (j2 - 1) * (NN - 1);
          pp[2 * e] = p1[e];
          pp[2 * e + 1] = p2[e];
          A[3 * e] = 0.25f;
          xe[2 * e] = i * dx;
          xe[2 * e + 1] = j * dx;
          e++;
        }
      }
    }
  }

  // declare sets, pointers, and datasets

  op_set nodes = op_decl_set(nnode, "nodes");
  op_set edges = op_decl_set(nedge, "edges");

  op_map ppedge = op_decl_map(edges, nodes, 2, pp, "ppedge");

  op_dat p_A = op_decl_dat(edges, 3, "double", A, "p_A");
  op_dat p_r = op_decl_dat(nodes, 2, "float", r, "p_r");
  op_dat p_u = op_decl_dat(nodes, 2, "float", u, "p_u");
  op_dat p_du = op_decl_dat(nodes, 3, "float", du, "p_du");

  alpha = 2.0f;
  op_decl_const(1, "float", &alpha);
  alpha = 1.0f;
  op_decl_const(1, "float", &alpha);

  op_diagnostic_output();

  // initialise timers for total execution wall time
  op_timers(&cpu_t1, &wall_t1);

  // main iteration loop

  float u_sum, u_max, beta = 1.0f;

  for (int iter = 0; iter < NITER; iter++) {
    op_par_loop(res, "res", edges,
                op_arg_dat(p_A, -1, OP_ID, 3, "double", OP_READ),
                op_arg_dat(p_u, 1, ppedge, 2, "float", OP_READ),
                op_arg_dat(p_du, 0, ppedge, 3, "float", OP_INC),
                op_arg_gbl(&beta, 1, "float", OP_READ));

    u_sum = 0.0f;
    u_max = 0.0f;
    op_par_loop(update, "update", nodes,
                op_arg_dat(p_r, -1, OP_ID, 2, "float", OP_READ),
                op_arg_dat(p_du, -1, OP_ID, 3, "float", OP_RW),
                op_arg_dat(p_u, -1, OP_ID, 2, "float", OP_INC),
                op_arg_gbl(&u_sum, 1, "float", OP_INC),
                op_arg_gbl(&u_max, 1, "float", OP_MAX));
    op_printf("\n u max/rms = %f %f \n\n", u_max, sqrt(u_sum / nnode));
  }

  op_timers(&cpu_t2, &wall_t2);

  // print out results
  op_printf("\n  Results after %d iterations:\n\n", NITER);

  op_fetch_data(p_u, u);

  for (int pass = 0; pass < 1; pass++) {
    for (int j = NN - 1; j > 0; j--) {
      for (int i = 1; i < NN; i++) {
        if (pass == 0)
          op_printf(" %7.4f", u[2 * (i - 1 + (j - 1) * (NN - 1))]);
        else if (pass == 1)
          op_printf(" %7.4f", du[i - 1 + (j - 1) * (NN - 1)]);
        else if (pass == 2)
          op_printf(" %7.4f", r[2 * (i - 1 + (j - 1) * (NN - 1))]);
      }
      op_printf("\n");
    }
    op_printf("\n");
  }

  op_timing_output();

  // print total time for niter interations
  op_printf("Max total runtime = %f\n", wall_t2 - wall_t1);

  int result = check_result<float>(u, NN, TOLERANCE);
  op_exit();

  free(pp);
  free(A);
  free(u);
  free(du);
  free(r);

  return result;
}
