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
// Unit test for reduction cases
//

//
// standard headers
//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//
// OP header file
//

#include "op_seq.h"

//
// kernel routines for parallel loops
//

#include "res_calc.h"
#include "update.h"

// main program

int main(int argc, char **argv) {
  // OP initialisation
  op_init(argc, argv, 2);

  int *becell, *ecell, *bound, *bedge, *edge, *cell;
  double *x, *q, *qold, *adt, *res;

  int nnode, ncell, nedge, nbedge;

  // timer
  double cpu_t1, cpu_t2, wall_t1, wall_t2;

  // read in airfoil grid

  op_printf("reading in data \n");

  FILE *fp;
  if ((fp = fopen("./new_grid.dat", "r")) == NULL) {
    op_printf("can't open file new_grid.dat\n");
    exit(-1);
  }

  if (fscanf(fp, "%d %d %d %d \n", &nnode, &ncell, &nedge, &nbedge) != 4) {
    op_printf("error reading from new_grid.dat\n");
    exit(-1);
  }

  cell = (int *)malloc(4 * ncell * sizeof(int));
  edge = (int *)malloc(2 * nedge * sizeof(int));
  ecell = (int *)malloc(2 * nedge * sizeof(int));
  bedge = (int *)malloc(2 * nbedge * sizeof(int));
  becell = (int *)malloc(nbedge * sizeof(int));
  bound = (int *)malloc(nbedge * sizeof(int));

  x = (double *)malloc(2 * nnode * sizeof(double));
  q = (double *)malloc(4 * ncell * sizeof(double));
  qold = (double *)malloc(4 * ncell * sizeof(double));
  res = (double *)malloc(4 * ncell * sizeof(double));
  adt = (double *)malloc(ncell * sizeof(double));

  for (int n = 0; n < nnode; n++) {
    if (fscanf(fp, "%lf %lf \n", &x[2 * n], &x[2 * n + 1]) != 2) {
      op_printf("error reading from new_grid.dat\n");
      exit(-1);
    }
  }

  for (int n = 0; n < ncell; n++) {
    if (fscanf(fp, "%d %d %d %d \n", &cell[4 * n], &cell[4 * n + 1],
               &cell[4 * n + 2], &cell[4 * n + 3]) != 4) {
      op_printf("error reading from new_grid.dat\n");
      exit(-1);
    }
  }

  for (int n = 0; n < nedge; n++) {
    if (fscanf(fp, "%d %d %d %d \n", &edge[2 * n], &edge[2 * n + 1],
               &ecell[2 * n], &ecell[2 * n + 1]) != 4) {
      op_printf("error reading from new_grid.dat\n");
      exit(-1);
    }
  }

  for (int n = 0; n < nbedge; n++) {
    if (fscanf(fp, "%d %d %d %d \n", &bedge[2 * n], &bedge[2 * n + 1],
               &becell[n], &bound[n]) != 4) {
      op_printf("error reading from new_grid.dat\n");
      exit(-1);
    }
  }

  fclose(fp);

  // declare sets, pointers, datasets

  op_set edges = op_decl_set(nedge, "edges");
  op_set cells = op_decl_set(ncell, "cells");

  op_map pecell = op_decl_map(edges, cells, 2, ecell, "pecell");
  op_dat p_res = op_decl_dat(cells, 4, "double", res, "p_res");

  int count1;
  int count2;

  op_diagnostic_output();

  // initialise timers for total execution wall time
  op_timers(&cpu_t1, &wall_t1);

  // indirect reduction
  count1 = 0;
  op_par_loop(res_calc, "res_calc", edges,
              op_arg_dat(p_res, 0, pecell, 4, "double", OP_INC),
              op_arg_gbl(&count1, 1, "int", OP_INC));
  op_printf("number of edges:: %d should be: %d \n", count1, nedge);
  if (count1 != nedge)
    op_printf("indirect reduction Failed\n");
  else
    op_printf("indirect reduction Passed\n");
  // direct reduction
  count2 = 0;
  op_par_loop(update, "update", cells,
              op_arg_dat(p_res, -1, OP_ID, 4, "double", OP_RW),
              op_arg_gbl(&count2, 1, "int", OP_INC));
  op_printf("number of cells: %d should be: %d \n", count2, ncell);
  if (count2 != ncell)
    op_printf("direct reduction Failed\n");
  else
    op_printf("direct reduction Passed\n");

  if (count1 == nedge && count2 == ncell)
    op_printf("Reduction application PASSED\n");
  else
    op_printf("Reduction application FAILED\n");

  op_timers(&cpu_t2, &wall_t2);
  op_timing_output();

  op_exit();

  free(cell);
  free(edge);
  free(ecell);
  free(bedge);
  free(becell);
  free(bound);
  free(x);
  free(q);
  free(qold);
  free(res);
  free(adt);
}
