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
// standard headers
//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// global constants

double gam, gm1, cfl, eps, mach, alpha, qinf[4];

//
// hdf5 header
//

#include "hdf5.h"

//
// op_par_loop declarations
//

#include "op_seq.h"

static void check_scan(int items_received, int items_expected) {
  if (items_received != items_expected) {
    printf("error reading from new_grid.dat\n");
    exit(-1);
  }
}

//
// main program
//

int main(int argc, char **argv) {
  // OP initialisation
  op_init(argc, argv, 2);

  int *becell, *ecell, *bound, *bedge, *edge, *cell;
  double *x, *q, *qold, *adt, *res;

  int nnode, ncell, nedge, nbedge;

  // set constants

  op_printf("initialising flow field\n");
  gam = 1.4f;
  gm1 = gam - 1.0f;
  cfl = 0.9f;
  eps = 0.05f;

  double mach = 0.4f;
  double alpha = 3.0f * atan(1.0f) / 45.0f;
  double p = 1.0f;
  double r = 1.0f;
  double u = sqrt(gam * p / r) * mach;
  double e = p / (r * gm1) + 0.5f * u * u;

  qinf[0] = r;
  qinf[1] = r * u;
  qinf[2] = 0.0f;
  qinf[3] = r * e;

  /**------------------------BEGIN  I/O -------------------**/

  char file[] = "new_grid.dat";
  char file_out[] = "new_grid_out.h5";

  /* read in grid from disk on root processor */
  FILE *fp;

  if ((fp = fopen(file, "r")) == NULL) {
    op_printf("can't open file %s\n", file);
    exit(-1);
  }

  check_scan(fscanf(fp, "%d %d %d %d \n", &nnode, &ncell, &nedge, &nbedge), 4);

  op_printf("reading in grid \n");
  op_printf("Global number of nodes, cells, edges, bedges = %d, %d, %d, %d\n",
            nnode, ncell, nedge, nbedge);

  cell = (int *)op_malloc(4 * ncell * sizeof(int));
  edge = (int *)op_malloc(2 * nedge * sizeof(int));
  ecell = (int *)op_malloc(2 * nedge * sizeof(int));
  bedge = (int *)op_malloc(2 * nbedge * sizeof(int));
  becell = (int *)op_malloc(nbedge * sizeof(int));
  bound = (int *)op_malloc(nbedge * sizeof(int));

  x = (double *)op_malloc(2 * nnode * sizeof(double));
  q = (double *)op_malloc(4 * ncell * sizeof(double));
  qold = (double *)op_malloc(4 * ncell * sizeof(double));
  res = (double *)op_malloc(4 * ncell * sizeof(double));
  adt = (double *)op_malloc(ncell * sizeof(double));

  for (int n = 0; n < nnode; n++) {
    check_scan(fscanf(fp, "%lf %lf \n", &x[2 * n], &x[2 * n + 1]), 2);
  }

  for (int n = 0; n < ncell; n++) {
    check_scan(fscanf(fp, "%d %d %d %d \n", &cell[4 * n], &cell[4 * n + 1],
                      &cell[4 * n + 2], &cell[4 * n + 3]),
               4);
  }

  for (int n = 0; n < nedge; n++) {
    check_scan(fscanf(fp, "%d %d %d %d \n", &edge[2 * n], &edge[2 * n + 1],
                      &ecell[2 * n], &ecell[2 * n + 1]),
               4);
  }

  for (int n = 0; n < nbedge; n++) {
    check_scan(fscanf(fp, "%d %d %d %d \n", &bedge[2 * n], &bedge[2 * n + 1],
                      &becell[n], &bound[n]),
               4);
  }

  // initialise flow field and residual

  for (int n = 0; n < ncell; n++) {
    for (int m = 0; m < 4; m++) {
      q[4 * n + m] = qinf[m];
      res[4 * n + m] = 0.0f;
    }
  }

  fclose(fp);

  cell = (int *)op_malloc(4 * ncell * sizeof(int));
  edge = (int *)op_malloc(2 * nedge * sizeof(int));
  ecell = (int *)op_malloc(2 * nedge * sizeof(int));
  bedge = (int *)op_malloc(2 * nbedge * sizeof(int));
  becell = (int *)op_malloc(nbedge * sizeof(int));
  bound = (int *)op_malloc(nbedge * sizeof(int));

  x = (double *)op_malloc(2 * nnode * sizeof(double));
  q = (double *)op_malloc(4 * ncell * sizeof(double));
  qold = (double *)op_malloc(4 * ncell * sizeof(double));
  res = (double *)op_malloc(4 * ncell * sizeof(double));
  adt = (double *)op_malloc(ncell * sizeof(double));

  /**------------------------END I/O  -----------------------**/

  /* FIXME: It's not clear to the compiler that sth. is going on behind the
     scenes here. Hence theses variables are reported as unused */

  op_set nodes = op_decl_set(nnode, "nodes");
  op_set edges = op_decl_set(nedge, "edges");
  op_set bedges = op_decl_set(nbedge, "bedges");
  op_set cells = op_decl_set(ncell, "cells");

  op_map pedge = op_decl_map(edges, nodes, 2, edge, "pedge");
  op_map pecell = op_decl_map(edges, cells, 2, ecell, "pecell");
  op_map pbedge = op_decl_map(bedges, nodes, 2, bedge, "pbedge");
  op_map pbecell = op_decl_map(bedges, cells, 1, becell, "pbecell");
  op_map pcell = op_decl_map(cells, nodes, 4, cell, "pcell");

  op_dat p_bound = op_decl_dat(bedges, 1, "int", bound, "p_bound");
  op_dat p_x = op_decl_dat(nodes, 2, "double", x, "p_x");
  op_dat p_q = op_decl_dat(cells, 4, "double", q, "p_q");
  op_dat p_qold = op_decl_dat(cells, 4, "double", qold, "p_qold");
  op_dat p_adt = op_decl_dat(cells, 1, "double", adt, "p_adt");
  op_dat p_res = op_decl_dat(cells, 4, "double", res, "p_res");

  /* Test out creating dataset within a nested path in an HDF5 file
  -- Remove when needing to create correct Airfoil mesh
  */
  op_dat p_x_test =
      op_decl_dat(nodes, 2, "double", x, "/group3/group2/group1/p_x_test");
  op_map pedge_test = op_decl_map(edges, nodes, 2, edge, "/group3/pedge_test");

  op_decl_const(1, "double", &gam);
  op_decl_const(1, "double", &gm1);
  op_decl_const(1, "double", &cfl);
  op_decl_const(1, "double", &eps);
  op_decl_const(1, "double", &mach);
  op_decl_const(1, "double", &alpha);
  op_decl_const(4, "double", qinf);

  /* Test functionality of fetching data of an op_dat to an HDF5 file*/
  op_fetch_data_hdf5_file(p_x_test, "test.h5");

  char name[128];
  int time = 0;
  sprintf(name, "states_%07i", (int)(time * 100.0));
  op_fetch_data_hdf5_file_path(p_x_test, "test.h5", name);
  sprintf(name, "/results/states_%07i", (int)(time * 100.0));
  op_fetch_data_hdf5_file_path(p_x_test, "test.h5", name);

  /* Test functionality of dumping all the sets,maps and dats to an HDF5 file*/
  op_dump_to_hdf5(file_out);
  op_write_const_hdf5("gam", 1, "double", (char *)&gam, "new_grid_out.h5");
  op_write_const_hdf5("gm1", 1, "double", (char *)&gm1, "new_grid_out.h5");
  op_write_const_hdf5("cfl", 1, "double", (char *)&cfl, "new_grid_out.h5");
  op_write_const_hdf5("eps", 1, "double", (char *)&eps, "new_grid_out.h5");
  op_write_const_hdf5("mach", 1, "double", (char *)&mach, "new_grid_out.h5");
  op_write_const_hdf5("alpha", 1, "double", (char *)&alpha, "new_grid_out.h5");
  op_write_const_hdf5("qinf", 4, "double", (char *)qinf, "new_grid_out.h5");

  op_exit();
}
