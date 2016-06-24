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

//
// mpi header file - included by user for user level mpi
//

#include <mpi.h>

// global constants

float gam, gm1, cfl, eps, mach, alpha, qinf[4];

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

#include "adt_calc.h"
#include "bres_calc.h"
#include "res_calc.h"
#include "save_soln.h"
#include "update.h"

//
// op_par_loop declarations
//

#include "op_seq.h"

static void scatter_float_array(float *g_array, float *l_array, int comm_size,
                                int g_size, int l_size, int elem_size) {
  int *sendcnts = (int *)malloc(comm_size * sizeof(int));
  int *displs = (int *)malloc(comm_size * sizeof(int));
  int disp = 0;

  for (int i = 0; i < comm_size; i++) {
    sendcnts[i] = elem_size * compute_local_size(g_size, comm_size, i);
  }
  for (int i = 0; i < comm_size; i++) {
    displs[i] = disp;
    disp = disp + sendcnts[i];
  }

  MPI_Scatterv(g_array, sendcnts, displs, MPI_FLOAT, l_array,
               l_size * elem_size, MPI_FLOAT, MPI_ROOT, MPI_COMM_WORLD);

  free(sendcnts);
  free(displs);
}

static void scatter_int_array(int *g_array, int *l_array, int comm_size,
                              int g_size, int l_size, int elem_size) {
  int *sendcnts = (int *)malloc(comm_size * sizeof(int));
  int *displs = (int *)malloc(comm_size * sizeof(int));
  int disp = 0;

  for (int i = 0; i < comm_size; i++) {
    sendcnts[i] = elem_size * compute_local_size(g_size, comm_size, i);
  }
  for (int i = 0; i < comm_size; i++) {
    displs[i] = disp;
    disp = disp + sendcnts[i];
  }

  MPI_Scatterv(g_array, sendcnts, displs, MPI_INT, l_array, l_size * elem_size,
               MPI_INT, MPI_ROOT, MPI_COMM_WORLD);

  free(sendcnts);
  free(displs);
}

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

  // MPI for user I/O
  int my_rank;
  int comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  int *becell, *ecell, *bound, *bedge, *edge, *cell;
  float *x, *q, *qold, *adt, *res;

  int nnode, ncell, nedge, nbedge;

  /**------------------------BEGIN  I/O -------------------**/

  char file[] = "new_grid.dat";
  char file_out[] = "new_grid_out.h5";

  /* read in grid from disk on root processor */
  FILE *fp;

  if ((fp = fopen(file, "r")) == NULL) {
    op_printf("can't open file %s\n", file);
    exit(-1);
  }

  int g_nnode, g_ncell, g_nedge, g_nbedge;

  check_scan(
      fscanf(fp, "%d %d %d %d \n", &g_nnode, &g_ncell, &g_nedge, &g_nbedge), 4);

  int *g_becell = 0, *g_ecell = 0, *g_bound = 0, *g_bedge = 0, *g_edge = 0,
      *g_cell = 0;
  float *g_x = 0, *g_q = 0, *g_qold = 0, *g_adt = 0, *g_res = 0;

  // set constants

  op_printf("initialising flow field\n");
  gam = 1.4f;
  gm1 = gam - 1.0f;
  cfl = 0.9f;
  eps = 0.05f;

  float mach = 0.4f;
  float alpha = 3.0f * atan(1.0f) / 45.0f;
  float p = 1.0f;
  float r = 1.0f;
  float u = sqrt(gam * p / r) * mach;
  float e = p / (r * gm1) + 0.5f * u * u;

  qinf[0] = r;
  qinf[1] = r * u;
  qinf[2] = 0.0f;
  qinf[3] = r * e;

  op_printf("reading in grid \n");
  op_printf("Global number of nodes, cells, edges, bedges = %d, %d, %d, %d\n",
            g_nnode, g_ncell, g_nedge, g_nbedge);

  if (my_rank == MPI_ROOT) {
    g_cell = (int *)malloc(4 * g_ncell * sizeof(int));
    g_edge = (int *)malloc(2 * g_nedge * sizeof(int));
    g_ecell = (int *)malloc(2 * g_nedge * sizeof(int));
    g_bedge = (int *)malloc(2 * g_nbedge * sizeof(int));
    g_becell = (int *)malloc(g_nbedge * sizeof(int));
    g_bound = (int *)malloc(g_nbedge * sizeof(int));

    g_x = (float *)malloc(2 * g_nnode * sizeof(float));
    g_q = (float *)malloc(4 * g_ncell * sizeof(float));
    g_qold = (float *)malloc(4 * g_ncell * sizeof(float));
    g_res = (float *)malloc(4 * g_ncell * sizeof(float));
    g_adt = (float *)malloc(g_ncell * sizeof(float));

    for (int n = 0; n < g_nnode; n++) {
      check_scan(fscanf(fp, "%f %f \n", &g_x[2 * n], &g_x[2 * n + 1]), 2);
    }

    for (int n = 0; n < g_ncell; n++) {
      check_scan(fscanf(fp, "%d %d %d %d \n", &g_cell[4 * n],
                        &g_cell[4 * n + 1], &g_cell[4 * n + 2],
                        &g_cell[4 * n + 3]),
                 4);
    }

    for (int n = 0; n < g_nedge; n++) {
      check_scan(fscanf(fp, "%d %d %d %d \n", &g_edge[2 * n],
                        &g_edge[2 * n + 1], &g_ecell[2 * n],
                        &g_ecell[2 * n + 1]),
                 4);
    }

    for (int n = 0; n < g_nbedge; n++) {
      check_scan(fscanf(fp, "%d %d %d %d \n", &g_bedge[2 * n],
                        &g_bedge[2 * n + 1], &g_becell[n], &g_bound[n]),
                 4);
    }

    // initialise flow field and residual

    for (int n = 0; n < g_ncell; n++) {
      for (int m = 0; m < 4; m++) {
        g_q[4 * n + m] = qinf[m];
        g_res[4 * n + m] = 0.0f;
      }
    }
  }

  fclose(fp);

  nnode = compute_local_size(g_nnode, comm_size, my_rank);
  ncell = compute_local_size(g_ncell, comm_size, my_rank);
  nedge = compute_local_size(g_nedge, comm_size, my_rank);
  nbedge = compute_local_size(g_nbedge, comm_size, my_rank);

  op_printf(
      "Number of nodes, cells, edges, bedges on process %d = %d, %d, %d, %d\n",
      my_rank, nnode, ncell, nedge, nbedge);

  /*Allocate memory to hold local sets, mapping tables and data*/
  cell = (int *)malloc(4 * ncell * sizeof(int));
  edge = (int *)malloc(2 * nedge * sizeof(int));
  ecell = (int *)malloc(2 * nedge * sizeof(int));
  bedge = (int *)malloc(2 * nbedge * sizeof(int));
  becell = (int *)malloc(nbedge * sizeof(int));
  bound = (int *)malloc(nbedge * sizeof(int));

  x = (float *)malloc(2 * nnode * sizeof(float));
  q = (float *)malloc(4 * ncell * sizeof(float));
  qold = (float *)malloc(4 * ncell * sizeof(float));
  res = (float *)malloc(4 * ncell * sizeof(float));
  adt = (float *)malloc(ncell * sizeof(float));

  /* scatter sets, mappings and data on sets*/
  scatter_int_array(g_cell, cell, comm_size, g_ncell, ncell, 4);
  scatter_int_array(g_edge, edge, comm_size, g_nedge, nedge, 2);
  scatter_int_array(g_ecell, ecell, comm_size, g_nedge, nedge, 2);
  scatter_int_array(g_bedge, bedge, comm_size, g_nbedge, nbedge, 2);
  scatter_int_array(g_becell, becell, comm_size, g_nbedge, nbedge, 1);
  scatter_int_array(g_bound, bound, comm_size, g_nbedge, nbedge, 1);

  scatter_float_array(g_x, x, comm_size, g_nnode, nnode, 2);
  scatter_float_array(g_q, q, comm_size, g_ncell, ncell, 4);
  scatter_float_array(g_qold, qold, comm_size, g_ncell, ncell, 4);
  scatter_float_array(g_res, res, comm_size, g_ncell, ncell, 4);
  scatter_float_array(g_adt, adt, comm_size, g_ncell, ncell, 1);

  /*Freeing memory allocated to gloabal arrays on rank 0
    after scattering to all processes*/
  if (my_rank == MPI_ROOT) {
    free(g_cell);
    free(g_edge);
    free(g_ecell);
    free(g_bedge);
    free(g_becell);
    free(g_bound);
    free(g_x);
    free(g_q);
    free(g_qold);
    free(g_adt);
    free(g_res);
  }

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
  op_dat p_x = op_decl_dat(nodes, 2, "float", x, "p_x");
  op_dat p_q = op_decl_dat(cells, 4, "float", q, "p_q");
  op_dat p_qold = op_decl_dat(cells, 4, "float", qold, "p_qold");
  op_dat p_adt = op_decl_dat(cells, 1, "float", adt, "p_adt");
  op_dat p_res = op_decl_dat(cells, 4, "float", res, "p_res");

  op_decl_const(1, "float", &gam);
  op_decl_const(1, "float", &gm1);
  op_decl_const(1, "float", &cfl);
  op_decl_const(1, "float", &eps);
  op_decl_const(1, "float", &mach);
  op_decl_const(1, "float", &alpha);
  op_decl_const(4, "float", qinf);

  op_dump_to_hdf5(file_out);
  op_write_const_hdf5("gam", 1, "float", (char *)&gam, "new_grid_out.h5");
  op_write_const_hdf5("gm1", 1, "float", (char *)&gm1, "new_grid_out.h5");
  op_write_const_hdf5("cfl", 1, "float", (char *)&cfl, "new_grid_out.h5");
  op_write_const_hdf5("eps", 1, "float", (char *)&eps, "new_grid_out.h5");
  op_write_const_hdf5("mach", 1, "float", (char *)&mach, "new_grid_out.h5");
  op_write_const_hdf5("alpha", 1, "float", (char *)&alpha, "new_grid_out.h5");
  op_write_const_hdf5("qinf", 4, "float", (char *)qinf, "new_grid_out.h5");

  // create halos - for sanity check
  op_halo_create();

  op_exit();
}
