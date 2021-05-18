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
//     Nonlinear airfoil lift calculation
//
//     Written by Mike Giles, 2010-2011, based on FORTRAN code
//     by Devendra Ghate and Mike Giles, 2005
//
//     Extended to MPI by Gihan Mudalige March 2011

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

double gam, gm1, cfl, eps, mach, alpha, qinf[4];

//
// OP header file
//

#include "op_lib_mpi.h"
#include "op_seq.h"

//
// kernel routines for parallel loops
//

#include "adt_calc.h"
#include "bres_calc.h"
#include "res_calc.h"
#include "save_soln.h"
#include "update.h"

//
// user declared functions
//

static int compute_local_size(int global_size, int mpi_comm_size,
                              int mpi_rank) {
  int local_size = global_size / mpi_comm_size;
  int remainder = (int)fmod(global_size, mpi_comm_size);

  if (mpi_rank < remainder) {
    local_size = local_size + 1;
  }
  return local_size;
}

static void scatter_double_array(double *g_array, double *l_array,
                                 int comm_size, int g_size, int l_size,
                                 int elem_size) {
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

  MPI_Scatterv(g_array, sendcnts, displs, MPI_DOUBLE, l_array,
               l_size * elem_size, MPI_DOUBLE, MPI_ROOT, MPI_COMM_WORLD);

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
    op_printf("error reading from new_grid.dat\n");
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

  // timer
  double cpu_t1, cpu_t2, wall_t1, wall_t2;

  int *becell, *ecell, *bound, *bedge, *edge, *cell;
  double *x, *q, *qold, *adt, *res;

  int nnode, ncell, nedge, nbedge, niter;
  double rms;

  /**------------------------BEGIN I/O and PARTITIONING -------------------**/

  op_timers(&cpu_t1, &wall_t1);

  /* read in grid from disk on root processor */
  FILE *fp;

  if ((fp = fopen("new_grid.dat", "r")) == NULL) {
    op_printf("can't open file new_grid.dat\n");
    exit(-1);
  }

  int g_nnode, g_ncell, g_nedge, g_nbedge;

  check_scan(
      fscanf(fp, "%d %d %d %d \n", &g_nnode, &g_ncell, &g_nedge, &g_nbedge), 4);

  int *g_becell = 0, *g_ecell = 0, *g_bound = 0, *g_bedge = 0, *g_edge = 0,
      *g_cell = 0;
  double *g_x = 0, *g_q = 0, *g_qold = 0, *g_adt = 0, *g_res = 0;

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

    g_x = (double *)malloc(2 * g_nnode * sizeof(double));
    g_q = (double *)malloc(4 * g_ncell * sizeof(double));
    g_qold = (double *)malloc(4 * g_ncell * sizeof(double));
    g_res = (double *)malloc(4 * g_ncell * sizeof(double));
    g_adt = (double *)malloc(g_ncell * sizeof(double));

    for (int n = 0; n < g_nnode; n++) {
      check_scan(fscanf(fp, "%lf %lf \n", &g_x[2 * n], &g_x[2 * n + 1]), 2);
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

  x = (double *)malloc(2 * nnode * sizeof(double));
  q = (double *)malloc(4 * ncell * sizeof(double));
  qold = (double *)malloc(4 * ncell * sizeof(double));
  res = (double *)malloc(4 * ncell * sizeof(double));
  adt = (double *)malloc(ncell * sizeof(double));

  /* scatter sets, mappings and data on sets*/
  scatter_int_array(g_cell, cell, comm_size, g_ncell, ncell, 4);
  scatter_int_array(g_edge, edge, comm_size, g_nedge, nedge, 2);
  scatter_int_array(g_ecell, ecell, comm_size, g_nedge, nedge, 2);
  scatter_int_array(g_bedge, bedge, comm_size, g_nbedge, nbedge, 2);
  scatter_int_array(g_becell, becell, comm_size, g_nbedge, nbedge, 1);
  scatter_int_array(g_bound, bound, comm_size, g_nbedge, nbedge, 1);

  scatter_double_array(g_x, x, comm_size, g_nnode, nnode, 2);
  scatter_double_array(g_q, q, comm_size, g_ncell, ncell, 4);
  scatter_double_array(g_qold, qold, comm_size, g_ncell, ncell, 4);
  scatter_double_array(g_res, res, comm_size, g_ncell, ncell, 4);
  scatter_double_array(g_adt, adt, comm_size, g_ncell, ncell, 1);

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

  op_timers(&cpu_t2, &wall_t2);
  op_printf("Max total file read time = %f\n", wall_t2 - wall_t1);

  /**------------------------END I/O and PARTITIONING -----------------------**/

  // declare sets, pointers, datasets and global constants

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

  op_decl_const(1, "double", &gam);
  op_decl_const(1, "double", &gm1);
  op_decl_const(1, "double", &cfl);
  op_decl_const(1, "double", &eps);
  op_decl_const(1, "double", &mach);
  op_decl_const(1, "double", &alpha);
  op_decl_const(4, "double", qinf);

  op_diagnostic_output();

  // trigger partitioning and halo creation routines
  op_partition("PTSCOTCH", "KWAY", cells, pecell, p_x);
  // op_partition("PARMETIS", "KWAY", cells, pecell, p_x);

  // initialise timers for total execution wall time
  op_timers(&cpu_t1, &wall_t1);

  niter = 1000;
  for (int iter = 1; iter <= niter; iter++) {

    // save old flow solution
    op_par_loop(save_soln, "save_soln", cells,
                op_arg_dat(p_q, -1, OP_ID, 4, "double", OP_READ),
                op_arg_dat(p_qold, -1, OP_ID, 4, "double", OP_WRITE));

    //  predictor/corrector update loop

    for (int k = 0; k < 2; k++) {

      //    calculate area/timstep
      op_par_loop(adt_calc, "adt_calc", cells,
                  op_arg_dat(p_x, 0, pcell, 2, "double", OP_READ),
                  op_arg_dat(p_x, 1, pcell, 2, "double", OP_READ),
                  op_arg_dat(p_x, 2, pcell, 2, "double", OP_READ),
                  op_arg_dat(p_x, 3, pcell, 2, "double", OP_READ),
                  op_arg_dat(p_q, -1, OP_ID, 4, "double", OP_READ),
                  op_arg_dat(p_adt, -1, OP_ID, 1, "double", OP_WRITE));

      //    calculate flux residual
      op_par_loop(res_calc, "res_calc", edges,
                  op_arg_dat(p_x, 0, pedge, 2, "double", OP_READ),
                  op_arg_dat(p_x, 1, pedge, 2, "double", OP_READ),
                  op_arg_dat(p_q, 0, pecell, 4, "double", OP_READ),
                  op_arg_dat(p_q, 1, pecell, 4, "double", OP_READ),
                  op_arg_dat(p_adt, 0, pecell, 1, "double", OP_READ),
                  op_arg_dat(p_adt, 1, pecell, 1, "double", OP_READ),
                  op_arg_dat(p_res, 0, pecell, 4, "double", OP_INC),
                  op_arg_dat(p_res, 1, pecell, 4, "double", OP_INC));

      op_par_loop(bres_calc, "bres_calc", bedges,
                  op_arg_dat(p_x, 0, pbedge, 2, "double", OP_READ),
                  op_arg_dat(p_x, 1, pbedge, 2, "double", OP_READ),
                  op_arg_dat(p_q, 0, pbecell, 4, "double", OP_READ),
                  op_arg_dat(p_adt, 0, pbecell, 1, "double", OP_READ),
                  op_arg_dat(p_res, 0, pbecell, 4, "double", OP_INC),
                  op_arg_dat(p_bound, -1, OP_ID, 1, "int", OP_READ));

      //    update flow field

      rms = 0.0;

      op_par_loop(update, "update", cells,
                  op_arg_dat(p_qold, -1, OP_ID, 4, "double", OP_READ),
                  op_arg_dat(p_q, -1, OP_ID, 4, "double", OP_WRITE),
                  op_arg_dat(p_res, -1, OP_ID, 4, "double", OP_RW),
                  op_arg_dat(p_adt, -1, OP_ID, 1, "double", OP_READ),
                  op_arg_gbl(&rms, 1, "double", OP_INC));
    }

    // print iteration history
    rms = sqrt(rms / (double)g_ncell);
    if (iter % 100 == 0)
      op_printf(" %d  %10.5e \n", iter, rms);

    if (iter % 1000 == 0 &&
        g_ncell == 720000) { // defailt mesh -- for validation testing
      // op_printf(" %d  %3.16f \n",iter,rms);
      double diff = fabs((100.0 * (rms / 0.0001060114637578)) - 100.0);
      op_printf("\n\nTest problem with %d cells is within %3.15E %% of the "
                "expected solution\n",
                720000, diff);
      if (diff < 0.00001) {
        op_printf("This test is considered PASSED\n");
      } else {
        op_printf("This test is considered FAILED\n");
      }
    }
  }

  op_timers(&cpu_t2, &wall_t2);

  // output the result dat array to files
  op_print_dat_to_txtfile(p_q, "out_grid_mpi.dat"); // ASCI
  op_print_dat_to_binfile(p_q, "out_grid_mpi.bin"); // Binary

  // write given op_dat's indicated segment of data to a memory block in the
  // order it was originally
  // arranged (i.e. before partitioning and reordering)
  double *q_part = (double *)op_malloc(sizeof(double) * op_get_size(cells) * 4);
  op_fetch_data_idx(p_q, q_part, 0, op_get_size(cells) - 1);
  free(q_part);

  op_timing_output();
  op_printf("Max total runtime = %f\n", wall_t2 - wall_t1);

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
