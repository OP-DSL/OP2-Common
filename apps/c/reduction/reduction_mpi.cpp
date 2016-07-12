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

//
// OP header file
//

#include "op_lib_mpi.h"
#include "op_seq.h"

//
// kernel routines for parallel loops
//

#include "res_calc.h"
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

  op_set edges = op_decl_set(nedge, "edges");
  op_set cells = op_decl_set(ncell, "cells");

  op_map pecell = op_decl_map(edges, cells, 2, ecell, "pecell");
  op_dat p_res = op_decl_dat(cells, 4, "double", res, "p_res");

  int count;

  // trigger partitioning and halo creation routines
  op_partition("PTSCOTCH", "KWAY", cells, pecell, NULL);

  op_diagnostic_output();

  // initialise timers for total execution wall time
  op_timers(&cpu_t1, &wall_t1);

  // indirect reduction
  count = 0;
  op_par_loop(res_calc, "res_calc", edges,
              op_arg_dat(p_res, 0, pecell, 4, "double", OP_INC),
              op_arg_gbl(&count, 1, "int", OP_INC));
  op_printf("number of edges:: %d should be: %d \n", count, g_nedge);
  if (count != g_nedge)
    op_printf("indirect reduction FAILED\n");
  else
    op_printf("indirect reduction PASSED\n");
  // direct reduction
  count = 0;
  op_par_loop(update, "update", cells,
              op_arg_dat(p_res, -1, OP_ID, 4, "double", OP_RW),
              op_arg_gbl(&count, 1, "int", OP_INC));
  op_printf("number of cells: %d should be: %d \n", count, g_ncell);
  if (count != g_ncell)
    op_printf("direct reduction FAILED\n");
  else
    op_printf("direct reduction PASSED\n");

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
