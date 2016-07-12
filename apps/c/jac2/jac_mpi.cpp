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

//
// mpi header file - included by user for user level mpi
//

#include <mpi.h>

// global constants

float alpha;

// jac header file

#include "check_result.h"

//
// OP header file
//

#include "op_lib_mpi.h"
#include "op_seq.h"

//
// kernel routines for parallel loops
//

#include "res.h"
#include "update.h"

// Error tolerance in checking correctness

#define TOLERANCE 1e-12

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

void scatter_double_array(double *g_array, double *l_array, int comm_size,
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

// define problem size

#define NN 6
#define NITER 2

// main program

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

  int *pp;
  float *r, *u, *du;
  double *A;

  int nnode, nedge;

  /**------------------------BEGIN I/O and PARTITIONING ---------------------**/

  int g_nnode, g_nedge, g_n, g_e;

  g_nnode = (NN - 1) * (NN - 1);
  g_nedge = (NN - 1) * (NN - 1) + 4 * (NN - 1) * (NN - 2);

  int *g_pp = 0;
  float *g_r = 0, *g_u = 0, *g_du = 0;
  double *g_A = 0;

  op_printf("Global number of nodes, edges = %d, %d\n", g_nnode, g_nedge);

  if (my_rank == MPI_ROOT) {
    g_pp = (int *)malloc(sizeof(int) * 2 * g_nedge);

    g_A = (double *)malloc(sizeof(double) * g_nedge);
    g_r = (float *)malloc(sizeof(float) * g_nnode);
    g_u = (float *)malloc(sizeof(float) * g_nnode);
    g_du = (float *)malloc(sizeof(float) * g_nnode);

    // create matrix and r.h.s., and set coordinates needed for renumbering /
    // partitioning

    g_e = 0;

    for (int i = 1; i < NN; i++) {
      for (int j = 1; j < NN; j++) {
        g_n = i - 1 + (j - 1) * (NN - 1);
        g_r[g_n] = 0.0f;
        g_u[g_n] = 0.0f;
        g_du[g_n] = 0.0f;

        g_pp[2 * g_e] = g_n;
        g_pp[2 * g_e + 1] = g_n;
        g_A[g_e] = -1.0f;
        g_e++;

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
            g_r[g_n] += 0.25f;
          } else {
            g_pp[2 * g_e] = g_n;
            g_pp[2 * g_e + 1] = i2 - 1 + (j2 - 1) * (NN - 1);
            g_A[g_e] = 0.25f;
            g_e++;
          }
        }
      }
    }
  }

  /* Compute local sizes */
  nnode = compute_local_size(g_nnode, comm_size, my_rank);
  nedge = compute_local_size(g_nedge, comm_size, my_rank);
  op_printf("Number of nodes, edges on process %d = %d, %d\n", my_rank, nnode,
            nedge);

  /*Allocate memory to hold local sets, mapping tables and data*/
  pp = (int *)malloc(2 * sizeof(int) * nedge);

  A = (double *)malloc(nedge * sizeof(double));
  r = (float *)malloc(nnode * sizeof(float));
  u = (float *)malloc(nnode * sizeof(float));
  du = (float *)malloc(nnode * sizeof(float));

  /* scatter sets, mappings and data on sets*/
  scatter_int_array(g_pp, pp, comm_size, g_nedge, nedge, 2);
  scatter_double_array(g_A, A, comm_size, g_nedge, nedge, 1);
  scatter_float_array(g_r, r, comm_size, g_nnode, nnode, 1);
  scatter_float_array(g_u, u, comm_size, g_nnode, nnode, 1);
  scatter_float_array(g_du, du, comm_size, g_nnode, nnode, 1);

  /*Freeing memory allocated to gloabal arrays on rank 0
    after scattering to all processes*/
  if (my_rank == MPI_ROOT) {
    free(g_pp);
    free(g_A);
    free(g_r);
    free(g_u);
    free(g_du);
  }

  /**------------------------END I/O and PARTITIONING ---------------------**/

  // declare sets, pointers, and datasets

  op_set nodes = op_decl_set(nnode, "nodes");
  op_set edges = op_decl_set(nedge, "edges");

  op_map ppedge = op_decl_map(edges, nodes, 2, pp, "ppedge");

  op_dat p_A = op_decl_dat(edges, 1, "double", A, "p_A");
  op_dat p_r = op_decl_dat(nodes, 1, "float", r, "p_r");
  op_dat p_u = op_decl_dat(nodes, 1, "float", u, "p_u");
  op_dat p_du = op_decl_dat(nodes, 1, "float", du, "p_du");

  alpha = 2.0f;
  op_decl_const(1, "float", &alpha);
  alpha = 1.0f;
  op_decl_const(1, "float", &alpha);

  op_diagnostic_output();

  // trigger partitioning and halo creation routines
  op_partition("PTSCOTCH", "KWAY", NULL, NULL, NULL);

  // initialise timers for total execution wall time
  op_timers(&cpu_t1, &wall_t1);

  // main iteration loop

  float u_sum, u_max, beta = 1.0f;

  for (int iter = 0; iter < NITER; iter++) {
    op_par_loop(res, "res", edges,
                op_arg_dat(p_A, -1, OP_ID, 1, "double", OP_READ),
                op_arg_dat(p_u, 1, ppedge, 1, "float", OP_READ),
                op_arg_dat(p_du, 0, ppedge, 1, "float", OP_INC),
                op_arg_gbl(&beta, 1, "float", OP_READ));

    u_sum = 0.0f;
    u_max = 0.0f;
    op_par_loop(update, "update", nodes,
                op_arg_dat(p_r, -1, OP_ID, 1, "float", OP_READ),
                op_arg_dat(p_du, -1, OP_ID, 1, "float", OP_RW),
                op_arg_dat(p_u, -1, OP_ID, 1, "float", OP_INC),
                op_arg_gbl(&u_sum, 1, "float", OP_INC),
                op_arg_gbl(&u_max, 1, "float", OP_MAX));

    op_printf("\n u max/rms = %f %f \n\n", u_max, sqrt(u_sum / g_nnode));
  }

  op_timers(&cpu_t2, &wall_t2);

  // get results data array
  op_fetch_data(p_u, u);

  // output the result dat array to files
  op_print_dat_to_txtfile(p_u, "out_grid_mpi.dat"); // ASCI
  op_print_dat_to_binfile(p_u, "out_grid_mpi.bin"); // Binary

  printf("solution on rank %d\n", my_rank);
  for (int i = 0; i < nnode; i++) {
    printf(" %7.4f", u[i]);
    fflush(stdout);
  }
  printf("\n");

  // print each mpi process's timing info for each kernel
  op_timing_output();

  // print total time for niter interations
  op_printf("Max total runtime = %f\n", wall_t2 - wall_t1);

  // gather results from all ranks and check
  float *ug = (float *)malloc(sizeof(float) * op_get_size(nodes));
  op_fetch_data_idx(p_u, ug, 0, op_get_size(nodes) - 1);
  int result = check_result<float>(ug, NN, TOLERANCE);
  free(ug);

  op_exit();

  free(u);
  free(pp);
  free(A);
  free(r);
  free(du);

  return result;
}
