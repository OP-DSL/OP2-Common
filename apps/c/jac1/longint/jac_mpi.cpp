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
#include <limits>
//
// mpi header file - included by user for user level mpi
//

#include <mpi.h>

// global constants

double alpha;

//
// OP header file
//

#include "op_lib_mpi.h"
#include "op_seq.h"

//
// kernel routines for parallel loops
//
#include "user_types.h"
#include "res.h"
#include "update.h"

// Error tolerance in checking correctness

#define TOLERANCE 1e-12



// This function performs a distributed validation where each rank checks its own local portion
template <class T> int distributed_check_result(T *local_u, idx_g_t nn, idx_g_t node_start, idx_g_t nnode, T tol) {
  // Check results against reference u solution
  // Each rank only validates its local portion
  int local_failed = 0;
  int global_failed = 0;
  int my_rank;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  
  for (idx_g_t local_idx = 0; local_idx < nnode; local_idx++) {
    // Calculate global (i,j) coordinates from the local index
    idx_g_t global_idx = node_start + local_idx;
    idx_g_t j = global_idx / (nn - 1) + 1;
    idx_g_t i = global_idx % (nn - 1) + 1;
    
    // Apply the same validation logic based on the global coordinates
    T expected_value;
    
    if (((i == 1) && (j == 1)) || ((i == 1) && (j == (nn - 1))) ||
        ((i == (nn - 1)) && (j == 1)) || ((i == (nn - 1)) && (j == (nn - 1)))) {
      // Corners of domain
      expected_value = (T)0.625;
    } else if ((i == 1 && j == 2) || (i == 2 && j == 1) ||
               (i == 1 && j == (nn - 2)) || (i == 2 && j == (nn - 1)) ||
               (i == (nn - 2) && j == 1) || (i == (nn - 1) && j == 2) ||
               (i == (nn - 2) && j == (nn - 1)) || ((i == (nn - 1)) && (j == (nn - 2)))) {
      // Horizontally or vertically-adjacent to a corner
      expected_value = (T)0.4375;
    } else if ((i == 2 && j == 2) || (i == 2 && j == (nn - 2)) ||
               (i == (nn - 2) && j == 2) || ((i == (nn - 2)) && (j == (nn - 2)))) {
      // Diagonally adjacent to a corner
      expected_value = (T)0.125;
    } else if ((i == 1) || (j == 1) || (i == (nn - 1)) || (j == (nn - 1))) {
      // On some other edge node
      expected_value = (T)0.3750;
    } else if ((i == 2) || (j == 2) || (i == (nn - 2)) || (j == (nn - 2))) {
      // On some other node that is 1 node from the edge
      expected_value = (T)0.0625;
    } else {
      // 2 or more nodes from the edge
      expected_value = (T)0.0;
    }
    
    // Check if the value matches the expected value
    if (fabs(local_u[local_idx] - expected_value) > tol) {
      op_printf("Failure on rank %d: i=%ld, j=%ld, expected: %f, actual: %f\n", 
                my_rank, i, j, (double)expected_value, (double)local_u[local_idx]);
      local_failed = 1;
      break;  // Exit after the first failure to avoid flooding output
    }
  }
  
  // Use MPI_Allreduce to combine validation results from all ranks
  MPI_Allreduce(&local_failed, &global_failed, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
  
  if (my_rank == MPI_ROOT) {
    if (!global_failed)
      op_printf("\nDistributed results check PASSED on all ranks!\n");
    else
      op_printf("\nDistributed results check FAILED on at least one rank!\n");
  }
  
  return global_failed;
}

//
// user declared functions
//

static idx_g_t compute_local_size(idx_g_t global_size, idx_g_t mpi_comm_size,
                              idx_g_t mpi_rank) {
  idx_g_t local_size = global_size / mpi_comm_size;
  idx_g_t remainder = global_size % mpi_comm_size;

  if (mpi_rank < remainder) {
    local_size = local_size + 1;
  }
  return local_size;
}

static idx_g_t compute_local_offset(idx_g_t global_size, idx_g_t mpi_comm_size,
                                    idx_g_t mpi_rank) {
  idx_g_t base = global_size / mpi_comm_size;
  idx_g_t remainder = global_size % mpi_comm_size;

  if (mpi_rank < remainder) {
    return mpi_rank * (base + 1);
  } else {
    return remainder * (base + 1) + (mpi_rank - remainder) * base;
  }
}


// define problem size

const idx_g_t NN = ((size_t)1)<<15;
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

  idx_g_t *pp;
  double *A, *r, *u, *du;

  idx_g_t nnode, nedge;

  /**------------------------BEGIN I/O and PARTITIONING ---------------------**/

  idx_g_t g_nnode, g_nedge, g_n, g_e;


  /**--------------------- BEGIN DISTRIBUTED INITIALIZATION ---------------------**/

  /* Global mesh sizes */
  g_nnode = (NN - 1) * (NN - 1);
  // Note: The global number of edges is given by 
  //      g_nedge = (NN - 1)^2 + 4*(NN - 1)*(NN - 2)
  // but here we do not need the global arrays at all

  op_printf("Global number of nodes = %ld\n", g_nnode);

  /* Compute local node partition.
     Assume compute_local_size returns the number of nodes for my_rank
     and compute_local_offset returns the starting global node index. */
  nnode = compute_local_size(g_nnode, comm_size, my_rank);
  idx_g_t node_start = compute_local_offset(g_nnode, comm_size, my_rank);

  /* First, count the number of edges that will be created on this process.
     (Each node always gets a self edge plus one edge per interior neighbor.) */
  idx_g_t nedge_local = 0;
  for (idx_g_t local = 0; local < nnode; local++) {
    idx_g_t global = node_start + local;
    // Recover (i,j) from the global node index
    // global = (i - 1) + (j - 1) * (NN - 1) with i,j in [1, NN-1]
    idx_g_t j = global / (NN - 1) + 1;
    idx_g_t i = global % (NN - 1) + 1;

    nedge_local++;  // self edge always exists

    for (int pass = 0; pass < 4; pass++) {
      idx_g_t i2 = i, j2 = j;
      if (pass == 0)
        i2=i2-1;
      else if (pass == 1)
        i2++;
      else if (pass == 2)
        j2=j2-1;
      else if (pass == 3)
        j2++;

      /* If the neighbor is on the boundary, update the right‐hand side,
         otherwise an edge will be created */
      if ((i2 == 0) || (i2 == NN) || (j2 == 0) || (j2 == NN)) {
        ;  // no extra edge is added
      } else {
        nedge_local++;
      }
    }
  }
  nedge = nedge_local;

  op_printf("Process %d: number of nodes, edges = %ld, %ld\n", my_rank, nnode, nedge);

  /* Allocate local arrays */
  pp = (idx_g_t *)malloc(2 * nedge * sizeof(idx_g_t));
  A  = (double *)malloc(nedge * sizeof(double));
  r  = (double *)malloc(nnode * sizeof(double));
  u  = (double *)malloc(nnode * sizeof(double));
  du = (double *)malloc(nnode * sizeof(double));

  /* Now fill in the local arrays.
Note: We are still using global indices in pp so that the assembled matrix
remains consistent. */
  idx_g_t edge_counter = 0;
  for (idx_g_t local = 0; local < nnode; local++) {
    idx_g_t global = node_start + local;
    idx_g_t j = global / (NN - 1) + 1;
    idx_g_t i = global % (NN - 1) + 1;

    /* Initialize local vectors */
    r[local]  = 0.0;
    u[local]  = 0.0;
    du[local] = 0.0;

    /* Self edge */
    pp[2 * edge_counter]     = global;
    pp[2 * edge_counter + 1] = global;
    A[edge_counter]          = -1.0;
    edge_counter++;

    /* Loop over the 4 neighbor directions */
    for (int pass = 0; pass < 4; pass++) {
      idx_g_t i2 = i, j2 = j;
      if (pass == 0)
        i2=i2-1;
      else if (pass == 1)
        i2++;
      else if (pass == 2)
        j2=j2-1;
      else if (pass == 3)
        j2++;

      if ((i2 == 0) || (i2 == NN) || (j2 == 0) || (j2 == NN)) {
        /* If the neighbor is on the boundary, update the right-hand side */
        r[local] += 0.25;
      } else {
        /* Otherwise, add an edge from global to the neighbor */
        idx_g_t neighbor = (i2 - 1) + (j2 - 1) * (NN - 1);
        pp[2 * edge_counter]     = global;
        pp[2 * edge_counter + 1] = neighbor;
        A[edge_counter]          = 0.25;
        edge_counter++;
      }
    }
  }
  /**------------------------END I/O and PARTITIONING ---------------------**/

  // declare sets, pointers, and datasets

  op_set nodes = op_decl_set(nnode, "nodes");
  op_set edges = op_decl_set(nedge, "edges");

  op_map ppedge = op_decl_map_long(edges, nodes, 2, pp, "ppedge");

  op_dat p_A = op_decl_dat(edges, 1, "double", A, "p_A");
  op_dat p_r = op_decl_dat(nodes, 1, "double", r, "p_r");
  op_dat p_u = op_decl_dat(nodes, 1, "double", u, "p_u");
  op_dat p_du = op_decl_dat(nodes, 1, "double", du, "p_du");

  alpha = 1.0f;
  op_decl_const(1, "double", &alpha);

  // op_diagnostic_output();

  // trigger partitioning and halo creation routines
  op_partition("PARMETIS", "KWAY", edges, ppedge, NULL);

  // initialise timers for total execution wall time
  op_timers(&cpu_t1, &wall_t1);

  // main iteration loop

  double u_sum, u_max, beta = 1.0f;

  for (int iter = 0; iter < NITER; iter++) {
    op_par_loop(res, "res", edges,
        op_arg_dat(p_A, -1, OP_ID, 1, "double", OP_READ),
        op_arg_dat(p_u, 1, ppedge, 1, "double", OP_READ),
        op_arg_dat(p_du, 0, ppedge, 1, "double", OP_INC),
        op_arg_gbl(&beta, 1, "double", OP_READ),
        op_arg_idx(-1, OP_ID),
        op_arg_idx(0, ppedge),
        op_arg_idx(1, ppedge));


    u_sum = 0.0f;
    u_max = 0.0f;
    op_par_loop(update, "update", nodes,
        op_arg_dat(p_r, -1, OP_ID, 1, "double", OP_READ),
        op_arg_dat(p_du, -1, OP_ID, 1, "double", OP_RW),
        op_arg_dat(p_u, -1, OP_ID, 1, "double", OP_INC),
        op_arg_idx(-1, OP_ID),
        op_arg_gbl(&u_sum, 1, "double", OP_INC),
        op_arg_gbl(&u_max, 1, "double", OP_MAX));

    op_printf("\n u max/rms = %f %f \n\n", u_max, sqrt(u_sum / (double)(size_t)g_nnode));
  }

  op_timers(&cpu_t2, &wall_t2);

  // output the result dat array to files
  // op_print_dat_to_txtfile(p_u, "out_grid_mpi.dat"); // ASCI
  // op_print_dat_to_binfile(p_u, "out_grid_mpi.bin"); // Binary

  // printf("solution on rank %d\n", my_rank);
  // for (idx_g_t i = 0; i < nnode; i++) {
  //   printf(" %7.4f", u[i]);
  //   fflush(stdout);
  // }
  // printf("\n");

  // print each mpi process's timing info for each kernel
  op_timing_output();

  // print total time for niter interations
  op_printf("Max total runtime = %f\n", wall_t2 - wall_t1);

  // fetch local results
  op_fetch_data(p_u, u);
  
  // Perform distributed validation using local u array directly
  int result = distributed_check_result<double>(u, NN, node_start, nnode, TOLERANCE);
  
  MPI_Barrier(MPI_COMM_WORLD);
  op_exit();

  free(u);
  // free(pp);
  free(A);
  free(r);
  free(du);

  return result;
}
