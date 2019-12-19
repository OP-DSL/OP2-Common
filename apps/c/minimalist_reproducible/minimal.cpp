
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

double alpha;

// jac header file

//
// OP header file
//

#include "op_seq.h"

//
// kernel routines for parallel loops
//
#include "increment_log.h"

// Error tolerance in checking correctness

#define TOLERANCE 1e-12

// define problem size

#define NN 6
#define NITER 2




int main(int argc, char **argv) {
  // OP initialisation
  op_init(argc, argv, 5);

  // timer
  double cpu_t1, cpu_t2, wall_t1, wall_t2;

  int nnode, nedge, n, e;

  nnode = (NN - 1) * (NN - 1);
  nedge = (NN - 1) * (NN - 1) + 4 * (NN - 1) * (NN - 2);

  int *pp = (int *)malloc(sizeof(int) * 2 * nedge);

  double *A = (double *)malloc(sizeof(double) * nedge);
  double *r = (double *)malloc(sizeof(double) * nnode);

  // create matrix and r.h.s., and set coordinates needed for renumbering /
  // partitioning

  e = 0;

  for (int i = 1; i < NN; i++) {
    for (int j = 1; j < NN; j++) {
      n = i - 1 + (j - 1) * (NN - 1);
      pp[2 * e] = n;
      pp[2 * e + 1] = n;
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
          ;
        } else {
          pp[2 * e] = n;
          pp[2 * e + 1] = i2 - 1 + (j2 - 1) * (NN - 1);
          e++;
        }
      }
    }
  }
  
  for (int i = 0; i < nnode; i++) {
    r[i]=3;
  }
  
  
  for (int i = 0; i < nedge; i++) {
    A[i]=i+2;
  }
  
  

  // declare sets, pointers, and datasets

  op_set nodes = op_decl_set(nnode, "nodes");
  op_set edges = op_decl_set(nedge, "edges");

  op_map ppedge = op_decl_map(edges, nodes, 2, pp, "ppedge");

  op_dat p_A = op_decl_dat(edges, 1, "double", A, "p_A");
  op_dat p_r = op_decl_dat(nodes, 1, "double", r, "p_r");

  alpha = 1.0f;
  op_decl_const(1, "double", &alpha);

  op_diagnostic_output();

  // initialise timers for total execution wall time
  op_timers(&cpu_t1, &wall_t1);

  // main iteration loop

  double u_sum, u_max, beta = 1.0f;
  
  op_partition("PARMETIS", "KWAY", edges, ppedge, p_r);
  create_reversed_mapping();
  
  op_par_loop(increment_log,"increment_log",edges,
        op_arg_dat(p_A,-1,OP_ID,1, "double", OP_READ),
        op_arg_dat(p_r,0,ppedge,1,"double",OP_RW),
        op_arg_dat(p_r,1,ppedge,1,"double",OP_RW));
  
  
  op_fetch_data_hdf5_file(p_r,"repr_kimenet.h5");
  
  op_exit();
  
  
  
}