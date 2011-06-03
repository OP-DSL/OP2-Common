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
// test program for new OPlus2 development
//

//
// standard headers
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// global constants

float alpha;

//
// OP header file
//

#include "op_seq.h"

//
// kernel routines for parallel loops
//

#include "res.h"
#include "update.h"


// define problem size

#define NN       6
#define NITER    2


// main program

int main(int argc, char **argv){

  int   nnode, nedge, n, e;
  float dx;

  nnode = (NN-1)*(NN-1);
  nedge = (NN-1)*(NN-1) + 4*(NN-1)*(NN-2);
  dx    = 1.0f / ((float) NN);

  int    *pp = (int *)malloc(sizeof(int)*2*nedge);

  float  *A  = (float *)malloc(sizeof(float)*nedge);
  float  *r  = (float *)malloc(sizeof(float)*nnode);
  float  *u  = (float *)malloc(sizeof(float)*nnode);
  float  *du = (float *)malloc(sizeof(float)*nnode);

  // create matrix and r.h.s., and set coordinates needed for renumbering / partitioning

  e = 0;

  for (int i=1; i<NN; i++) {
    for (int j=1; j<NN; j++) {
      n         = i-1 + (j-1)*(NN-1);
      r[n]      = 0.0f;
      u[n]      = 0.0f;
      du[n]     = 0.0f;

      pp[2*e]   = n;
      pp[2*e+1] = n;
      A[e]      = -1.0f;
      e++;

      for (int pass=0; pass<4; pass++) {
        int i2 = i;
        int j2 = j;
        if (pass==0) i2 += -1;
        if (pass==1) i2 +=  1;
        if (pass==2) j2 += -1;
        if (pass==3) j2 +=  1;

        if ( (i2==0) || (i2==NN) || (j2==0) || (j2==NN) ) {
          r[n] += 0.25f;
	}
        else {
          pp[2*e]   = n;
          pp[2*e+1] = i2-1 + (j2-1)*(NN-1);
          A[e]      = 0.25f;
          e++;
        }
      }
    }
  }

  // OP initialisation

  op_init(argc,argv,5);

  // declare sets, pointers, and datasets

  op_set nodes = op_decl_set(nnode, "nodes");
  op_set edges = op_decl_set(nedge, "edges");

  op_map ppedge = op_decl_map(edges,nodes,2,pp, "ppedge");

  op_dat p_A  = op_decl_dat(edges,1,"float",A,  "p_A" );
  op_dat p_r  = op_decl_dat(nodes,1,"float",r,  "p_r" );
  op_dat p_u  = op_decl_dat(nodes,1,"float",u,  "p_u" );
  op_dat p_du = op_decl_dat(nodes,1,"float",du, "p_du");

  alpha = 1.0f;
  op_decl_const(1,"float",&alpha);

  op_diagnostic_output();

  // main iteration loop

  float u_sum, u_max, beta = 1.0f;

  for (int iter=0; iter<NITER; iter++) {
    op_par_loop(res,"res", edges,
                op_arg_dat(p_A, -1,OP_ID,  1,"float",OP_READ),
                op_arg_dat(p_u,  1,ppedge, 1,"float",OP_READ),
                op_arg_dat(p_du, 0,ppedge, 1,"float",OP_INC ),
                op_arg_gbl(&beta,1,"float",OP_READ));

    u_sum = 0.0f;
    u_max = 0.0f;
    op_par_loop(update,"update", nodes,
                op_arg_dat(p_r,  -1,OP_ID, 1,"float",OP_READ),
                op_arg_dat(p_du, -1,OP_ID, 1,"float",OP_RW  ),
                op_arg_dat(p_u,  -1,OP_ID, 1,"float",OP_INC ),
                op_arg_gbl(&u_sum,1,"float",OP_INC),
                op_arg_gbl(&u_max,1,"float",OP_MAX));
    printf("\n u max/rms = %f %f \n\n",u_max, sqrt(u_sum/nnode));
  }

  // print out results

  printf("\n  Results after %d iterations:\n\n",NITER);

  op_fetch_data(p_u);
  /*
  op_fetch_data(p_du);
  op_fetch_data(p_r);
  */

  for (int pass=0; pass<1; pass++) {
    /*
    if(pass==0)      printf("\narray u\n");
    else if(pass==1) printf("\narray du\n");
    else if(pass==2) printf("\narray r\n");
    */

    for (int j=NN-1; j>0; j--) {
      for (int i=1; i<NN; i++) {
        if (pass==0)
	  printf(" %7.4f",u[i-1 + (j-1)*(NN-1)]);
        else if (pass==1)
          printf(" %7.4f",du[i-1 + (j-1)*(NN-1)]);
        else if (pass==2)
          printf(" %7.4f",r[i-1 + (j-1)*(NN-1)]);
      }
      printf("\n");
    }
    printf("\n");
  }

  op_timing_output();

  op_exit();

  // free allocated arrays

  free(pp);
  free(A);
  free(r);
  free(u);
  free(du);
}
