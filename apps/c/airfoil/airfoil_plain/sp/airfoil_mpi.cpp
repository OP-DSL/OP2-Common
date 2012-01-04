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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

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

//
// kernel routines for parallel loops
//

#include "save_soln.h"
#include "adt_calc.h"
#include "res_calc.h"
#include "bres_calc.h"
#include "update.h"

//
// op_par_loop declarations
//

#include "op_seq.h"

//
//user declared functions
//

static int compute_local_size (int global_size, int mpi_comm_size, int mpi_rank )
{
  int local_size = global_size/mpi_comm_size;
  int remainder = (int)fmod(global_size,mpi_comm_size);

  if (mpi_rank < remainder)
  {
    local_size = local_size + 1;
  }
  return local_size;
}

static void scatter_float_array(float* g_array, float* l_array, int comm_size, int g_size,
                                 int l_size, int elem_size)
{
  int* sendcnts = (int *) malloc(comm_size*sizeof(int));
  int* displs = (int *) malloc(comm_size*sizeof(int));
  int disp = 0;

  for(int i = 0; i<comm_size; i++)
  {
    sendcnts[i] =   elem_size*compute_local_size (g_size, comm_size, i);
  }
  for(int i = 0; i<comm_size; i++)
  {
    displs[i] =   disp;
    disp = disp + sendcnts[i];
  }

  MPI_Scatterv(g_array, sendcnts, displs, MPI_FLOAT, l_array,
      l_size*elem_size, MPI_FLOAT, MPI_ROOT,  MPI_COMM_WORLD );

  free(sendcnts);
  free(displs);
}

static void scatter_int_array(int* g_array, int* l_array, int comm_size, int g_size,
                              int l_size, int elem_size)
{
  int* sendcnts = (int *) malloc(comm_size*sizeof(int));
  int* displs = (int *) malloc(comm_size*sizeof(int));
  int disp = 0;

  for(int i = 0; i<comm_size; i++)
  {
    sendcnts[i] =   elem_size*compute_local_size (g_size, comm_size, i);
  }
  for(int i = 0; i<comm_size; i++)
  {
    displs[i] =   disp;
    disp = disp + sendcnts[i];
  }

  MPI_Scatterv(g_array, sendcnts, displs, MPI_INT, l_array,
      l_size*elem_size, MPI_INT, MPI_ROOT,  MPI_COMM_WORLD );

  free(sendcnts);
  free(displs);
}

static void check_scan(int items_received, int items_expected)
{
  if(items_received != items_expected) {
    op_printf("error reading from new_grid.dat\n");
    exit(-1);
  }
}

//
// main program
//

int main(int argc, char **argv)
{
  // OP initialisation
  op_init(argc,argv,2);

  //MPI for user I/O
  int my_rank;
  int comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  //timer
  double cpu_t1, cpu_t2, wall_t1, wall_t2;

  int    *becell, *ecell,  *bound, *bedge, *edge, *cell;
  float  *x, *q, *qold, *adt, *res;

  int    nnode,ncell,nedge,nbedge,niter;
  float  rms;

  /**------------------------BEGIN I/O and PARTITIONING -------------------**/

  op_timers(&cpu_t1, &wall_t1);

  /* read in grid from disk on root processor */
  FILE *fp;

  if ( (fp = fopen("new_grid.dat","r")) == NULL) {
    op_printf("can't open file new_grid.dat\n"); exit(-1);
  }

  int   g_nnode,g_ncell,g_nedge,g_nbedge;

  check_scan(fscanf(fp,"%d %d %d %d \n",&g_nnode, &g_ncell, &g_nedge, &g_nbedge), 4);

  int *g_becell = 0, *g_ecell = 0, *g_bound = 0, *g_bedge = 0, *g_edge = 0, *g_cell = 0;
  float *g_x = 0,*g_q = 0, *g_qold = 0, *g_adt = 0, *g_res = 0;

  // set constants

  op_printf("initialising flow field\n");
  gam = 1.4f;
  gm1 = gam - 1.0f;
  cfl = 0.9f;
  eps = 0.05f;

  float mach  = 0.4f;
  float alpha = 3.0f*atan(1.0f)/45.0f;
  float p     = 1.0f;
  float r     = 1.0f;
  float u     = sqrt(gam*p/r)*mach;
  float e     = p/(r*gm1) + 0.5f*u*u;

  qinf[0] = r;
  qinf[1] = r*u;
  qinf[2] = 0.0f;
  qinf[3] = r*e;

  op_printf("reading in grid \n");
  op_printf("Global number of nodes, cells, edges, bedges = %d, %d, %d, %d\n"
      ,g_nnode,g_ncell,g_nedge,g_nbedge);

  if(my_rank == MPI_ROOT) {
    g_cell   = (int *) malloc(4*g_ncell*sizeof(int));
    g_edge   = (int *) malloc(2*g_nedge*sizeof(int));
    g_ecell  = (int *) malloc(2*g_nedge*sizeof(int));
    g_bedge  = (int *) malloc(2*g_nbedge*sizeof(int));
    g_becell = (int *) malloc(  g_nbedge*sizeof(int));
    g_bound  = (int *) malloc(  g_nbedge*sizeof(int));

    g_x      = (float *) malloc(2*g_nnode*sizeof(float));
    g_q      = (float *) malloc(4*g_ncell*sizeof(float));
    g_qold   = (float *) malloc(4*g_ncell*sizeof(float));
    g_res    = (float *) malloc(4*g_ncell*sizeof(float));
    g_adt    = (float *) malloc(  g_ncell*sizeof(float));

    for (int n=0; n<g_nnode; n++){
      check_scan(fscanf(fp,"%f %f \n",&g_x[2*n], &g_x[2*n+1]), 2);
    }

    for (int n=0; n<g_ncell; n++) {
      check_scan(fscanf(fp,"%d %d %d %d \n",&g_cell[4*n  ], &g_cell[4*n+1],
            &g_cell[4*n+2], &g_cell[4*n+3]), 4);
    }

    for (int n=0; n<g_nedge; n++) {
      check_scan(fscanf(fp,"%d %d %d %d \n",&g_edge[2*n],&g_edge[2*n+1],
            &g_ecell[2*n],&g_ecell[2*n+1]), 4);
    }

    for (int n=0; n<g_nbedge; n++) {
      check_scan(fscanf(fp,"%d %d %d %d \n",&g_bedge[2*n],&g_bedge[2*n+1],
            &g_becell[n],&g_bound[n]), 4);
    }

    //initialise flow field and residual

    for (int n=0; n<g_ncell; n++) {
      for (int m=0; m<4; m++) {
        g_q[4*n+m] = qinf[m];
        g_res[4*n+m] = 0.0f;
      }
    }
  }

  fclose(fp);

  nnode = compute_local_size (g_nnode, comm_size, my_rank);
  ncell = compute_local_size (g_ncell, comm_size, my_rank);
  nedge = compute_local_size (g_nedge, comm_size, my_rank);
  nbedge = compute_local_size (g_nbedge, comm_size, my_rank);

  op_printf("Number of nodes, cells, edges, bedges on process %d = %d, %d, %d, %d\n"
      ,my_rank,nnode,ncell,nedge,nbedge);

  /*Allocate memory to hold local sets, mapping tables and data*/
  cell   = (int *) malloc(4*ncell*sizeof(int));
  edge   = (int *) malloc(2*nedge*sizeof(int));
  ecell  = (int *) malloc(2*nedge*sizeof(int));
  bedge  = (int *) malloc(2*nbedge*sizeof(int));
  becell = (int *) malloc(  nbedge*sizeof(int));
  bound  = (int *) malloc(  nbedge*sizeof(int));

  x      = (float *) malloc(2*nnode*sizeof(float));
  q      = (float *) malloc(4*ncell*sizeof(float));
  qold   = (float *) malloc(4*ncell*sizeof(float));
  res    = (float *) malloc(4*ncell*sizeof(float));
  adt    = (float *) malloc(  ncell*sizeof(float));

  /* scatter sets, mappings and data on sets*/
  scatter_int_array(g_cell, cell, comm_size, g_ncell,ncell, 4);
  scatter_int_array(g_edge, edge, comm_size, g_nedge,nedge, 2);
  scatter_int_array(g_ecell, ecell, comm_size, g_nedge,nedge, 2);
  scatter_int_array(g_bedge, bedge, comm_size, g_nbedge,nbedge, 2);
  scatter_int_array(g_becell, becell, comm_size, g_nbedge,nbedge, 1);
  scatter_int_array(g_bound, bound, comm_size, g_nbedge,nbedge, 1);

  scatter_float_array(g_x, x, comm_size, g_nnode,nnode, 2);
  scatter_float_array(g_q, q, comm_size, g_ncell,ncell, 4);
  scatter_float_array(g_qold, qold, comm_size, g_ncell,ncell, 4);
  scatter_float_array(g_res, res, comm_size, g_ncell,ncell, 4);
  scatter_float_array(g_adt, adt, comm_size, g_ncell,ncell, 1);

  /*Freeing memory allocated to gloabal arrays on rank 0
    after scattering to all processes*/
  if(my_rank == MPI_ROOT) {
    free(g_cell);
    free(g_edge);
    free(g_ecell);
    free(g_bedge);
    free(g_becell);
    free(g_bound);
    free(g_x );
    free(g_q);
    free(g_qold);
    free(g_adt);
    free(g_res);
  }

  op_timers(&cpu_t2, &wall_t2);
  op_printf("Max total file read time = %f\n", wall_t2-wall_t1);

  /**------------------------END I/O and PARTITIONING -----------------------**/

  // declare sets, pointers, datasets and global constants

  op_set nodes  = op_decl_set(nnode,  "nodes");
  op_set edges  = op_decl_set(nedge,  "edges");
  op_set bedges = op_decl_set(nbedge, "bedges");
  op_set cells  = op_decl_set(ncell,  "cells");

  op_map pedge   = op_decl_map(edges, nodes,2,edge,  "pedge");
  op_map pecell  = op_decl_map(edges, cells,2,ecell, "pecell");
  op_map pbedge  = op_decl_map(bedges,nodes,2,bedge, "pbedge");
  op_map pbecell = op_decl_map(bedges,cells,1,becell,"pbecell");
  op_map pcell   = op_decl_map(cells, nodes,4,cell,  "pcell");

  op_dat p_bound = op_decl_dat(bedges,1,"int"  ,bound,"p_bound");
  op_dat p_x     = op_decl_dat(nodes ,2,"float",x    ,"p_x");
  op_dat p_q     = op_decl_dat(cells ,4,"float",q    ,"p_q");
  op_dat p_qold  = op_decl_dat(cells ,4,"float",qold ,"p_qold");
  op_dat p_adt   = op_decl_dat(cells ,1,"float",adt  ,"p_adt");
  op_dat p_res   = op_decl_dat(cells ,4,"float",res  ,"p_res");

  op_decl_const(1,"float",&gam  );
  op_decl_const(1,"float",&gm1  );
  op_decl_const(1,"float",&cfl  );
  op_decl_const(1,"float",&eps  );
  op_decl_const(1,"float",&mach );
  op_decl_const(1,"float",&alpha);
  op_decl_const(4,"float",qinf  );

  op_diagnostic_output();

  //trigger partitioning and halo creation routines
  op_partition("PTSCOTCH", "KWAY", NULL, pecell, p_x);

  //initialise timers for total execution wall time
  op_timers(&cpu_t1, &wall_t1);

  niter = 1000;
  for(int iter=1; iter<=niter; iter++) {

    //save old flow solution
    op_par_loop(save_soln,"save_soln", cells,
        op_arg_dat(p_q,   -1,OP_ID, 4,"float",OP_READ ),
        op_arg_dat(p_qold,-1,OP_ID, 4,"float",OP_WRITE));

    //  predictor/corrector update loop

    for(int k=0; k<2; k++) {

      //    calculate area/timstep

      op_par_loop(adt_calc,"adt_calc",cells,
                  op_arg_dat(p_x,  -4,pcell, 2,"float",OP_READ ),
                  op_arg_dat(p_q,  -1,OP_ID, 4,"float",OP_READ ),
                  op_arg_dat(p_adt,-1,OP_ID, 1,"float",OP_WRITE));

      //    calculate flux residual

      op_par_loop(res_calc,"res_calc",edges,
                  op_arg_dat(p_x,   -2,pedge, 2,"float",OP_READ),
                  op_arg_dat(p_q,   -2,pecell,4,"float",OP_READ),
                  op_arg_dat(p_adt, -2,pecell,1,"float",OP_READ),
                  op_arg_dat(p_res, -2,pecell,4,"float",OP_INC ));

      op_par_loop(bres_calc,"bres_calc",bedges,
                  op_arg_dat(p_x,    -2,pbedge, 2,"float",OP_READ),
                  op_arg_dat(p_q,     0,pbecell,4,"float",OP_READ),
                  op_arg_dat(p_adt,   0,pbecell,1,"float",OP_READ),
                  op_arg_dat(p_res,   0,pbecell,4,"float",OP_INC ),
                  op_arg_dat(p_bound,-1,OP_ID  ,1,"int",  OP_READ));

      //    update flow field

      rms = 0.0;

      op_par_loop(update,"update",cells,
          op_arg_dat(p_qold,-1,OP_ID, 4,"float",OP_READ ),
          op_arg_dat(p_q,   -1,OP_ID, 4,"float",OP_WRITE),
          op_arg_dat(p_res, -1,OP_ID, 4,"float",OP_RW   ),
          op_arg_dat(p_adt, -1,OP_ID, 1,"float",OP_READ ),
          op_arg_gbl(&rms,1,"float",OP_INC));
    }

    //print iteration history
    rms = sqrt(rms/(float) g_ncell);
    if (iter%100 == 0)
      op_printf("%d  %10.5e \n",iter,rms);
  }

  op_timers(&cpu_t2, &wall_t2);

  //get results data array - perhaps can be later handled by a remporary dat
  //op_dat temp = op_mpi_get_data(p_q);

  //output the result dat array to files
  //print_dat_tofile(temp, "out_grid.dat"); //ASCI
  //print_dat_tobinfile(temp, "out_grid.bin"); //Binary

  op_timing_output();

  //print total time for niter interations
  op_printf("Max total runtime = %f\n",wall_t2-wall_t1);
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

