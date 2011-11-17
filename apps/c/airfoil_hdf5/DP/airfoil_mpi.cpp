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

double gam, gm1, cfl, eps, mach, alpha, qinf[4];

//
// OP header file
//

#include "op_lib_mpi.h"
#include "op_lib_cpp.h"

//
//hdf5 header
//
#include "hdf5.h"


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
#include "op_mpi_seq.h"


//
// main program
//
int main(int argc, char **argv){
    
    int my_rank;
    int comm_size;
	
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
	
    //timer
    double cpu_t1, cpu_t2, wall_t1, wall_t2;
    double time;
    double max_time;
  
    int    niter;
    double  rms;
    
    op_timers(&cpu_t1, &wall_t1);
    
    // set constants
    if(my_rank == MPI_ROOT )printf("initialising flow field\n");
    gam = 1.4f;
    gm1 = gam - 1.0f;
    cfl = 0.9f;
    eps = 0.05f;

    double mach  = 0.4f;
    double alpha = 3.0f*atan(1.0f)/45.0f;  
    double p     = 1.0f;
    double r     = 1.0f;
    double u     = sqrt(gam*p/r)*mach;
    double e     = p/(r*gm1) + 0.5f*u*u;

    qinf[0] = r;
    qinf[1] = r*u;
    qinf[2] = 0.0f;
    qinf[3] = r*e;
	
    // OP initialisation
    op_init(argc,argv,2);

    /**------------------------BEGIN Parallel I/O -------------------**/
    
    char file[] = "new_grid.h5";//"new_grid-26mil.h5";//"new_grid.h5";
    
    // declare sets, pointers, datasets and global constants - reading in from file
    op_set nodes  = op_decl_set_hdf5(file, "nodes");
    op_set edges  = op_decl_set_hdf5(file,  "edges");
    op_set bedges = op_decl_set_hdf5(file, "bedges");
    op_set cells  = op_decl_set_hdf5(file,  "cells");

    op_map pedge   = op_decl_map_hdf5(edges, nodes, 2, file, "pedge");
    op_map pecell  = op_decl_map_hdf5(edges, cells,2, file, "pecell");
    op_map pbedge  = op_decl_map_hdf5(bedges,nodes,2, file, "pbedge");
    op_map pbecell = op_decl_map_hdf5(bedges,cells,1, file, "pbecell");
    op_map pcell   = op_decl_map_hdf5(cells, nodes,4, file, "pcell");

    op_dat p_bound = op_decl_dat_hdf5(bedges,1,"int"  ,file,"p_bound");
    op_dat p_x     = op_decl_dat_hdf5(nodes ,2,"double",file,"p_x");
    op_dat p_q     = op_decl_dat_hdf5(cells ,4,"double",file,"p_q");
    op_dat p_qold  = op_decl_dat_hdf5(cells ,4,"double",file,"p_qold");
    op_dat p_adt   = op_decl_dat_hdf5(cells ,1,"double",file,"p_adt");
    op_dat p_res   = op_decl_dat_hdf5(cells ,4,"double",file,"p_res");

    /**------------------------END Parallel I/O  -----------------------**/
    
    op_timers(&cpu_t2, &wall_t2); 
    time = wall_t2-wall_t1;
    MPI_Reduce(&time,&max_time,1,MPI_DOUBLE, MPI_MAX,MPI_ROOT, MPI_COMM_WORLD);
    if(my_rank==MPI_ROOT)printf("Max total file read time = %f\n",max_time); 

    op_decl_const(1,"double",&gam  );
    op_decl_const(1,"double",&gm1  );
    op_decl_const(1,"double",&cfl  );
    op_decl_const(1,"double",&eps  );
    op_decl_const(1,"double",&mach );
    op_decl_const(1,"double",&alpha);
    op_decl_const(4,"double",qinf  );

    op_diagnostic_output();

    //write back original data just to compare you read the file correctly 
    //do an h5diff between new_grid_writeback.h5 and new_grid.h5 to 
    //compare two hdf5 files 
    op_write_hdf5("new_grid_out.h5");
    
    //partition with ParMetis
    //op_partition_geom(p_x);
    //op_partition_random(cells);
    //op_partition_kway(pecell);
    //op_partition_geomkway(p_x, pcell);
        
    //partition with PT-Scotch
    op_partition_ptscotch(pecell);
    
    //create halos
    op_halo_create();    
    
    int g_ncell = 0;
    int* sizes = (int *)malloc(sizeof(int)*comm_size);
    MPI_Allgather(&cells->size, 1, MPI_INT, sizes, 1, MPI_INT, MPI_COMM_WORLD);
    for(int i = 0; i<comm_size; i++)g_ncell = g_ncell + sizes[i];
    free(sizes);
    
    //initialise timers for total execution wall time
    op_timers(&cpu_t1, &wall_t1); 
    
    niter = 1000;
    for(int iter=1; iter<=niter; iter++) {
    	
    	//save old flow solution
    	op_par_loop(save_soln,"save_soln", cells,
    	    op_arg_dat(p_q,   -1,OP_ID, 4,"double",OP_READ ),
    	    op_arg_dat(p_qold,-1,OP_ID, 4,"double",OP_WRITE));

    	//  predictor/corrector update loop

    	for(int k=0; k<2; k++) {
    	   
    	    //    calculate area/timstep
    	    op_par_loop(adt_calc,"adt_calc",cells,
                  op_arg_dat(p_x,   0,pcell, 2,"double",OP_READ ),
                  op_arg_dat(p_x,   1,pcell, 2,"double",OP_READ ),
                  op_arg_dat(p_x,   2,pcell, 2,"double",OP_READ ),
                  op_arg_dat(p_x,   3,pcell, 2,"double",OP_READ ),
                  op_arg_dat(p_q,  -1,OP_ID, 4,"double",OP_READ ),
                  op_arg_dat(p_adt,-1,OP_ID, 1,"double",OP_WRITE));
                        
            //    calculate flux residual
            op_par_loop(res_calc,"res_calc",edges,
                  op_arg_dat(p_x,    0,pedge, 2,"double",OP_READ),
                  op_arg_dat(p_x,    1,pedge, 2,"double",OP_READ),
                  op_arg_dat(p_q,    0,pecell,4,"double",OP_READ),
                  op_arg_dat(p_q,    1,pecell,4,"double",OP_READ),
                  op_arg_dat(p_adt,  0,pecell,1,"double",OP_READ),
                  op_arg_dat(p_adt,  1,pecell,1,"double",OP_READ),
                  op_arg_dat(p_res,  0,pecell,4,"double",OP_INC ),
                  op_arg_dat(p_res,  1,pecell,4,"double",OP_INC ));
            
            op_par_loop(bres_calc,"bres_calc",bedges,
                  op_arg_dat(p_x,     0,pbedge, 2,"double",OP_READ),
                  op_arg_dat(p_x,     1,pbedge, 2,"double",OP_READ),
                  op_arg_dat(p_q,     0,pbecell,4,"double",OP_READ),
                  op_arg_dat(p_adt,   0,pbecell,1,"double",OP_READ),
                  op_arg_dat(p_res,   0,pbecell,4,"double",OP_INC ),
                  op_arg_dat(p_bound,-1,OP_ID  ,1,"int",  OP_READ));
            
            //    update flow field

            rms = 0.0;

            op_par_loop(update,"update",cells,
                  op_arg_dat(p_qold,-1,OP_ID, 4,"double",OP_READ ),
                  op_arg_dat(p_q,   -1,OP_ID, 4,"double",OP_WRITE),
                  op_arg_dat(p_res, -1,OP_ID, 4,"double",OP_RW   ),
                  op_arg_dat(p_adt, -1,OP_ID, 1,"double",OP_READ ),
                  op_arg_gbl(&rms,1,"double",OP_INC));
           
        }
        //print iteration history
        if(my_rank==MPI_ROOT)
        {
            rms = sqrt(rms/(double) g_ncell);
            if (iter%100 == 0)
            	printf("%d  %10.5e \n",iter,rms);
        }
        
    }
    op_timers(&cpu_t2, &wall_t2);
       
    //output the result dat array to files 
    //op_write_hdf5("new_grid_out.h5");
    
    //compress using
    // ~/hdf5/bin/h5repack -f GZIP=9 new_grid.h5 new_grid_pack.h5
    
    //free memory allocated to halos
    op_halo_destroy(); 
        
    //return all op_dats, op_maps back to original element order
    op_partition_reverse(); 
    
    //print each mpi process's timing info for each kernel
    op_mpi_timing_output();
    //print total time for niter interations
    time = wall_t2-wall_t1;
    MPI_Reduce(&time,&max_time,1,MPI_DOUBLE, MPI_MAX,MPI_ROOT, MPI_COMM_WORLD);
    if(my_rank==MPI_ROOT)printf("Max total runtime = %f\n",max_time);    
    
    op_exit();
    MPI_Finalize();   //user mpi finalize
}

 


