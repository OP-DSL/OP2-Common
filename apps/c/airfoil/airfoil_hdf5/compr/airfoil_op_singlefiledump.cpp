
//
// standard headers
//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// global constants

long double gam, gm1, cfl, eps, mach, alpha, qinf[4];

//
// OP header file
//

#include "op_seq.h"

//
// kernel routines for parallel loops
//

// main program

int main(int argc, char **argv) {

  // OP initialisation
  op_init(argc, argv, 2);

  int renumber = 0;
  for (int i = 1; i < argc; ++i)
    if (strcmp(argv[i],"-renumber")==0) {
      op_printf("Enabling renumbering\n");
      renumber = 1;
    }

  //int niter;
  //long double rms;

  // timer
  double cpu_t1, cpu_t2, wall_t1, wall_t2;

  // set constants and initialise flow field and residual
  op_printf("initialising flow field \n");

  char file[] = "new_grid.h5";
  char f_in[] = "p_q-double_2.8m_1000it.h5";
  char ofilename[] = "airfoil_2.8m_3.vtk";
  //char ofilename[] = "airfoil_45k_1_2.vtk";

  // declare sets, pointers, datasets and global constants

  op_set nodes = op_decl_set_hdf5(file, "nodes");
  op_set edges = op_decl_set_hdf5(file, "edges");
  //op_set bedges = op_decl_set_hdf5(file, "bedges");
  op_set cells = op_decl_set_hdf5(file, "cells");

  //op_map pedge = op_decl_map_hdf5(edges, nodes, 2, file, "pedge");
  op_map pecell = op_decl_map_hdf5(edges, cells, 2, file, "pecell");
  //op_map pbedge = op_decl_map_hdf5(bedges, nodes, 2, file, "pbedge");
  //op_map pbecell = op_decl_map_hdf5(bedges, cells, 1, file, "pbecell");
  op_map pcell = op_decl_map_hdf5(cells, nodes, 4, file, "pcell");

  op_map m_test = op_decl_map_hdf5(cells, nodes, 4, file, "m_test");
  if (m_test == NULL)
    printf("m_test not found\n");

  //op_dat p_bound = op_decl_dat_hdf5(bedges, 1, "int", file, "p_bound");
  op_dat p_x = op_decl_dat_hdf5(nodes, 2, "long double", file, "p_x");
  //op_dat p_q = op_decl_dat_hdf5(cells, 4, "long double", file, "p_q");
  //op_dat p_qold = op_decl_dat_hdf5(cells, 4, "long double", file, "p_qold");
  //op_dat p_adt = op_decl_dat_hdf5(cells, 1, "long double", file, "p_adt");
  //op_dat p_res = op_decl_dat_hdf5(cells, 4, "long double", file, "p_res");

  
  op_dat p_q = op_decl_dat_hdf5(cells, 4, "double", f_in, "p_q");

  op_dat p_test = op_decl_dat_hdf5(cells, 4, "long double", file, "p_test");
  if (p_test == NULL)
    printf("p_test not found\n");

  op_get_const_hdf5("gam", 1, "long double", (char *)&gam, "new_grid.h5");
  op_get_const_hdf5("gm1", 1, "long double", (char *)&gm1, "new_grid.h5");
  op_get_const_hdf5("cfl", 1, "long double", (char *)&cfl, "new_grid.h5");
  op_get_const_hdf5("eps", 1, "long double", (char *)&eps, "new_grid.h5");
  op_get_const_hdf5("mach", 1, "long double", (char *)&mach, "new_grid.h5");
  op_get_const_hdf5("alpha", 1, "long double", (char *)&alpha, "new_grid.h5");
  op_get_const_hdf5("qinf", 4, "long double", (char *)&qinf, "new_grid.h5");

  op_decl_const(1, "long double", &gam);
  op_decl_const(1, "long double", &gm1);
  op_decl_const(1, "long double", &cfl);
  op_decl_const(1, "long double", &eps);
  op_decl_const(1, "long double", &mach);
  op_decl_const(1, "long double", &alpha);
  op_decl_const(4, "long double", qinf);

  op_diagnostic_output();

  // write back original data just to compare you read the file correctly
  // do an h5diff between new_grid_out.h5 and new_grid.h5 to
  // compare two hdf5 files
  op_dump_to_hdf5("new_grid_out.h5");

  op_write_const_hdf5("gam", 1, "long double", (char *)&gam, "new_grid_out.h5");
  op_write_const_hdf5("gm1", 1, "long double", (char *)&gm1, "new_grid_out.h5");
  op_write_const_hdf5("cfl", 1, "long double", (char *)&cfl, "new_grid_out.h5");
  op_write_const_hdf5("eps", 1, "long double", (char *)&eps, "new_grid_out.h5");
  op_write_const_hdf5("mach", 1, "long double", (char *)&mach, "new_grid_out.h5");
  op_write_const_hdf5("alpha", 1, "long double", (char *)&alpha, "new_grid_out.h5");
  op_write_const_hdf5("qinf", 4, "long double", (char *)qinf, "new_grid_out.h5");

  // trigger partitioning and halo creation routines
  op_partition("PTSCsdfsOTCH", "KWAY", edges, pecell, p_x);
  // op_partition("PARMETIS", "KWAY", edges, pecell, p_x);
  if (renumber) op_renumber(pecell);

#define PDIM 2
 // int g_ncell = op_get_size(cells);

  // initialise timers for total execution wall time
  op_timers(&cpu_t1, &wall_t1);

  // main time-marching loop


  op_timers(&cpu_t2, &wall_t2);
  FILE *of;
  of = fopen(ofilename,"w");
  //of = fopen("airfoil_reldiff_single_genseqvsvec_2.8m.vtk","w");
  fprintf(of,"# vtk DataFile Version 3.0\n2D vector data\nASCII\nDATASET UNSTRUCTURED_GRID\n\nPOINTS %d float\n",p_x->set->size);

  for (int i=0; i<p_x->set->size; i++)
  //for (int i=0; i<10; i++)
  {
    fprintf(of,"%.25f   %.25f 0.0\n",(double)(((long double*)p_x->data)[2*i]),(double)(((long double*)p_x->data)[2*i+1]));
  }

  fprintf(of,"\nCELLS %d %d\n",pcell->from->size,pcell->from->size*5);
  for (int i=0; i<pcell->from->size;i++){
  //for (int i=0; i<10; i++){
    fprintf(of,"%d %d %d %d %d\n",4,pcell->map[i*4+0],pcell->map[i*4+1],pcell->map[i*4+2],pcell->map[i*4+3]);
  }

  fprintf(of,"\nCELL_TYPES %d\n",pcell->from->size);
  for (int i=0; i<pcell->from->size; i++){
  //for (int i=0; i<10; i++){
    fprintf(of,"9\n");
  }

  //fprintf(of,"\nCELL_DATA %d\n",pcell->from->size);
  //fprintf(of,"SCALARS pressure double 1\nLOOKUP_TABLE default\n");
  fprintf(of,"\nCELL_DATA %d\n",pcell->from->size);
//  fprintf(of,"SCALARS scalars float 1\nLOOKUP_TABLE default\n");
//  for (int i=0; i<p_q->set->size; i++){
//    fprintf(of,"%25f",(double)i);
//  }

  fprintf(of,"\nVECTORS vectors float\nLOOKUP_TABLE default\n");
  for (int i=0; i<p_q->set->size; i++){
    
      fprintf(of,"%.25f   %.25f  0.0\n",(double)((double*)(p_q->data))[i*4+1],(double)((double*)(p_q->data))[i*4+2]);
      
  }
  fclose(of);

  // printf("Root process = %d\n",op_is_root());

  // output the result dat array to files
  // op_dump_to_hdf5("new_grid_out.h5"); //writes data as it is held on each
  // process (under MPI)

  // compress using
  // ~/hdf5/bin/h5repack -f GZIP=9 new_grid.h5 new_grid_pack.h5

  op_timing_output();
  op_printf("Max total runtime = %f\n", wall_t2 - wall_t1);
  op_exit();
}
