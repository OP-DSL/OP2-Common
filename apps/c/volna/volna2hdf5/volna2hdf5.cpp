//============================================================================
// Name        : volna2hdf5.cpp
// Author      : Endre Laszlo
// Version     :
// Copyright   :
// Description :
//============================================================================

#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<string.h>
#include<fstream>
#include<iostream>
#include<string>
#include<map>

#include<hdf5.h>
#include<hdf5_hl.h>

//
// Sequential OP2 function declarations
//
#include "op_seq.h"

//
// VOLNA function declarations
//
#include "simulation.hpp"
#include "paramFileParser.hpp"
#include "event.hpp"
//
// Define meta data
//
#define N_STATEVAR 4
#define MESH_DIM 2
#define N_NODESPERCELL 3
#define N_CELLSPEREDGE 2

//
//helper functions
//
#define check_hdf5_error(err) __check_hdf5_error(err, __FILE__, __LINE__)
void __check_hdf5_error(herr_t err, const char *file, const int line) {
  if (err < 0) {
    printf("%s(%i) : OP2_HDF5_error() Runtime API error %d.\n", file,
        line, (int) err);
    exit(-1);
  }
}

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Wrong parameters! Please specify the VOLNA configuration "
        "script filename with the *.vln extension, "
        "e.g. ./volna2hdf5 bump.vln \n");
    exit(-1);
  }

  //
  ////////////// INIT VOLNA TO GAIN DATA IMPORT //////////////
  //

  printf("Importing data from VOLNA Framework ...\n");
  //  op_mesh_io_import(filename_msh, 2, 3, &x, &cell, &edge, &ecell,
  //      &bedge, &becell, &bound, &nnode, &ncell, &nedge, &nbedge);

  char const * const file = argv[1];
  spirit::file_iterator<> file_it(file);

  Simulation sim;
  ParamFileController fileParser;

  std::cerr << file << "\n";
  std::cerr << "EPSILON value: " << EPS << " \n";
  std::cerr << "Parsing parameter file... ";

  fileParser.parse(file_it, file_it.make_end(), sim);
  //sim.MeshFileName = "stlaurent_35k.msh";

  std::cerr << "Final time is : " << sim.FinalTime << std::endl;
  std::cerr << "Mesh fileName is : " << sim.MeshFileName << std::endl;
  std::cerr << "Initial value formula is : '" << sim.InitFormulas.eta
      << "'" << std::endl;
  std::cerr << "Bathymetry formula is : '"
      << sim.InitFormulas.bathymetry << "'" << std::endl;
  std::cerr << "Horizontal velocity formula is : '"
      << sim.InitFormulas.U << "'" << std::endl;
  std::cerr << "Vertical velocity formula is : '" << sim.InitFormulas.V
      << "'" << std::endl;

  int num_events = sim.events.size();

  std::vector<float> timer_start(events.size());
  std::vector<float> timer_end(events.size());
  std::vector<float> timer_step(events.size());
  std::vector<int> timer_istart(events.size());
  std::vector<int> timer_iend(events.size());
  std::vector<int> timer_istep(events.size());

  std::vector<float> event_location_x(events.size());
  std::vector<float> event_location_y(events.size());
  std::vector<int> event_post_update(events.size());
  std::vector<std::string> event_className(events.size());
  std::vector<std::string> event_formula(events.size());
  std::vector<std::string> event_streamName(events.size());

  for ( int i = 0; i < events.size(); i++ ) {
    TimerParams t_p;
    EventParams e_p;
    sim.events[i].dump(e_p);
    sim.events[i].timer.dump(e_p);
    timer_start[i] = t_p.start;
    timer_step[i] = t_p.step;
    timer_end[i] = t_p.end;
    timer_start[i] = t_p.istart;
    timer_istep[i] = t_p.istep;
    timer_iend[i] = t_p.iend;
    event_location_x[i] = e_p.location_x;
    event_location_y[i] = e_p.location_y;
    event_post_update[i] = e_p.event_post_update;
    event_className[i] = e_p.className;
    event_streamName[i] = e_p.streamName;
    event_formula[i] = e_p.formula;
  }
  // Initialize simulation: load mesh, calculate geometry data
  sim.init();

  //
  ////////////// INITIALIZE OP2 DATA /////////////////////
  //

  op_printf("Initializing OP2...\n");
  op_init(argc, argv, 2);

  int *cell = NULL; // Node IDs of cells
  int *ecell = NULL; // Cell IDs of edge
  int *ccell = NULL; // Cell IDs of neighbouring cells
  int *cedge = NULL; // Edge IDs of cells
  double *ccent = NULL; // Cell centre vectors
  double *carea = NULL; // Cell area
  double *enorm = NULL; // Edge normal vectors. Pointing from left cell to the right???
  double *ecent = NULL; // Edge center vectors
  double *eleng = NULL; // Edge length
  double *x = NULL; // Node coordinates in 2D
  double *w = NULL; // Conservative variables
  //double *wold = NULL; // Old conservative variables
  //double *res = NULL; // Per edge residuals
  //double *dt = NULL; // Time step

  // Number of nodes, cells, edges and iterations
  int nnode = 0, ncell = 0, nedge = 0, niter = 0;

  // Use this variable to obtain the no. nodes.
  // E.g. sim.mesh.Nodes.size() would return invalid data
  nnode = sim.mesh.NPoints;
  ncell = sim.mesh.NVolumes;
  nedge = sim.mesh.NFaces;

  printf("GMSH file data statistics: \n");
  printf("  No. nodes = %d\n", nnode);
  printf("  No. cells = %d\n", ncell);
  printf("Connectivity data statistics: \n");
  printf("  No. of edges           = %d\n", nedge);
  //printf("  No. of boundary edges  = %d\n\n", nbedge);

  // Arrays for mapping data
  cell = (int*) malloc(N_NODESPERCELL * ncell * sizeof(int));
  ecell = (int*) malloc(N_CELLSPEREDGE * nedge * sizeof(int));
  ccell = (int*) malloc(N_NODESPERCELL * ncell * sizeof(int));
  cedge = (int*) malloc(N_NODESPERCELL * ncell * sizeof(int));
  ccent = (double*) malloc(MESH_DIM * ncell * sizeof(double));
  carea = (double*) malloc(ncell * sizeof(double));
  enorm = (double*) malloc(MESH_DIM * nedge * sizeof(double));
  ecent = (double*) malloc(MESH_DIM * nedge * sizeof(double));
  eleng = (double*) malloc(nedge * sizeof(double));
  x = (double*) malloc(MESH_DIM * nnode * sizeof(double));
  w = (double*) malloc(N_STATEVAR * ncell * sizeof(double));
  //wold = (double*) malloc(N_STATEVAR * ncell * sizeof(double));
  //res = (double*) malloc(N_STATEVAR * nedge * sizeof(double));

  //
  ////////////// USE VOLNA FOR DATA IMPORT //////////////
  //

  int i = 0;
  std::cout << "Number of nodes according to sim.mesh.Nodes.size() = "
      << sim.mesh.Nodes.size() << std::endl;
  std::cout << "Number of nodes according to sim.mesh.NPoints      = "
      << sim.mesh.NPoints << "  <=== Accept this one! " << std::endl;

  // Import node coordinates
  for (i = 0; i < sim.mesh.NPoints; i++) {
    x[i * MESH_DIM] = sim.mesh.Nodes[i].x();
    x[i * MESH_DIM + 1] = sim.mesh.Nodes[i].y();
    //    std::cout << i << "  x,y,z = " << sim.mesh.Nodes[i].x() << " "
    //        << sim.mesh.Nodes[i].y() << " " << sim.mesh.Nodes[i].z()
    //        << endl;
  }

  // Boost arrays for temporarly storing mesh data
  boost::array<int, N_NODESPERCELL> vertices;
  boost::array<int, N_NODESPERCELL> neighbors;
  boost::array<int, N_NODESPERCELL> facet_ids;
  for (i = 0; i < sim.mesh.NVolumes; i++) {

    vertices = sim.mesh.Cells[i].vertices();
    neighbors = sim.mesh.Cells[i].neighbors();
    facet_ids = sim.mesh.Cells[i].facets();

    cell[i * N_NODESPERCELL] = vertices[0];
    cell[i * N_NODESPERCELL + 1] = vertices[1];
    cell[i * N_NODESPERCELL + 2] = vertices[2];

    ccell[i * N_NODESPERCELL] = neighbors[0];
    ccell[i * N_NODESPERCELL + 1] = neighbors[1];
    ccell[i * N_NODESPERCELL + 2] = neighbors[2];

    cedge[i * N_NODESPERCELL] = facet_ids[0];
    cedge[i * N_NODESPERCELL + 1] = facet_ids[1];
    cedge[i * N_NODESPERCELL + 2] = facet_ids[2];

    ccent[i * N_NODESPERCELL] = sim.mesh.CellCenters.x(i);
    ccent[i * N_NODESPERCELL + 1] = sim.mesh.CellCenters.y(i);

    carea[i] = sim.mesh.CellVolumes(i);

    w[i * N_STATEVAR] = sim.CellValues.H(i);
    w[i * N_STATEVAR + 1] = sim.CellValues.U(i);
    w[i * N_STATEVAR + 2] = sim.CellValues.V(i);
    w[i * N_STATEVAR + 3] = sim.CellValues.Zb(i);

    //    std::cout << "Cell " << i << " nodes = " << vertices[0] << " "
    //        << vertices[1] << " " << vertices[2] << std::endl;
    //    std::cout << "Cell " << i << " neighbours = " << neighbors[0]
    //        << " " << neighbors[1] << " " << neighbors[2] << std::endl;
    //    std::cout << "Cell " << i << " facets  = " << facet_ids[0] << " "
    //        << facet_ids[1] << " " << facet_ids[2] << std::endl;
    //    std::cout << "Cell " << i << " center  = [ "
    //        << ccent[i * N_NODESPERCELL] << " , "
    //        << ccent[i * N_NODESPERCELL + 1] << " ]" << std::endl;
    //    std::cout << "Cell " << i << " area  = " << carea[i] << std::endl;
    std::cout << "Cell " << i << " w = [H u v Zb] = [ "
        << w[i * N_STATEVAR] << " " << w[i * N_STATEVAR + 1] << " "
        << w[i * N_STATEVAR + 2] << " " << w[i * N_STATEVAR + 3]
                                             << std::endl;
  }

  // Store edge data: edge-cell map, edge normal vectors
  for (i = 0; i < sim.mesh.NFaces; i++) {
    ecell[i * N_CELLSPEREDGE] = sim.mesh.Facets[i].LeftCell();
    ecell[i * N_CELLSPEREDGE + 1] = sim.mesh.Facets[i].RightCell();

    enorm[i * N_CELLSPEREDGE] = sim.mesh.FacetNormals.x(i);
    enorm[i * N_CELLSPEREDGE + 1] = sim.mesh.FacetNormals.y(i);

    ecent[i * N_CELLSPEREDGE] = sim.mesh.FacetCenters.x(i);
    ecent[i * N_CELLSPEREDGE + 1] = sim.mesh.FacetCenters.y(i);

    eleng[i] = sim.mesh.FacetVolumes(i);

    //    std::cout << "Edge " << i << "   left cell = "
    //        << sim.mesh.Facets[i].LeftCell() << "   right cell = "
    //        << sim.mesh.Facets[i].RightCell() << std::endl;
    //    std::cout << "Edge " << i << "   normal vector = [ "
    //        << enorm[i * N_CELLSPEREDGE] << " , "
    //        << enorm[i * N_CELLSPEREDGE + 1] << " ]" << std::endl;
    //    std::cout << "Edge " << i << "   center vector = [ "
    //        << ecent[i * N_CELLSPEREDGE] << " , "
    //        << ecent[i * N_CELLSPEREDGE + 1] << " ]" << std::endl;
    //    std::cout << "Edge " << i << "   length =  " << eleng[i]
    //        << std::endl;
  }

  std::cout << "Number of edges according to sim.mesh.Nodes.size() = "
      << sim.mesh.Facets.size() << std::endl;
  std::cout << "Number of edges according to ssim.mesh.NFaces      = "
      << sim.mesh.NFaces << "  <=== Accept this one! " << std::endl;

  //
  // Define OP2 sets
  //
  op_set nodes = op_decl_set(nnode, "nodes");
  op_set edges = op_decl_set(nedge, "edges");
  op_set cells = op_decl_set(ncell, "cells");

  //
  // Define OP2 set maps
  //
  op_map pcell = op_decl_map(cells, nodes, N_NODESPERCELL, cell,
      "pcell");
  op_map pecell = op_decl_map(edges, cells, N_CELLSPEREDGE, ecell,
      "pecell");
  op_map pccell = op_decl_map(cells, cells, N_NODESPERCELL, ccell,
      "pccell");
  op_map pcedge = op_decl_map(cells, edges, N_NODESPERCELL, cedge,
      "pcedge");

  //
  // Define OP2 datasets
  //
  op_dat p_ccent = op_decl_dat(cells, MESH_DIM, "double", ccent,
      "p_ccent");
  op_dat p_carea = op_decl_dat(cells, 1, "double", carea, "p_carea");
  op_dat p_enorm = op_decl_dat(edges, MESH_DIM, "double", enorm,
      "p_enorm");
  op_dat p_ecent = op_decl_dat(edges, MESH_DIM, "double", ecent,
      "p_ecent");
  op_dat p_eleng = op_decl_dat(edges, 1, "double", eleng, "p_eleng");
  op_dat p_x = op_decl_dat(nodes, MESH_DIM, "double", x, "p_x");
  op_dat p_w = op_decl_dat(cells, N_STATEVAR, "double", w, "p_w");
  //  op_dat p_wold = op_decl_dat(cells, N_STATEVAR, "double", wold,
  //      "p_wold");
  //  op_dat p_res = op_decl_dat(cells, N_STATEVAR, "double", res,
  //      "p_res");

  //
  // Define HDF5 filename
  //
  char *filename_msh; // = "stlaurent_35k.msh";
  char *filename_h5; // = "stlaurent_35k.h5";
  filename_msh = strdup(sim.MeshFileName.c_str());
  filename_h5 = strndup(filename_msh, strlen(filename_msh) - 4);
  strcat(filename_h5, ".h5");

  //
  // Write mesh and geometry data to HDF5
  //
  op_write_hdf5(filename_h5);

  //
  // Read constants and write to HDF5
  //
  double cfl = sim.CFL; // CFL condition
  op_write_const_hdf5("cfl", 1, "double", (char *) &cfl, filename_h5);
  // Final time: as defined by Volna the end of real-time simulation
  double ftime = sim.FinalTime;
  op_write_const_hdf5("ftime", 1, "double", (char *) &ftime,
      filename_h5);
  double dtmax = sim.Dtmax; // Maximum timestep
  op_write_const_hdf5("dtmax", 1, "double", (char *) &dtmax,
      filename_h5);
  double g = 9.81; // Gravity constant
  op_write_const_hdf5("g", 1, "double", (char *) &g, filename_h5);

  //WRITING VALUES MANUALLY
  hid_t h5file;
  herr_t status;
  h5file = H5Fopen(filename_h5, H5F_ACC_RDWR, H5P_DEFAULT);

  int rank = 1;
  const hsize_t dims = 1;

  check_hdf5_error(
      H5LTmake_dataset_float(h5file, "BoreParams/x0", 1, &dims,
          (float*)&sim.bore_params.x0));
  check_hdf5_error(
      H5LTmake_dataset_float(h5file, "BoreParams/Hl", 1, &dims,
          (float*)&sim.bore_params.Hl));
  check_hdf5_error(
      H5LTmake_dataset_float(h5file, "BoreParams/ul", 1, &dims,
          (float*)&sim.bore_params.ul));
  check_hdf5_error(
      H5LTmake_dataset_float(h5file, "BoreParams/vl", 1, &dims,
          (float*)&sim.bore_params.vl));
  check_hdf5_error(
      H5LTmake_dataset_float(h5file, "BoreParams/S", 1, &dims,
          (float*)&sim.bore_params.S));
  check_hdf5_error(
      H5LTmake_dataset_float(h5file, "GaussianLandslideParams/A", 1,
          &dims, (float*)&sim.gaussian_landslide_params.A));
  check_hdf5_error(
      H5LTmake_dataset_float(h5file, "GaussianLandslideParams/v", 1,
          &dims, (float*)&sim.gaussian_landslide_params.v));
  check_hdf5_error(
      H5LTmake_dataset_float(h5file, "GaussianLandslideParams/lx", 1,
          &dims, (float*)&sim.gaussian_landslide_params.lx));
  check_hdf5_error(
      H5LTmake_dataset_float(h5file, "GaussianLandslideParams/ly", 1,
          &dims, (float*)&sim.gaussian_landslide_params.ly));

  check_hdf5_error(
        H5LTmake_dataset_int(h5file, "numEvents", 1,
            &dims, &num_events));

  check_hdf5_error(H5LTmake_dataset(h5file, "timer_start", 1, &num_events, H5T_NATIVE_FLOAT, &timer_start[0]));
  check_hdf5_error(H5LTmake_dataset(h5file, "timer_end", 1, &num_events, H5T_NATIVE_FLOAT, &timer_end[0]));
  check_hdf5_error(H5LTmake_dataset(h5file, "timer_step", 1, &num_events, H5T_NATIVE_FLOAT, &timer_step[0]));
  check_hdf5_error(H5LTmake_dataset(h5file, "timer_istart", 1, &num_events, H5T_NATIVE_INT, &timer_istart[0]));
  check_hdf5_error(H5LTmake_dataset(h5file, "timer_iend", 1, &num_events, H5T_NATIVE_INT, &timer_iend[0]));
  check_hdf5_error(H5LTmake_dataset(h5file, "timer_istep", 1, &num_events, H5T_NATIVE_INT, &timer_istep[0]));

  check_hdf5_error(H5LTmake_dataset(h5file, "event_location_x", 1, &num_events, H5T_NATIVE_FLOAT, &event_location_x[0]));
  check_hdf5_error(H5LTmake_dataset(h5file, "event_location_y", 1, &num_events, H5T_NATIVE_FLOAT, &event_location_y[0]));
  check_hdf5_error(H5LTmake_dataset(h5file, "event_post_update", 1, &num_events, H5T_NATIVE_INT, &event_post_update[0]));

  char buffer[18];
  for (int i = 0; i < event_className.size(); i++) {
    memset(buffer,0,18);
    sprintf(buffer, "event_className%d",i);
    check_hdf5_error(H5LTmake_dataset(h5file, buffer, event_className[i].c_str()));
    memset(buffer,0,18);
    sprintf(buffer, "event_formula%d",i);
    check_hdf5_error(H5LTmake_dataset(h5file, buffer, event_formula[i].c_str()));
    memset(buffer,0,18);
    sprintf(buffer, "event_streamName%d",i);
    check_hdf5_error(H5LTmake_dataset(h5file, buffer, event_streamName[i].c_str()));
  }

  //...
  check_hdf5_error(H5Fclose(h5file));

  free(cell);
  free(ecell);
  free(ccell);
  free(cedge);
  free(ccent);
  free(carea);
  free(enorm);
  free(ecent);
  free(eleng);
  free(x);
  free(w);
  //free(wold);
  //free(res);

  op_exit();
}

