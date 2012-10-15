#include "volna_common.h"
#include "EvolveValuesRK2_1.h"
#include "EvolveValuesRK2_2.h"
#include "simulation_1.h"
//these are not const, we just don't want to pass them around
double timestamp = 0.0;
int itercount = 0;

//constants
double CFL, g, EPS;


int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Wrong parameters! Please specify the OP2 HDF5 data file "
        "name, that was created by volna2hdf5 tool "
        "script filename with the *.vln extension, "
        "e.g. ./volna filename.h5 \n");
    exit(-1);
  }

  op_init(argc, argv, 2);

  EPS = 1e-6; //for doubles 1e-11
  GaussianLandslideParams gaussian_landslide_params;
  BoreParams bore_params;
  hid_t file;
  herr_t status;
  const char *filename_h5 = argv[1]; // = "stlaurent_35k.h5";
  file = H5Fopen(filename_h5, H5F_ACC_RDONLY, H5P_DEFAULT);

  check_hdf5_error(H5LTread_dataset_double(file, "BoreParamsx0", &bore_params.x0));
  check_hdf5_error(H5LTread_dataset_double(file, "BoreParamsHl", &bore_params.Hl));
  check_hdf5_error(H5LTread_dataset_double(file, "BoreParamsul", &bore_params.ul));
  check_hdf5_error(H5LTread_dataset_double(file, "BoreParamsvl", &bore_params.vl));
  check_hdf5_error(H5LTread_dataset_double(file, "BoreParamsS", &bore_params.S));
  check_hdf5_error(H5LTread_dataset_double(file, "GaussianLandslideParamsA", &gaussian_landslide_params.A));
  check_hdf5_error(H5LTread_dataset_double(file, "GaussianLandslideParamsv", &gaussian_landslide_params.v));
  check_hdf5_error(H5LTread_dataset_double(file, "GaussianLandslideParamslx", &gaussian_landslide_params.lx));
  check_hdf5_error(H5LTread_dataset_double(file, "GaussianLandslideParamsly", &gaussian_landslide_params.ly));

  int num_events = 0;

  check_hdf5_error(H5LTread_dataset_int(file, "numEvents", &num_events));
  std::vector<TimerParams> timers(num_events);
  std::vector<EventParams> events(num_events);

  read_events_hdf5(file, num_events, &timers, &events);

  check_hdf5_error(H5Fclose(file));


  /*
   * Define HDF5 filename
   */
//  char *filename_msh; // = "stlaurent_35k.msh";
//  char *filename_h5; // = "stlaurent_35k.h5";
//  filename_msh = strdup(sim.MeshFileName.c_str());
//  filename_h5 = strndup(filename_msh, strlen(filename_msh) - 4);
//  strcat(filename_h5, ".h5");

  /*
   * Define OP2 sets - Read mesh and geometry data from HDF5
   */
  op_set nodes = op_decl_set_hdf5(filename_h5, "nodes");
  op_set edges = op_decl_set_hdf5(filename_h5, "edges");
  op_set cells = op_decl_set_hdf5(filename_h5, "cells");

  /*
   * Define OP2 set maps
   */
  op_map cellsToNodes = op_decl_map_hdf5(cells, nodes, N_NODESPERCELL,
                                  filename_h5,
                                  "cellsToNodes");
  op_map edgesToCells = op_decl_map_hdf5(edges, cells, N_CELLSPEREDGE,
                                  filename_h5,
                                  "edgesToCells");
  op_map cellsToCells = op_decl_map_hdf5(cells, cells, N_NODESPERCELL,
                                  filename_h5,
                                  "cellsToCells");
  op_map cellsToEdges = op_decl_map_hdf5(cells, edges, N_NODESPERCELL,
                                  filename_h5,
                                  "cellsToEdges");

  /*
   * Define OP2 datasets
   */
  op_dat cellCenters = op_decl_dat_hdf5(cells, MESH_DIM, "double",
                                    filename_h5,
                                    "cellCenters");

  op_dat cellVolumes = op_decl_dat_hdf5(cells, 1, "double",
                                    filename_h5,
                                    "cellVolumes");

  op_dat edgeNormals = op_decl_dat_hdf5(edges, MESH_DIM, "double",
                                    filename_h5,
                                    "edgeNormals");

  op_dat edgeCenters = op_decl_dat_hdf5(edges, MESH_DIM, "double",
                                    filename_h5,
                                    "edgeCenters");

  op_dat edgeLength = op_decl_dat_hdf5(edges, 1, "double",
                                    filename_h5,
                                    "edgeLength");

  op_dat nodeCoords = op_decl_dat_hdf5(nodes, MESH_DIM, "double",
                                      filename_h5,
                                      "nodeCoords");

  op_dat values = op_decl_dat_hdf5(cells, N_STATEVAR, "double",
                                    filename_h5,
                                    "values");
  op_dat isBoundary = op_decl_dat_hdf5(edges, 1, "int",
                                    filename_h5,
                                    "isBoundary");

  /*
   * Read constants from HDF5
   */
  double ftime, dtmax;
  op_get_const_hdf5("CFL", 1, "double", (char *) &CFL, filename_h5);
//  op_get_const_hdf5("EPS", 1, "double", (char *) &EPS, filename_h5);
  // Final time: as defined by Volna the end of real-time simulation
  op_get_const_hdf5("ftime", 1, "double", (char *) &ftime, filename_h5);
  op_get_const_hdf5("dtmax", 1, "double", (char *) &dtmax, filename_h5);
  op_get_const_hdf5("g", 1, "double", (char *) &g, filename_h5);

  op_decl_const(1, "double", &CFL);
  op_decl_const(1, "double", &EPS);
  op_decl_const(1, "double", &g);

  op_dat temp_initEta        = NULL;
  op_dat temp_initBathymetry = NULL;

  //Very first Init loop
  for (int i = 0; i < events.size(); i++) {
      if (!strcmp(events[i].className.c_str(), "InitEta")) {
        if (strcmp(events[i].streamName.c_str(), ""))
          temp_initEta = op_decl_dat_hdf5(cells, 1, "double",
              filename_h5,
              "initEta");
      } else if (!strcmp(events[i].className.c_str(), "InitBathymetry")) {
        if (strcmp(events[i].streamName.c_str(), ""))
          temp_initBathymetry = op_decl_dat_hdf5(cells, 1, "double",
              filename_h5,
              "initBathymetry");
      }
  }

  op_diagnostic_output();

  op_partition("PTSCOTCH", "KWAY", cells, cellsToEdges, NULL);

  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers(&cpu_t1, &wall_t1);

  //Very first Init loop
  processEvents(&timers, &events, 1/*firstTime*/, 1/*update timers*/, 0.0/*=dt*/, 1/*remove finished events*/, 2/*init loop, not pre/post*/,
                     cells, values, cellVolumes, cellCenters, nodeCoords, cellsToNodes, temp_initEta, temp_initBathymetry, bore_params, gaussian_landslide_params);


  //Corresponding to CellValues and tmp in Simulation::run() (simulation.hpp)
  //and in and out in EvolveValuesRK2() (timeStepper.hpp)

  double *tmp_elem = NULL;
  op_dat values_new = op_decl_dat_temp(cells, 4, "double",tmp_elem,"values_new"); //tmp - cells - dim 4

  //temporary dats
  //EvolveValuesRK2
  op_dat midPointConservative = op_decl_dat_temp(cells, 4, "double", tmp_elem, "midPointConservative"); //temp - cells - dim 4
  op_dat inConservative = op_decl_dat_temp(cells, 4, "double", tmp_elem, "inConservative"); //temp - cells - dim 4
  op_dat outConservative = op_decl_dat_temp(cells, 4, "double", tmp_elem, "outConservative"); //temp - cells - dim 4
  op_dat midPoint = op_decl_dat_temp(cells, 4, "double", tmp_elem, "midPoint"); //temp - cells - dim 4
  //SpaceDiscretization
  op_dat leftCellValues = op_decl_dat_temp(edges, 4, "double", tmp_elem, "leftCellValues"); //temp - edges - dim 4
  op_dat rightCellValues = op_decl_dat_temp(edges, 4, "double", tmp_elem, "rightCellValues"); //temp - edges - dim 4
  op_dat interfaceBathy = op_decl_dat_temp(edges, 1, "double", tmp_elem, "interfaceBathy"); //temp - edges - dim 1
  op_dat bathySource = op_decl_dat_temp(edges, 2, "double", tmp_elem, "bathySource"); //temp - edges - dim 2 (left & right)
  op_dat edgeFluxes = op_decl_dat_temp(edges, 4, "double", tmp_elem, "edgeFluxes"); //temp - edges - dim 4
  //NumericalFluxes
  op_dat maxEdgeEigenvalues = op_decl_dat_temp(edges, 1, "double", tmp_elem, "maxEdgeEigenvalues"); //temp - edges - dim 1

  double timestep;

  while (timestamp < ftime) {

    processEvents(&timers, &events, 0, 0, 0.0, 0, 0,
                       cells, values, cellVolumes, cellCenters, nodeCoords, cellsToNodes, temp_initEta, temp_initBathymetry, bore_params, gaussian_landslide_params);
    //memset(values_new->data, 0, values_new->set->size * values_new->size);
    printf("Call to EvolveValuesRK2 CellValues H %g U %g V %g Zb %g\n", normcomp(values, 0), normcomp(values, 1),normcomp(values, 2),normcomp(values, 3));
    //Call to EvolveValuesRK2( CellValues, tmp, mesh, CFL, Params, dt, timer.t );
    //  void EvolveValuesRK2( const Values &in, Values &out, const Mesh &m,
    //            const RealType &CFL, const PhysicalParams &params,
    //            RealType &timestep, const RealType &t )
    { //begin EvolveValuesRK2

      double minTimestep = 0.0;

      //call to SpaceDiscretization( in, midPointConservative, m, params, minTimestep, t );
      spaceDiscretization(values, midPointConservative, &minTimestep,
          leftCellValues, rightCellValues, interfaceBathy,
          bathySource, edgeFluxes, maxEdgeEigenvalues,
          edgeNormals, edgeLength, cellVolumes, isBoundary,
          cells, edges, edgesToCells, cellsToEdges, 0);
      
      printf("Return of SpaceDiscretization #1 midPointConservative H %g U %g V %g Zb %g\n", normcomp(midPointConservative, 0), normcomp(midPointConservative, 1),normcomp(midPointConservative, 2),normcomp(midPointConservative, 3));

      double dT = CFL * minTimestep;

      op_par_loop(EvolveValuesRK2_1, "EvolveValuesRK2_2", cells,
          op_arg_gbl(&dT,1,"double", OP_READ),
          op_arg_dat(midPointConservative, -1, OP_ID, 4, "double", OP_RW),
          op_arg_dat(values, -1, OP_ID, 4, "double", OP_READ),
          op_arg_dat(inConservative, -1, OP_ID, 4, "double", OP_WRITE),
          op_arg_dat(midPoint, -1, OP_ID, 4, "double", OP_WRITE));

      double dummy = 0.0;
      
      
      //call to SpaceDiscretization( midPoint, outConservative, m, params, dummy_time, t );
      spaceDiscretization(midPoint, outConservative, &dummy,
          leftCellValues, rightCellValues, interfaceBathy,
          bathySource, edgeFluxes, maxEdgeEigenvalues,
          edgeNormals, edgeLength, cellVolumes, isBoundary,
          cells, edges, edgesToCells, cellsToEdges, 1);
      
      op_par_loop(EvolveValuesRK2_2, "EvolveValuesRK2_2", cells,
          op_arg_gbl(&dT,1,"double", OP_READ),
          op_arg_dat(outConservative, -1, OP_ID, 4, "double", OP_RW),
          op_arg_dat(inConservative, -1, OP_ID, 4, "double", OP_READ),
          op_arg_dat(midPointConservative, -1, OP_ID, 4, "double", OP_READ),
          op_arg_dat(values_new, -1, OP_ID, 4, "double", OP_WRITE));

      timestep = dT;
    } //end EvolveValuesRK2

    op_par_loop(simulation_1, "simulation_1", cells,
        op_arg_dat(values, -1, OP_ID, 4, "double", OP_WRITE),
        op_arg_dat(values_new, -1, OP_ID, 4, "double", OP_READ));

    
    printf("New cell values %g %g %g %g\n", normcomp(values, 0), normcomp(values, 1),normcomp(values, 2),normcomp(values, 3));
    timestep = timestep < dtmax ? timestep : dtmax;
    op_printf("timestep = %g\n", timestep);
    {
      int dim = values->dim;
      double *data = (double *)(values->data);
      double norm = 0.0;
      for (int i = 0; i < values->set->size; i++) {
        norm += (data[dim*i]+data[dim*i+3])*(data[dim*i]+data[dim*i+3]);
      }
      printf("H+Zb: %g\n", sqrt(norm));
    }

    itercount++;
    timestamp += timestep;
    //TODO: mesh.mathParser.updateTime( timer.t ); ??

    //processing events
    processEvents(&timers, &events, 0, 1, timestep, 1, 1,
                         cells, values, cellVolumes, cellCenters, nodeCoords, cellsToNodes, temp_initEta, temp_initBathymetry, bore_params, gaussian_landslide_params);
  }

  //simulation
  if (op_free_dat_temp(values_new) < 0)
        op_printf("Error: temporary op_dat %s cannot be removed\n",values_new->name);
  //EvolveValuesRK2
  if (op_free_dat_temp(midPointConservative) < 0)
          op_printf("Error: temporary op_dat %s cannot be removed\n",midPointConservative->name);
  if (op_free_dat_temp(inConservative) < 0)
          op_printf("Error: temporary op_dat %s cannot be removed\n",inConservative->name);
  if (op_free_dat_temp(outConservative) < 0)
          op_printf("Error: temporary op_dat %s cannot be removed\n",outConservative->name);
  if (op_free_dat_temp(midPoint) < 0)
          op_printf("Error: temporary op_dat %s cannot be removed\n",midPoint->name);
  //SpaceDiscretization
  if (op_free_dat_temp(leftCellValues) < 0)
          op_printf("Error: temporary op_dat %s cannot be removed\n",leftCellValues->name);
  if (op_free_dat_temp(rightCellValues) < 0)
          op_printf("Error: temporary op_dat %s cannot be removed\n",rightCellValues->name);
  if (op_free_dat_temp(interfaceBathy) < 0)
          op_printf("Error: temporary op_dat %s cannot be removed\n",interfaceBathy->name);
  if (op_free_dat_temp(bathySource) < 0)
          op_printf("Error: temporary op_dat %s cannot be removed\n",bathySource->name);
  if (op_free_dat_temp(edgeFluxes) < 0)
          op_printf("Error: temporary op_dat %s cannot be removed\n",edgeFluxes->name);
  //NumericalFluxes
  if (op_free_dat_temp(maxEdgeEigenvalues) < 0)
          op_printf("Error: temporary op_dat %s cannot be removed\n",maxEdgeEigenvalues->name);

  op_timers(&cpu_t2, &wall_t2);
  op_timing_output();
  op_printf("Max total runtime = \n%lf\n",wall_t2-wall_t1);

  op_exit();

  return 0;
}
