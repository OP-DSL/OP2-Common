#include <stdio.h>
#include <stdlib.h>
#include "op_seq.h"

struct BoreParams {
  float x0, Hl, ul, vl;
  float S;
};

struct GaussianLandslideParams {
  float A, v, lx, ly;
};

void InitEta(op_set cells, op_dat cellCenters, op_dat values, op_dat temp_initEta, int fromFile) {
  if (fromFile) {
    //overwrite values.H with values stored in temp_initEta
    int variable = 1; //bitmask 1 - H, 2 - U, 4 - V, 8 - Zb
    //TODO: we are only overwriting H, moving the whole thing
    op_par_loop(applyConst, "applyConst", cells,
        op_arg_dat(temp_initEta, -1, OP_ID, 4, "float", OP_READ),
        op_arg_dat(values, -1, OP_ID, 4, "float", OP_WRITE),
        op_arg_gbl(&variable, 1, "int", OP_READ));
  } else {
    //TODO: document the fact that this actually adds to the value of V
    // i.e. user should only access values[2]
    op_par_loop(initEta_formula, "initEta_formula", cells,
        op_arg_dat(cellCenters, -1, OP_ID, 2, "float", OP_READ),
        op_arg_dat(values, -1, OP_ID, 4, "float", OP_INC));
  }
}

void InitU(op_set cells, op_dat cellCenters, op_dat values) {
  //TODO: document the fact that this actually adds to the value of U
  // i.e. user should only access values[1]
  op_par_loop(initU_formula, "initU_formula", cells,
      op_arg_dat(cellCenters, -1, OP_ID, 2, "float", OP_READ),
      op_arg_dat(values, -1, OP_ID, 4, "float", OP_INC));

}

void InitV(op_set cells, op_dat cellCenters, op_dat values) {
  //TODO: document the fact that this actually adds to the value of V
  // i.e. user should only access values[2]
  op_par_loop(initV_formula, "initV_formula", cells,
      op_arg_dat(cellCenters, -1, OP_ID, 2, "float", OP_READ),
      op_arg_dat(values, -1, OP_ID, 4, "float", OP_INC));

}

void OutputSimulation(op_set points, op_set cells, op_dat p_x, op_dat values) {


}

void InitBathymetry(op_set cells, op_dat cellCenters, op_dat values, op_dat temp_initBathymetry, int fromFile, int firstTime) {
  if (firstTime) {
    int result = 0;
    int leftOperand = 0;
    int rightOperand = 3;
    int operation = 0; //0 +, 1 -, 2 *, 3 /
    op_par_loop(values_operation2, "values_operation2", cells,
        op_arg_dat(values, -1, OP_ID, 4, OP_RW).
        op_arg_gbl(&result, 1, "int", OP_READ),
        op_arg_gbl(&leftOperand, 1, "int", OP_READ),
        op_arg_gbl(&rightOperand, 1, "int", OP_READ),
        op_arg_gbl(&operation, 1, "int", OP_READ));
  }
  if (fromFile) {
    //overwrite values.H with values stored in temp_initEta
    int variable = 8; //bitmask 1 - H, 2 - U, 4 - V, 8 - Zb
    //TODO: we are only overwriting H, moving the whole thing
    op_par_loop(applyConst, "applyConst", cells,
        op_arg_dat(temp_initBathymetry, -1, OP_ID, 4, "float", OP_READ),
        op_arg_dat(values, -1, OP_ID, 4, "float", OP_RW),
        op_arg_gbl(&variable, 1, "int", OP_READ));
  } else {
    //TODO: document the fact that this actually sets to the value of Zb
    // i.e. user should only access values[3]
    op_par_loop(initBathymetry_formula, "initBathymetry_formula", cells,
        op_arg_dat(cellCenters, -1, OP_ID, 2, "float", OP_READ),
        op_arg_dat(values, -1, OP_ID, 4, "float", OP_INC));
  }

  op_par_loop(initBathymetry_update, "initBathymetry_update", cells,
        op_arg_dat(values, -1, OP_ID, 4, "float", OP_RW),
        op_arg_gbl(&firstTime, 1, "int", OP_READ));
}

void InitBore(op_set cells, op_dat cellCenters, op_dat values, BoreParams params) {
  float g = 9.81;
  float Fl = params.ul / sqrt( g * params.Hl );
  float Fs = params.S / sqrt( g * params.Hl );

  float r = .5 * ( sqrt( 1.0 + 8.0*( Fl - Fs )*(Fl - Fs ) ) - 1.0 );

  float Hr = r * params.Hl;
  float ur = S + ( params.ul - S ) / r;
  float vr = params.vl;
  ur *= -1.0;

  op_par_loop(initBore_select, "initBore_select", cells,
      op_arg_dat(values, -1, OP_ID, 4, "float", OP_RW),
      op_arg_dat(cellCenters, -1, OP_ID, 2, "float", OP_READ),
      op_arg_gbl(&params.x0, 1, "float", OP_READ),
      op_arg_gbl(&params.Hl, 1, "float", OP_READ),
      op_arg_gbl(&params.ul, 1, "float", OP_READ),
      op_arg_gbl(&params.vl, 1, "float", OP_READ),
      op_arg_gbl(&Hr, 1, "float", OP_READ),
      op_arg_gbl(&ur, 1, "float", OP_READ),
      op_arg_gbl(&vr, 1, "float", OP_READ));
}

void InitGaussianLandslide(op_set cells, op_dat cellCenters, op_dat values, float time, GaussianLandslideParams params) {
  //again, we only need Zb
  op_par_loop(initGaussianLandslide, "initGaussianLandslide", cells,
      op_arg_dat(cellCenters, -1, OP_ID, 2, "float",OP_READ),
      op_arg_dat(values, -1, OP_ID, 4, "float",OP_RW),
      op_arg_gbl(&mesh_xmin, 1, "float", OP_READ),
      op_arg_gbl(&params.A, 1, "float", OP_READ),
      op_arg_gbl(&time, 1, "float", OP_READ),
      op_arg_gbl(&params.lx, 1, "float", OP_READ),
      op_arg_gbl(&params.ly, 1, "float", OP_READ),
      op_arg_gbl(&params.v, 1, "float", OP_READ));

  if (firstTime) {
    int result = 0;
    int leftOperand = 0;
    int rightOperand = 3;
    int operation = 1; //0 +, 1 -, 2 *, 3 /
    op_par_loop(values_operation2, "values_operation2", cells,
        op_arg_dat(values, -1, OP_ID, 4, OP_RW).
        op_arg_gbl(&result, 1, "int", OP_READ),
        op_arg_gbl(&leftOperand, 1, "int", OP_READ),
        op_arg_gbl(&rightOperand, 1, "int", OP_READ),
        op_arg_gbl(&operation, 1, "int", OP_READ));
  }
}

void spaceDiscretization(op_dat data_in, op_dat data_out, float *minTimestep) {
  //    void SpaceDiscretization( const Values &in, Values &out, const Mesh
  //            &mesh, const PhysicalParams &params,
  //            RealType &minTimeStep, const RealType &t )
  { //begin SpaceDiscretization
    minTimestep = 0.0;
    op_dat leftCellValues; //temp - edges - dim 4
    op_dat rightCellValues; //temp - edges - dim 4
    op_dat interfaceBathy; //temp - edges - dim 1
    //call to FacetsValuesFromCellValues( in, leftCellValues, rightCellValues,
    //    interfaceBathy, mesh, t, params );
    //    void FacetsValuesFromCellValues( const Values &CellValues,
    //                                     Values &leftCellValues,
    //                                     Values &rightCellValues,
    //                                     ScalarValue &interfaceBathy,
    //                                     const Mesh &mesh, const RealType &t,
    //                                     const PhysicalParams &params )
    { //begin FacetsValuesFromCellValues
      op_par_loop(FacetsValuesFromCellValues, "FacetsValuesFromCellValues", edges,
          op_arg_dat(data_in, 0, ecell, 4, "float", OP_READ),
          op_arg_dat(data_in, 1, ecell, 4, "float", OP_READ),
          op_arg_dat(leftCellValues, -1, OP_ID, 4, "float", OP_WRITE),
          op_arg_dat(rightCellValues, -1, OP_ID, 4, "float", OP_WRITE),
          op_arg_dat(interfaceBathy, -1, OP_ID, 1, "float", OP_WRITE),
          op_arg_dat(edgeNormals, -1, OP_ID, 2, "float", OP_READ),
          op_arg_dat(isBoundary, -1, OP_ID, 1, "int", OP_READ));
    } //end FacetsValuesFromCellValues

    op_dat bathySource; //temp - edges - dim 2 (left & right)

    op_par_loop(SpaceDiscretization_1, "SpaceDiscretization_1", edges,
        op_arg_dat(leftCellValues, -1, OP_ID, 4, "float", OP_RW), // WE ONLY NEED H (RW) and Zb (READ)
        op_arg_dat(rightCellValues, -1, OP_ID, 4, "float", OP_RW), // WE ONLY NEED H (RW) and Zb (READ)
        op_arg_dat(interfaceBathy, -1, OP_ID, 1, "float", OP_READ),
        op_arg_dat(edgeLength, -1, OP_ID, 1, "float", OP_READ),
        op_arg_dat(bathySource, -1, OP_ID, 2, "float", OP_WRITE));

    op_dat edgeFluxes; //temp - edges - dim 4
    //call to NumericalFluxes( leftCellValues, rightCellValues,
    //                     params, mesh, edgeFluxes, minTimeStep );
    //      void NumericalFluxes( const Values &leftCellValues,
    //                            const Values &rightCellValues,
    //                            const PhysicalParams &params,
    //                            const Mesh &mesh, Values &out, RealType &minTimeStep )
    { //begin NumericalFluxes
      op_dat maxEdgeEigenvalues; //temp - edges - dim 1

      op_par_loop(NumericalFluxes_1, "NumericalFluxes_1", edges,
          op_arg_dat(leftCellValues, -1, OP_ID, 4, "float", OP_READ), // WE do not need Zb
          op_arg_dat(rightCellValues, -1, OP_ID, 4, "float", OP_READ), // WE do not need Zb
          op_arg_dat(edgeFluxes, -1, OP_ID, 4, "float", OP_WRITE),
          op_arg_dat(edgeLength, -1, OP_ID, 1, "float", OP_READ),
          op_arg_dat(edgeNormals, -1, OP_ID, 2, "float", OP_READ),
          op_arg_dat(maxEdgeEigenvalues, -1, OP_ID, 1, "float", OP_WRITE));

      op_par_loop(NumericalFluxes_2, "NumericalFluxes_2", cells,
          op_arg_dat(maxEdgeEigenvalues, -3, cedges, 1, "float", OP_READ),
          op_arg_dat(edgeLength, -3, cedges, 1, "float", OP_READ),
          op_arg_dat(cellVolumes, -1, OP_ID, 1, "float", OP_READ),
          op_arg_gbl(&minTimeStep,1,"float", OP_MIN));
    } //end NumericalFluxes
#warning should set data_out to 0???
    op_par_loop(SpaceDiscretization_2, "SpaceDiscretization_2", edges,
        op_arg_dat(data_out, 0, ecell, 4, "float", OP_INC), //again, Zb is not needed
        op_arg_dat(data_out, 1, ecell, 4, "float", OP_INC),
        op_arg_dat(edgeFluxes, -1, OP_ID, 4, "float", OP_READ),
        op_arg_dat(bathySource, -1, OP_ID, 2, "float", OP_WRITE),
        op_arg_dat(edgeNormals, -1, OP_ID, 2, "float", OP_READ),
        op_arg_dat(isBoundary, -1, OP_ID, 1, "int", OP_READ));

    op_par_loop(SpaceDiscretization_3, "SpaceDiscretization_3", cells,
        op_arg_dat(data_out, -1, OP_ID, 4, "float", OP_RW),
        op_arg_dat(cellVolumes, -1, OP_ID, 1, "float", OP_READ));
  } //end SpaceDiscretization
}

struct TimerParams {
  float start, end, step;
  unsigned int istart, iend, istep;
};

struct EventParams {
  float location_x, location_y;
  int post_update;
  std::string className;
  std::string formula;
  std::string streamName;
};

int main(void) {

  GaussianLandslideParams gaussian_landslide_params;
  BoreParams bore_params;
  hid_t file;
  herr_t status;
  file = H5Fopen(filename_h5, H5F_ACC_RDONLY, H5P_DEFAULT);

  check_hdf5_error(H5LTread_dataset_float(file, "BoreParamsx0", &bore_params.x0));
  check_hdf5_error(H5LTread_dataset_float(file, "BoreParamsHl", &bore_params.Hl));
  check_hdf5_error(H5LTread_dataset_float(file, "BoreParamsul", &bore_params.ul));
  check_hdf5_error(H5LTread_dataset_float(file, "BoreParamsvl", &bore_params.vl));
  check_hdf5_error(H5LTread_dataset_float(file, "BoreParamsS", &bore_params.S));
  check_hdf5_error(H5LTread_dataset_float(file, "GaussianLandslideParamsA", &gaussian_landslide_params.A));
  check_hdf5_error(H5LTread_dataset_float(file, "GaussianLandslideParamsv", &gaussian_landslide_params.v));
  check_hdf5_error(H5LTread_dataset_float(file, "GaussianLandslideParamslx", &gaussian_landslide_params.lx));
  check_hdf5_error(H5LTread_dataset_float(file, "GaussianLandslideParamsly", &gaussian_landslide_params.ly));

  int num_events = 0;

  check_hdf5_error(H5LTread_dataset_int(h5file, "numEvents", &num_events));

  std::vector<float> timer_start(num_events);
  std::vector<float> timer_end(num_events);
  std::vector<float> timer_step(num_events);
  std::vector<int> timer_istart(num_events);
  std::vector<int> timer_iend(num_events);
  std::vector<int> timer_istep(num_events);

  std::vector<float> event_location_x(num_events);
  std::vector<float> event_location_y(num_events);
  std::vector<int> event_post_update(num_events);
  std::vector<std::string> event_className(num_events);
  std::vector<std::string> event_formula(num_events);
  std::vector<std::string> event_streamName(num_events);

  const hsize_t num_events_hsize = num_events;
  check_hdf5_error(H5LTread_dataset(h5file, "timer_start", H5T_NATIVE_FLOAT, &timer_start[0]));
  check_hdf5_error(H5LTread_dataset(h5file, "timer_end", H5T_NATIVE_FLOAT, &timer_end[0]));
  check_hdf5_error(H5LTread_dataset(h5file, "timer_step", H5T_NATIVE_FLOAT, &timer_step[0]));
  check_hdf5_error(H5LTread_dataset(h5file, "timer_istart", H5T_NATIVE_INT, &timer_istart[0]));
  check_hdf5_error(H5LTread_dataset(h5file, "timer_iend", H5T_NATIVE_INT, &timer_iend[0]));
  check_hdf5_error(H5LTread_dataset(h5file, "timer_istep", H5T_NATIVE_INT, &timer_istep[0]));

  check_hdf5_error(H5LTread_dataset(h5file, "event_location_x", H5T_NATIVE_FLOAT, &event_location_x[0]));
  check_hdf5_error(H5LTread_dataset(h5file, "event_location_y", H5T_NATIVE_FLOAT, &event_location_y[0]));
  check_hdf5_error(H5LTread_dataset(h5file, "event_post_update", H5T_NATIVE_INT, &event_post_update[0]));

  std::vector<TimerParams> timers(num_events);
  std::vector<EventParams> events(num_events);

  /*
   * Convert Arrays to AoS
   */
  char buffer[18];
  char* eventbuffer;
  int length = 0;
  for (int i = 0; i < event_className.size(); i++) {
    timers[i].start = timer_start[i];
    timers[i].end = timer_end[i];
    timers[i].step = timer_step[i];
    timers[i].istart = timer_istart[i];
    timers[i].iend = timer_iend[i];
    timers[i].istep = timer_istep[i];

    events[i].location_x = event_location_x[i];
    events[i].location_y = event_location_y[i];
    events[i].post_update = event_post_update[i];

    memset(buffer,0,18);
    sprintf(buffer, "event_className%d",i);
    /*
     * If string can not handle a variable size char*, then use the commented lines
     */
    //check_hdf5_error(H5LTget_attribute_int(h5file, buffer, "length", &length));
    //eventbuffer = (char*)malloc(length);
    check_hdf5_error(H5LTread_dataset_string(h5file, buffer, events[i].className.c_str()));
    events[i].
    // free(eventbuffer);
    memset(buffer,0,18);
    sprintf(buffer, "event_formula%d",i);
    //check_hdf5_error(H5LTget_attribute_int(h5file, buffer, "length", &length));
    //eventbuffer = (char*)malloc(length);
    check_hdf5_error(H5LTread_dataset_string(h5file, buffer, events[i].formula.c_str()));
    // free(eventbuffer);
    memset(buffer,0,18);
    sprintf(buffer, "event_streamName%d",i);
    //heck_hdf5_error(H5LTget_attribute_int(h5file, buffer, "length", &length));
    //eventbuffer = (char*)malloc(length);
    check_hdf5_error(H5LTread_dataset_string(h5file, buffer, events[i].streamName.c_str()));
    // free(eventbuffer);
  }

  //...
  check_hdf5_error(H5Fclose(file));

  /*
   * Define HDF5 filename
   */
//  char *filename_msh; // = "stlaurent_35k.msh";
//  char *filename_h5; // = "stlaurent_35k.h5";
//  filename_msh = strdup(sim.MeshFileName.c_str());
//  filename_h5 = strndup(filename_msh, strlen(filename_msh) - 4);
//  strcat(filename_h5, ".h5");

  op_init(argc, argv, 2);

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
                                  "cellsToNodes");
  op_map cellsToCells = op_decl_map_hdf5(cells, cells, N_NODESPERCELL,
                                  filename_h5,
                                  "cellsToCells");
  op_map cellsToEdges = op_decl_map_hdf5(cells, edges, N_NODESPERCELL,
                                  filename_h5,
                                  "cellsToCells");

  /*
   * Define OP2 datasets
   */
  op_dat cellCenters = op_decl_dat_hdf5(cells, MESH_DIM, "float",
                                    filename_h5,
                                    "cellCenters");
  op_dat cellVolumes = op_decl_dat_hdf5(cells, 1, "float",
                                    filename_h5,
                                    "cellVolumes");
  op_dat edgeNormals = op_decl_dat_hdf5(edges, MESH_DIM, "float",
                                    filename_h5,
                                    "edgeNormals");
  op_dat edgeCenters = op_decl_dat_hdf5(edges, MESH_DIM, "float",
                                    filename_h5,
                                    "edgeCenters");
  op_dat edgeLength = op_decl_dat_hdf5(edges, 1, "float",
                                    filename_h5,
                                    "edgeLength");

  op_dat nodeCoords = op_decl_dat_hdf5(cells, N_STATEVAR, "float",
                                      filename_h5,
                                      "nodeCoords");

  op_dat values = op_decl_dat_hdf5(nodes, MESH_DIM, "float",
                                    filename_h5,
                                    "values");

  /*
   * Read constants and write to HDF5
   */
  op_get_const_hdf5("cfl", 1, "float", (char *) &cfl, filename_h5);
  // Final time: as defined by Volna the end of real-time simulation
  op_get_const_hdf5("ftime", 1, "float", (char *) &ftime, filename_h5);
  op_get_const_hdf5("dtmax", 1, "float", (char *) &dtmax, filename_h5);
  op_get_const_hdf5("g", 1, "float", (char *) &g, filename_h5);

  op_decl_const(1, "float", &cfl);
  op_decl_const(1, "float", &ftime);
  op_decl_const(1, "float", &dtmax);
  op_decl_const(1, "float", &g);

  op_diagnostic_output();


  //Corresponding to CellValues and tmp in Simulation::run() (simulation.hpp)
  //and in and out in EvolveValuesRK2() (timeStepper.hpp)

  op_dat values_new; //tmp - cells - dim 4

  float timestep = 0.0;
  int itercount = 0;
  while (time < FinalTime) {
    //Call to EvolveValuesRK2( CellValues, tmp, mesh, CFL, Params, dt, timer.t );
    //  void EvolveValuesRK2( const Values &in, Values &out, const Mesh &m,
    //            const RealType &CFL, const PhysicalParams &params,
    //            RealType &timestep, const RealType &t )
    { //begin EvolveValuesRK2
      op_dat midPointConservative; //temp - cells - dim 4
      float minTimestep = 0.0;
      //call to SpaceDiscretization( in, midPointConservative, m, params, minTimestep, t );
      spaceDiscretization(values, midPointConservative, &minTimestep);

      float dT = CFL * minTimestep;
      op_dat inConservative; //temp - volums - dim 4
      op_dat midPoint; //temp - volums - dim 4
      op_par_loop(EvolveValuesRK2_1, "EvolveValuesRK2_2", cells,
          op_arg_gbl(&dT,1,"float", OP_READ),
          op_arg_dat(midPointConservative, -1, OP_ID, 4, "float", OP_RW),
          op_arg_dat(values, -1, OP_ID, 4, "float", OP_READ),
          op_arg_dat(inConservative, -1, OP_ID, 4, "float", OP_WRITE),
          op_arg_dat(midPoint, -1, OP_ID, 4, "float", OP_WRITE));

      op_dat outConservative; //temp - cells - dim 4
      float dummy = 0.0;

      //call to SpaceDiscretization( midPoint, outConservative, m, params, dummy_time, t );
      spaceDiscretization(midPoint, outConservative, &dummy);

      op_par_loop(EvolveValuesRK2_2, "EvolveValuesRK2_2", cells,
          op_arg_gbl(&dT,1,"float", OP_READ),
          op_arg_dat(outConservative, -1, OP_ID, 4, "float", OP_RW),
          op_arg_dat(inConservative, -1, OP_ID, 4, "float", OP_READ),
          op_arg_dat(midPointConservative, -1, OP_ID, 4, "float", OP_READ),
          op_arg_dat(values_new, -1, OP_ID, 4, "float", OP_WRITE));
      timestep = dT;
    } //end EvolveValuesRK2
    op_par_loop(simulation_1, "simulation_1", cells,
        op_arg_dat(values, -1, OP_ID, 4, "float", OP_WRITE),
        op_arg_dat(values_new, -1, OP_ID, 4, "float", OP_READ));
    timestep = timestep < Dtmax ? timestep : Dtmax;
    //TODO update time +timestep
    itercount++;
  }
  return 0;
}
