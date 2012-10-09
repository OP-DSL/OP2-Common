#include <stdio.h>
#include <stdlib.h>
#include "op_seq.h"

struct BoreParams {
  double x0, Hl, ul, vl;
  double S;
};

struct GaussianLandslideParams {
  double A, v, lx, ly;
};

void InitEta(op_set volumes, op_dat cellCenters, op_dat values, op_dat temp_initEta, int fromFile) {
  if (fromFile) {
    //overwrite values.H with values stored in temp_initEta
    int variable = 1; //bitmask 1 - H, 2 - U, 4 - V, 8 - Zb
    //TODO: we are only overwriting H, moving the whole thing
    op_par_loop(applyConst, "applyConst", volumes,
        op_arg_dat(temp_initEta, -1, OP_ID, 4, "double", OP_READ),
        op_arg_dat(values, -1, OP_ID, 4, "double", OP_WRITE),
        op_arg_gbl(&variable, 1, "int", OP_READ));
  } else {
    //TODO: document the fact that this actually adds to the value of V
    // i.e. user should only access values[2]
    op_par_loop(initEta_formula, "initEta_formula", volumes,
        op_arg_dat(cellCenters, -1, OP_ID, 2, "double", OP_READ),
        op_arg_dat(values, -1, OP_ID, 4, "double", OP_INC));
  }
}

void InitU(op_set volumes, op_dat cellCenters, op_dat values) {
  //TODO: document the fact that this actually adds to the value of U
  // i.e. user should only access values[1]
  op_par_loop(initU_formula, "initU_formula", volumes,
      op_arg_dat(cellCenters, -1, OP_ID, 2, "double", OP_READ),
      op_arg_dat(values, -1, OP_ID, 4, "double", OP_INC));

}

void InitV(op_set volumes, op_dat cellCenters, op_dat values) {
  //TODO: document the fact that this actually adds to the value of V
  // i.e. user should only access values[2]
  op_par_loop(initV_formula, "initV_formula", volumes,
      op_arg_dat(cellCenters, -1, OP_ID, 2, "double", OP_READ),
      op_arg_dat(values, -1, OP_ID, 4, "double", OP_INC));

}

void InitBathymetry(op_set volumes, op_dat cellCenters, op_dat values, op_dat temp_initBathymetry, int fromFile, int firstTime) {
  if (firstTime) {
    int result = 0;
    int leftOperand = 0;
    int rightOperand = 3;
    int operation = 0; //0 +, 1 -, 2 *, 3 /
    op_par_loop(values_operation2, "values_operation2", volumes,
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
    op_par_loop(applyConst, "applyConst", volumes,
        op_arg_dat(temp_initBathymetry, -1, OP_ID, 4, "double", OP_READ),
        op_arg_dat(values, -1, OP_ID, 4, "double", OP_RW),
        op_arg_gbl(&variable, 1, "int", OP_READ));
  } else {
    //TODO: document the fact that this actually sets to the value of Zb
    // i.e. user should only access values[3]
    op_par_loop(initBathymetry_formula, "initBathymetry_formula", volumes,
        op_arg_dat(cellCenters, -1, OP_ID, 2, "double", OP_READ),
        op_arg_dat(values, -1, OP_ID, 4, "double", OP_INC));
  }

  op_par_loop(initBathymetry_update, "initBathymetry_update", volumes,
        op_arg_dat(values, -1, OP_ID, 4, "double", OP_RW),
        op_arg_gbl(&firstTime, 1, "int", OP_READ));
}

void InitBore(op_set volumes, op_dat cellCenters, op_dat values, BoreParams params) {
  double g = 9.81;
  double Fl = params.ul / sqrt( g * params.Hl );
  double Fs = params.S / sqrt( g * params.Hl );

  double r = .5 * ( sqrt( 1.0 + 8.0*( Fl - Fs )*(Fl - Fs ) ) - 1.0 );

  double Hr = r * params.Hl;
  double ur = S + ( params.ul - S ) / r;
  double vr = params.vl;
  ur *= -1.0;

  op_par_loop(initBore_select, "initBore_select", volumes,
      op_arg_dat(values, -1, OP_ID, 4, "double", OP_RW),
      op_arg_dat(cellCenters, -1, OP_ID, 2, "double", OP_READ),
      op_arg_gbl(&params.x0, 1, "double", OP_READ),
      op_arg_gbl(&params.Hl, 1, "double", OP_READ),
      op_arg_gbl(&params.ul, 1, "double", OP_READ),
      op_arg_gbl(&params.vl, 1, "double", OP_READ),
      op_arg_gbl(&Hr, 1, "double", OP_READ),
      op_arg_gbl(&ur, 1, "double", OP_READ),
      op_arg_gbl(&vr, 1, "double", OP_READ));
}

void InitGaussianLandslide(op_set volumes, op_dat cellCenters, op_dat values, double time, GaussianLandslideParams params) {
  //again, we only need Zb
  op_par_loop(initGaussianLandslide, "initGaussianLandslide", volumes,
      op_arg_dat(cellCenters, -1, OP_ID, 2, "double",OP_READ),
      op_arg_dat(values, -1, OP_ID, 4, "double",OP_RW),
      op_arg_gbl(&mesh_xmin, 1, "double", OP_READ),
      op_arg_gbl(&params.A, 1, "double", OP_READ),
      op_arg_gbl(&time, 1, "double", OP_READ),
      op_arg_gbl(&params.lx, 1, "double", OP_READ),
      op_arg_gbl(&params.ly, 1, "double", OP_READ),
      op_arg_gbl(&params.v, 1, "double", OP_READ));

  if (firstTime) {
    int result = 0;
    int leftOperand = 0;
    int rightOperand = 3;
    int operation = 1; //0 +, 1 -, 2 *, 3 /
    op_par_loop(values_operation2, "values_operation2", volumes,
        op_arg_dat(values, -1, OP_ID, 4, OP_RW).
        op_arg_gbl(&result, 1, "int", OP_READ),
        op_arg_gbl(&leftOperand, 1, "int", OP_READ),
        op_arg_gbl(&rightOperand, 1, "int", OP_READ),
        op_arg_gbl(&operation, 1, "int", OP_READ));
  }
}

void spaceDiscretization(op_dat data_in, op_dat data_out, double *minTimestep) {
  //    void SpaceDiscretization( const Values &in, Values &out, const Mesh
  //            &mesh, const PhysicalParams &params,
  //            RealType &minTimeStep, const RealType &t )
  { //begin SpaceDiscretization
    minTimestep = 0.0;
    op_dat LeftFacetValues; //temp - faces - dim 4
    op_dat RightFacetValues; //temp - faces - dim 4
    op_dat InterfaceBathy; //temp - faces - dim 1
    //call to FacetsValuesFromCellValues( in, LeftFacetValues, RightFacetValues,
    //    InterfaceBathy, mesh, t, params );
    //    void FacetsValuesFromCellValues( const Values &CellValues,
    //                                     Values &LeftFacetValues,
    //                                     Values &RightFacetValues,
    //                                     ScalarValue &InterfaceBathy,
    //                                     const Mesh &mesh, const RealType &t,
    //                                     const PhysicalParams &params )
    { //begin FacetsValuesFromCellValues
      op_par_loop(FacetsValuesFromCellValues, "FacetsValuesFromCellValues", faces,
          op_arg_dat(data_in, 0, ecell, 4, "double", OP_READ),
          op_arg_dat(data_in, 1, ecell, 4, "double", OP_READ),
          op_arg_dat(LeftFacetValues, -1, OP_ID, 4, "double", OP_WRITE),
          op_arg_dat(RightFacetValues, -1, OP_ID, 4, "double", OP_WRITE),
          op_arg_dat(InterfaceBathy, -1, OP_ID, 1, "double", OP_WRITE),
          op_arg_dat(mesh_FacetNormals, -1, OP_ID, 2, "double", OP_READ),
          op_arg_dat(mesh_isBoundary, -1, OP_ID, 1, "int", OP_READ));
    } //end FacetsValuesFromCellValues

    op_dat BathySource; //temp - faces - dim 2 (left & right)

    op_par_loop(SpaceDiscretization_1, "SpaceDiscretization_1", faces,
        op_arg_dat(LeftFacetValues, -1, OP_ID, 4, "double", OP_RW), // WE ONLY NEED H (RW) and Zb (READ)
        op_arg_dat(RightFacetValues, -1, OP_ID, 4, "double", OP_RW), // WE ONLY NEED H (RW) and Zb (READ)
        op_arg_dat(InterfaceBathy, -1, OP_ID, 1, "double", OP_READ),
        op_arg_dat(mesh_FacetVolumes, -1, OP_ID, 1, "double", OP_READ),
        op_arg_dat(BathySource, -1, OP_ID, 2, "double", OP_WRITE));

    op_dat FacetFluxes; //temp - faces - dim 4
    //call to NumericalFluxes( LeftFacetValues, RightFacetValues,
    //                     params, mesh, FacetFluxes, minTimeStep );
    //      void NumericalFluxes( const Values &LeftFacetValues,
    //                            const Values &RightFacetValues,
    //                            const PhysicalParams &params,
    //                            const Mesh &mesh, Values &out, RealType &minTimeStep )
    { //begin NumericalFluxes
      op_dat maxFacetEigenvalues; //temp - faces - dim 1

      op_par_loop(NumericalFluxes_1, "NumericalFluxes_1", faces,
          op_arg_dat(LeftFacetValues, -1, OP_ID, 4, "double", OP_READ), // WE do not need Zb
          op_arg_dat(RightFacetValues, -1, OP_ID, 4, "double", OP_READ), // WE do not need Zb
          op_arg_dat(FacetFluxes, -1, OP_ID, 4, "double", OP_WRITE),
          op_arg_dat(mesh_FacetVolumes, -1, OP_ID, 1, "double", OP_READ),
          op_arg_dat(mesh_FacetNormals, -1, OP_ID, 2, "double", OP_READ),
          op_arg_dat(maxFacetEigenvalues, -1, OP_ID, 1, "double", OP_WRITE));

      op_par_loop(NumericalFluxes_2, "NumericalFluxes_2", volumes,
          op_arg_dat(maxFacetEigenvalues, -3, cedges, 1, "double", OP_READ),
          op_arg_dat(mesh_FacetVolumes, -3, cedges, 1, "double", OP_READ),
          op_arg_dat(mesh_CellVolumes, -1, OP_ID, 1, "double", OP_READ),
          op_arg_gbl(&minTimeStep,1,"double", OP_MIN));
    } //end NumericalFluxes
#warning should set data_out to 0???
    op_par_loop(SpaceDiscretization_2, "SpaceDiscretization_2", faces,
        op_arg_dat(data_out, 0, ecell, 4, "doube", OP_INC), //again, Zb is not needed
        op_arg_dat(data_out, 1, ecell, 4, "doube", OP_INC),
        op_arg_dat(FacetFluxes, -1, OP_ID, 4, "double", OP_READ),
        op_arg_dat(BathySource, -1, OP_ID, 2, "double", OP_WRITE),
        op_arg_dat(mesh_FacetNormals, -1, OP_ID, 2, "double", OP_READ),
        op_arg_dat(mesh_isBoundary, -1, OP_ID, 1, "int", OP_READ));

    op_par_loop(SpaceDiscretization_3, "SpaceDiscretization_3", volumes,
        op_arg_dat(MidPointConservative, -1, OP_ID, 4, "doube", OP_RW),
        op_arg_dat(mesh_CellVolumes, -1, OP_ID, 1, "double", OP_READ));
  } //end SpaceDiscretization
}

int main(void) {

  GaussianLandslideParams gaussian_landslide_params;
  BoreParams bore_params;
  hid_t file;
  herr_t status;
  file = H5Fopen(filename_h5, H5F_ACC_RDONLY, H5P_DEFAULT);

  check_hdf5_error(H5LTmake_dataset_double(file, "BoreParams/x0", &bore_params.x0));
  check_hdf5_error(H5LTmake_dataset_double(file, "BoreParams/Hl", &bore_params.Hl));
  check_hdf5_error(H5LTmake_dataset_double(file, "BoreParams/ul", &bore_params.ul));
  check_hdf5_error(H5LTmake_dataset_double(file, "BoreParams/vl", &bore_params.vl));
  check_hdf5_error(H5LTmake_dataset_double(file, "BoreParams/S", &bore_params.S));
  check_hdf5_error(H5LTmake_dataset_double(file, "GaussianLandslideParams/A", &gaussian_landslide_params.A));
  check_hdf5_error(H5LTmake_dataset_double(file, "GaussianLandslideParams/v", &gaussian_landslide_params.v));
  check_hdf5_error(H5LTmake_dataset_double(file, "GaussianLandslideParams/lx", &gaussian_landslide_params.lx));
  check_hdf5_error(H5LTmake_dataset_double(file, "GaussianLandslideParams/ly", &gaussian_landslide_params.ly));

  //...
  check_hdf5_error(H5Fclose(file));

  //Corresponding to CellValues and tmp in Simulation::run() (simulation.hpp)
  //and in and out in EvolveValuesRK2() (timeStepper.hpp)
  op_dat values_old; //CellValues - volumes - dim 4
  op_dat values_new; //tmp - volumes - dim 4

  double timestep = 0.0;
  int itercount = 0;
  while (time < FinalTime) {
    //Call to EvolveValuesRK2( CellValues, tmp, mesh, CFL, Params, dt, timer.t );
    //  void EvolveValuesRK2( const Values &in, Values &out, const Mesh &m,
    //            const RealType &CFL, const PhysicalParams &params,
    //            RealType &timestep, const RealType &t )
    { //begin EvolveValuesRK2
      op_dat MidPointConservative; //temp - volumes - dim 4
      double minTimestep = 0.0;
      //call to SpaceDiscretization( in, MidPointConservative, m, params, minTimestep, t );
      spaceDiscretization(values_old, MidPointConservative, &minTimestep);

      double dT = CFL * minTimestep;
      op_dat inConservative; //temp - volums - dim 4
      op_dat MidPoint; //temp - volums - dim 4
      op_par_loop(EvolveValuesRK2_1, "EvolveValuesRK2_2", volumes,
          op_arg_gbl(&dT,1,"double", OP_READ),
          op_arg_dat(MidPointConservative, -1, OP_ID, 4, "double", OP_RW),
          op_arg_dat(values_old, -1, OP_ID, 4, "double", OP_READ),
          op_arg_dat(inConservative, -1, OP_ID, 4, "double", OP_WRITE),
          op_arg_dat(MidPoint, -1, OP_ID, 4, "double", OP_WRITE));

      op_dat outConservative; //temp - volumes - dim 4
      double dummy = 0.0;

      //call to SpaceDiscretization( MidPoint, outConservative, m, params, dummy_time, t );
      spaceDiscretization(MidPoint, outConservative, &dummy);

      op_par_loop(EvolveValuesRK2_2, "EvolveValuesRK2_2", volumes,
          op_arg_gbl(&dT,1,"double", OP_READ),
          op_arg_dat(outConservative, -1, OP_ID, 4, "double", OP_RW),
          op_arg_dat(inConservative, -1, OP_ID, 4, "double", OP_READ),
          op_arg_dat(MidPointConservative, -1, OP_ID, 4, "double", OP_READ),
          op_arg_dat(values_new, -1, OP_ID, 4, "double", OP_WRITE));
      timestep = dT;
    } //end EvolveValuesRK2
    op_par_loop(simulation_1, "simulation_1", volumes,
        op_arg_dat(values_old, -1, OP_ID, 4, "double", OP_WRITE),
        op_arg_dat(values_new, -1, OP_ID, 4, "double", OP_READ));
    timestep = timestep < Dtmax ? timestep : Dtmax;
    //TODO update time +timestep
    itercount++;
  }
  return 0;
}
