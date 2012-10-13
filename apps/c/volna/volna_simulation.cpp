#include "volna_common.h"
#include "FacetsValuesFromCellValues.h"
#include "NumericalFluxes_1.h"
#include "NumericalFluxes_2.h"
#include "SpaceDiscretization_1.h"
#include "SpaceDiscretization_2.h"
#include "SpaceDiscretization_3.h"

void spaceDiscretization(op_dat data_in, op_dat data_out, float *minTimestep,
    op_dat leftCellValues, op_dat rightCellValues, op_dat interfaceBathy,
    op_dat bathySource, op_dat edgeFluxes, op_dat maxEdgeEigenvalues,
    op_dat edgeNormals, op_dat edgeLength, op_dat cellVolumes, op_dat isBoundary,
    op_set cells, op_set edges, op_map edgesToCells, op_map cellsToEdges) {
  //    void SpaceDiscretization( const Values &in, Values &out, const Mesh
  //            &mesh, const PhysicalParams &params,
  //            RealType &minTimeStep, const RealType &t )
  { //begin SpaceDiscretization
    *minTimestep = INFINITY;
    //decl left/rightCellValues, interfaceBathy
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
          op_arg_dat(data_in, 0, edgesToCells, 4, "float", OP_READ),
          op_arg_dat(data_in, 1, edgesToCells, 4, "float", OP_READ),
          op_arg_dat(leftCellValues, -1, OP_ID, 4, "float", OP_WRITE),
          op_arg_dat(rightCellValues, -1, OP_ID, 4, "float", OP_WRITE),
          op_arg_dat(interfaceBathy, -1, OP_ID, 1, "float", OP_WRITE),
          op_arg_dat(edgeNormals, -1, OP_ID, 2, "float", OP_READ),
          op_arg_dat(isBoundary, -1, OP_ID, 1, "int", OP_READ));
    } //end FacetsValuesFromCellValues
    printf("FacetsValuesFromCellValues Left H %lf U %lf V %lf Zb %lf InterfaceBathy %lf\n", normcomp(leftCellValues,0), normcomp(leftCellValues, 1),normcomp(leftCellValues, 2),normcomp(leftCellValues, 3),normcomp(interfaceBathy,0));
    //decl bathySource

    op_par_loop(SpaceDiscretization_1, "SpaceDiscretization_1", edges,
        op_arg_dat(leftCellValues, -1, OP_ID, 4, "float", OP_RW), // WE ONLY NEED H (RW) and Zb (READ)
        op_arg_dat(rightCellValues, -1, OP_ID, 4, "float", OP_RW), // WE ONLY NEED H (RW) and Zb (READ)
        op_arg_dat(interfaceBathy, -1, OP_ID, 1, "float", OP_READ),
        op_arg_dat(edgeLength, -1, OP_ID, 1, "float", OP_READ),
        op_arg_dat(bathySource, -1, OP_ID, 2, "float", OP_WRITE));

    //decl edgeFluxes
    //call to NumericalFluxes( leftCellValues, rightCellValues,
    //                     params, mesh, edgeFluxes, minTimeStep );
    //      void NumericalFluxes( const Values &leftCellValues,
    //                            const Values &rightCellValues,
    //                            const PhysicalParams &params,
    //                            const Mesh &mesh, Values &out, RealType &minTimeStep )
    { //begin NumericalFluxes
      //decl maxEdgeEigenvalues
        printf("Numerical fluxes input left %lf %lf %lf %lf, right %lf %lf %lf %lf, edgeFluxes, %lf %lf %lf %lf\n", normcomp(leftCellValues, 0), normcomp(leftCellValues, 1), normcomp(leftCellValues, 2), normcomp(leftCellValues, 3), normcomp(rightCellValues, 0),  normcomp(rightCellValues, 1), normcomp(rightCellValues, 2), normcomp(rightCellValues, 3), normcomp(edgeFluxes, 0), normcomp(edgeFluxes, 1), normcomp(edgeFluxes, 2), normcomp(edgeFluxes, 3));
      op_par_loop(NumericalFluxes_1, "NumericalFluxes_1", edges,
          op_arg_dat(leftCellValues, -1, OP_ID, 4, "float", OP_READ), // WE do not need Zb
          op_arg_dat(rightCellValues, -1, OP_ID, 4, "float", OP_READ), // WE do not need Zb
          op_arg_dat(edgeFluxes, -1, OP_ID, 4, "float", OP_WRITE),
          op_arg_dat(edgeLength, -1, OP_ID, 1, "float", OP_READ),
          op_arg_dat(edgeNormals, -1, OP_ID, 2, "float", OP_READ),
          op_arg_dat(maxEdgeEigenvalues, -1, OP_ID, 1, "float", OP_WRITE));
  printf("maxFacetEigenvalues %lf edgeLen %lf cellVol %lf\n", normcomp(maxEdgeEigenvalues, 0), normcomp(edgeLength, 0), normcomp(cellVolumes, 0));
      op_par_loop(NumericalFluxes_2, "NumericalFluxes_2", cells,
          op_arg_dat(maxEdgeEigenvalues, -3, cellsToEdges, 1, "float", OP_READ),
          op_arg_dat(edgeLength, -3, cellsToEdges, 1, "float", OP_READ),
          op_arg_dat(cellVolumes, -1, OP_ID, 1, "float", OP_READ),
          op_arg_dat(data_out, -1, OP_ID, 4, "float", OP_WRITE),
          op_arg_gbl(minTimestep,1,"float", OP_MIN));
    } //end NumericalFluxes

    op_par_loop(SpaceDiscretization_2, "SpaceDiscretization_2", edges,
        op_arg_dat(data_out, 0, edgesToCells, 4, "float", OP_INC), //again, Zb is not needed
        op_arg_dat(data_out, 1, edgesToCells, 4, "float", OP_INC),
        op_arg_dat(edgeFluxes, -1, OP_ID, 4, "float", OP_READ),
        op_arg_dat(bathySource, -1, OP_ID, 2, "float", OP_READ),
        op_arg_dat(edgeNormals, -1, OP_ID, 2, "float", OP_READ),
        op_arg_dat(isBoundary, -1, OP_ID, 1, "int", OP_READ));

    op_par_loop(SpaceDiscretization_3, "SpaceDiscretization_3", cells,
        op_arg_dat(data_out, -1, OP_ID, 4, "float", OP_RW),
        op_arg_dat(cellVolumes, -1, OP_ID, 1, "float", OP_READ));
  } //end SpaceDiscretization
}
