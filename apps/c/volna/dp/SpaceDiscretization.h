inline void SpaceDiscretization(double *left, //OP_INC
              double *right, //OP_INC
              double *edgeFluxes, //OP_READ
              double *bathySource, //OP_READ
              double *edgeNormals, int *isRightBoundary, double **cellVolumes //OP_READ
)
{
  left[0] -= (edgeFluxes[0])/cellVolumes[0][0];
  left[1] -= (edgeFluxes[1] + bathySource[0] * edgeNormals[0])/cellVolumes[0][0];
  left[2] -= (edgeFluxes[2] + bathySource[0] * edgeNormals[1])/cellVolumes[0][0];

  if (!*isRightBoundary) {
    right[0] += edgeFluxes[0]/cellVolumes[1][0];
    right[1] += (edgeFluxes[1] + bathySource[1] * edgeNormals[0])/cellVolumes[1][0];
    right[2] += (edgeFluxes[2] + bathySource[1] * edgeNormals[1])/cellVolumes[1][0];
  }
}
