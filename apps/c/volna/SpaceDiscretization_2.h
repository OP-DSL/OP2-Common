void SpaceDiscretization_2(double *left, //OP_INC
              double *right, //OP_INC
              double *FacetFluxes, //OP_READ
              double *BathySource, //OP_READ
              double *mesh_FacetNormals, int isRightBoundary //OP_READ
)
{
  left[0] -= FacetFluxes[0];
  left[1] -= (FacetFluxes[1] + BathySource[0] * *mesh_FacetNormals[0]);
  left[2] -= (FacetFluxes[2] + BathySource[0] * *mesh_FacetNormals[1]);

  if (!isRightBoundary) {
    right[0] += FacetFluxes[0];
    right[0] += (FacetFluxes[1] + BathySource[1] * *mesh_FacetNormals[0]);
    right[0] += (FacetFluxes[2] + BathySource[1] * *mesh_FacetNormals[1]);
  }
}