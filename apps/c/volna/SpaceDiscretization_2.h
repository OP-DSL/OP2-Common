void SpaceDiscretization_2(float *left, //OP_INC
              float *right, //OP_INC
              float *FacetFluxes, //OP_READ
              float *BathySource, //OP_READ
              float *mesh_FacetNormals, int *isRightBoundary //OP_READ
)
{
  left[0] -= FacetFluxes[0];
  left[1] -= (FacetFluxes[1] + BathySource[0] * mesh_FacetNormals[0]);
  left[2] -= (FacetFluxes[2] + BathySource[0] * mesh_FacetNormals[1]);

  if (!*isRightBoundary) {
    right[0] += FacetFluxes[0];
    right[1] += (FacetFluxes[1] + BathySource[1] * mesh_FacetNormals[0]);
    right[2] += (FacetFluxes[2] + BathySource[1] * mesh_FacetNormals[1]);
  }
}
