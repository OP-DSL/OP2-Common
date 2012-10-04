void SpaceDiscretization_2(double *H_left, double *U_left, double *V_left, //OP_INC
              double *H_right, double *U_right, double *V_right, //OP_INC
              double *FacetFluxes_H, double *FacetFluxes_U, double *FacetFluxes_V, //OP_READ
              double *BathySourceLeft, double *BathySourceRight, //OP_READ
              double *mesh_FacetNormals_x, double *mesh_FacetNormals_y, int isRightBoundary //OP_READ
)
{
  *H_left -= *FacetFluxes_H;
  *U_left -= (*FacetFluxes_U + *BathySourceLeft * *mesh_FacetNormals_x);
  *V_left -= (*FacetFluxes_V + *BathySourceLeft * *mesh_FacetNormals_y);

  if (!isRightBoundary) {
    *H_right += *FacetFluxes_H;
    *U_right += (*FacetFluxes_U + *BathySourceRight * *mesh_FacetNormals_x);
    *V_right += (*FacetFluxes_V + *BathySourceRight * *mesh_FacetNormals_y);
  }
}