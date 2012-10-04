void SpaceDiscretization_1(double *LeftFacetValues_H, double *LeftFacetValues_Zb, //RW, READ
              double *RightFacetValues_H, double *RightFacetValues_Zb, //RW, READ
              double *InterfaceBathy, double *mesh_FacetVolumes, //READ
              double *BathySourceLeft, double *BathySourceRight //OP_WRITE
)
{
  *BathySourceLeft = .5 * params_g * (*LeftFacetValues_H)*(*LeftFacetValues_H);
  *BathySourceRight = .5 * params_g * (*RightFacetValues_H)*(*RightFacetValues_H);
  *LeftFacetValues_H = (*LeftFacetValues_H + *LeftFacetValues_Zb - *InterfaceBathy);
  *LeftFacetValues_H = *LeftFacetValues_H > 0.0 ? *LeftFacetValues_H : 0.0;
  *RightFacetValues_H = (*RightFacetValues_H + *RightFacetValues_Zb - *InterfaceBathy);
  *RightFacetValues_H = *RightFacetValues_H > 0.0 ? *RightFacetValues_H : 0.0;

  *BathySourceLeft -= .5 * params_g * (*LeftFacetValues_H)*(*LeftFacetValues_H);
  *BathySourceRight -= .5 * params_g * (*RightFacetValues_H)*(*RightFacetValues_H);
  *BathySourceLeft *= *mesh_FacetVolumes;
  *BathySourceRight *= *mesh_FacetVolumes;
}