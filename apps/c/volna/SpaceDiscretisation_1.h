void SpaceDiscretization_1(double *LeftFacetValues,//double *LeftFacetValues_H, double *LeftFacetValues_Zb, //RW, READ
              double *RightFacetValues, //double *RightFacetValues_H, double *RightFacetValues_Zb, //RW, READ
              double *InterfaceBathy, double *mesh_FacetVolumes, //READ
              double *BathySource //OP_WRITE
)
{
  BathySource[0] = .5 * params_g * (*LeftFacetValues_H)*(*LeftFacetValues_H);
  BathySource[1] = .5 * params_g * (*RightFacetValues_H)*(*RightFacetValues_H);
  *LeftFacetValues_H = (*LeftFacetValues_H + *LeftFacetValues_Zb - *InterfaceBathy);
  *LeftFacetValues_H = *LeftFacetValues_H > 0.0 ? *LeftFacetValues_H : 0.0;
  *RightFacetValues_H = (*RightFacetValues_H + *RightFacetValues_Zb - *InterfaceBathy);
  *RightFacetValues_H = *RightFacetValues_H > 0.0 ? *RightFacetValues_H : 0.0;

  BathySource[0] -= .5 * params_g * (*LeftFacetValues_H)*(*LeftFacetValues_H);
  BathySource[1] -= .5 * params_g * (*RightFacetValues_H)*(*RightFacetValues_H);
  BathySource[0] *= *mesh_FacetVolumes;
  BathySource[1] *= *mesh_FacetVolumes;
}