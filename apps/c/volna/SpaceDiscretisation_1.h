void SpaceDiscretization_1(float *LeftFacetValues,//float *LeftFacetValues_H, float *LeftFacetValues_Zb, //RW, READ
              float *RightFacetValues, //float *RightFacetValues_H, float *RightFacetValues_Zb, //RW, READ
              float *InterfaceBathy, float *mesh_FacetVolumes, //READ
              float *BathySource //OP_WRITE
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