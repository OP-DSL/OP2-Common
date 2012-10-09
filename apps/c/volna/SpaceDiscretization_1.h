void SpaceDiscretization_1(float *LeftFacetValues,//float LeftFacetValues[0], float LeftFacetValues[3], //RW, READ
              float *RightFacetValues, //float RightFacetValues[0], float RightFacetValues[3], //RW, READ
              float *InterfaceBathy, float *mesh_FacetVolumes, //READ
              float *BathySource //OP_WRITE
)
{
  BathySource[0] = .5f * g * (LeftFacetValues[0])*(LeftFacetValues[0]);
  BathySource[1] = .5f * g * (RightFacetValues[0])*(RightFacetValues[0]);
  LeftFacetValues[0] = (LeftFacetValues[0] + LeftFacetValues[3] - *InterfaceBathy);
  LeftFacetValues[0] = LeftFacetValues[0] > 0.0f ? LeftFacetValues[0] : 0.0f;
  RightFacetValues[0] = (RightFacetValues[0] + RightFacetValues[3] - *InterfaceBathy);
  RightFacetValues[0] = RightFacetValues[0] > 0.0f ? RightFacetValues[0] : 0.0f;

  BathySource[0] -= .5f * g * (LeftFacetValues[0])*(LeftFacetValues[0]);
  BathySource[1] -= .5f * g * (RightFacetValues[0])*(RightFacetValues[0]);
  BathySource[0] *= *mesh_FacetVolumes;
  BathySource[1] *= *mesh_FacetVolumes;
}
