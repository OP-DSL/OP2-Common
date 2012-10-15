void SpaceDiscretization_1(double *LeftFacetValues,//double LeftFacetValues[0], double LeftFacetValues[3], //RW, READ
              double *RightFacetValues, //double RightFacetValues[0], double RightFacetValues[3], //RW, READ
              double *InterfaceBathy, double *mesh_FacetVolumes, //READ
              double *BathySource //OP_WRITE
)
{
  BathySource[0] = .5 * g * (LeftFacetValues[0])*(LeftFacetValues[0]);
  BathySource[1] = .5 * g * (RightFacetValues[0])*(RightFacetValues[0]);
  LeftFacetValues[0] = (LeftFacetValues[0] + LeftFacetValues[3] - *InterfaceBathy);
  LeftFacetValues[0] = LeftFacetValues[0] > 0.0 ? LeftFacetValues[0] : 0.0;
  RightFacetValues[0] = (RightFacetValues[0] + RightFacetValues[3] - *InterfaceBathy);
  RightFacetValues[0] = RightFacetValues[0] > 0.0 ? RightFacetValues[0] : 0.0;

  BathySource[0] -= .5 * g * (LeftFacetValues[0])*(LeftFacetValues[0]);
  BathySource[1] -= .5 * g * (RightFacetValues[0])*(RightFacetValues[0]);
  BathySource[0] *= *mesh_FacetVolumes;
  BathySource[1] *= *mesh_FacetVolumes;
}
