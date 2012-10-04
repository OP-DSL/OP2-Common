void FacetsValuesFromCellValues(double *CellLeft_H, double *CellLeft_U, double *CellLeft_V, double *CellLeft_Zb,
                double *CellRight_H, double *CellRight_U, double *CellRight_V, double *CellRight_Zb,
                double *LeftFacetValues_H, double *LeftFacetValues_U, double *LeftFacetValues_V, double *LeftFacetValues_Zb, //OP_WRITE
                double *RightFacetValues_H, double *RightFacetValues_U, double *RightFacetValues_V, double *RightFacetValues_Zb, //OP_WRITE
                double *mesh_FacetNormals_x, double *mesh_FacetNormals_y, int *isRightBoundary //OP_READ

//EZ PARA, the FacetNormals are only required on boudnary edges.... redundant data movement - split???
 )
{
  *LeftFacetValues_H = *CellLeft_H;
  *LeftFacetValues_U = *CellLeft_U;
  *LeftFacetValues_V = *CellLeft_V;
  *LeftFacetValues_Zb = *CellLeft_Zb;

  if (!isRightBoundary) {
    *RightFacetValues_H = *CellRight_H;
    *RightFacetValues_U = *CellRight_U;
    *RightFacetValues_V = *CellRight_V;
    *RightFacetValues_Zb = *CellRight_Zb;
  } else {
    *RightFacetValues_Zb = *CellLeft_Zb;
    double nx = *mesh_FacetNormals_x;
    double ny = *mesh_FacetNormals_y;
    double inNormalVelocity = *CellLeft_U * nx + *CellLeft_V * ny;
    double inTangentVelocity = *CellLeft_U * ny + *CellLeft_V * nx;
    //WALL
    *RightFacetValues_H = *CellLeft_H;
    double outNormalVelocity = -1.0 * inNormalVelocity;
    double outTangetVelocity = inTangentVeolcity;
    //TODO: HEIGHTSUBC, FLOWSUBC

    *RightFacetValues_U = outNormalVelocity * nx - outTangentVelocity * ny;
    *RightFacetValues_V = outNormalVelocity * ny - outTangentVelocity * nx;
  }

  *InterfaceBathy = *LeftFacetValues_Zb > *RightFacetValues_Zb ? *LeftFacetValues_Zb : *RightFacetValues_Zb;
}