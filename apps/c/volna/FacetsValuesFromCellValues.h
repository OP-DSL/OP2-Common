void FacetsValuesFromCellValues(float *CellLeft, float *CellRight,
                float *LeftFacetValues, //OP_WRITE
                float *RightFacetValues, //OP_WRITE
                float *InterfaceBathy, //OP_WRITE
                float *mesh_FacetNormals, int *isRightBoundary //OP_READ
                /*,float *t*/
//EZ PARA, the FacetNormals are only required on boudnary edges.... redundant data movement - split???
 )
{
  LeftFacetValues[0] = CellLeft[0];
  LeftFacetValues[1] = CellLeft[1];
  LeftFacetValues[2] = CellLeft[2];
  LeftFacetValues[3] = CellLeft[3];

  if (!*isRightBoundary) {
    RightFacetValues[0] = CellRight[0];
    RightFacetValues[1] = CellRight[1];
    RightFacetValues[2] = CellRight[2];
    RightFacetValues[3] = CellRight[3];
  } else {
    RightFacetValues[3] = CellLeft[3];
    float nx = mesh_FacetNormals[0];
    float ny = mesh_FacetNormals[1];
    float inNormalVelocity = CellLeft[1] * nx + CellLeft[2] * ny;
    float inTangentVelocity = CellLeft[1] * ny + CellLeft[2] * nx;

    float outNormalVelocity;
    float outTangentVelocity;

    //WALL
    RightFacetValues[0] = CellLeft[0];
    outNormalVelocity = -1.0 * inNormalVelocity;
    outTangentVelocity = inTangentVelocity;


    /* //HEIGHTSUBC
    RightFacetValues[0] = -1.0 * RightFacetValues[3];
    RightFacetValues[0] += 0.1 * sin(10.0*t);
    outNormalVelocity = inNormalVelocity;
    outNormalVelocity +=
        2.0 * sqrt( g * CellLeft[0] );
    outNormalVelocity -=
        2.0 * sqrt( g * RightFacetValues[0] );

    outTangentVelocity = inTangentVelocity;
    */ //end HEIGHTSUBC

    /* //FLOWSUBC
    outNormalVelocity = 1;

    //RightFacetValues[0] = - RightFacetValues[3];

    RightFacetValues[0] = (inNormalVelocity - outNormalVelocity);
    RightFacetValues[0] *= .5 / sqrt( g );

    RightFacetValues[0] += sqrt( CellLeft[0] );

    outTangentVelocity = inTangentVelocity;
    */

    RightFacetValues[1] = outNormalVelocity * nx - outTangentVelocity * ny;
    RightFacetValues[2] = outNormalVelocity * ny - outTangentVelocity * nx;
  }

  *InterfaceBathy = LeftFacetValues[3] > RightFacetValues[3] ? LeftFacetValues[3] : RightFacetValues[3];
}
