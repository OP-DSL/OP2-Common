void computeFluxes(double *CellLeft, double *CellRight,
                                double *mesh_FacetVolumes, double *mesh_FacetNormals,
                                int *isRightBoundary, //OP_READ
                                double *BathySource, double *out, //OP_WRITE
                                double *maxFacetEigenvalues) //OP_WRITE
{
  //begin FacetsValuesFromCellValues
  double leftFacetValues[4];
  double rightFacetValues[4];
  double InterfaceBathy;
  leftFacetValues[0] = CellLeft[0];
  leftFacetValues[1] = CellLeft[1];
  leftFacetValues[2] = CellLeft[2];
  leftFacetValues[3] = CellLeft[3];
  
  if (!*isRightBoundary) {
    rightFacetValues[0] = CellRight[0];
    rightFacetValues[1] = CellRight[1];
    rightFacetValues[2] = CellRight[2];
    rightFacetValues[3] = CellRight[3];
  } else {
    rightFacetValues[3] = CellLeft[3];
    double nx = mesh_FacetNormals[0];
    double ny = mesh_FacetNormals[1];
    double inNormalVelocity = CellLeft[1] * nx + CellLeft[2] * ny;
    double inTangentVelocity = -1.0 *  CellLeft[1] * ny + CellLeft[2] * nx;
    
    double outNormalVelocity = 0.0;
    double outTangentVelocity = 0.0;
    
    //WALL
    rightFacetValues[0] = CellLeft[0];
    outNormalVelocity = -1.0 * inNormalVelocity;
    outTangentVelocity = inTangentVelocity;
    
    
    /* //HEIGHTSUBC
     rightFacetValues[0] = -1.0 * rightFacetValues[3];
     rightFacetValues[0] += 0.1 * sin(10.0*t);
     outNormalVelocity = inNormalVelocity;
     outNormalVelocity +=
     2.0 * sqrt( g * CellLeft[0] );
     outNormalVelocity -=
     2.0 * sqrt( g * rightFacetValues[0] );
     
     outTangentVelocity = inTangentVelocity;
     */ //end HEIGHTSUBC
    
    /* //FLOWSUBC
     outNormalVelocity = 1;
     
     //rightFacetValues[0] = - rightFacetValues[3];
     
     rightFacetValues[0] = (inNormalVelocity - outNormalVelocity);
     rightFacetValues[0] *= .5 / sqrt( g );
     
     rightFacetValues[0] += sqrt( CellLeft[0] );
     
     outTangentVelocity = inTangentVelocity;
     */
    
    rightFacetValues[1] = outNormalVelocity * nx - outTangentVelocity * ny;
    rightFacetValues[2] = outNormalVelocity * ny + outTangentVelocity * nx;
  }
  
  InterfaceBathy = leftFacetValues[3] > rightFacetValues[3] ? leftFacetValues[3] : rightFacetValues[3];
  //SpaceDiscretization_1
  BathySource[0] = .5 * g * (leftFacetValues[0]*leftFacetValues[0]);
  BathySource[1] = .5 * g * (rightFacetValues[0]*rightFacetValues[0]);
  leftFacetValues[0] = (leftFacetValues[0] + leftFacetValues[3] - InterfaceBathy);
  leftFacetValues[0] = leftFacetValues[0] > 0.0 ? leftFacetValues[0] : 0.0;
  rightFacetValues[0] = (rightFacetValues[0] + rightFacetValues[3] - InterfaceBathy);
  rightFacetValues[0] = rightFacetValues[0] > 0.0 ? rightFacetValues[0] : 0.0;
  //NumericalFluxes_1
  BathySource[0] -= .5 * g * (leftFacetValues[0]*leftFacetValues[0]);
  BathySource[1] -= .5 * g * (rightFacetValues[0]*rightFacetValues[0]);
  BathySource[0] *= *mesh_FacetVolumes;
  BathySource[1] *= *mesh_FacetVolumes;
  double cL = sqrt(g * leftFacetValues[0]);
  cL = cL > 0.0 ? cL : 0.0;
  double cR = sqrt(g * rightFacetValues[0]);
  cR = cR > 0.0 ? cR : 0.0;
  
  double uLn = leftFacetValues[1] * mesh_FacetNormals[0] + leftFacetValues[2] * mesh_FacetNormals[1];
  double uRn = rightFacetValues[1] * mesh_FacetNormals[0] + rightFacetValues[2] * mesh_FacetNormals[1];
  
  double unStar = 0.5 * (uLn + uRn) - 0.25* (cL+cR);
  double cStar = 0.5 * (cL + cR) - 0.25* (uLn-uRn);
  
  double sL = (uLn - cL) < (unStar - cStar) ? (uLn - cL) : (unStar - cStar);
  double sLMinus = sL < 0.0 ? sL : 0.0;
  
  double sR = (uRn + cR) > (unStar + cStar) ? (uRn + cR) : (unStar + cStar);
  double sRPlus = sR > 0.0 ? sR : 0.0;
  
  sL = leftFacetValues[0] < EPS ? uRn - 2.0*cR : sL; // is this 2.0 or 2? (i.e. double/int)
  sR = leftFacetValues[0] < EPS ? uRn + cR : sR;
  
  sR = rightFacetValues[0] < EPS ? uLn + 2.0*cL : sR; // is this 2.0 or 2? (i.e. double/int)
  sL = rightFacetValues[0] < EPS ? uLn - cL : sL;
  
  double sRMinussL = sRPlus - sLMinus;
  sRMinussL = sRMinussL < EPS ? EPS : sRMinussL;
  
  double t1 = sRPlus / sRMinussL;
  //assert( ( 0 <= t1 ) && ( t1 <= 1 ) );
  
  double t2 = ( -1.0 * sLMinus ) / sRMinussL;
  //assert( ( 0 <= t2 ) && ( t2 <= 1 ) );
  
  double t3 = ( sRPlus * sLMinus ) / sRMinussL;
  
  double LeftFluxes_H, LeftFluxes_U, LeftFluxes_V;
  //inlined ProjectedPhysicalFluxes(leftFacetValues, Normals, params, LeftFluxes);
  double HuDotN = (leftFacetValues[0] * leftFacetValues[1]) * mesh_FacetNormals[0] +
  (leftFacetValues[0] * leftFacetValues[2]) * mesh_FacetNormals[1];
  
  LeftFluxes_H = HuDotN;
  LeftFluxes_U = HuDotN * leftFacetValues[1];
  LeftFluxes_V = HuDotN * leftFacetValues[2];
  
  LeftFluxes_U += (.5 * g * mesh_FacetNormals[0] ) * ( leftFacetValues[0] * leftFacetValues[0] );
  LeftFluxes_V += (.5 * g * mesh_FacetNormals[1] ) * ( leftFacetValues[0] * leftFacetValues[0] );
  //end of inlined
  
  double RightFluxes_H, RightFluxes_U, RightFluxes_V;
  //inlined ProjectedPhysicalFluxes(rightFacetValues, Normals, params, RightFluxes);
  HuDotN = (rightFacetValues[0] * rightFacetValues[1] * mesh_FacetNormals[0]) +
  (rightFacetValues[0] * rightFacetValues[2] * mesh_FacetNormals[1]);
  
  RightFluxes_H =   HuDotN;
  RightFluxes_U =   HuDotN * rightFacetValues[1];
  RightFluxes_V =   HuDotN * rightFacetValues[2];
  
  RightFluxes_U += (.5 * g * mesh_FacetNormals[0] ) * ( rightFacetValues[0] * rightFacetValues[0] );
  RightFluxes_V += (.5 * g * mesh_FacetNormals[1] ) * ( rightFacetValues[0] * rightFacetValues[0] );
  //end of inlined
  
  
  out[0] =
  ( t1 * LeftFluxes_H ) +
  ( t2 * RightFluxes_H ) +
  ( t3 * ( rightFacetValues[0] - leftFacetValues[0] ) );
  
  out[1] =
  ( t1 * LeftFluxes_U ) +
  ( t2 * RightFluxes_U ) +
  ( t3 * ( (rightFacetValues[0] * rightFacetValues[1]) -
          (leftFacetValues[0] * leftFacetValues[1]) ) );
  
  out[2] =
  ( t1 * LeftFluxes_V ) +
  ( t2 * RightFluxes_V ) +
  ( t3 * ( (rightFacetValues[0] * rightFacetValues[2]) -
          (leftFacetValues[0] * leftFacetValues[2]) ) );
  
  out[0] *= *mesh_FacetVolumes;
  out[1] *= *mesh_FacetVolumes;
  out[2] *= *mesh_FacetVolumes;
  out[3] = 0.0;
  
  double maximum = fabs(uLn + cL);
  maximum = maximum > fabs(uLn - cL) ? maximum : fabs(uLn - cL);
  maximum = maximum > fabs(uRn + cR) ? maximum : fabs(uRn + cR);
  maximum = maximum > fabs(uRn - cR) ? maximum : fabs(uRn - cR);
  *maxFacetEigenvalues = maximum;
}
