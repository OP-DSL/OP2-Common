void computeFluxes(double *cellLeft, double *cellRight,
                                double *edgeLength, double *edgeNormals,
                                int *isRightBoundary, //OP_READ
                                double *bathySource, double *out, //OP_WRITE
                                double *maxEdgeEigenvalues) //OP_WRITE
{
  //begin EdgesValuesFromCellValues
  double leftCellValues[4];
  double rightCellValues[4];
  double InterfaceBathy;
  leftCellValues[0] = cellLeft[0];
  leftCellValues[1] = cellLeft[1];
  leftCellValues[2] = cellLeft[2];
  leftCellValues[3] = cellLeft[3];

  if (!*isRightBoundary) {
    rightCellValues[0] = cellRight[0];
    rightCellValues[1] = cellRight[1];
    rightCellValues[2] = cellRight[2];
    rightCellValues[3] = cellRight[3];
  } else {
    rightCellValues[3] = cellLeft[3];
    double nx = edgeNormals[0];
    double ny = edgeNormals[1];
    double inNormalVelocity = cellLeft[1] * nx + cellLeft[2] * ny;
    double inTangentVelocity = -1.0 *  cellLeft[1] * ny + cellLeft[2] * nx;

    double outNormalVelocity = 0.0;
    double outTangentVelocity = 0.0;

    //WALL
    rightCellValues[0] = cellLeft[0];
    outNormalVelocity = -1.0 * inNormalVelocity;
    outTangentVelocity = inTangentVelocity;


    /* //HEIGHTSUBC
     rightCellValues[0] = -1.0 * rightCellValues[3];
     rightCellValues[0] += 0.1 * sin(10.0*t);
     outNormalVelocity = inNormalVelocity;
     outNormalVelocity +=
     2.0 * sqrt( g * cellLeft[0] );
     outNormalVelocity -=
     2.0 * sqrt( g * rightCellValues[0] );

     outTangentVelocity = inTangentVelocity;
     */ //end HEIGHTSUBC

    /* //FLOWSUBC
     outNormalVelocity = 1;

     //rightCellValues[0] = - rightCellValues[3];

     rightCellValues[0] = (inNormalVelocity - outNormalVelocity);
     rightCellValues[0] *= .5 / sqrt( g );

     rightCellValues[0] += sqrt( cellLeft[0] );

     outTangentVelocity = inTangentVelocity;
     */

    rightCellValues[1] = outNormalVelocity * nx - outTangentVelocity * ny;
    rightCellValues[2] = outNormalVelocity * ny + outTangentVelocity * nx;
  }

  InterfaceBathy = leftCellValues[3] > rightCellValues[3] ? leftCellValues[3] : rightCellValues[3];
  //SpaceDiscretization_1
  bathySource[0] = .5 * g * (leftCellValues[0]*leftCellValues[0]);
  bathySource[1] = .5 * g * (rightCellValues[0]*rightCellValues[0]);
  leftCellValues[0] = (leftCellValues[0] + leftCellValues[3] - InterfaceBathy);
  leftCellValues[0] = leftCellValues[0] > 0.0 ? leftCellValues[0] : 0.0;
  rightCellValues[0] = (rightCellValues[0] + rightCellValues[3] - InterfaceBathy);
  rightCellValues[0] = rightCellValues[0] > 0.0 ? rightCellValues[0] : 0.0;
  //NumericalFluxes_1
  bathySource[0] -= .5 * g * (leftCellValues[0]*leftCellValues[0]);
  bathySource[1] -= .5 * g * (rightCellValues[0]*rightCellValues[0]);
  bathySource[0] *= *edgeLength;
  bathySource[1] *= *edgeLength;
  double cL = sqrt(g * leftCellValues[0]);
  cL = cL > 0.0 ? cL : 0.0;
  double cR = sqrt(g * rightCellValues[0]);
  cR = cR > 0.0 ? cR : 0.0;

  double uLn = leftCellValues[1] * edgeNormals[0] + leftCellValues[2] * edgeNormals[1];
  double uRn = rightCellValues[1] * edgeNormals[0] + rightCellValues[2] * edgeNormals[1];

  double unStar = 0.5 * (uLn + uRn) - 0.25* (cL+cR);
  double cStar = 0.5 * (cL + cR) - 0.25* (uLn-uRn);

  double sL = (uLn - cL) < (unStar - cStar) ? (uLn - cL) : (unStar - cStar);
  double sLMinus = sL < 0.0 ? sL : 0.0;

  double sR = (uRn + cR) > (unStar + cStar) ? (uRn + cR) : (unStar + cStar);
  double sRPlus = sR > 0.0 ? sR : 0.0;

  sL = leftCellValues[0] < EPS ? uRn - 2.0*cR : sL; // is this 2.0 or 2? (i.e. double/int)
  sR = leftCellValues[0] < EPS ? uRn + cR : sR;

  sR = rightCellValues[0] < EPS ? uLn + 2.0*cL : sR; // is this 2.0 or 2? (i.e. double/int)
  sL = rightCellValues[0] < EPS ? uLn - cL : sL;

  double sRMinussL = sRPlus - sLMinus;
  sRMinussL = sRMinussL < EPS ? EPS : sRMinussL;

  double t1 = sRPlus / sRMinussL;
  //assert( ( 0 <= t1 ) && ( t1 <= 1 ) );

  double t2 = ( -1.0 * sLMinus ) / sRMinussL;
  //assert( ( 0 <= t2 ) && ( t2 <= 1 ) );

  double t3 = ( sRPlus * sLMinus ) / sRMinussL;

  double LeftFluxes_H, LeftFluxes_U, LeftFluxes_V;
  //inlined ProjectedPhysicalFluxes(leftCellValues, Normals, params, LeftFluxes);
  double HuDotN = (leftCellValues[0] * leftCellValues[1]) * edgeNormals[0] +
  (leftCellValues[0] * leftCellValues[2]) * edgeNormals[1];

  LeftFluxes_H = HuDotN;
  LeftFluxes_U = HuDotN * leftCellValues[1];
  LeftFluxes_V = HuDotN * leftCellValues[2];

  LeftFluxes_U += (.5 * g * edgeNormals[0] ) * ( leftCellValues[0] * leftCellValues[0] );
  LeftFluxes_V += (.5 * g * edgeNormals[1] ) * ( leftCellValues[0] * leftCellValues[0] );
  //end of inlined

  double RightFluxes_H, RightFluxes_U, RightFluxes_V;
  //inlined ProjectedPhysicalFluxes(rightCellValues, Normals, params, RightFluxes);
  HuDotN = (rightCellValues[0] * rightCellValues[1] * edgeNormals[0]) +
  (rightCellValues[0] * rightCellValues[2] * edgeNormals[1]);

  RightFluxes_H =   HuDotN;
  RightFluxes_U =   HuDotN * rightCellValues[1];
  RightFluxes_V =   HuDotN * rightCellValues[2];

  RightFluxes_U += (.5 * g * edgeNormals[0] ) * ( rightCellValues[0] * rightCellValues[0] );
  RightFluxes_V += (.5 * g * edgeNormals[1] ) * ( rightCellValues[0] * rightCellValues[0] );
  //end of inlined


  out[0] =
  ( t1 * LeftFluxes_H ) +
  ( t2 * RightFluxes_H ) +
  ( t3 * ( rightCellValues[0] - leftCellValues[0] ) );

  out[1] =
  ( t1 * LeftFluxes_U ) +
  ( t2 * RightFluxes_U ) +
  ( t3 * ( (rightCellValues[0] * rightCellValues[1]) -
          (leftCellValues[0] * leftCellValues[1]) ) );

  out[2] =
  ( t1 * LeftFluxes_V ) +
  ( t2 * RightFluxes_V ) +
  ( t3 * ( (rightCellValues[0] * rightCellValues[2]) -
          (leftCellValues[0] * leftCellValues[2]) ) );

  out[0] *= *edgeLength;
  out[1] *= *edgeLength;
  out[2] *= *edgeLength;
  out[3] = 0.0;

  double maximum = fabs(uLn + cL);
  maximum = maximum > fabs(uLn - cL) ? maximum : fabs(uLn - cL);
  maximum = maximum > fabs(uRn + cR) ? maximum : fabs(uRn + cR);
  maximum = maximum > fabs(uRn - cR) ? maximum : fabs(uRn - cR);
  *maxEdgeEigenvalues = maximum;
}
