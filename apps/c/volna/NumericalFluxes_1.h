void NumericalFluxes_1(double *LeftFacetValues, //OP_READ
           double *RightFacetValues, //OP_READ
           double *out, //OP_WRITE
           double *FacetVolumes, //OP_READ,
           double *Normals, //OP_READ,
           double *maxFacetEigenvalues //OP_WRITE
)
{
  double cL = sqrt(params_g * LeftFacetValues[0]);
  cL = cL > 0.0 ? cL : 0.0;
  double cR = sqrt(params_g * RightFacetValues[0]);
  cR = cR > 0.0 ? cR : 0.0;

  double uLn = LeftFacetValues[1] * Normals[0] + LeftFacetValues[2] * Normals[1];
  double uRn = RightFacetValues[1] * Normals[0] + RightFacetValues[2] * Normals[1];

  double unStar = 0.5 * (uLn + uRn) - 0.25* (cL+cR);
  double cStar = 0.5 * (cL + cR) - 0.25* (uLn+uRn);

  double sL = (uLn - cL) < (unStar - cStar) ? (uLn - cL) : (unStar - cStar);
  double sLMinus = sL < 0.0 ? sL : 0.0;

  double sR = (uRn + cR) > (unStar + cStar) ? (uRn + cR) : (unStar + cStar);
  double sRPlus = sR > 0.0 ? sR : 0.0;

  sL = LeftFacetValues[0] < EPS ? uRn - 2.0*cR : sL; // is this 2.0 or 2? (i.e. double/int)
  sR = LeftFacetValues[0] < EPS ? uRn + cR : sR;

  sR = RightFacetValues[0] < EPS ? uLn + 2.0*cL : sR; // is this 2.0 or 2? (i.e. double/int)
  sL = RightFacetValues[0] < EPS ? uLn - cL : sL;

  double sRMinussL = sRPlus - sLMinus;
  sRMinussL = sRMinussL < EPS ? EPS : sRMinussL;

  double t1 = sRPlus / sRMinussL;
  //assert( ( 0 <= t1 ) && ( t1 <= 1 ) );

  double t2 = ( -1.0 * sLMinus ) / sRMinussL;
  //assert( ( 0 <= t2 ) && ( t2 <= 1 ) );

  double t3 = ( sRPlus * sLMinus ) / sRMinussL;

  double LeftFluxes_H, LeftFluxes_U, LeftFluxes_V;
  //inlined ProjectedPhysicalFluxes(LeftFacetValues, Normals, params, LeftFluxes);
  double HuDotN = (LeftFacetValues[0] * LeftFacetValues[1] * Normals[0]) +
          (LeftFacetValues[0] * LeftFacetValues[2] * Normals[1]);

  LeftFluxes_H = HuDotN;
  LeftFluxes_U = HuDotN * LeftFacetValues[1];
  LeftFluxes_V = HuDotN * LeftFacetValues[2];

  LeftFluxes_U += (.5 * params_g * Normals[0] ) * ( LeftFacetValues[0] * LeftFacetValues[0] );
  LeftFluxes_V += (.5 * params_g * Normals[1] ) * ( LeftFacetValues[0] * LeftFacetValues[0] );
  //end of inlined

  double RightFluxes_H, RightFluxes_U, RightFluxes_V;
  //inlined ProjectedPhysicalFluxes(RightFacetValues, Normals, params, RightFluxes);
  double HuDotN = (RightFacetValues[0] * RightFacetValues[1] * Normals[0]) +
          (RightFacetValues[0] * RightFacetValues[2] * Normals[1]);

  RightFluxes_H =   HuDotN;
  RightFluxes_U =   HuDotN * RightFacetValues[1];
  RightFluxes_V =   HuDotN * RightFacetValues[2];

  RightFluxes_U += (.5 * params_g * Normals[0] ) * ( RightFacetValues[0] * RightFacetValues[0] );
  RightFluxes_V += (.5 * params_g * Normals[1] ) * ( RightFacetValues[0] * RightFacetValues[0] );
  //end of inlined


  out[0] =
    ( t1 * LeftFluxes_H ) +
    ( t2 * RightFluxes_H ) +
    ( t3 * ( RightFacetValues[0] - LeftFacetValues[0] ) );

  out[1] =
    ( t1 * LeftFluxes_U ) +
    ( t2 * RightFluxes_U ) +
    ( t3 * ( (RightFacetValues[0] * RightFacetValues[1]) -
    (LeftFacetValues[0] * LeftFacetValues[1]) ) );

  out[2] =
    ( t1 * LeftFluxes_V ) +
    ( t2 * RightFluxes_V ) +
    ( t3 * ( (RightFacetValues[0] * RightFacetValues[2]) -
    (LeftFacetValues[0] * LeftFacetValues[2]) ) );

  out[0] *= *FacetVolumes;
  out[1] *= *FacetVolumes;
  out[2] *= *FacetVolumes;
  out[3] = 0.0;

  double maximum = abs(uLn + cL);
  maximum = maximum > abs(uLn - cL) ? maximum : abs(uLn - cL);
  maximum = maximum > abs(uRn + cR) ? maximum : abs(uRn + cR);
  maximum = maximum > abs(uRn - cR) ? maximum : abs(uRn - cR);
  *maxFacetEigenvalues = maximum;
}
