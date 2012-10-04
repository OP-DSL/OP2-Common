void NumericalFluxes_1(double *LeftFacetValues_H, double *LeftFacetValues_U, double *LeftFacetValues_V, //OP_READ
           double *RightFacetValues_H, double *RightFacetValues_U, double *RightFacetValues_V, //OP_READ
          d ouble *out_H, double *out_U, double *out_V, double *out_Zb, //OP_WRITE
          double *FacetVolumes //OP_READ,
          double *maxFacetEigenvalues //OP_WRITE
)
{
  double cL = sqrt(params_g * *LeftFacetValues_H);
  cL = cL > 0.0 ? cL : 0.0;
  double cR = sqrt(params_g * *RightFacetValues_H);
  cR = cR > 0.0 ? cR : 0.0;

  double uLn = *LeftFacetValues_U * *Normals_x + *LeftFacetValues_V * *Normals_y;
  double uRn = *RightFacetValues_U * *Normals_x + *RightFacetValues_V * *Normals_y;

  double unStar = 0.5 * (uLn + uRn) - 0.25* (cL+cR);
  double cStar = 0.5 * (cL + cR) - 0.25* (uLn+uRn);

  double sL = (uLn - cL) < (unStar - cStar) ? (uLn - cL) : (unStar - cStar);
  double sLMinus = sL < 0.0 ? sL : 0.0;

  double sR = (uRn + cR) > (unStar + cStar) ? (uRn + cR) : (unStar + cStar);
  double sRPlus = sR > 0.0 ? sR : 0.0;

  sL = *LeftFacetValues_H < EPS ? uRn - 2.0*cR : sL; // is this 2.0 or 2? (i.e. double/int)
  sR = *LeftFacetValues_H < EPS ? uRn + cR : sR;

  sR = *RightFacetValues_H < EPS ? uLn + 2.0*cL : sR; // is this 2.0 or 2? (i.e. double/int)
  sL = *RightFacetValues_H < EPS ? uLn - cL : sL;

  double sRMinussL = sRPlus - sLMinus;
  sRMinussL = sRMinussL < EPS ? EPS : sRMinussL;

  double t1 = sRPlus / sRMinussL;
  //assert( ( 0 <= t1 ) && ( t1 <= 1 ) );

  double t2 = ( -1.0 * sLMinus ) / sRMinussL;
  //assert( ( 0 <= t2 ) && ( t2 <= 1 ) );

  double t3 = ( sRPlus * sLMinus ) / sRMinussL;

  double LeftFluxes_H, LeftFluxes_U, LeftFluxes_V;
  //inlined ProjectedPhysicalFluxes(LeftFacetValues, Normals, params, LeftFluxes);
  double HuDotN = (*LeftFacetValues_H * *LeftFacetValues_U * *Normals_x) +
          (*LeftFacetValues_H * *LeftFacetValues_V * *Normals_y);

  LeftFluxes_H = HuDotN;
  LeftFluxes_U = HuDotN * *LeftFacetValues_U;
  LeftFluxes_V = HuDotN * *LeftFacetValues_V;

  LeftFluxes_U += (.5 * params_g * *Normals_x ) * ( *LeftFacetValues_H * *LeftFacetValues_H );
  LeftFluxes_V += (.5 * params_g * *Normals_y ) * ( *LeftFacetValues_H * *LeftFacetValues_H );
  //end of inlined

  double RightFluxes_H, RightFluxes_U, RightFluxes_V;
  //inlined ProjectedPhysicalFluxes(RightFacetValues, Normals, params, RightFluxes);
  double HuDotN = (*RightFacetValues_H * *RightFacetValues_U * *Normals_x) +
          (*RightFacetValues_H * *RightFacetValues_V * *Normals_y);

  RightFluxes_H =   HuDotN;
  RightFluxes_U =   HuDotN * *RightFacetValues_U;
  RightFluxes_V =   HuDotN * *RightFacetValues_V;

  RightFluxes_U += (.5 * params_g * *Normals_x ) * ( *RightFacetValues_H * *RightFacetValues_H );
  RightFluxes_V += (.5 * params_g * *Normals_y ) * ( *RightFacetValues_H * *RightFacetValues_H );
  //end of inlined


  *out_H =
    ( t1 * LeftFluxes_H ) +
    ( t2 * RightFluxes_H ) +
    ( t3 * ( *RightFacetValues_H - *LeftFacetValues_H ) );

  *out_U =
    ( t1 * LeftFluxes_U ) +
    ( t2 * RightFluxes_U ) +
    ( t3 * ( (*RightFacetValues_H * *RightFacetValues_U) -
    (*LeftFacetValues_H * *LeftFacetValues_U) ) );

  *out_V =
    ( t1 * LeftFluxes_V ) +
    ( t2 * RightFluxes_V ) +
    ( t3 * ( (*RightFacetValues_H * *RightFacetValues_V) -
    (*LeftFacetValues_H * *LeftFacetValues_V) ) );

  *out_H *= *FacetVolumes;
  *out_U *= *FacetVolumes;
  *out_V *= *FacetVolumes;
  *out_Zb = 0.0;

  double maximum = abs(uLn + cL);
  maximum = maximum > abs(uLn - cL) ? maximum : abs(uLn - cL);
  maximum = maximum > abs(uRn + cR) ? maximum : abs(uRn + cR);
  maximum = maximum > abs(uRn - cR) ? maximum : abs(uRn - cR);
  *maxFacetEigenvalues = maximum;
}
