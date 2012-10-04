void EvolveValuesRK2_2(double dT, double *outConservative_H, double *outConservative_U, double *outConservative_V, double *outConservative_Zb, //OP_RW, discard
            double *inConservative_H, double *inConservative_U, double *inConservative_V, double *inConservative_Zb, //OP_READ, discard
            double *MidPointConservative_H, double *MidPointConservative_U, double *MidPointConservative_v, double *MidPointConservative_Zb, //OP_READ, discard
            double *out_H, double *out_U, double *out_V, double *out_Zb) //OP_WRITE

{
  *outConservative_H = 0.5*(*outConservative_H * dt + *MidPointConservative_H + *inConservative_H);
  *outConservative_U = 0.5*(*outConservative_U * dt + *MidPointConservative_U + *inConservative_U);
  *outConservative_V = 0.5*(*outConservative_V * dt + *MidPointConservative_V + *inConservative_V);

  *outConservative_H = *outConservative_H <= EPS ? EPS : *outConservative_H;
  *outConservative_Zb = *inConservative_Zb;

  //call to ToPhysicalVariables inlined
  double TruncatedH = *outConservative_H < EPS ? EPS : *outConservative_H;
  *out_H = *outConservative_H;
  *out_U = *outConservative_U / TruncatedH;
  *out_V = *outConservative_V / TruncatedH;
  *out_Zb = *outConservative_Zb;
}
