void EvolveValuesRK2_1(double dT, double *MidPointConservative, //OP_RW //temp
            double *in, //OP_READ
            double *inConservative, //OP_WRITE //temp
            double *MidPoint) //OP_WRITE
{
  MidPointConservative[0] *= dt;
  MidPointConservative[1] *= dt;
  MidPointConservative[2] *= dt;

  //call to ToConservativeVariables inlined
  inConservative[0] = in[0];
  inConservative[1] = in[0] * (in[1]);
  inConservative[2] = in[0] * (in[2]);
  inConservative[3] = in[3];

  MidPointConservative[0] += inConservative[0];
  MidPointConservative[1] += inConservative[1];
  MidPointConservative[2] += inConservative[2];
  MidPointConservative[3] += inConservative[3];

  //call to ToPhysicalVariables inlined
  double TruncatedH = MidPointConservative[0] < EPS ? EPS : MidPointConservative[0];
  MidPoint[0] = MidPointConservative[0];
  MidPoint[1] = MidPointConservative[1] / TruncatedH;
  MidPoint[2] = MidPointConservative[2] / TruncatedH;
  MidPoint[3] = MidPointConservative[3];
}
