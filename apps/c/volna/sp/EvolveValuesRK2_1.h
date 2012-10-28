inline void EvolveValuesRK2_1(const float *dT, float *midPointConservative, //OP_RW //temp
            float *in, //OP_READ
            float *inConservative, //OP_WRITE //temp
            float *midPoint) //OP_WRITE
{
  midPointConservative[0] *= *dT;
  midPointConservative[1] *= *dT;
  midPointConservative[2] *= *dT;

  //call to ToConservativeVariables inlined
  inConservative[0] = in[0];
  inConservative[1] = in[0] * (in[1]);
  inConservative[2] = in[0] * (in[2]);
  inConservative[3] = in[3];

  midPointConservative[0] += inConservative[0];
  midPointConservative[1] += inConservative[1];
  midPointConservative[2] += inConservative[2];
  midPointConservative[3] += inConservative[3];

  //call to ToPhysicalVariables inlined
  float TruncatedH = midPointConservative[0] < EPS ? EPS : midPointConservative[0];
  midPoint[0] = midPointConservative[0];
  midPoint[1] = midPointConservative[1] / TruncatedH;
  midPoint[2] = midPointConservative[2] / TruncatedH;
  midPoint[3] = midPointConservative[3];
}
