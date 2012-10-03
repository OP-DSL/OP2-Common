void EvolveValuesRK2_1(double dT, double *MidPointConservative_H, double *MidPointConservative_U, double *MidPointConservative_v, double *MidPointConservative_Zb, //OP_RW //temp
                       double *in_H, double *in_U, double *in_V,  double *in_Zb, //OP_READ
                       double *inConservative_H, double *inConservative_U, double *inConservative_V, double *inConservative_Zb, //OP_WRITE //temp
                       double *MidPoint_H, double *MidPoint_U, double *MidPoint_V, double *MidPoint_Zb) //OP_WRITE
{
    *MidPointConservative_H *= dt;
    *MidPointConservative_U *= dt;
    *MidPointConservative_V *= dt;

    //call to ToConservativeVariables  inlined
    *inConservative_H = *in_H;
    *inConservative_U = *in_H * (*in_U);
    *inConservative_V = *in_H * (*in_V);
    *inConservative_Zb = *in_Zb;

    *MidPointConservative_H += *inConservative_H;
    *MidPointConservative_U += *inConservative_U;
    *MidPointConservative_V += *inConservative_V;
    *MidPointConservative_Zb += *inConservative_Zb;

    //call to ToPhysicalVariables inlined
    double TruncatedH = *MidPointConservative_H < EPS ? EPS : *MidPointConservative_H;
    *MidPoint_H = *MidPointConservative_H;
    *MidPoint_U = *MidPointConservative_U / TruncatedH;
    *MidPoint_V = *MidPointConservative_V / TruncatedH;
    *MidPoint_Zb = *MidPointConservative_Zb;
}
