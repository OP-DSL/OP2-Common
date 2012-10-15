void initBore_select(double *values, double *center,
                     double *x0,
                     double *Hl,
                     double *ul,
                     double *vl,
                     double *Hr,
                     double *ur,
                     double *vr) {
  values[0] = center[0] < *x0 ? *Hl : *Hr;
  values[1] = center[0] < *x0 ? *ul : *ur;
  values[2] = center[0] < *x0 ? *vl : *vr;
}
