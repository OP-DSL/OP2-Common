inline void initBore_select(double *values, double *center,
                     const double *x0,
                     const double *Hl,
                     const double *ul,
                     const double *vl,
                     const double *Hr,
                     const double *ur,
                     const double *vr) {
  values[0] = center[0] < *x0 ? *Hl : *Hr;
  values[1] = center[0] < *x0 ? *ul : *ur;
  values[2] = center[0] < *x0 ? *vl : *vr;
}
