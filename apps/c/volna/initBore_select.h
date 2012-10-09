void initBore_select(float *values, float *center,
                     float *x0,
                     float *Hl,
                     float *ul,
                     float *vl,
                     float *Hr,
                     float *ur,
                     float *vl) {
  values[0] = center[0] < *x0 ? *Hl : *Hr;
  values[1] = center[0] < *x0 ? *ul : *ur;
  values[2] = center[0] < *x0 ? *vl : *vr;
}
