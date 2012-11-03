inline void initBore_select(float *values, float *center,
                     const float *x0,
                     const float *Hl,
                     const float *ul,
                     const float *vl,
                     const float *Hr,
                     const float *ur,
                     const float *vr) {
  values[0] = center[0] < *x0 ? *Hl : *Hr;
  values[1] = center[0] < *x0 ? *ul : *ur;
  values[2] = center[0] < *x0 ? *vl : *vr;
}
