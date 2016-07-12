inline void update(const float *qold, float *q, float *res, const float *adt,
                   float *rms) {
  float del, adti;

  adti = 1.0f / (*adt);

  for (int n = 0; n < 4; n++) {
    del = adti * res[n];
    q[n] = qold[n] - del;
    res[n] = 0.0f;
    *rms += del * del;
  }
}
