//outlined elemental kernel - update
inline void update(const double *qold, double *q, double *res,
                   const double *adt, double *rms) {
  double del, adti;

  adti = 1.0f / (*adt);

  for (int n = 0; n < 4; n++) {
    del = adti * res[n];
    q[n] = qold[n] - del;
    res[n] = 0.0f;
    *rms += del * del;
  }
}
