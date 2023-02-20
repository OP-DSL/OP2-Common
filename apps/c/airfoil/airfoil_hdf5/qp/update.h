inline void update(const long double *qold, long double *q, long double *res,
                   const long double *adt, long double *rms) {
  long double del, adti, rmsl;

  rmsl = 0.0L;
  adti = 1.0L / (*adt);

  for (int n = 0; n < 4; n++) {
    del = adti * res[n];
    q[n] = qold[n] - del;
    res[n] = 0.0L;
    rmsl += del * del;
    //*rms=MAX(*rms,abs(del));
  }
  *rms += rmsl;
}
