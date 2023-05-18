inline void update(const double *qold, double *q, double *res,
                   const double *adt) {
  double del[4], adti;

  adti = 1.0f / (*adt);
  
  #pragma omp simd
  for (int n = 0; n < 4; n++) {
    del[n] = adti * res[n];
    q[n] = qold[n] - del[n];
    res[n] = 0.0f;
    // printf("n=%d qold[n]=%f q[n]=%f  del[n]=%f\n", n, qold[n], q[n],  del[n]);
    //*rms += del * del;
  }
}
