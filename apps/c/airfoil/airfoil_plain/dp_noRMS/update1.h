inline void update1(const double *qold, double *q, double *res,
                   const double *adt, double *rms) {
  double del[4], adti;

  adti = 1.0f / (*adt);

  #pragma omp simd
  for (int n = 0; n < 4; n++) {
    //del = adti * res[n];
    del[n] = qold[n] - q[n];
    //res[n] = 0.0f;
    //*rms += del * del;
  }
  
  for (int n = 0; n < 4; n++) {
    *rms += del[n]*del[n];
  }
}
