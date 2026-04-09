inline void update(const double *qold, double *q, double *res,
                   const double *adt, double *rms, double *maxerr, int *idx, int *errloc) {
  double del, adti;

  adti = 1.0f / (*adt);

  for (int n = 0; n < 4; n++) {
    del = adti * res[n];
    q[n] = qold[n] - del;
    res[n] = 0.0f;
    double sqdel = del * del;
    *rms += sqdel;

    if (sqdel > *maxerr) {
      *maxerr = sqdel;
      // *errloc = *idx; 
      // Uncomment above when op_arg_info is fully supported
      // As of now, seq through templates and GPU versions do not support op_arg_info
    }
  }
}