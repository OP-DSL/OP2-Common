inline void update(double *qold, double *q, double *res, double *adt, double *rms){
  double del, adti;

  adti = 1.0f/(*adt);

  for (int n=0; n<4; n++) {
    del    = adti*res[n];
    q[n]   = qold[n] - del;
    res[n] = 0.0f;
    *rms  += del*del;
  }
}
