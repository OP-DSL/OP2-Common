inline void update(double *phim, double *res, double *u, double *rms) {
  *phim -= *u;
  *res = 0.0;
  *rms += (*u) * (*u);
}
