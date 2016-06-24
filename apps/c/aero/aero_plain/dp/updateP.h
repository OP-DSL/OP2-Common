inline void updateP(double *r, double *p, const double *beta) {
  *p = (*beta) * (*p) + (*r);
}
