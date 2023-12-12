inline void updateP(const double *r, double *p, const double *beta) {
  *p = (*beta) * (*p) + (*r);
}
