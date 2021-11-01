inline void updateUR(double *u, double *r, const double *p, double *v,
                     const double *alpha) {
  *u += (*alpha) * (*p);
  *r -= (*alpha) * (*v);
  *v = 0.0f;
}
