inline void update(const double *r, double *du, double *u, double *u_sum,
                   double *u_max) {
  *u += *du + alpha * (*r);
  *du = 0.0f;
  *u_sum += (*u) * (*u);
  *u_max = MAX(*u_max, *u);
}
