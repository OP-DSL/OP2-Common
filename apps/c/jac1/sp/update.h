inline void update(const float *r, float *du, float *u, float *u_sum,
                   float *u_max) {
  *u += *du + alpha * (*r);
  *du = 0.0f;
  *u_sum += (*u) * (*u);
  *u_max = MAX(*u_max, *u);
}
