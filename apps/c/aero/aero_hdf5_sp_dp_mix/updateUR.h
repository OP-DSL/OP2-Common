inline void updateUR(float *u, float *r, const float *p, float *v,
                     const float *alpha) {
  *u += (*alpha) * (*p);
  *r -= (*alpha) * (*v);
  *v = 0.0f;
}
