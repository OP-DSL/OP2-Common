inline void update(float *phim, float *res, const float *u, float *rms) {
  *phim -= *u;
  *res = 0.0f;
  *rms += (*u) * (*u);
}
