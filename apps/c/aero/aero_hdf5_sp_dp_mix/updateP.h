inline void updateP(const float *r, float *p, const float *beta) {
  *p = (*beta) * (*p) + (*r);
}
