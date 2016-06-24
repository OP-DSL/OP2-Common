inline void res(const float *A, const float *u, float *du, const float *beta) {
  *du += (*beta) * (*A) * (*u);
}
