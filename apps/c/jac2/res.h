inline void res(const double *A, const float *u, float *du, const float *beta) {
  *du += (float)((*beta) * (*A) * (*u));
}
