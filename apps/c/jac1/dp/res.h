inline void res(const double *A, const double *u, double *du,
                const double *beta) {
  *du += (*beta) * (*A) * (*u);
}
