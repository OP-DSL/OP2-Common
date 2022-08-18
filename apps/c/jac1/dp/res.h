inline void res(const double *A, const double *u, double *du,
                const double *beta, const int *index, const int *idx_ppedge0,
                const int *idx_ppedge1) {
  *du += (*beta) * (*A) * (*u);
  printf("edge %d, nodes %d, %d\n", *index, *idx_ppedge0, *idx_ppedge1);
}
