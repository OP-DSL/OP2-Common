inline void save_soln(const double *q, double *qold) {
  for (int n = 0; n < 4; n++)
    qold[n] = q[n];
}
