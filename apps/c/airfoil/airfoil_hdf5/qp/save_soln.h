inline void save_soln(const long double *q, long double *qold) {
  for (int n = 0; n < 4; n++)
    qold[n] = q[n];
}
