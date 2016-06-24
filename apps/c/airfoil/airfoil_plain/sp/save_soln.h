inline void save_soln(const float *q, float *qold) {
  for (int n = 0; n < 4; n++)
    qold[n] = q[n];
}
