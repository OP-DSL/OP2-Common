inline void bres_calc(const float *x1, const float *x2, const float *q1,
                      const float *adt1, float *res1, const int *bound) {
  float dx, dy, mu, ri, p1, vol1, p2, vol2, f;

  dx = x1[0] - x2[0];
  dy = x1[1] - x2[1];

  ri = 1.0f / q1[0];
  p1 = gm1 * (q1[3] - 0.5f * ri * (q1[1] * q1[1] + q1[2] * q1[2]));

  vol1 = ri * (q1[1] * dy - q1[2] * dx);

  ri = 1.0f / qinf[0];
  p2 = gm1 * (qinf[3] - 0.5f * ri * (qinf[1] * qinf[1] + qinf[2] * qinf[2]));
  vol2 = ri * (qinf[1] * dy - qinf[2] * dx);

  mu = (*adt1) * eps;

  f = 0.5f * (vol1 * q1[0] + vol2 * qinf[0]) + mu * (q1[0] - qinf[0]);
  res1[0] += *bound == 1 ? 0.0f : f;
  f = 0.5f * (vol1 * q1[1] + p1 * dy + vol2 * qinf[1] + p2 * dy) +
      mu * (q1[1] - qinf[1]);
  res1[1] += *bound == 1 ? p1 * dy : f;
  f = 0.5f * (vol1 * q1[2] - p1 * dx + vol2 * qinf[2] - p2 * dx) +
      mu * (q1[2] - qinf[2]);
  res1[2] += *bound == 1 ? -p1 * dx : f;
  f = 0.5f * (vol1 * (q1[3] + p1) + vol2 * (qinf[3] + p2)) +
      mu * (q1[3] - qinf[3]);
  res1[3] += *bound == 1 ? 0.0f : f;
}
