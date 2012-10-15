void initBathymetry_update(double *values, int *firstTime) {
  if (*firstTime)
    values[0] -= values[3];

  values[0] = values[0] < EPS ? EPS : values[0];
}
