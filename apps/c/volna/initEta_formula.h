void initEta_formula(double *coords, double *values, const double *time) {
  double x = coords[0];
  double y = coords[1];
  double t = *time;
  //insert user formula here
  double val = x*x+y*y<1.0;//0.1f* exp(-1.0 * x*x - y*y);
  //printf("values[0] %lf += val %lf @ x %lf y %lf\n", values[0], val, x, y);
  values[0] += val;
}
