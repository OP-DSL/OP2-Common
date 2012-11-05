inline void initEta_formula(float *coords, float *values, const float *time) {
  float x = coords[0];
  float y = coords[1];
  float t = *time;
  //insert user formula here
  //float val = x*x+y*y<1.0;//0.1f* exp(-1.0 * x*x - y*y);
  float val = 0.5f*exp(-x*x-y*y);
  //printf("values[0] %lf += val %lf @ x %lf y %lf\n", values[0], val, x, y);
  values[0] += val;
}
