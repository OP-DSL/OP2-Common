void initEta_formula(float *coords, float *values, float *time) {
  float x = coords[0];
  float y = coords[1];
  float t = *time;
  //insert user formula here
  float val = 0.1f* expf(-1.0f * x*x - y*y);
  values[0] += val;
}
