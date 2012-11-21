void initBathymetry_formula(float *coords, float *values, const float *time) {
  float x = coords[0];
  float y = coords[1];
  float t = *time;
  //insert user formula here
  float val = .2*(-5.0-x)*(x<0)-(x>=0)+.2*(t<1)*exp(-(x+3.0-2.0*t)*(x+3.0-2.0*t)-y*y)+.2*(t>=1)*exp(-(x+1.0)*(x+1.0)-y*y);///*...*/0.0;
  values[3] = val;
}
