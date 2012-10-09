void initGaussianLandslide( float *center, float *values, float *mesh_xmin, float *A, float *t, float *lx, float *ly, float *v) {
  float x = center[0];
  float y = center[1];
  values[3] = (*mesh_xmin-x)*(x<0.0)-5.0*(x>=0.0)+
      *A*(*t<1./v)*exp(-1.0* *lx* *lx*(x+3.0-*v**t)*(x+3.0-*v**t)-*ly**ly*y*y)
      +*A*(*t>=1./(*v))*exp(-*lx*(x+3.0-1.0)**lx*(x+3.0-1.0)-*ly**ly*y*y);
}
