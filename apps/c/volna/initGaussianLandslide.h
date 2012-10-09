void initGaussianLandslide( double *center, double *values, double *mesh_xmin, double *A, double *t, double *lx, double *ly, double *v) {
  double x = center[0];
  double y = center[1];
  values[3] = (*mesh_xmin-x)*(x<0.0)-5.0*(x>=0.0)+
      *A*(*t<1./v)*exp(-1.0* *lx* *lx*(x+3.0-*v**t)*(x+3.0-*v**t)-*ly**ly*y*y)
      +*A*(*t>=1./(*v))*exp(-*lx*(x+3.0-1.0)**lx*(x+3.0-1.0)-*ly**ly*y*y);
}
