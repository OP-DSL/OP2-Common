void initGaussianLandslide( double *center, double *values, const double *mesh_xmin, const double *A, const double *t, const double *lx, const double *ly, const double *v) {
  double x = center[0];
  double y = center[1];
  values[3] = (*mesh_xmin-x)*(x<0.0)-5.0*(x>=0.0)+
      *A*(*t<1.0/(*v))*exp(-1.0* *lx* *lx*(x+3.0-*v**t)*(x+3.0-*v**t)-*ly**ly*y*y)
      +*A*(*t>=1.0/(*v))*exp(-*lx*(x+3.0-1.0)**lx*(x+3.0-1.0)-*ly**ly*y*y);
}
