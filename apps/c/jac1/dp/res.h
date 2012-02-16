inline void res(double *A, double *u, double *du, const double *beta){
  *du += (*beta)*(*A)*(*u);
}
