inline void res(double *A, float *u, float *du, const float *beta){
  *du += (*beta)*(*A)*(*u);
}
