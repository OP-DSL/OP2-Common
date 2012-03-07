inline void res(double *A, float *u, float *du, const float *beta){
  *du += (float)((*beta)*(*A)*(*u));
}

