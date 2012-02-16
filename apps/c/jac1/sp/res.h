inline void res(float *A, float *u, float *du, const float *beta){
  *du += (*beta)*(*A)*(*u);
}
