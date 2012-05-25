inline void spMV(double **v, double *K, double **p){
  double localsum = 0;
  for (int j=0; j<4; j++) {
    localsum = 0;
    for (int k = 0; k<4; k++) {
      localsum += K[j*4+k] * p[k][0];
    }
    v[j][0] += localsum;
  }
}
