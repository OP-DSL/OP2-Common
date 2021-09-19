inline void res_calc1(double *q1,
                     double *q2,
                     double *res1, double *res2) {

  double r1[1], r2[1];                     
  for(int i = 0; i < 1; i++){
      r1[i] = q1[i];
      r2[i] = q2[i];
      res1[i] += r1[i] + 1;
      res2[i] += r2[i] - 1;
  }
}
