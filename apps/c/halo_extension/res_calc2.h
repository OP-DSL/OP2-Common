inline void res_calc2(double *res1, double *res2, 
                      double *node1,
                      double *node2) {
  for(int i = 0; i < 2; i++){
      node1[i] += res1[0] + 1;
      node2[i] += res2[0] - 1;
  }
}