inline void bres_calc(double **x, double **res, int *bound) {
  if (*bound == 2) { //far field
    res[0][0] = 0;
    res[1][0] = 0;
  } else if (*bound == 4) {
    double Dk   = (1.0+0.5*gm1*minf*minf)/(1.0+0.5*gm1*mfan*mfan);
    double rfan = pow(Dk,gm1i);
    double cfan = pow(Dk,0.5);
    double mass = rfan*cfan*mfan;
    for (int i = 0; i<2; i++) {
      double N[] = {Ng1[i], Ng1[2+i]}; //N    = mesh.Ng1   (:,ig).';
      double N_xi[] = {Ng1_xi[i], Ng1_xi[2+i]}; //N_xi = mesh.Ng1_xi(:,ig).';
      double r = N[0]*x[0][1]+N[1]*x[1][1]; //r    = N   *rs;
      double r_xi = N_xi[0]*x[0][1]+N_xi[1]*x[1][1]; //r_xi = N_xi*rs;
      double wt1 = wtg1[i]*r_xi*r; //wt1 = mesh.wtg1(ig)*r_xi*r;
      res[0][0] -= wt1*mass*N[0]; //res = res - wt1*mass*N';
      res[1][0] -= wt1*mass*N[1];
    }
  }
}
