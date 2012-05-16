inline void res_calc(double **x, double **phim, double *K, /*double *Kt,*/ double **res) {
  for (int j = 0;j<4;j++) {
    for (int k = 0;k<4;k++) {
      K[stride*(j*4+k)] = 0;
    }
  }
  for (int i = 0; i<4; i++) { //for each gauss point
    //double N[] = {Ng2[i], Ng2[4+i], Ng2[8+i], Ng2[12+i]};
    double N_xi[] = {Ng2_xi[4*i], Ng2_xi[4*i+1], Ng2_xi[4*i+2], Ng2_xi[4*i+3], Ng2_xi[4*i+16], Ng2_xi[4*i+17], Ng2_xi[4*i+18], Ng2_xi[4*i+19]};
    double x_xi[] = {N_xi[0]*x[0][0]+N_xi[1]*x[1][0]+N_xi[2]*x[2][0]+N_xi[3]*x[3][0],
            N_xi[0]*x[0][1]+N_xi[1]*x[1][1]+N_xi[2]*x[2][1]+N_xi[3]*x[3][1],
            N_xi[4]*x[0][0]+N_xi[5]*x[1][0]+N_xi[6]*x[2][0]+N_xi[7]*x[3][0],
            N_xi[4]*x[0][1]+N_xi[5]*x[1][1]+N_xi[6]*x[2][1]+N_xi[7]*x[3][1]};
    double det_x_xi = x_xi[0]*x_xi[3]-x_xi[1]*x_xi[2];
    double N_x[] = {x_xi[3]*N_xi[0]-x_xi[1]*N_xi[4], x_xi[3]*N_xi[1]-x_xi[1]*N_xi[5], x_xi[3]*N_xi[2]-x_xi[1]*N_xi[6], x_xi[3]*N_xi[3]-x_xi[1]*N_xi[7],
             x_xi[0]*N_xi[4]-x_xi[2]*N_xi[0], x_xi[0]*N_xi[5]-x_xi[2]*N_xi[1], x_xi[0]*N_xi[6]-x_xi[2]*N_xi[2], x_xi[0]*N_xi[7]-x_xi[2]*N_xi[3]};
    for (int j = 0;j<8;j++)
      N_x[j] /= det_x_xi;

    double r = 1.0;
    //for (int j = 0;j<4;j++)
    //  r+=N[j]*x[j][1];

    double wt1 = wtg2[i]*det_x_xi*r;
    //double wt2 = wtg2[i]*det_x_xi/r;

    double u[2] = {0.0, 0.0};
    for (int j = 0;j<4;j++) {
      u[0] += N_x[j]*phim[j][0];
      u[1] += N_x[4+j]*phim[j][0];
    }

    double Dk = 1.0 + 0.5*gm1*(m2-(u[0]*u[0]+u[1]*u[1]));
    double rho = pow(Dk,gm1i); //wow this might be problematic -> go to log?
    double rc2 = rho/Dk;

    for (int j = 0;j<4;j++) {
      res[j][0] += wt1*rho*(u[0]*N_x[j] + u[1]*N_x[4+j]);
    }
    for (int j = 0;j<4;j++) {
      for (int k = 0;k<4;k++) {
        K[stride*(j*4+k)] += wt1*rho*(N_x[j]*N_x[k]+N_x[4+j]*N_x[4+k]) - wt1*rc2*(u[0]*N_x[j] + u[1]*N_x[4+j])*(u[0]*N_x[k] + u[1]*N_x[4+k]);
        //Kt[j*4+k] += wt2*rho*N[j]*N[k];
      }
    }
  }
}
