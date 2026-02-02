inline void res_calc(const double *x0, const double *x1, const double *x2, const double *x3, 
                     const double *phim0, const double *phim1, const double *phim2, const double *phim3, 
                     double *K, /*double *Kt,*/ double *res0, double *res1, double *res2, double *res3) {
  double x[4][2], phim[4];
  x[0][0] = x0[0]; x[1][0] = x1[0]; x[2][0] = x2[0]; x[3][0] = x3[0]; 
  x[0][1] = x0[1]; x[1][1] = x1[1]; x[2][1] = x2[1]; x[3][1] = x3[1]; 
  phim[0] = phim0[0]; phim[1] = phim1[0]; phim[2] = phim2[0]; phim[3] = phim3[0]; 

  for (int j = 0; j < 4; j++) {
    for (int k = 0; k < 4; k++) {
      K[j * 4 + k] = 0;
    }
  }
  for (int i = 0; i < 4; i++) { // for each gauss point
    double det_x_xi = 0;
    double N_x[8];

    double a = 0;
    for (int m = 0; m < 4; m++)
      det_x_xi += Ng2_xi[4 * i + 16 + m] * x[m][1];
    for (int m = 0; m < 4; m++)
      N_x[m] = det_x_xi * Ng2_xi[4 * i + m];

    a = 0;
    for (int m = 0; m < 4; m++)
      a += Ng2_xi[4 * i + m] * x[m][0];
    for (int m = 0; m < 4; m++)
      N_x[4 + m] = a * Ng2_xi[4 * i + 16 + m];

    det_x_xi *= a;

    a = 0;
    for (int m = 0; m < 4; m++)
      a += Ng2_xi[4 * i + m] * x[m][1];
    for (int m = 0; m < 4; m++)
      N_x[m] -= a * Ng2_xi[4 * i + 16 + m];

    double b = 0;
    for (int m = 0; m < 4; m++)
      b += Ng2_xi[4 * i + 16 + m] * x[m][0];
    for (int m = 0; m < 4; m++)
      N_x[4 + m] -= b * Ng2_xi[4 * i + m];

    det_x_xi -= a * b;

    for (int j = 0; j < 8; j++)
      N_x[j] /= det_x_xi;

    double wt1 = wtg2[i] * det_x_xi;
    // double wt2 = wtg2[i]*det_x_xi/r;

    double u[2] = {0.0, 0.0};
    for (int j = 0; j < 4; j++) {
      u[0] += N_x[j] * phim[j];
      u[1] += N_x[4 + j] * phim[j];
    }

    double Dk = 1.0 + 0.5 * gm1 * (m2 - (u[0] * u[0] + u[1] * u[1]));
    double rho = pow(Dk, gm1i); // wow this might be problematic -> go to log?
    double rc2 = rho / Dk;

    res0[0] += wt1 * rho * (u[0] * N_x[0] + u[1] * N_x[4 + 0]);
    res1[0] += wt1 * rho * (u[0] * N_x[1] + u[1] * N_x[4 + 1]);
    res2[0] += wt1 * rho * (u[0] * N_x[2] + u[1] * N_x[4 + 2]);
    res3[0] += wt1 * rho * (u[0] * N_x[3] + u[1] * N_x[4 + 3]);

    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 4; k++) {
        K[j * 4 + k] +=
            wt1 * rho * (N_x[j] * N_x[k] + N_x[4 + j] * N_x[4 + k]) -
            wt1 * rc2 * (u[0] * N_x[j] + u[1] * N_x[4 + j]) *
                (u[0] * N_x[k] + u[1] * N_x[4 + k]);
      }
    }
  }
}
