inline void spMV(double **v, const double *K, const double **p) {
  //     double localsum = 0;
  //  for (int j=0; j<4; j++) {
  //         localsum = 0;
  //         for (int k = 0; k<4; k++) {
  //                 localsum += OP2_STRIDE(K, (j*4+k)] * p[k][0];
  //         }
  //         v[j][0] += localsum;
  //     }
  // }
  //
  //  for (int j=0; j<4; j++) {
  //    v[j][0] += OP2_STRIDE(K, (j*4+j)] * p[j][0];
  //         for (int k = j+1; k<4; k++) {
  //      double mult = OP2_STRIDE(K, (j*4+k)];
  //             v[j][0] += mult * p[k][0];
  //      v[k][0] += mult * p[j][0];
  //         }
  //     }
  // }
  v[0][0] += K[0] * p[0][0];
  v[0][0] += K[1] * p[1][0];
  v[1][0] += K[1] * p[0][0];
  v[0][0] += K[2] * p[2][0];
  v[2][0] += K[2] * p[0][0];
  v[0][0] += K[3] * p[3][0];
  v[3][0] += K[3] * p[0][0];
  v[1][0] += K[4 + 1] * p[1][0];
  v[1][0] += K[4 + 2] * p[2][0];
  v[2][0] += K[4 + 2] * p[1][0];
  v[1][0] += K[4 + 3] * p[3][0];
  v[3][0] += K[4 + 3] * p[1][0];
  v[2][0] += K[8 + 2] * p[2][0];
  v[2][0] += K[8 + 3] * p[3][0];
  v[3][0] += K[8 + 3] * p[2][0];
  v[3][0] += K[15] * p[3][0];
}
