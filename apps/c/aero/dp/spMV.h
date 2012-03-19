inline void spMV(double **v, double *K, double **p, int stride){
  //     double localsum = 0;
  //  for (int j=0; j<4; j++) {
  //         localsum = 0;
  //         for (int k = 0; k<4; k++) {
  //                 localsum += K[stride*(j*4+k)] * p[k][0];
  //         }
  //         v[j][0] += localsum;
  //     }
  // }
  //
  //  for (int j=0; j<4; j++) {
  //    v[j][0] += K[stride*(j*4+j)] * p[j][0];
  //         for (int k = j+1; k<4; k++) {
  //      double mult = K[stride*(j*4+k)];
  //             v[j][0] += mult * p[k][0];
  //      v[k][0] += mult * p[j][0];
  //         }
  //     }
  // }
  v[0][0] += K[stride*0] * p[0][0];
  v[0][0] += K[stride*1] * p[1][0];
  v[1][0] += K[stride*1] * p[0][0];
  v[0][0] += K[stride*2] * p[2][0];
  v[2][0] += K[stride*2] * p[0][0];
  v[0][0] += K[stride*3] * p[3][0];
  v[3][0] += K[stride*3] * p[0][0];
  v[1][0] += K[stride*(4+1)] * p[1][0];
  v[1][0] += K[stride*(4+2)] * p[2][0];
  v[2][0] += K[stride*(4+2)] * p[1][0];
  v[1][0] += K[stride*(4+3)] * p[3][0];
  v[3][0] += K[stride*(4+3)] * p[1][0];
  v[2][0] += K[stride*(8+2)] * p[2][0];
  v[2][0] += K[stride*(8+3)] * p[3][0];
  v[3][0] += K[stride*(8+3)] * p[2][0];
  v[3][0] += K[stride*(15)] * p[3][0];
}
