inline void spMV(double **v, double *K, double **p){
  //     double localsum = 0;
  //  for (int j=0; j<4; j++) {
  //         localsum = 0;
  //         for (int k = 0; k<4; k++) {
  //                 localsum += K[op2_stride*(j*4+k)] * p[k][0];
  //         }
  //         v[j][0] += localsum;
  //     }
  // }
  //
  //  for (int j=0; j<4; j++) {
  //    v[j][0] += K[op2_stride*(j*4+j)] * p[j][0];
  //         for (int k = j+1; k<4; k++) {
  //      double mult = K[op2_stride*(j*4+k)];
  //             v[j][0] += mult * p[k][0];
  //      v[k][0] += mult * p[j][0];
  //         }
  //     }
  // }
  v[0][0] += K[op2_stride*0] * p[0][0];
  v[0][0] += K[op2_stride*1] * p[1][0];
  v[1][0] += K[op2_stride*1] * p[0][0];
  v[0][0] += K[op2_stride*2] * p[2][0];
  v[2][0] += K[op2_stride*2] * p[0][0];
  v[0][0] += K[op2_stride*3] * p[3][0];
  v[3][0] += K[op2_stride*3] * p[0][0];
  v[1][0] += K[op2_stride*(4+1)] * p[1][0];
  v[1][0] += K[op2_stride*(4+2)] * p[2][0];
  v[2][0] += K[op2_stride*(4+2)] * p[1][0];
  v[1][0] += K[op2_stride*(4+3)] * p[3][0];
  v[3][0] += K[op2_stride*(4+3)] * p[1][0];
  v[2][0] += K[op2_stride*(8+2)] * p[2][0];
  v[2][0] += K[op2_stride*(8+3)] * p[3][0];
  v[3][0] += K[op2_stride*(8+3)] * p[2][0];
  v[3][0] += K[op2_stride*(15)] * p[3][0];
}
