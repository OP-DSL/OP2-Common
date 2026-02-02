inline void spMV(double *v0, double *v1, double *v2, double *v3, const double *K, 
                 const double *p0, const double *p1, const double *p2, const double *p3) {
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
  v0[0] += K[0] * p0[0];
  v0[0] += K[1] * p1[0];
  v1[0] += K[1] * p0[0];
  v0[0] += K[2] * p2[0];
  v2[0] += K[2] * p0[0];
  v0[0] += K[3] * p3[0];
  v3[0] += K[3] * p0[0];
  v1[0] += K[4 + 1] * p1[0];
  v1[0] += K[4 + 2] * p2[0];
  v2[0] += K[4 + 2] * p1[0];
  v1[0] += K[4 + 3] * p3[0];
  v3[0] += K[4 + 3] * p1[0];
  v2[0] += K[8 + 2] * p2[0];
  v2[0] += K[8 + 3] * p3[0];
  v3[0] += K[8 + 3] * p2[0];
  v3[0] += K[15] * p3[0];
}
