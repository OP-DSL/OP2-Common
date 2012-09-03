inline void res_calc(double *x[2], double *q[4], double *adt[1],double *res[4]) {
  double dx,dy,mu, ri, p1,vol1, p2,vol2, f;

  dx = x[0][0] - x[1][0];
  dy = x[0][1] - x[1][1];

  ri   = 1.0f/q[0][0];
  p1   = gm1*(q[0][3]-0.5f*ri*(q[0][1]*q[0][1]+q[0][2]*q[0][2]));
  vol1 =  ri*(q[0][1]*dy - q[0][2]*dx);

  ri   = 1.0f/q[1][0];
  p2   = gm1*(q[1][3]-0.5f*ri*(q[1][1]*q[1][1]+q[1][2]*q[1][2]));
  vol2 =  ri*(q[1][1]*dy - q[1][2]*dx);

  mu = 0.5f*(adt[0][0]+adt[1][0])*eps;

  f = 0.5f*(vol1* q[0][0]         + vol2* q[1][0]        ) + mu*(q[0][0]-q[1][0]);
  res[0][0] += f;
  res[1][0] -= f;
  f = 0.5f*(vol1* q[0][1] + p1*dy + vol2* q[1][1] + p2*dy) + mu*(q[0][1]-q[1][1]);
  res[0][1] += f;
  res[1][1] -= f;
  f = 0.5f*(vol1* q[0][2] - p1*dx + vol2* q[1][2] - p2*dx) + mu*(q[0][2]-q[1][2]);
  res[0][2] += f;
  res[1][2] -= f;
  f = 0.5f*(vol1*(q[0][3]+p1)     + vol2*(q[1][3]+p2)    ) + mu*(q[0][3]-q[1][3]);
  res[0][3] += f;
  res[1][3] -= f;
}
