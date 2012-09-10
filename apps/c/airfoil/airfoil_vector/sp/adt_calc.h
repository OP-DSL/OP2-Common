inline void adt_calc(float *x[2], float q[4], float * adt){
  float dx,dy, ri,u,v,c;

  ri =  1.0f/q[0];
  u  =   ri*q[1];
  v  =   ri*q[2];
  c  = sqrt(gam*gm1*(ri*q[3]-0.5f*(u*u+v*v)));

  dx = x[1][0] - x[0][0];
  dy = x[1][1] - x[0][1];
  *adt  = fabs(u*dy-v*dx) + c*sqrt(dx*dx+dy*dy);

  dx = x[2][0] - x[1][0];
  dy = x[2][1] - x[1][1];
  *adt += fabs(u*dy-v*dx) + c*sqrt(dx*dx+dy*dy);

  dx = x[3][0] - x[2][0];
  dy = x[3][1] - x[2][1];
  *adt += fabs(u*dy-v*dx) + c*sqrt(dx*dx+dy*dy);

  dx = x[0][0] - x[3][0];
  dy = x[0][1] - x[3][1];
  *adt += fabs(u*dy-v*dx) + c*sqrt(dx*dx+dy*dy);

  *adt = (*adt) / cfl;
}
