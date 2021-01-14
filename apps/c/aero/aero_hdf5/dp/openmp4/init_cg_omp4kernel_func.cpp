//
// auto-generated by op2.py
//

void init_cg_omp4_kernel(
  double *data0,
  int dat0size,
  double *arg1,
  double *data2,
  int dat2size,
  double *data3,
  int dat3size,
  double *data4,
  int dat4size,
  int count,
  int num_teams,
  int nthread){

  double arg1_l = *arg1;
#pragma omp target teams distribute parallel for schedule(static,1)\
     num_teams(num_teams) thread_limit(nthread) map(to:data0[0:dat0size],data2[0:dat2size],data3[0:dat3size],data4[0:dat4size])\
    map(tofrom: arg1_l) reduction(+:arg1_l)
  for ( int n_op=0; n_op<count; n_op++ ){
    //variable mapping
    const double *r = &data0[1*n_op];
    double *c = &arg1_l;
    double *u = &data2[1*n_op];
    double *v = &data3[1*n_op];
    double *p = &data4[1*n_op];

    //inline function
    
    *c += (*r) * (*r);
    *p = *r;
    *u = 0;
    *v = 0;
    //end inline func
  }

  *arg1 = arg1_l;
}
