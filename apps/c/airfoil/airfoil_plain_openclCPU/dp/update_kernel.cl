//#include "op_opencl_reduction.h"
typedef enum { OP_READ, OP_WRITE, OP_RW, OP_INC, OP_MIN, OP_MAX } op_access;
typedef double T;
#define reduction OP_INC
#define WARP_SIZE 32

#define ZERO_double 0.0

inline void update(
		__global const double * restrict qold,
		__global       double * restrict q,
		__global       double * restrict res,
		__global const double * restrict adt,
		               double *rms){
  double del, adti;

  adti = 1.0f/(*adt);

  for (int n=0; n<4; n++) {
    del    = adti*res[n];
    q[n]   = qold[n] - del;
    res[n] = 0.0f;
    *rms  += del*del;
  }
}

// OpenCL kernel function
__kernel void op_opencl_update(
  __global const double * restrict arg0,
  __global       double * restrict arg1,
  __global       double * restrict arg2,
  __global const double * restrict arg3,
  __global       double * restrict arg4,
                 int    set_size) {

  double arg4_l[1];

  for (int d=0; d<1; d++) {
    arg4_l[d]=ZERO_double;
  }

  // process set elements
  int n=get_global_id(0);
  if(n>=set_size) return;

    update(  &arg0[n * 4],
             &arg1[n * 4],
             &arg2[n * 4],
             &arg3[n * 1],
             &arg4_l[0]);

  arg4[get_group_id(0)*1*64] += arg4_l[0];

  // global reductions
//  int d=0;
//  for(d=0; d<1; d++) { 
//    //op_reductionCPU<OP_INC>(&arg4[d+get_group_id(0)*1],arg4_l[d]);
//    //op_reductionCPU<OP_INC,double>(&arg4[d],arg4_l[d]);
//    //op_reduction_openclCPU(&arg4[d+get_group_id(0)*1],arg4_l[d],set_size);
//
//    double sum = ZERO_double;
//    if(get_local_id(0) == 0) {
//      int i;
//      for(i=0; i<8; i++) {
//        sum += dat_s[i];
//      }
//      arg4[d+get_group_id(0)*1] += sum;
//    }
//    
//    //if(get_local_id(0) == 0) 
//    //printf("arg4 = %e \n", arg4[d+get_group_id(0)*1]);
//  }
//      if(get_local_id(0) ==0)
//      printf("arg4 = %e \n", arg4[0+get_group_id(0)*1]);
}
