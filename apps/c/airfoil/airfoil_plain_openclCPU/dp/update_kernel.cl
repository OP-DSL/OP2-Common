//#include "op_opencl_reduction.h"
typedef enum { OP_READ, OP_WRITE, OP_RW, OP_INC, OP_MIN, OP_MAX } op_access;
typedef double T;
#define reduction OP_INC
#define WARP_SIZE 32

#define ZERO_double 0.0f

//inline void op_reduction_openclCPU(__global volatile T * dat_g, T dat_l, int set_size) {
//  __local double temp[2048];
//  T   dat_t;
//
//  barrier(CLK_LOCAL_MEM_FENCE);     /* important to finish all previous activity */
//
//  int tid   = get_local_id(0);
//  temp[tid] = dat_l;
//
//  // first, cope with blockDim.x perhaps not being a power of 2
//
//  barrier(CLK_LOCAL_MEM_FENCE);
//
//  int d = 1 << (31 - clz(((int)get_local_size(0)-1)) );
//  // d = blockDim.x/2 rounded up to nearest power of 2
//
//  if ( tid+d < get_local_size(0) ) {
//    dat_t = temp[tid+d];
//
//    switch ( reduction ) {
//      case OP_INC:
//        dat_l = dat_l + dat_t;
//        break;
//      case OP_MIN:
//        if ( dat_t < dat_l ) dat_l = dat_t;
//        break;
//      case OP_MAX:
//        if ( dat_t > dat_l ) dat_l = dat_t;
//        break;
//    }
//
//    temp[tid] = dat_l;
//  }
//
//  // second, do reductions involving more than one warp
//
//  for (d >>= 1 ; d > WARP_SIZE; d >>= 1 ) {
//    barrier(CLK_LOCAL_MEM_FENCE);
//
//    if ( tid < d ) {
//      dat_t = temp[tid+d];
//
//      switch ( reduction ) {
//        case OP_INC:
//          dat_l = dat_l + dat_t;
//          break;
//        case OP_MIN:
//          if ( dat_t < dat_l ) dat_l = dat_t;
//          break;
//        case OP_MAX:
//          if ( dat_t > dat_l ) dat_l = dat_t;
//          break;
//      }
//
//      temp[tid] = dat_l;
//    }
//  }
//
//  // third, do reductions involving just one warp
//
//  //__syncthreads();
//  barrier(CLK_LOCAL_MEM_FENCE);
//
//  if ( tid < WARP_SIZE ) {
//    for ( ; d > 0; d >>= 1 ) {
//      if ( tid < d ) {
//        dat_t = temp[tid+d];
//
//        switch ( reduction ) {
//          case OP_INC:
//            dat_l = dat_l + dat_t;
//            break;
//          case OP_MIN:
//            if ( dat_t < dat_l ) dat_l = dat_t;
//            break;
//          case OP_MAX:
//            if ( dat_t > dat_l ) dat_l = dat_t;
//            break;
//         }
//
//        temp[tid] = dat_l;
//      }
//    }
//
//    // finally, update global reduction variable
//
//    if ( tid == 0 ) {
//      switch ( reduction ) {
//        case OP_INC:
//          *dat_g = *dat_g + dat_l;
//          break;
//        case OP_MIN:
//          if ( dat_l < *dat_g ) *dat_g = dat_l;
//          break;
//        case OP_MAX:
//          if ( dat_l > *dat_g ) *dat_g = dat_l;
//          break;
//      }
//    }
//  }
//}



//  __local T dat_s[2048];
//  dat_s[get_local_id(0)] = dat_l;
//  barrier(CLK_LOCAL_MEM_FENCE);
//
//  const int block   = 8;
////  int   local_index = get_local_id(0) * block;
////  int   global_index = get_global_id(0) * block;
//  T dat_t = ZERO_double;//dat_g[global_index];
////  dat_l  = INFINITY;
////  int upper_bound = (get_global_id(0) + 1) * block;
////  if (upper_bound > set_size) upper_bound = set_size;
//  // reduce step one: inside block
////  while (global_index < upper_bound) {
////    dat_t += dat_s[local_index] ;
////      dat_t = dat_t + dat_l;
////    local_index++;
////    global_index++;
////  }
////  dat_s[get_local_id(0)] = dat_t;
//  int i;
//  for(i=get_local_id(0); i<get_local_size(0); i++) {
//    dat_t += dat_s[i];
//  }
//
//
////  // reduce step two: inside work-group
////  if(get_local_id(0) == 0) {
////  int i;
////  for(i=1; i<get_local_size(0); i++) {
////    dat_s[0] += dat_s[i];
////  } 
//}

  // reduce step three: write out workgroup reduction result
  //dat_g[get_global_id(0)/get_local_size(0)] = get_global_id(0);///get_local_size(0);//get_group_id(0);//] = get_global_offset(0);//get_local_size(0);//dat_s[0];  
//  dat_g[get_group_id(0)+1] = 22;//get_group_id(0);//22.0f;//dat_s[0];  
//if(get_local_id(0) <200)  dat_g[get_local_id(0)] = get_local_id(0);//22.0f;//dat_s[0];  
//printf("get_group_id(0) = %d\n", get_global_id(0));//g[get_group_id(0)]);
//printf("dat_g[0] = %e\n", dat_s[0]);//g[get_group_id(0)]);
//if(get_local_id(0) == 0) *dat_g = dat_s[0];  
 
//  printf("dat_l = %e \n", dat_s[0]);
//  result[get_group_id(0)] = dat_l;
// printf("get_group_id(0) = %d: dat_g[] = %e\n", get_local_id(0),(double)dat_l);//g[get_group_id(0)]);
  //printf("a = %e \n", (double)dat_s[0]);
//}

inline void update(__global const double * restrict qold, __global double * restrict q, __global double * restrict res, __global const double * restrict adt, double *rms){
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

  double arg0_l[4];
  double arg1_l[4];
  double arg2_l[4];
  double arg4_l[1];

  for (int d=0; d<1; d++) {
    arg4_l[d]=ZERO_double;
 }

  // process set elements
  int n=get_global_id(0);
  if(n>=set_size) return;
//  for (int n=get_local_id(0)+get_group_id(0)*get_local_size(0); n<set_size; n+=get_local_size(0)*get_num_groups(0)) {

    update(  &arg0[n * 4],
             &arg1[n * 4],
             &arg2[n * 4],
             &arg3[n * 1],
             &arg4_l[0]);

//  }
//  dat_s[get_local_id(0) % 8] += arg4_l[0];
//  barrier(CLK_LOCAL_MEM_FENCE);
//arg4_l[0] += 1.0f;
//   AtomicAdd(&dat_s[0],1.0f);// ++;//+= arg4_l[0];
  arg4[get_group_id(0)*1*64] += arg4_l[0];//dat_s[0];
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
