// user function definition in separate header file
//#include "save_soln.h"
inline void save_soln(double *q, double *qold){
  for (int n=0; n<4; n++) qold[n] = q[n];
}

///* Header to make Clang compatible with OpenCL */
//#define __global __attribute__((address_space(1)))
//
//int get_global_id(int index);
//int get_local_id(int index);
//int get_local_size(int index);
//int get_num_groups(int index);
//
//#define OP_WARPSIZE 32


/*
 * min / max definitions
 */

#ifndef MIN
#define MIN(a,b) ((a<b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a,b) ((a>b) ? (a) : (b))
#endif

// OpenCL kernel function
__kernel void op_opencl_save_soln(
    __global const double * restrict arg0,
    __global       double * restrict arg1,
    int   set_size) {
  
  double arg0_l[4];
  double arg1_l[4];
  //double4 arg0_l;
  //double4 arg1_l;
  int   tid = get_local_id(0) % OP_WARPSIZE;
//  int   tid = threadIdx.x%OP_WARPSIZE;

//  __local char shared[];
//  extern __shared__ char shared[];

  //__local char *arg_s = shared + offset_s*(get_local_id(0) / OP_WARPSIZE);

  // process set elements

  for (int n=get_global_id(0); n<set_size; n+=get_local_size(0)*get_num_groups(0)) {

    int offset = n - tid;
    int nelems = MIN(OP_WARPSIZE,set_size-offset);

    // copy data into shared memory, then into local
    //#pragma unroll(4)
    for (int m=0; m<4; m++)
      arg0_l[m] = arg0[tid+m*nelems+offset*4];

    //#pragma unroll(4)
    //for (int m=0; m<4; m++)
    //  ((__local double *)arg_s)[tid+m*nelems] = arg0[tid+m*nelems+offset*4];

    //#pragma unroll(4)
    //for (int m=0; m<4; m++)
    //  arg0_l[m] = ((__local double *)arg_s)[m+tid*4];

    // user-supplied kernel call
    //#pragma unroll(4)
    //for (int m=0; m<4; m++) arg1_l[m] = arg0_l[m];

    save_soln(  arg0_l,
        arg1_l );
    //arg1_l = arg0_l;

    // copy back into shared memory, then to device
    //#pragma unroll(4)
    for (int m=0; m<4; m++)
      arg1[tid+m*nelems+offset*4] = arg1_l[m];

    //#pragma unroll(4)
    //for (int m=0; m<4; m++)
    //  ((__local double *)arg_s)[m+tid*4] = arg1_l[m];

    //#pragma unroll(4)
    //for (int m=0; m<4; m++)
    //  arg1[tid+m*nelems+offset*4] = ((__local double *)arg_s)[tid+m*nelems];

  }

}
