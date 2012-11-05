// user function definition in separate header file
#include "save_soln.h"

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
__kernel void op_opencl_save_soln(__global float *arg0,
    __global float *arg1,
    int   offset_s,
    int   set_size,
    __local char* shared) {
  
  float arg0_l[4];
  float arg1_l[4];
  int   tid = get_local_id(0) % OP_WARPSIZE;
//  int   tid = threadIdx.x%OP_WARPSIZE;

//  __local char shared[];
//  extern __shared__ char shared[];

  __local char *arg_s = shared + offset_s*(get_local_id(0) / OP_WARPSIZE);
//  char *arg_s = shared + offset_s*(threadIdx.x/OP_WARPSIZE);

  // process set elements

  for (int n=get_global_id(0);
      n<set_size; n+=get_local_size(0)*get_num_groups(0)) {

    int offset = n - tid;
    int nelems = MIN(OP_WARPSIZE,set_size-offset);

    // copy data into shared memory, then into local
    for (int m=0; m<4; m++)
      ((__local float *)arg_s)[tid+m*nelems] = arg0[tid+m*nelems+offset*4];

    for (int m=0; m<4; m++)
      arg0_l[m] = ((__local float *)arg_s)[m+tid*4];

    // user-supplied kernel call
    for (int m=0; m<4; m++) arg1_l[m] = arg0_l[m];
    save_soln(  arg0_l,
        arg1_l );

    // copy back into shared memory, then to device

    for (int m=0; m<4; m++)
      ((__local float *)arg_s)[m+tid*4] = arg1_l[m];

    for (int m=0; m<4; m++)
      arg1[tid+m*nelems+offset*4] = ((__local float *)arg_s)[tid+m*nelems];

  }

}
