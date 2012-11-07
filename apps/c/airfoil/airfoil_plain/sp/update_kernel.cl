inline void update(float *qold, float *q, float *res, __global float *adt, float *rms){
  float del, adti;

  adti = 1.0f/(*adt);

  for (int n=0; n<4; n++) {
    del    = adti*res[n];
    q[n]   = qold[n] - del;
    res[n] = 0.0f;
    *rms  += del*del;
  }
}

#define OP_WARPSIZE 32

// OpenCL kernel function

__kernel void op_opencl_update(
  __global float *arg0,
  __global float *arg1,
  __global float *arg2,
  __global float *arg3,
  __global float *arg4,
  int   offset_s,
  int   set_size,
  __local char*  shared ) {

  float arg0_l[4];
  float arg1_l[4];
  float arg2_l[4];
  float arg4_l[1];
  for (int d=0; d<1; d++) arg4_l[d]=ZERO_float;
  int   tid = get_local_id(0)%OP_WARPSIZE;
//  int   tid = threadIdx.x%OP_WARPSIZE;

  __local char* arg_s = shared + offset_s*(get_local_id(0)/OP_WARPSIZE);
//  char *arg_s = shared + offset_s*(threadIdx.x/OP_WARPSIZE);

  // process set elements

  for (int n=get_local_id(0)+get_group_id(0)*get_local_size(0);
       n<set_size; n+=get_local_size(0)*get_num_groups(0)) {
//  for (int n=threadIdx.x+blockIdx.x*blockDim.x;
//       n<set_size; n+=blockDim.x*gridDim.x) {

    int offset = n - tid;
    int nelems = MIN(OP_WARPSIZE,set_size-offset);

    // copy data into shared memory, then into local

    for (int m=0; m<4; m++)
      ((__local float *)arg_s)[tid+m*nelems] = arg0[tid+m*nelems+offset*4];

    for (int m=0; m<4; m++)
      arg0_l[m] = ((__local float *)arg_s)[m+tid*4];

    for (int m=0; m<4; m++)
      ((__local float *)arg_s)[tid+m*nelems] = arg2[tid+m*nelems+offset*4];

    for (int m=0; m<4; m++)
      arg2_l[m] = ((__local float *)arg_s)[m+tid*4];


    // user-supplied kernel call


    update(  arg0_l,
             arg1_l,
             arg2_l,
             arg3+n,
             arg4_l );

    // copy back into shared memory, then to device

    for (int m=0; m<4; m++)
      ((__local float *)arg_s)[m+tid*4] = arg1_l[m];

    for (int m=0; m<4; m++)
      arg1[tid+m*nelems+offset*4] = ((__local float *)arg_s)[tid+m*nelems];

    for (int m=0; m<4; m++)
      ((__local float *)arg_s)[m+tid*4] = arg2_l[m];

    for (int m=0; m<4; m++)
      arg2[tid+m*nelems+offset*4] = ((__local float *)arg_s)[tid+m*nelems];

  }

  // global reductions

//  for(int d=0; d<1; d++)
//    op_reduction_OP_INC(&arg4[d+blockIdx.x*1],arg4_l[d]);
}
