//#include "res_calc.h"

#define ROUND_UP(bytes) (((bytes) + 15) & ~15)
#define ZERO_float 0.0f

inline void res_calc(__global const float * restrict x1, __global const float * restrict x2, __global const  float * restrict q1, __global const  float * restrict q2,
                      __global const float * restrict adt1, __global const float * restrict adt2, float * restrict res1, float * restrict res2, float gm1, float eps) {

  float dx,dy,mu, ri, p1,vol1, p2,vol2, f;

  dx = x1[0] - x2[0];
  dy = x1[1] - x2[1];

  ri   = 1.0f/q1[0];
  p1   = gm1*(q1[3]-0.5f*ri*(q1[1]*q1[1]+q1[2]*q1[2]));
  vol1 =  ri*(q1[1]*dy - q1[2]*dx);

  ri   = 1.0f/q2[0];
  p2   = gm1*(q2[3]-0.5f*ri*(q2[1]*q2[1]+q2[2]*q2[2]));
  vol2 =  ri*(q2[1]*dy - q2[2]*dx);

  mu = 0.5f*((*adt1)+(*adt2))*eps;

  f = 0.5f*(vol1* q1[0]         + vol2* q2[0]        ) + mu*(q1[0]-q2[0]);
  res1[0] += f;
  res2[0] -= f;
  f = 0.5f*(vol1* q1[1] + p1*dy + vol2* q2[1] + p2*dy) + mu*(q1[1]-q2[1]);
  res1[1] += f;
  res2[1] -= f;
  f = 0.5f*(vol1* q1[2] - p1*dx + vol2* q2[2] - p2*dx) + mu*(q1[2]-q2[2]);
  res1[2] += f;
  res2[2] -= f;
  f = 0.5f*(vol1*(q1[3]+p1)     + vol2*(q2[3]+p2)    ) + mu*(q1[3]-q2[3]);
  res1[3] += f;
  res2[3] -= f;
}

// OpenCL kernel function

//__attribute__((vec_type_hint(float8)))
__kernel void op_opencl_res_calc(
  __global const float* restrict ind_arg0,
  __global const float* restrict ind_arg1,
  __global const float* restrict ind_arg2,
  __global float *ind_arg3,
  __global const int   * restrict ind_arg0_map_data,
  __global const int   * restrict ind_arg1_map_data,
  __global const int   * restrict ind_arg2_map_data,
  __global const int   * restrict ind_arg3_map_data,
           int    block_offset,
  __global const int   * restrict blkmap,
  __global const int   * restrict offset,
  __global const int   * restrict nelems,
  __global const int   * restrict ncolors,
  __global const int   * restrict colors,
           int    nblocks,
           int    set_size,
  __constant float* gm1,
  __constant float* eps) {

  float arg6_l[4];
  float arg7_l[4];
  //__local int    nelems2, ncolor;
  //__local int    nelem, offset_b;
  int    /*nelems2,*/ ncolor;
  int    nelem, offset_b;

  if (get_group_id(0)+get_group_id(1)*get_num_groups(0) >= nblocks) return;

//  if (get_local_id(0) == 0) {
    // get sizes and shift pointers and direct-mapped data
    int blockId = blkmap[get_group_id(0) + get_group_id(1)*get_num_groups(0)  + block_offset];

    nelem    = nelems[blockId];
    offset_b = offset[blockId];

//    nelems2  = get_local_size(0)*(1+(nelem-1)/get_local_size(0));
    ncolor   = ncolors[blockId];
//  }

//  barrier(CLK_LOCAL_MEM_FENCE); // make sure all of above completed

  const int n=get_local_id(0);
//  if(n>=nelem) return;
//  if(get_local_id(0)>=nelem) return;
//  for (int n=get_local_id(0); n<nelems2; n+=get_local_size(0)) {
    
    int col2 = -1;

    int map0idx;
    int map1idx;
    int map2idx;
    int map3idx;

    if (n<nelem) {

      //initialise local variables
      for ( int d=0; d<4; d++ ){
        arg6_l[d] = ZERO_float;
      }
      for ( int d=0; d<4; d++ ){
        arg7_l[d] = ZERO_float;
      }
   

      map0idx = ind_arg0_map_data[n + offset_b + set_size * 0];
      map1idx = ind_arg0_map_data[n + offset_b + set_size * 1];
      map2idx = ind_arg2_map_data[n + offset_b + set_size * 0];
      map3idx = ind_arg2_map_data[n + offset_b + set_size * 1];
    
      // user-supplied kernel call
      res_calc(
        &ind_arg0[2 * map0idx], /* *x1    */
        &ind_arg0[2 * map1idx], /* *x2    */
        &ind_arg1[4 * map2idx], /* *q1    */
        &ind_arg1[4 * map3idx], /* *q2    */
        &ind_arg2[1 * map2idx], /* *adt1  */
        &ind_arg2[1 * map3idx], /* *adt2  */
        arg6_l,                   /* *res1  */
        arg7_l,                   /* *res2  */
        *gm1,
        *eps);

      col2 = colors[n+offset_b];
    } // end if
   
    //store local variables - colored increments
    for ( int col=0; col<ncolor; col++ ){
      if (col2==col) {
        for ( int d=0; d<4; d++ ){
          //AtomicAdd(&ind_arg3[d+map2idx*4], arg6_l[d]);
          //AtomicAdd(&ind_arg3[d+map3idx*4], arg7_l[d]);            
          ind_arg3[d+map2idx*4] += arg6_l[d];
          ind_arg3[d+map3idx*4] += arg7_l[d];
        }
      }
      //barrier(CLK_GLOBAL_MEM_FENCE);//__syncthreads();
    }
//    for ( int col=0; col<ncolor; col++ ){
////      if (col2==col) {
//        for ( int d=0; d<4; d++ ){
//          ind_arg3[d+map2idx*4] += arg6_l[d];
//        }
//        for ( int d=0; d<4; d++ ){
//          ind_arg3[d+map3idx*4] += arg7_l[d];
//        }
////      }
//      barrier(CLK_GLOBAL_MEM_FENCE);//__syncthreads();
//    }
//  } // end for
}




//  float arg6_l[4];
//  float arg7_l[4];
//
//  __global int* __local ind_arg0_map;
//  __local int ind_arg0_size;
//  __global int* __local ind_arg1_map;
//  __local int ind_arg1_size;
//  __global int* __local ind_arg2_map;
//  __local int ind_arg2_size;
//  __global int* __local ind_arg3_map;
//  __local int ind_arg3_size;
//  __local float* __local ind_arg0_s;
//  __local float* __local ind_arg1_s;
//  __local float* __local ind_arg2_s;
//  __local float* __local ind_arg3_s;
//  __local int    nelems2, ncolor;
//  __local int    nelem, offset_b;
//
//
//	if (get_group_id(0)+get_group_id(1)*get_num_groups(0) >= nblocks) return;
//	if (get_local_id(0) == 0) {
//
//    // get sizes and shift pointers and direct-mapped data
//
//  	int blockId = blkmap[get_group_id(0) + get_group_id(1)*get_num_groups(0)  + block_offset];
//
//    nelem    = nelems[blockId];
//		 offset_b = offset[blockId];
//
//    nelems2  = get_local_size(0)*(1+(nelem-1)/get_local_size(0));
////    nelems2  = blockDim.x*(1+(nelem-1)/blockDim.x);
//    ncolor   = ncolors[blockId];
//
//    ind_arg0_size = ind_arg_sizes[0+blockId*4];
//    ind_arg1_size = ind_arg_sizes[1+blockId*4];
//    ind_arg2_size = ind_arg_sizes[2+blockId*4];
//    ind_arg3_size = ind_arg_sizes[3+blockId*4];
//
//    ind_arg0_map = &ind_map[0*set_size] + ind_arg_offs[0+blockId*4];
//    ind_arg1_map = &ind_map[2*set_size] + ind_arg_offs[1+blockId*4];
//    ind_arg2_map = &ind_map[4*set_size] + ind_arg_offs[2+blockId*4];
//    ind_arg3_map = &ind_map[6*set_size] + ind_arg_offs[3+blockId*4];
//
//    // set shared memory pointers
//
//    int nbytes = 0;
//    ind_arg0_s = (__local float *) &shared[nbytes];
//    nbytes    += ROUND_UP(ind_arg0_size*sizeof(float)*2);
//    ind_arg1_s = (__local float *) &shared[nbytes];
//    nbytes    += ROUND_UP(ind_arg1_size*sizeof(float)*4);
//    ind_arg2_s = (__local float *) &shared[nbytes];
//    nbytes    += ROUND_UP(ind_arg2_size*sizeof(float)*1);
//    ind_arg3_s = (__local float *) &shared[nbytes];
//	}
//
//  barrier(CLK_LOCAL_MEM_FENCE); // make sure all of above completed
//
//  // copy indirect datasets into shared memory or zero increment
//
//  for (int n=get_local_id(0); n<ind_arg0_size*2; n+=get_local_size(0))
//    ind_arg0_s[n] = ind_arg0[n%2+ind_arg0_map[n/2]*2];
//
//  for (int n=get_local_id(0); n<ind_arg1_size*4; n+=get_local_size(0))
//    ind_arg1_s[n] = ind_arg1[n%4+ind_arg1_map[n/4]*4];
//
//  for (int n=get_local_id(0); n<ind_arg2_size*1; n+=get_local_size(0))
//    ind_arg2_s[n] = ind_arg2[n%1+ind_arg2_map[n/1]*1];
//
//  for (int n=get_local_id(0); n<ind_arg3_size*4; n+=get_local_size(0))
//    ind_arg3_s[n] = ZERO_float;
//
//  barrier(CLK_LOCAL_MEM_FENCE);
//
//  // process set elements
//
////  int offset2 = get_local_id(0);
////  int stride = get_local_size(0);
////  int count = nelems2 / stride;
// 
//
////  for (int n=0; n<count ; n++) {
//   
//int n = get_local_id(0); 
////  for (int n=get_local_id(0); n<nelems2; n+=get_local_size(0)) {
//  int col2 = -1;
//
////    if (n*stride+offset2<nelem) {
//    if (n<nelem) {
//
//      // initialise local variables
//
//      #pragma unroll(4)
//      for (int d=0; d<4; d++)
//        arg6_l[d] = ZERO_float;
//      #pragma unroll(4)
//      for (int d=0; d<4; d++)
//        arg7_l[d] = ZERO_float;
//
//
//      // user-supplied kernel call
//
////      res_calc(  ind_arg0_s+arg_map[0*set_size+n*stride+offset2+offset_b]*2,
////                 ind_arg0_s+arg_map[1*set_size+n*stride+offset2+offset_b]*2,
////                 ind_arg1_s+arg_map[2*set_size+n*stride+offset2+offset_b]*4,
////                 ind_arg1_s+arg_map[3*set_size+n*stride+offset2+offset_b]*4,
////                 ind_arg2_s+arg_map[4*set_size+n*stride+offset2+offset_b]*1,
////                 ind_arg2_s+arg_map[5*set_size+n*stride+offset2+offset_b]*1,
////                 arg6_l,
////                 arg7_l,
////                 *gm1,
////                 *eps);
//
//      res_calc(  ind_arg0_s+arg_map[0*set_size+n+offset_b]*2,
//                 ind_arg0_s+arg_map[1*set_size+n+offset_b]*2,
//                 ind_arg1_s+arg_map[2*set_size+n+offset_b]*4,
//                 ind_arg1_s+arg_map[3*set_size+n+offset_b]*4,
//                 ind_arg2_s+arg_map[4*set_size+n+offset_b]*1,
//                 ind_arg2_s+arg_map[5*set_size+n+offset_b]*1,
//                 arg6_l,
//                 arg7_l,
//                *gm1,
//                 *eps);
//
////       __local float *x1 = ind_arg0_s+arg_map[0*set_size+n+offset_b]*2;
////       __local float *x2 = ind_arg0_s+arg_map[1*set_size+n+offset_b]*2;
////       __local float *q1 = ind_arg1_s+arg_map[2*set_size+n+offset_b]*4;
////       __local float *q2 = ind_arg1_s+arg_map[3*set_size+n+offset_b]*4;
////       __local float *adt1 = ind_arg2_s+arg_map[4*set_size+n+offset_b]*1;
////       __local float *adt2 = ind_arg2_s+arg_map[5*set_size+n+offset_b]*1;
////       float *res1 = arg6_l;
////       float *res2 = arg7_l;
////
////       float dx,dy,mu, ri, p1,vol1, p2,vol2, f;
////     
////       dx = x1[0] - x2[0];
////       dy = x1[1] - x2[1];
////     
////       ri   = 1.0f/q1[0];
////       p1   = (*gm1)*(q1[3]-0.5f*ri*(q1[1]*q1[1]+q1[2]*q1[2]));
////       vol1 =  ri*(q1[1]*dy - q1[2]*dx);
////     
////       ri   = 1.0f/q2[0];
////       p2   = (*gm1)*(q2[3]-0.5f*ri*(q2[1]*q2[1]+q2[2]*q2[2]));
////       vol2 =  ri*(q2[1]*dy - q2[2]*dx);
////     
////       mu = 0.5f*((*adt1)+(*adt2))*(*eps);
////     
////       f = 0.5f*(vol1* q1[0]         + vol2* q2[0]        ) + mu*(q1[0]-q2[0]);
////       res1[0] += f;
////       res2[0] -= f;
////       f = 0.5f*(vol1* q1[1] + p1*dy + vol2* q2[1] + p2*dy) + mu*(q1[1]-q2[1]);
////       res1[1] += f;
////       res2[1] -= f;
////       f = 0.5f*(vol1* q1[2] - p1*dx + vol2* q2[2] - p2*dx) + mu*(q1[2]-q2[2]);
////       res1[2] += f;
////       res2[2] -= f;
////       f = 0.5f*(vol1*(q1[3]+p1)     + vol2*(q2[3]+p2)    ) + mu*(q1[3]-q2[3]);
////       res1[3] += f;
////       res2[3] -= f;
//
////  col2 = colors[n*stride+offset2+offset_b];
//    	col2 = colors[n+offset_b];
//    }
//
//    // store local variables
//
//    int arg6_map;
//    int arg7_map;
//
//    if (col2>=0) {
////      arg6_map = convert_int(arg_map[6*set_size+n+offset_b]);
////      arg7_map = convert_int(arg_map[7*set_size+n+offset_b]);
////      arg6_map = (int) arg_map[6*set_size+n+offset_b];
////      arg7_map = (int) arg_map[7*set_size+n+offset_b];
//      arg6_map = arg_map[6*set_size+n+offset_b];
//      arg7_map = arg_map[7*set_size+n+offset_b];
////      arg6_map = arg_map[6*set_size+n+offset_b];
////      arg7_map = 1;
////      arg7_map = arg_map[7*set_size+n+offset_b];
//    }
//
//    for (int col=0; col<ncolor; col++) {
//      if (col2==col) {
//        for (int d=0; d<4; d++)
//            ind_arg3_s[d+arg6_map*4] += arg6_l[d];
//        for (int d=0; d<4; d++)
//            ind_arg3_s[d+arg7_map*4] += arg7_l[d];
//      }
//      barrier(CLK_LOCAL_MEM_FENCE);
//    }
//
////  }
//
//  // apply pointered write/increment
//
//  for (int n=get_local_id(0); n<ind_arg3_size*4; n+=get_local_size(0))
//    ind_arg3[n%4+ind_arg3_map[n/4]*4] += ind_arg3_s[n];
//
//
////IACA_END
//
//
//}



////#include "res_calc.h"
//
//#define ROUND_UP(bytes) (((bytes) + 15) & ~15)
//#define ZERO_float 0.0f
//
//inline void res_calc(__local float *x1, __local float *x2, __local  float *q1, __local  float *q2,
//                      __local float *adt1, __local float *adt2, float *res1, float *res2, float gm1, float eps) {
//
//  float dx,dy,mu, ri, p1,vol1, p2,vol2, f;
//
//  dx = x1[0] - x2[0];
//  dy = x1[1] - x2[1];
//
//  ri   = 1.0f/q1[0];
//  p1   = gm1*(q1[3]-0.5f*ri*(q1[1]*q1[1]+q1[2]*q1[2]));
//  vol1 =  ri*(q1[1]*dy - q1[2]*dx);
//
//  ri   = 1.0f/q2[0];
//  p2   = gm1*(q2[3]-0.5f*ri*(q2[1]*q2[1]+q2[2]*q2[2]));
//  vol2 =  ri*(q2[1]*dy - q2[2]*dx);
//
//  mu = 0.5f*((*adt1)+(*adt2))*eps;
//
//  f = 0.5f*(vol1* q1[0]         + vol2* q2[0]        ) + mu*(q1[0]-q2[0]);
//  res1[0] += f;
//  res2[0] -= f;
//  f = 0.5f*(vol1* q1[1] + p1*dy + vol2* q2[1] + p2*dy) + mu*(q1[1]-q2[1]);
//  res1[1] += f;
//  res2[1] -= f;
//  f = 0.5f*(vol1* q1[2] - p1*dx + vol2* q2[2] - p2*dx) + mu*(q1[2]-q2[2]);
//  res1[2] += f;
//  res2[2] -= f;
//  f = 0.5f*(vol1*(q1[3]+p1)     + vol2*(q2[3]+p2)    ) + mu*(q1[3]-q2[3]);
//  res1[3] += f;
//  res2[3] -= f;
//}
//
//
//// OpenCL kernel function
//
//__kernel void op_opencl_res_calc(
//  __global float *ind_arg0,
//  __global float *ind_arg1,
//  __global float *ind_arg2,
//  __global float *ind_arg3,
//  __global int   *ind_map,
//  __global int *arg_map,
//  __global int   *ind_arg_sizes,
//  __global int   *ind_arg_offs,
//  int    block_offset,
//  __global int   *blkmap,
//  __global int   *offset,
//  __global int   *nelems,
//  __global int   *ncolors,
//  __global int   *colors,
//  int   nblocks,
//  int   set_size,
//  __local char*  shared,
//  __constant float* gm1,
//  __constant float* eps) {
//
//  float arg6_l[4];
//  float arg7_l[4];
//
//  __global int* __local ind_arg0_map;
//  __local int ind_arg0_size;
//  __global int* __local ind_arg1_map;
//  __local int ind_arg1_size;
//  __global int* __local ind_arg2_map;
//  __local int ind_arg2_size;
//  __global int* __local ind_arg3_map;
//  __local int ind_arg3_size;
//  __local float* __local ind_arg0_s;
//  __local float* __local ind_arg1_s;
//  __local float* __local ind_arg2_s;
//  __local float* __local ind_arg3_s;
//  __local int    nelems2, ncolor;
//  __local int    nelem, offset_b;
//
//
//  if (get_group_id(0)+get_group_id(1)*get_num_groups(0) >= nblocks) return;
//  if (get_local_id(0) == 0) {
//
//    // get sizes and shift pointers and direct-mapped data
//
//    int blockId = blkmap[get_group_id(0) + get_group_id(1)*get_num_groups(0)  + block_offset];
//
//    nelem    = nelems[blockId];
//    offset_b = offset[blockId];
//
//    nelems2  = get_local_size(0)*(1+(nelem-1)/get_local_size(0));
////    nelems2  = blockDim.x*(1+(nelem-1)/blockDim.x);
//    ncolor   = ncolors[blockId];
//
//    ind_arg0_size = ind_arg_sizes[0+blockId*4];
//    ind_arg1_size = ind_arg_sizes[1+blockId*4];
//    ind_arg2_size = ind_arg_sizes[2+blockId*4];
//    ind_arg3_size = ind_arg_sizes[3+blockId*4];
//
//    ind_arg0_map = &ind_map[0*set_size] + ind_arg_offs[0+blockId*4];
//    ind_arg1_map = &ind_map[2*set_size] + ind_arg_offs[1+blockId*4];
//    ind_arg2_map = &ind_map[4*set_size] + ind_arg_offs[2+blockId*4];
//    ind_arg3_map = &ind_map[6*set_size] + ind_arg_offs[3+blockId*4];
//
//    // set shared memory pointers
//
//    int nbytes = 0;
//    ind_arg0_s = (__local float *) &shared[nbytes];
//    nbytes    += ROUND_UP(ind_arg0_size*sizeof(float)*2);
//    ind_arg1_s = (__local float *) &shared[nbytes];
//    nbytes    += ROUND_UP(ind_arg1_size*sizeof(float)*4);
//    ind_arg2_s = (__local float *) &shared[nbytes];
//    nbytes    += ROUND_UP(ind_arg2_size*sizeof(float)*1);
//    ind_arg3_s = (__local float *) &shared[nbytes];
//  }
//
//  barrier(CLK_LOCAL_MEM_FENCE); // make sure all of above completed
//
//  // copy indirect datasets into shared memory or zero increment
//
//  for (int n=get_local_id(0); n<ind_arg0_size*2; n+=get_local_size(0))
//    ind_arg0_s[n] = ind_arg0[n%2+ind_arg0_map[n/2]*2];
//
//  for (int n=get_local_id(0); n<ind_arg1_size*4; n+=get_local_size(0))
//    ind_arg1_s[n] = ind_arg1[n%4+ind_arg1_map[n/4]*4];
//
//  for (int n=get_local_id(0); n<ind_arg2_size*1; n+=get_local_size(0))
//    ind_arg2_s[n] = ind_arg2[n%1+ind_arg2_map[n/1]*1];
//
//  for (int n=get_local_id(0); n<ind_arg3_size*4; n+=get_local_size(0))
//    ind_arg3_s[n] = ZERO_float;
//
//  barrier(CLK_LOCAL_MEM_FENCE);
//
//  // process set elements
//
//  for (int n=get_local_id(0); n<nelems2; n+=get_local_size(0)) {
//    int col2 = -1;
//
//    if (n<nelem) {
//
//      // initialise local variables
//
//      for (int d=0; d<4; d++)
//        arg6_l[d] = ZERO_float;
//      for (int d=0; d<4; d++)
//        arg7_l[d] = ZERO_float;
//
//
//
//
//      // user-supplied kernel call
//
//
//      res_calc(  ind_arg0_s+arg_map[0*set_size+n+offset_b]*2,
//                 ind_arg0_s+arg_map[1*set_size+n+offset_b]*2,
//                 ind_arg1_s+arg_map[2*set_size+n+offset_b]*4,
//                 ind_arg1_s+arg_map[3*set_size+n+offset_b]*4,
//                 ind_arg2_s+arg_map[4*set_size+n+offset_b]*1,
//                 ind_arg2_s+arg_map[5*set_size+n+offset_b]*1,
//                 arg6_l,
//                 arg7_l,
//                 *gm1,
//                 *eps);
//
//      col2 = colors[n+offset_b];
//    }
//
//    // store local variables
//
//      int arg6_map;
//      int arg7_map;
//
//      if (col2>=0) {
//        arg6_map = arg_map[6*set_size+n+offset_b];
//        arg7_map = arg_map[7*set_size+n+offset_b];
//      }
//
//    for (int col=0; col<ncolor; col++) {
//      if (col2==col) {
//        for (int d=0; d<4; d++)
//          ind_arg3_s[d+arg6_map*4] += arg6_l[d];
//        for (int d=0; d<4; d++)
//          ind_arg3_s[d+arg7_map*4] += arg7_l[d];
//      }
//      barrier(CLK_LOCAL_MEM_FENCE);
//    }
//
//  }
//
//  // apply pointered write/increment
//
//  for (int n=get_local_id(0); n<ind_arg3_size*4; n+=get_local_size(0))
//    ind_arg3[n%4+ind_arg3_map[n/4]*4] += ind_arg3_s[n];
//
//}
