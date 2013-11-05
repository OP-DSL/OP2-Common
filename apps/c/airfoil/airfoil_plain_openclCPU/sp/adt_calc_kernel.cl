// OpenCL kernel function
//#include "adt_calc.h"

//extern __constant float gam;
//extern __constant float gm1;
//extern __constant float cfl;

//void adt_calc(__local float *x1, __local float *x2, __local float *x3, __local float *x4, __global float *q, __global float *adt,
void adt_calc(__global const float * restrict x1, 
              __global const float * restrict x2, 
              __global const float * restrict x3, 
              __global const float * restrict x4, 
              __global const float * restrict q, 
              __global float * restrict adt, 
              float gam, 
              float gm1, 
              float cfl){

  float dx,dy, ri,u,v,c;

  ri =  1.0f/q[0];
  u  =   ri*q[1];
  v  =   ri*q[2];
  c  = sqrt(gam*gm1*(ri*q[3]-0.5f*(u*u+v*v)));

  dx = x2[0] - x1[0];
  dy = x2[1] - x1[1];
  *adt = fabs(u*dy-v*dx) + c*sqrt(dx*dx+dy*dy);

  dx = x3[0] - x2[0];
  dy = x3[1] - x2[1];
  *adt += fabs(u*dy-v*dx) + c*sqrt(dx*dx+dy*dy);

  dx = x4[0] - x3[0];
  dy = x4[1] - x3[1];
  *adt += fabs(u*dy-v*dx) + c*sqrt(dx*dx+dy*dy);

  dx = x1[0] - x4[0];
  dy = x1[1] - x4[1];
  *adt += fabs(u*dy-v*dx) + c*sqrt(dx*dx+dy*dy);

  *adt = (*adt) / cfl;
}

__kernel void op_opencl_adt_calc(
  __global const float * restrict ind_arg0, /* x */
  __global const int   * restrict ind_arg0_map_data,
  __global const float * restrict arg4,
  __global       float * restrict arg5, /* adt */
           const int              block_offset,
  __global const int   * restrict blkmap,
  __global const int   * restrict offset,
  __global const int   * restrict nelems,
  __global const int   * restrict ncolors,
  __global const int   * restrict colors,
           int    nblocks,
           int    set_size,
  __constant float *gam,
  __constant float *gm1,
  __constant float *cfl) {


  //__local  int       nelem, offset_b;
  int       nelem, offset_b;

  //int blockId;
  if (get_group_id(0)+get_group_id(1)*get_num_groups(0) >= nblocks) return;

//  if (get_local_id(0) == 0) {

    // get sizes and shift pointers and direct-mapped data

    int blockId = blkmap[get_group_id(0) + get_group_id(1)*get_num_groups(0)  + block_offset];

    nelem    = nelems[blockId];
    offset_b = offset[blockId];

    //ind_arg0_size = ind_arg_sizes[0+blockId*1];
    //ind_arg0_map = &ind_map[0*set_size] + ind_arg_offs[0+blockId*1];

    // set shared memory pointers

    //int nbytes = 0;
    //ind_arg0_s = (__local float *) &shared[nbytes];
//  }

//  arg5[get_global_id(0)+offset_b] = nelem;
//  barrier(CLK_GLOBAL_MEM_FENCE);

// barrier(CLK_LOCAL_MEM_FENCE);

//
//  // copy indirect datasets into shared memory or zero increment
//

//  for (int n=get_local_id(0); n<ind_arg0_size*2; n+=get_local_size(0))
//    ind_arg0_s[n] = ind_arg0[n%2+ind_arg0_map[n/2]*2];
//  barrier(CLK_GLOBAL_MEM_FENCE);

//
//  // process set elements
//

  int map0idx;
  int map1idx;
  int map2idx;
  int map3idx;
  const int n=get_local_id(0);
  
  if(n>=nelem) return;
  
//  for (int n=get_local_id(0); n<nelem; n+=get_local_size(0)) {
    map0idx = ind_arg0_map_data[n + offset_b + set_size * 0];
    map1idx = ind_arg0_map_data[n + offset_b + set_size * 1];
    map2idx = ind_arg0_map_data[n + offset_b + set_size * 2];
    map3idx = ind_arg0_map_data[n + offset_b + set_size * 3];
      // user-supplied kernel call
      adt_calc(  &ind_arg0[2 * map0idx], /* x1  */
                 &ind_arg0[2 * map1idx], /* x2  */
                 &ind_arg0[2 * map2idx], /* x3  */
                 &ind_arg0[2 * map3idx], /* x4  */
                 &arg4[(n+offset_b)*4],  /* q   */
                 &arg5[(n+offset_b)*1],  /* adt */
                 *gam,
                 *gm1,
                 *cfl);
   
      //__local float *x1 = ind_arg0_s+arg_map[0*set_size+n+offset_b]*2;
      //__local float *x2 = ind_arg0_s+arg_map[1*set_size+n+offset_b]*2;
      //__local float *x3 = ind_arg0_s+arg_map[2*set_size+n+offset_b]*2;
      //__local float *x4 = ind_arg0_s+arg_map[3*set_size+n+offset_b]*2;
      //__global float *q = arg4+(n+offset_b)*4;
      //__global float *adt = arg5+(n+offset_b)*1;


      // float dx,dy, ri,u,v,c;

      // ri =  1.0f/q[0];
      // u  =   ri*q[1];
      // v  =   ri*q[2];
      // c  = sqrt((*gam)*(*gm1)*(ri*q[3]-0.5f*(u*u+v*v)));

      // dx = x2[0] - x1[0];
      // dy = x2[1] - x1[1];
      // *adt = fabs(u*dy-v*dx) + c*sqrt(dx*dx+dy*dy);

      // dx = x3[0] - x2[0];
      // dy = x3[1] - x2[1];
      // *adt += fabs(u*dy-v*dx) + c*sqrt(dx*dx+dy*dy);

      // dx = x4[0] - x3[0];
      // dy = x4[1] - x3[1];
      // *adt += fabs(u*dy-v*dx) + c*sqrt(dx*dx+dy*dy);

      // dx = x1[0] - x4[0];
      // dy = x1[1] - x4[1];
      // *adt += fabs(u*dy-v*dx) + c*sqrt(dx*dx+dy*dy);

      // *adt = (*adt) / (*cfl);
//  } // end for
}

//  __global int* __local ind_arg0_map;
//
//  __local int ind_arg0_size;
//
//  __local float* __local ind_arg0_s;
//
//  __local int    nelem, offset_b;
//
//  int blockId;
//  if (get_group_id(0)+get_group_id(1)*get_num_groups(0) >= nblocks) return;
//
//  if (get_local_id(0) == 0) {
//
//    // get sizes and shift pointers and direct-mapped data
//
//    blockId = blkmap[get_group_id(0) + get_group_id(1)*get_num_groups(0)  + block_offset];
//
//    nelem    = nelems[blockId];
//    offset_b = offset[blockId];
//
//    ind_arg0_size = ind_arg_sizes[0+blockId*1];
//
//    ind_arg0_map = &ind_map[0*set_size] + ind_arg_offs[0+blockId*1];
//
//    // set shared memory pointers
//
//    int nbytes = 0;
//    ind_arg0_s = (__local float *) &shared[nbytes];
//  }
//
////  arg5[get_global_id(0)+offset_b] = nelem;
////  barrier(CLK_GLOBAL_MEM_FENCE);
//
//  barrier(CLK_LOCAL_MEM_FENCE);
//
//
////  ind_arg0_s[get_local_id(0)] = get_global_id(0);
////
////  barrier(CLK_LOCAL_MEM_FENCE);
////  if(get_local_id(0)>10)
////    arg5[get_global_id(0)] = (float)ind_arg0_s[get_local_id(0)-10];
////
////  return;
//
////  arg5[get_global_id(0)] = (float)get_global_id(0);
////  if(ind_arg0_s == 0) {
////    arg5[get_global_id(0)] = 0.1;
////    return;
////  }
////  return;
////  int n = 0;
////int n = ind_arg0_size*2-1;
////  ind_arg0_s[get_local_id(0)] = ind_arg0[n%2+ind_arg0_map[n/2]*2];
////
////  // copy indirect datasets into shared memory or zero increment
////
//
//
//  for (int n=get_local_id(0); n<ind_arg0_size*2; n+=get_local_size(0))
//
//    ind_arg0_s[n] = ind_arg0[n%2+ind_arg0_map[n/2]*2];
//
//  barrier(CLK_GLOBAL_MEM_FENCE);
//
//
//
//  ////  __syncthreads();
////
////  // process set elements
////
//
//
//
////  __private int a = (int)ind_arg0_map[0];
////  arg5[get_global_id(0)] = (float)a;
//
////  int n=get_local_id(0);
//  for (int n=get_local_id(0); n<nelem; n+=get_local_size(0)) {
//      // user-supplied kernel call
//   //   adt_calc(  ind_arg0_s+arg_map[0*set_size+n+offset_b]*2,
//   //              ind_arg0_s+arg_map[1*set_size+n+offset_b]*2,
//   //              ind_arg0_s+arg_map[2*set_size+n+offset_b]*2,
//   //              ind_arg0_s+arg_map[3*set_size+n+offset_b]*2,
//   //              arg4+(n+offset_b)*4,
//   //              arg5+(n+offset_b)*1,
//   //              *gam,
//   //              *gm1,
//   //              *cfl);
//   
//      __local float *x1 = ind_arg0_s+arg_map[0*set_size+n+offset_b]*2;
//      __local float *x2 = ind_arg0_s+arg_map[1*set_size+n+offset_b]*2;
//      __local float *x3 = ind_arg0_s+arg_map[2*set_size+n+offset_b]*2;
//      __local float *x4 = ind_arg0_s+arg_map[3*set_size+n+offset_b]*2;
//      __global float *q = arg4+(n+offset_b)*4;
//      __global float *adt = arg5+(n+offset_b)*1;
//
//
//       float dx,dy, ri,u,v,c;
//
//       ri =  1.0f/q[0];
//       u  =   ri*q[1];
//       v  =   ri*q[2];
//       c  = sqrt((*gam)*(*gm1)*(ri*q[3]-0.5f*(u*u+v*v)));
//
//       dx = x2[0] - x1[0];
//       dy = x2[1] - x1[1];
//       *adt = fabs(u*dy-v*dx) + c*sqrt(dx*dx+dy*dy);
//
//       dx = x3[0] - x2[0];
//       dy = x3[1] - x2[1];
//       *adt += fabs(u*dy-v*dx) + c*sqrt(dx*dx+dy*dy);
//
//       dx = x4[0] - x3[0];
//       dy = x4[1] - x3[1];
//       *adt += fabs(u*dy-v*dx) + c*sqrt(dx*dx+dy*dy);
//
//       dx = x1[0] - x4[0];
//       dy = x1[1] - x4[1];
//       *adt += fabs(u*dy-v*dx) + c*sqrt(dx*dx+dy*dy);
//
//       *adt = (*adt) / (*cfl);
//
////      arg5[(n+offset_b)*1] = (float) 1.1;
////      printf("abc\n");
////      arg5[(n+offset_b)*1] = (int) ind_arg0_s+arg_map[0*set_size+n+offset_b]*2;
//  }
////  arg5[get_global_id(0)+offset_b] = nelem;
//}
